"""
ModelWithGradiend: Abstract base class for combining a base model with a GRADIEND model.

Generic interface across modalities (text, vision, ...). Owns:
- base_model (frozen or trainable; modality-specific)
- gradiend (encoder/decoder over gradients)
- interpretation metadata (source/target)
- glue logic: create_gradients -> encode / rewrite_base_model
- persistence of gradiend_context.json + feature_class_encoding_direction.json
"""

from __future__ import annotations

import os
import copy
import json
import csv
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Iterator, List, Optional, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import Parameter

from gradiend.util import unwrap_model
from gradiend.util.logging import get_logger
from gradiend.model import ParamMappedGradiendModel
from gradiend.model.core import build_gradiend_from_base_model
from gradiend.model._source_target import validate_source_target
from gradiend.model.utils import (
    get_hf_device_map,
    resolve_device_config_for_model,
    read_gradiend_context,
    write_gradiend_context,
)

logger = get_logger(__name__)

def effective_rewrite_learning_rate(learning_rate: float, source: str) -> float:
    """
    Learning rate applied in :meth:`ModelWithGradiend.rewrite_base_model`.

    CONTRACT (do not change without explicit design review):
    - Return the nominal ``learning_rate`` unchanged for every ``source``.
    - Rewrite orientation is **only** controlled by ``feature_factor`` passed to the decoder,
      not by negating LR for ``source="alternative"`` or any other source.
    """
    validate_source_target("source", source)
    return float(learning_rate)


def _gradiend_checkpoint_diagnostic(load_directory: str) -> str:
    if not load_directory or not isinstance(load_directory, str):
        return "path is not a string"
    if not os.path.isdir(load_directory):
        return "path is not an existing directory"
    cfg_path = os.path.join(load_directory, "config.json")
    if not os.path.isfile(cfg_path):
        return "config.json is missing"
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as exc:
        return f"config.json is not valid JSON ({exc})"
    except OSError as exc:
        return f"config.json could not be read ({exc})"
    if not isinstance(cfg, dict):
        return f"config.json is a {type(cfg).__name__}, expected object"
    if "architecture" not in cfg:
        return f"config.json keys={sorted(cfg.keys())}"
    return "ok"


def _is_gradiend_checkpoint(load_directory: str) -> bool:
    """
    True if load_directory is a local GRADIEND checkpoint (config.json with 'architecture' key).
    False for HF model IDs, decoder_mlm_head, or other base-model paths.
    """
    if not load_directory or not isinstance(load_directory, str):
        return False
    if not os.path.isdir(load_directory):
        return False
    cfg_path = os.path.join(load_directory, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        return "architecture" in cfg
    except (json.JSONDecodeError, OSError):
        return False


# Paths we have already logged a non-convergent warning for (avoid duplicate logs when same model is loaded multiple times)
_convergence_warning_logged: set = set()


def _check_convergence_warning(model_path: str) -> None:
    """
    Check if a model did not converge during training and log a warning.
    
    First checks training.json for convergence_info. If not found, falls back to
    checking seed_report.json for multi-seed runs. Warns at most once per path per process.
    
    Args:
        model_path: Path to the model directory being loaded.
    """
    norm_path = os.path.normpath(os.path.abspath(model_path))
    if norm_path in _convergence_warning_logged:
        return
    # First, try to load convergence info from training.json
    training_json_path = os.path.join(model_path, "training.json")
    if os.path.isfile(training_json_path):
        try:
            with open(training_json_path, "r") as f:
                training_data = json.load(f)
            
            convergence_info = training_data.get("convergence_info")
            if convergence_info is not None:
                converged = convergence_info.get("converged", True)
                convergent_count = convergence_info.get("convergent_count")
                min_convergent_seeds = convergence_info.get("min_convergent_seeds")
                convergence_metric = convergence_info.get("convergence_metric", "correlation")
                threshold = convergence_info.get("threshold")
                
                # Convergence is OK if at least one convergent run was present
                # Only warn if convergent_count is 0 (or None) and min_convergent_seeds requires convergence
                actual_count = convergent_count if convergent_count is not None else 0
                if actual_count == 0 and min_convergent_seeds is not None and min_convergent_seeds > 0:
                    _convergence_warning_logged.add(norm_path)
                    logger.warning(
                        "Loading model from non-convergent training: "
                        "converged=False (convergent_count=%s, required: %s) for metric=%s threshold=%.4f. ",
                        actual_count,
                        min_convergent_seeds,
                        convergence_metric,
                        threshold if threshold is not None else 0.0,
                    )
                return
        except Exception as e:
            logger.debug("Could not check convergence status from training.json: %s", e)
    
    # Fallback: check seed_report.json for multi-seed runs
    possible_paths = [
        os.path.join(model_path, "seeds", "seed_report.json"),  # model_path is experiment_dir
        os.path.join(model_path, "..", "seeds", "seed_report.json"),  # model_path is a seed dir
        os.path.join(os.path.dirname(model_path), "seeds", "seed_report.json"),  # model_path is output_dir, seeds is sibling
    ]
    
    # Also check if model_path itself is in a seeds directory
    if "seeds" in model_path:
        parts = model_path.split(os.sep)
        seeds_idx = None
        for i, part in enumerate(parts):
            if part == "seeds":
                seeds_idx = i
                break
        if seeds_idx is not None:
            # Reconstruct path up to seeds, then add seed_report.json
            seeds_dir = os.sep.join(parts[:seeds_idx + 1])
            possible_paths.append(os.path.join(seeds_dir, "seed_report.json"))
    
    seed_report_path = None
    for path in possible_paths:
        normalized = os.path.normpath(path)
        if os.path.isfile(normalized):
            seed_report_path = normalized
            break
    
    if seed_report_path is None:
        # No convergence info found - could be old checkpoint or convergence not tracked
        return
    
    try:
        with open(seed_report_path, "r") as f:
            report = json.load(f)
        
        convergent_count = report.get("convergent_count", 0)
        convergence_metric = report.get("convergence_metric", "correlation")
        threshold = report.get("threshold")
        min_convergent_seeds = report.get("min_convergent_seeds")
        
        if convergent_count == 0 and min_convergent_seeds is not None and min_convergent_seeds > 0:
            _convergence_warning_logged.add(norm_path)
            logger.warning(
                "Loading model from non-convergent multi-seed training: "
                "convergent_count=0 (required: %s) for metric=%s threshold=%.4f. ",
                min_convergent_seeds,
                convergence_metric,
                threshold if threshold is not None else 0.0,
            )
    except Exception as e:
        # Silently ignore errors reading seed_report.json (could be corrupted, etc.)
        logger.debug("Could not check convergence status from %s: %s", seed_report_path, e)


def _normalize_param_name(name: str) -> str:
    # tolerate wrappers like DDP/compiled modules
    prefixes = ("module.", "_orig_mod.")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if name.startswith(p):
                name = name[len(p):]
                changed = True
    return name

def _build_param_lookup_named(model: torch.nn.Module):
    # exact + normalized keys, plus unambiguous suffix aliases for cases where
    # training wrappers expose backbone-local names but evaluation rewrites the
    # original HF model with an extra container prefix such as "model.".
    lookup = {}
    suffix_candidates = {}
    for n, p in model.named_parameters():
        lookup[n] = p
        normalized = _normalize_param_name(n)
        lookup.setdefault(normalized, p)
        parts = normalized.split(".")
        for i in range(1, len(parts) - 1):
            suffix = ".".join(parts[i:])
            suffix_candidates.setdefault(suffix, []).append(p)

    for suffix, params in suffix_candidates.items():
        unique = []
        for p in params:
            if not any(p is existing for existing in unique):
                unique.append(p)
        if len(unique) == 1:
            lookup.setdefault(suffix, unique[0])
    return lookup

def _first_param_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


class ModelWithGradiend(nn.Module, ABC):
    """
    Abstract base class that combines a base model (neural network) with a GRADIEND model.

    The GRADIEND model holds encoder/decoder weights. This adapter:
    - interprets GRADIEND IO (source/target),
    - defines how gradients are created (create_gradients),
    - provides encode() and rewrite_base_model(),
    - persists adapter-level config next to the GRADIEND checkpoint.

    Important refactor invariant:
    - self.gradiend.param_map is a dict mapping each base parameter name to a param-spec:
        {"shape": tuple[int,...], "repr": "all"|"mask"|"indices", ("mask": BoolTensor), ("indices": LongTensor)}
      Construction-time normalization happens here (adapter), since shapes come from base_model.

    Subclasses must implement:
    - create_gradients(...)
    - _save_model(...)
    - _load_model(...)
    - _create_gradiend(...)
    """

    def __init__(
        self,
        base_model,
        gradiend: ParamMappedGradiendModel,
        *,
        base_model_device=None,
        device_encoder=None,
        device_decoder=None,
        base_model_device_map=None,
        base_model_max_memory=None,
        gradient_creator=None,
        source: str = "factual",
        target: str = "diff",
    ):
        validate_source_target("source", source)
        validate_source_target("target", target)

        super().__init__()
        self.base_model = base_model
        self.gradiend = gradiend
        self._source = source
        self._target = target

        self._gradient_creator = gradient_creator or self


        self.base_model_device_map = base_model_device_map or get_hf_device_map(base_model)
        self.base_model_max_memory = base_model_max_memory
        self.base_model_is_sharded = self.base_model_device_map not in (None, False)

        if gradiend.encoder is not None:
            self.base_model_device = base_model_device or gradiend.encoder[0].linear.weight.device
        else:
            self.base_model_device = base_model_device or gradiend.device_encoder
        if self.base_model_is_sharded:
            self.base_model_device = None
        else:
            self.base_model.to(self.base_model_device)
        self.gradiend.to(device_encoder=device_encoder, device_decoder=device_decoder)

        # Stable parameter map for shapes + updates
        self.param_lookup = _build_param_lookup_named(self.base_model)

        self._ensure_gradiend_param_map_spec()
        self._sync_base_requires_grad_to_param_map()

        self._enhancer_mask_cache = {}
        self.feature_class_encoding_direction: Optional[Dict[str, int]] = None
        self._base_gradient_lock = threading.RLock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_base_gradient_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._base_gradient_lock = threading.RLock()

    @contextmanager
    def exclusive_base_gradient_access(self):
        """Serialize base-model tokenization, forward, and backward across threads."""
        self._base_gradient_lock.acquire()
        try:
            yield
        finally:
            self._base_gradient_lock.release()

    def to(self, device: object = None, *, torch_dtype: Optional[torch.dtype] = None) -> "ModelWithGradiend":
        """Move base_model and gradiend to the given device and/or GRADIEND torch_dtype."""
        dev = torch.device(device) if isinstance(device, str) else device
        move_base = not getattr(self, "base_model_is_sharded", False)
        if dev is not None and torch_dtype is not None:
            if move_base:
                self.base_model.to(device=dev, dtype=torch_dtype)
                self.base_model_device = dev
            self.gradiend.to(dev, torch_dtype=torch_dtype)
        elif dev is not None:
            if move_base:
                self.base_model.to(dev)
                self.base_model_device = dev
            self.gradiend.to(dev)
        elif torch_dtype is not None:
            if move_base:
                self.base_model.to(dtype=torch_dtype)
            self.gradiend.to(torch_dtype=torch_dtype)
        return self

    def cpu(self) -> "ModelWithGradiend":
        """Move base_model and gradiend to CPU."""
        return self.to("cpu")

    def cuda(self, device: Optional[object] = None) -> "ModelWithGradiend":
        """Move base_model and gradiend to CUDA. device: None (default cuda), int (cuda:N), or str/torch.device."""
        if device is None:
            return self.to("cuda")
        if isinstance(device, int):
            return self.to(f"cuda:{device}")
        return self.to(device)

    def place_for_evaluation(
        self,
        *,
        device: Optional[Union[str, torch.device]] = None,
        device_encoder: Optional[Union[str, torch.device]] = None,
        device_decoder: Optional[Union[str, torch.device]] = None,
        device_base_model: Optional[Union[str, torch.device]] = None,
        encoder_decoder_same_device: bool = False,
    ) -> "ModelWithGradiend":
        """
        Place base model and GRADIEND on evaluation devices.

        Used when an explicit ``device`` is passed to evaluation methods, or when
        you call :meth:`cuda` / :meth:`to` yourself. Fresh :meth:`from_pretrained`
        loads already use :func:`~gradiend.model.utils.resolve_device_config_for_model`
        (CUDA when available). This does not run automatically when a cached
        in-memory model was moved to CPU.
        """
        if device is not None:
            dev = torch.device(device) if isinstance(device, str) else device
            if dev.type == "cpu":
                return self.cpu()

        device_encoder, device_decoder, device_base_model, _ = resolve_device_config_for_model(
            device=device,
            device_encoder=device_encoder,
            device_decoder=device_decoder,
            device_base_model=device_base_model,
            encoder_decoder_same_device=encoder_decoder_same_device,
        )

        if not getattr(self, "base_model_is_sharded", False):
            self.base_model.to(device_base_model)
            self.base_model_device = device_base_model
        self.gradiend.to(device_encoder=device_encoder, device_decoder=device_decoder)
        return self

    # ----------------------------
    # Construction-time normalization
    # ----------------------------
    def _ensure_gradiend_param_map_spec(self) -> None:
        """
        Validate gradiend.param_map is in spec-dict format.

        param_map must be a dict of per-parameter specs including "shape" and "repr".
        """
        g = self.gradiend
        param_map = getattr(g, "param_map", None)

        if isinstance(param_map, dict):
            # Optionally enrich missing shapes (but don't change repr)
            changed = False
            for name, spec in param_map.items():
                if not isinstance(spec, dict):
                    raise TypeError(f"gradiend.param_map[{name!r}] must be a dict spec, got {type(spec)}")
                if "shape" not in spec:
                    spec["shape"] = tuple(self.param_lookup[name].shape)
                    changed = True
                if "repr" not in spec:
                    raise ValueError(f"gradiend.param_map[{name!r}] missing required key 'repr'")
            if changed:
                g.param_map = param_map
            # input_dim should already match, but don't silently recompute (bugs should surface)
            return

        raise TypeError(f"gradiend.param_map must be dict-spec, got {type(param_map)}")

    def _active_param_map_names(self) -> set:
        active = set()
        for name, spec in self.gradiend.param_map.items():
            r = spec.get("repr")
            if r == "all":
                shape = spec.get("shape")
                if shape is None or int(torch.tensor(tuple(shape)).prod().item()) > 0:
                    active.add(_normalize_param_name(name))
            elif r == "mask":
                mask = spec.get("mask")
                if torch.is_tensor(mask) and bool(mask.any().item()):
                    active.add(_normalize_param_name(name))
            elif r == "indices":
                indices = spec.get("indices")
                if torch.is_tensor(indices) and int(indices.numel()) > 0:
                    active.add(_normalize_param_name(name))
            else:
                raise ValueError(f"Unknown gradiend.param_map repr {r!r} for {name!r}")
        return active

    def _sync_base_requires_grad_to_param_map(self) -> None:
        """Require base gradients only for parameters represented by the current param_map."""
        active_names = self._active_param_map_names()
        base = self._get_base_forward_model()
        changed = 0
        trainable = 0
        total = 0
        for name, param in base.named_parameters():
            total += 1
            should_train = _normalize_param_name(name) in active_names
            if bool(param.requires_grad) != should_train:
                param.requires_grad = should_train
                changed += 1
            if should_train:
                trainable += 1
        logger.debug(
            "Synced base requires_grad to GRADIEND param_map: trainable_params=%s/%s changed=%s.",
            trainable,
            total,
            changed,
        )

    def __str__(self) -> str:
        g = self.gradiend
        g_summary = f"input_dim={g.input_dim}, latent_dim={g.latent_dim}" if g else "gradiend=None"
        return f"ModelWithGradiend(source={self._source!r}, target={self._target!r}, gradiend({g_summary}))"

    # ----------------------------
    # Metadata helpers
    # ----------------------------
    def set_feature_class_encoding_direction(self, class_labels: Dict[str, Any]) -> None:
        """
        Set feature_class_encoding_direction from configuration (class_labels). Set-once.

        Direction is taken directly from class_labels: +1, -1, or 0 (neutral).
        """
        if self.feature_class_encoding_direction is not None:
            return
        if not class_labels:
            logger.warning("No class_labels provided for feature_class_encoding_direction")
            return

        direction = {}
        for class_name, expected_label in class_labels.items():
            val = expected_label if isinstance(expected_label, (int, float)) else float(expected_label)
            direction[class_name] = 1 if val > 0 else (-1 if val < 0 else 0)

        self.feature_class_encoding_direction = direction
        logger.info(f"Set feature_class_encoding_direction: {direction}")

    def get_weight_importance(self, part: str = "decoder-weight") -> "torch.Tensor":
        """
        Return per-input-dimension importance from GRADIEND weights.

        Args:
            part: Which component to use for importance aggregation:
                - "encoder-weight": L1 over encoder weight columns
                - "decoder-weight": L1 over decoder weight rows
                - "decoder-bias": absolute decoder bias
                - "decoder-sum": absolute(sum(weight_row) + bias)

        Returns:
            1D CPU float tensor of length input_dim, where higher means more
            influential according to the chosen aggregation.
        """
        return self.gradiend.get_weight_importance(part=part)

    def get_topk_weights(self, part: str = "decoder-weight", topk: int = 1000) -> List[int]:
        """
        Return top-k base-global indices by importance.

        Args:
            part: Importance source passed to get_weight_importance.
                Options: "encoder-weight", "decoder-weight", "decoder-bias", "decoder-sum".
            topk: Number of indices to return (clipped to input_dim) or a proportion in (0, 1].

        Returns:
            List of base-global input indices (length k) sorted by descending importance (base-global
            index means the index in the flattened input space corresponding to the base model parameters,
            not the GRADIEND input space, such that differently pruned GRADIEND models are comparable).
        """
        local_idx = self.gradiend.get_topk_weights(part=part, topk=topk)
        if not local_idx:
            return []
        base_map = self.gradiend._get_base_global_index_map()
        idx_t = torch.as_tensor(local_idx, dtype=torch.long)
        return base_map[idx_t].tolist()

    def get_topk_feature_metadata(
        self,
        part: str = "decoder-weight",
        topk: Union[int, float] = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Return human-readable metadata for the top-k most important GRADIEND input dimensions.

        Each row includes both the base-global index and the corresponding base-model
        parameter name plus parameter-local coordinates.
        """
        local_idx = self.gradiend.get_topk_weights(part=part, topk=topk)
        if not local_idx:
            return []

        base_map = self.gradiend._get_base_global_index_map()
        importance = self.gradiend.get_weight_importance(part=part)
        rows: List[Dict[str, Any]] = []

        for rank, local_index in enumerate(local_idx, start=1):
            local_index = int(local_index)
            base_global_index = int(base_map[local_index].item())
            decoded = self.gradiend.decode_base_global_index(base_global_index)
            rows.append({
                "rank": rank,
                "part": part,
                "importance": float(importance[local_index].item()),
                "local_input_index": local_index,
                "base_global_index": base_global_index,
                **decoded,
            })

        return rows

    def export_topk_features(
        self,
        output_path: str,
        *,
        part: str = "decoder-weight",
        topk: Union[int, float] = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Export top-k feature metadata to JSON or CSV based on output_path suffix.

        Supported suffixes: .json, .csv
        """
        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError("output_path must be a non-empty string")

        rows = self.get_topk_feature_metadata(part=part, topk=topk)
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        suffix = os.path.splitext(output_path)[1].lower()
        if suffix == ".json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
        elif suffix == ".csv":
            fieldnames = [
                "rank",
                "part",
                "importance",
                "local_input_index",
                "base_global_index",
                "param_name",
                "flat_index_in_param",
                "param_row",
                "param_col",
                "coords",
                "param_shape",
                "param_numel",
            ]
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    csv_row = dict(row)
                    csv_row["coords"] = json.dumps(csv_row["coords"])
                    csv_row["param_shape"] = json.dumps(csv_row["param_shape"])
                    writer.writerow(csv_row)
        else:
            raise ValueError("output_path must end with .json or .csv")

        return rows

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return GRADIEND parameters (adapter exposes GRADIEND weights as trainable parameters)."""
        return self.gradiend.parameters(recurse=recurse)

    @property
    def name(self):
        return os.path.basename(self.gradiend.name_or_path)

    @property
    def source(self) -> str:
        return self._source

    @property
    def target(self) -> str:
        return self._target

    @property
    def gradient_creator(self):
        return self._gradient_creator

    @property
    def param_map_hash(self):
        return self.gradiend.param_map_hash

    # ----------------------------
    # Abstract API
    # ----------------------------
    @abstractmethod
    def create_gradients(self, *args, **kwargs):
        """
        Create GRADIEND input gradients for a modality-specific example.

        Expected to run the base model forward/backward and return either:
        - a 1D tensor in GRADIEND input space, or
        - a dict of per-parameter gradient tensors compatible with the GRADIEND param_map.

        Subclasses decide how to build inputs, labels, and loss for their modality.
        """
        raise NotImplementedError()

    # ----------------------------
    # Encoding
    # ----------------------------
    def encode(self, input, label=None, return_float=False):
        """
        Encode input to latent space.

        Supports:
        - raw modality input (e.g. str) -> create_gradients -> encode
        - already-created gradient tensor in GRADIEND input space
        """
        if not isinstance(input, torch.Tensor):
            assert label is not None, "Label must be provided if input is not a tensor!"
            input = self.create_gradients(input, label)
        elif hasattr(input, "to"):
            input = input.to(self.gradiend.device_encoder, dtype=self.gradiend.torch_dtype)

        encoded = self.gradiend.encoder(input)

        if return_float:
            if hasattr(encoded, "tolist"):
                encoded = encoded.tolist()
            if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
                encoded = encoded[0]
            return float(encoded)
        return encoded

    # ----------------------------
    # Model modification
    # ----------------------------
    def rewrite_base_model(self, learning_rate, feature_factor, part="decoder"):
        """
        Rewrite the base model by applying GRADIEND-derived updates.

        General form: ``base_model + learning_rate * enhancer(part, feature_factor)``.

        CONTRACT: ``feature_factor`` (nominal latent scale) sets rewrite orientation.
        ``learning_rate`` is never negated by ``source``; see ``effective_rewrite_learning_rate``.

        part:
        - 'decoder'        : uses decoder(feature_factor)
        - 'decoder-weight' : uses weight vector of decoder
        - 'decoder-bias'   : uses decoder bias vector
        - 'decoder-sum'    : uses decoder (weight + bias) vector
        - 'encoder-weight' : uses encoder weights
        """
        if not isinstance(feature_factor, list):
            feature_factor = [feature_factor]

        applied_learning_rate = effective_rewrite_learning_rate(learning_rate, self.source)

        # Work on the unwrapped model, not a training wrapper.
        source_model = unwrap_model(self.base_model)

        # Deep-copy the plain model
        enhanced_model = copy.deepcopy(source_model)

        model_device = _first_param_device(enhanced_model)
        param_lookup = _build_param_lookup_named(enhanced_model)

        def _get_param(name: str):
            p = param_lookup.get(name)
            if p is not None:
                return p
            p = param_lookup.get(_normalize_param_name(name))
            if p is not None:
                return p
            p = param_lookup.get(f"module.{name}")  # extra fallback
            if p is not None:
                return p
            raise KeyError(f"Parameter '{name}' not found in enhanced_model")

        if part == "decoder":
            enhancer = self.gradiend.decoder(
                torch.tensor(feature_factor, dtype=torch.float, device=model_device)
            )
        elif part in {"decoder-bias", "decoder-sum", "decoder-weight", "encoder-weight"}:
            enhancer = self.gradiend.get_update_vector(part).to(model_device)
        else:
            raise ValueError(
                "part must be 'decoder', 'decoder-bias', 'decoder-sum', 'decoder-weight', or 'encoder-weight', "
                f"got {part!r}"
            )

        idx = 0
        with torch.no_grad():
            for param_name, spec in self.gradiend.param_map.items():
                p = _get_param(param_name)
                r = spec["repr"]

                if r == "all":
                    n = p.numel()
                    chunk = enhancer[..., idx: idx + n].to(p.device)
                    chunk = chunk.unsqueeze(0) if chunk.dim() == 1 and p.dim() > 0 else chunk
                    p.add_(applied_learning_rate * chunk.reshape(p.shape))
                    idx += n

                elif r == "mask":
                    m = spec["mask"].to(device=p.device).bool()
                    n = int(m.sum().item())
                    update_values = enhancer[idx: idx + n].to(p.device)
                    update_tensor = torch.zeros_like(m, dtype=update_values.dtype, device=p.device)
                    update_tensor[m] = update_values
                    p.add_(applied_learning_rate * update_tensor)
                    idx += n

                elif r == "indices":
                    flat_idx = spec["indices"].to(device=p.device, dtype=torch.long)
                    n = int(flat_idx.numel())
                    update_values = enhancer[idx: idx + n].to(p.device)
                    update_tensor = torch.zeros(p.numel(), dtype=update_values.dtype, device=p.device)
                    update_tensor[flat_idx] = update_values
                    p.add_(applied_learning_rate * update_tensor.reshape(p.shape))
                    idx += n

                else:
                    raise ValueError(f"Unknown param repr {r!r} for param {param_name}")

        if idx != enhancer.numel():
            raise ValueError(
                f"Inconsistent enhancer length vs mapping (used {idx}, enhancer has {enhancer.numel()})")

        return enhanced_model

    def with_original_base_model(self, new_base: nn.Module) -> "ModelWithGradiend":
        """
        Return a copy of this ModelWithGradiend with base_model replaced by new_base.

        Used when the base model has a specialized head for training but evaluation
        should use the original underlying model. The gradiend and other attributes
        (name_or_path, tokenizer, etc.) are preserved.

        Args:
            new_base: The original/base model to use for evaluation.

        Returns:
            A new ModelWithGradiend instance with base_model=new_base, same gradiend,
            and param_lookup recomputed from new_base.
        """
        other = copy.copy(self)
        other.base_model = new_base
        # Recompute param_lookup so rewrite_base_model works with the new base
        other.param_lookup = _build_param_lookup_named(new_base)
        other._sync_base_requires_grad_to_param_map()
        return other

    def get_enhancer_mask(self, topk, part="decoder-weight"):
        cache_key = f"{topk}_{part}"
        if cache_key in self._enhancer_mask_cache:
            return self._enhancer_mask_cache[cache_key]

        vec = self.gradiend.get_update_vector(part=part)
        vec_len = vec.numel()

        if topk == 0.0:
            mask = torch.zeros(vec_len, dtype=torch.bool, device=vec.device)
            self._enhancer_mask_cache[cache_key] = mask
            return mask

        k = int(topk) if topk > 1.0 else int(topk * vec_len)
        k = max(0, min(k, vec_len))
        if k == 0:
            mask = torch.zeros(vec_len, dtype=torch.bool, device=vec.device)
            self._enhancer_mask_cache[cache_key] = mask
            return mask

        indices = self.get_topk_weights(part=part, topk=k)
        mask = torch.zeros(vec_len, dtype=torch.bool, device=vec.device)
        mask[indices] = True
        self._enhancer_mask_cache[cache_key] = mask
        return mask

    def invert_encoding(self, *, update_direction: bool = True) -> None:
        """
        Invert encoder direction by flipping encoder/decoder weight signs.

        ``update_direction`` controls whether ``feature_class_encoding_direction`` metadata
        is flipped as well:

        - **Training normalization** (``NormalizationCallback``): ``update_direction=False``.
          Only weights are flipped so correlation becomes positive; semantic class→sign
          metadata in ``feature_class_encoding_direction`` stays fixed. Do **not** set True here.
        - **Manual / user-driven correction**: ``update_direction=True`` so metadata stays
          consistent with weights.

        Decoder rewrite orientation is still set via ``feature_factor`` at eval time, not by
        changing this flag during training normalization.
        """
        enc = self.gradiend.encoder[0]
        dec = self.gradiend.decoder[0]
        enc_lin = getattr(enc, "linear", enc)
        dec_lin = getattr(dec, "linear", dec)
        with torch.no_grad():
            enc_lin.weight.mul_(-1)
            if enc_lin.bias is not None:
                enc_lin.bias.mul_(-1)
            dec_lin.weight.mul_(-1)
            if dec_lin.bias is not None:
                dec_lin.bias.mul_(-1)
        if update_direction and isinstance(self.feature_class_encoding_direction, dict):
            self.feature_class_encoding_direction = {
                k: (-v if isinstance(v, (int, float)) else v)
                for k, v in self.feature_class_encoding_direction.items()
            }


    # ----------------------------
    # Saving (unchanged naming; GRADIEND writes config + training.json itself)
    # ----------------------------
    def save_pretrained(self, save_directory, **kwargs):
        """
        Save base model artifacts + GRADIEND metadata and weights.

        Writes:
        - gradiend_context.json (source/target and optional feature_class_encoding_direction)
        - subclass hook _save_model(...)
        - gradiend.save_pretrained(...)
        """
        write_gradiend_context(
            save_directory,
            self._source,
            self._target,
            feature_class_encoding_direction=self.feature_class_encoding_direction,
        )

        base_model_id = getattr(self.base_model, "name_or_path", None)
        if not base_model_id:
            raise ValueError("base_model.name_or_path is required to save a GRADIEND checkpoint")
        self.gradiend.kwargs["base_model"] = base_model_id

        self._save_model(save_directory, **kwargs)
        self.gradiend.save_pretrained(save_directory, **kwargs)

    @abstractmethod
    def _save_model(self, save_directory, **kwargs):
        """
        Subclass hook to persist base-model artifacts.

        Implementations typically save the base model, and any modality-specific
        files needed to restore the model at load time (e.g., tokenizer).
        """
        pass

    @classmethod
    @abstractmethod
    def _load_model(
        cls,
        load_directory: str,
        base_model_id: Optional[str] = None,
        gradiend_kwargs: Optional[Dict[str, Any]] = None,
        base_model: Optional[Any] = None,
        **kwargs,
    ) -> tuple:
        """
        Subclass hook to load the base model and modality-specific components.

        When base_model_id is set, load_directory is a GRADIEND checkpoint dir and the base model
        should be loaded from base_model_id (and gradiend_kwargs may contain e.g. tokenizer path),
        unless base_model is provided to reuse a shared base model instance.
        When base_model_id is None, load_directory is the base model path/name; load base model from it.

        Returns:
            (base_model, *extra) where extra are modality-specific training_args for the subclass constructor
            (e.g. tokenizer for text).
        """
        pass


    @classmethod
    def _resolve_device_config(cls, load_directory: str, **kwargs) -> Tuple[Any, Any, Any, bool]:
        """Hook for resolving device placement for base/encoder/decoder."""
        return resolve_device_config_for_model(
            device=kwargs.get("device"),
            device_encoder=kwargs.get("device_encoder"),
            device_decoder=kwargs.get("device_decoder"),
            device_base_model=kwargs.get("device_base_model"),
            encoder_decoder_same_device=kwargs.get("encoder_decoder_same_device", False),
        )

    @classmethod
    def _get_device_config(cls, load_directory: str, **kwargs) -> Dict[str, Any]:
        """
        Optional hook: return device_encoder, device_decoder, device_base_model (or similar) for loading.
        Default uses resolve_device_config_for_model; subclasses may override _resolve_device_config.
        """
        device_encoder, device_decoder, device_base_model, _ = cls._resolve_device_config(load_directory, **kwargs)
        return {
            "device_encoder": device_encoder,
            "device_decoder": device_decoder,
            "base_model_device": device_base_model,
            "base_model_device_map": kwargs.get("base_model_device_map"),
            "base_model_max_memory": kwargs.get("base_model_max_memory"),
        }


    @classmethod
    def _create_gradiend(cls, base_model: Any, load_directory: str, **kwargs) -> ParamMappedGradiendModel:
        """
        Create a new ParamMappedGradiendModel when loading a path that is not a GRADIEND checkpoint.

        Uses modality-agnostic build_gradiend_from_base_model (backbone vs head split).
        When pre_prune_config is set, uses lazy_init=True so encoder/decoder are built only after prune.
        Subclasses may override for custom behavior.
        """
        create_kwargs = dict(kwargs)
        lazy_init = bool(create_kwargs.get("pre_prune_config") is not None)
        return build_gradiend_from_base_model(
            base_model,
            load_directory,
            param_map=create_kwargs.pop("param_map", None),
            params=create_kwargs.pop("params", None),
            lazy_init=lazy_init,
            **create_kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        load_directory: Any,
        *,
        require_gradiend_model: bool = False,
        feature_definition: Optional[Any] = None,
        **kwargs: Any,
    ) -> "ModelWithGradiend":
        """
        Load a ModelWithGradiend from a directory (GRADIEND checkpoint) or create new from base model path.

        Common logic: normalize path, try ParamMappedGradiendModel.from_pretrained, then either load base via
        _load_model(..., base_model_id=..., gradiend_kwargs=..., base_model=...) or _load_model + _create_gradiend;
        load gradiend_context.json (source/target), instantiate cls(...), restore feature_class_encoding_direction.
        Modality-specific loading is in _get_device_config, _load_model, _create_gradiend.

        Args:
            load_directory: Directory path or model identifier to load from.
            require_gradiend_model: If True, load_directory must be a GRADIEND checkpoint.
                Raises FileNotFoundError (or ValueError) if not found. Default: False.
            feature_definition: Optional FeatureLearningDefinition instance. When provided, uses its
                pair and classes attributes to set feature_class_encoding_direction on the model.
            **kwargs: Additional arguments passed to _load_model and _create_gradiend.
                Optional ``base_model`` can be used to reuse an already loaded base model.
        """
        if not isinstance(load_directory, str) and not hasattr(load_directory, "name_or_path"):
            raise TypeError(
                "load_directory must be a string or an object with name_or_path, got {}".format(type(load_directory))
            )
        load_directory_str = load_directory if isinstance(load_directory, str) else getattr(load_directory, "name_or_path")
        if isinstance(kwargs.get("param_map"), list) and len(kwargs["param_map"]) == 1 and isinstance(kwargs["param_map"][0], list):
            kwargs["param_map"] = kwargs["param_map"][0]
        # Merge TrainingArguments into kwargs so params etc. are passed to _create_gradiend
        training_args = kwargs.pop("training_args", None)
        def _training_arg_value(key: str, default: Any = None) -> Any:
            if training_args is None:
                return default
            if isinstance(training_args, dict):
                return training_args.get(key, default)
            return getattr(training_args, key, default)

        if training_args is not None:
            gradiend_keys = (
                "params", "param_map", "trust_remote_code", "torch_dtype",
                "activation_encoder", "activation_decoder", "bias_decoder", "latent_dim",
                "encoder_decoder_same_device", "pre_prune_config",
                "base_model_device_map", "base_model_max_memory",
                "prediction_objective",
                "decoder_sequence_cloze_rhs_window",
                "source", "target",
            )
            for key in gradiend_keys:
                val = _training_arg_value(key, None)
                if val is not None and key not in kwargs:
                    kwargs.setdefault(key, val)
        require_gradiend_model = kwargs.pop("require_gradiend_model", require_gradiend_model)
        feature_definition = kwargs.pop("feature_definition", feature_definition)
        base_model_device_map = kwargs.pop("base_model_device_map", None)
        base_model_max_memory = kwargs.pop("base_model_max_memory", None)
        device_config = cls._get_device_config(
            load_directory_str,
            base_model_device_map=base_model_device_map,
            base_model_max_memory=base_model_max_memory,
            **kwargs,
        )
        gradiend_device_config = {
            k: v for k, v in device_config.items()
            if k not in {"base_model_device", "base_model_device_map", "base_model_max_memory"}
        }

        if _is_gradiend_checkpoint(load_directory_str):
            # Load as GRADIEND checkpoint
            gradiend = ParamMappedGradiendModel.from_pretrained(load_directory_str, **gradiend_device_config)
            base_model_id = gradiend.base_model_id
            base_model, *extra = cls._load_model(
                load_directory_str,
                base_model_id=base_model_id,
                gradiend_kwargs=gradiend.kwargs,
                **kwargs,
                **device_config,
            )
            if kwargs.get("param_map") and getattr(gradiend, "param_map", None) != kwargs["param_map"]:
                raise ValueError(
                    "Provided param_map {} do not match model training_args {}".format(
                        kwargs["param_map"], gradiend.param_map
                    )
                )
        else:
            # Base model path (HF id, decoder_mlm_head, etc.) — load base model and create fresh GRADIEND
            if require_gradiend_model:
                diagnostic = _gradiend_checkpoint_diagnostic(load_directory_str)
                raise FileNotFoundError(
                    f"Expected GRADIEND checkpoint at {load_directory_str}, but path is not a valid GRADIEND directory "
                    f"({diagnostic}). Pass require_gradiend_model=False to allow base model loading."
                )
            logger.debug(
                "Path is not a GRADIEND checkpoint -> loading as base model",
            )
            gradiend = None
            load_arg = load_directory if not isinstance(load_directory, str) else load_directory_str
            base_model, *extra = cls._load_model(
                load_arg,
                **kwargs,
                **device_config,
            )
            create_gradiend_kwargs = dict(kwargs)
            create_gradiend_kwargs.pop("base_model", None)
            gradiend = cls._create_gradiend(
                base_model,
                load_directory_str,
                **create_gradiend_kwargs,
                **gradiend_device_config,
            )

        # Source/target: from adapter_config when loading checkpoint, else kwargs
        if gradiend is not None and getattr(gradiend, "name_or_path", None) == load_directory_str:
            source, target, feature_class_encoding_direction_from_context = read_gradiend_context(load_directory_str)
        else:
            source = kwargs.get("source", _training_arg_value("source", "factual"))
            target = kwargs.get("target", _training_arg_value("target", "diff"))
            feature_class_encoding_direction_from_context = None

        gradient_creator = kwargs.pop("gradient_creator", None)
        model = cls(
            base_model,
            gradiend,
            *extra,
            source=source,
            target=target,
            gradient_creator=gradient_creator,
            **device_config,
        )
        model.name_or_path = load_directory_str

        if feature_class_encoding_direction_from_context is not None:
            model.feature_class_encoding_direction = feature_class_encoding_direction_from_context
        elif feature_definition is not None:
            pair = getattr(feature_definition, "pair", None)
            classes = getattr(feature_definition, "classes", None) or []
            if pair and len(pair) >= 2:
                class_labels = {pair[0]: 1.0, pair[1]: -1.0}
                for c in classes:
                    if c not in class_labels:
                        class_labels[c] = 0.0
                model.set_feature_class_encoding_direction(class_labels)

        model._post_init_from_pretrained()
        
        # Check if this is a non-convergent multi-seed model and warn
        _check_convergence_warning(load_directory_str)
        
        return model

    def _post_init_from_pretrained(self):
        """
        Optional hook called after from_pretrained builds the instance (e.g. to freeze base model layers).
        Subclasses may override; default no-op.
        """
        pass

    def prune_gradiend(
            self,
            *,
            topk: Optional[float] = None,
            threshold: Optional[float] = None,
            mask: Optional[torch.Tensor] = None,
            part: str = "decoder-weight",
            importance: Optional[torch.Tensor] = None,
            keep_idx: Optional[torch.Tensor] = None,
            keep_idx_sorted_unique: bool = False,
            inplace: bool = True,
            return_mask: bool = False,
    ) -> Union["ModelWithGradiend", Tuple["ModelWithGradiend", torch.Tensor]]:
        """
        Prune GRADIEND input space by selecting important input dimensions and physically reducing
        gradiend.input_dim. Converts `gradiend.param_map` list -> dict internally; the pruned gradiend
        will have dict(param -> bool mask).

        Selection is applied in fixed order: mask -> threshold -> topk.

        Args:
            topk: int (absolute) or float in (0,1] (relative fraction of remaining dims).
            threshold: keep dims with importance >= threshold (importance from get_weight_importance(part) or importance arg).
            mask: optional bool mask of shape (gradiend.input_dim,) in current GRADIEND input space.
            part: 'encoder-weight' | 'decoder-weight' | 'decoder-bias' | 'decoder-sum' (delegated to get_weight_importance when importance is None).
            importance: optional 1D tensor (e.g. from pre-prune gradient mean); when provided, used instead of get_weight_importance(part).
            keep_idx: optional 1D tensor of input-space indices to keep. Bypasses dense mask/importance materialization.
            inplace: if True, mutate self; else return a deepcopy with pruned gradiend.
            return_mask: if True, also return the final combined_mask (in original input space).

        Returns:
            model (self or copy) or (model, combined_mask) if return_mask=True
        """
        kwargs = {
            "topk": topk,
            "threshold": threshold,
            "mask": mask,
            "part": part,
            "importance": importance,
            "keep_idx": keep_idx,
            "return_mask": return_mask,
        }
        if keep_idx_sorted_unique:
            kwargs["keep_idx_sorted_unique"] = True

        if inplace:
            res = self.gradiend.prune(**kwargs, inplace=False)
            if return_mask:
                self.gradiend, combined_mask = res
                self._sync_base_requires_grad_to_param_map()
                return self, combined_mask
            self.gradiend = res
            self._sync_base_requires_grad_to_param_map()
            return self

        out = copy.deepcopy(self)
        res = out.gradiend.prune(**kwargs, inplace=False)
        if return_mask:
            out.gradiend, combined_mask = res
            out._sync_base_requires_grad_to_param_map()
            return out, combined_mask
        out.gradiend = res
        out._sync_base_requires_grad_to_param_map()
        return out


    def __len__(self):
        """Length of the GRADIEND input space (after pruning)."""
        return self.pruned_length()

    def pruned_length(self):
        """Length of the GRADIEND input space (after pruning)."""
        return len(self.gradiend)


    def unpruned_length(self):
        """Length of the original GRADIEND input space (before pruning)."""
        return self.gradiend.unpruned_length()

    def _move_batch_to_device(self, batch, device):
        if torch.is_tensor(batch):
            return batch.to(device, non_blocking=True)
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)(self._move_batch_to_device(v, device) for v in batch)
        return batch

    def _get_base_forward_model(self):
        return self.base_model

    def _get_base_forward_device(self):
        m = self._get_base_forward_model()
        base = m.module if hasattr(m, "module") else m
        hf_device = getattr(base, "device", None)
        if hf_device is not None:
            try:
                return hf_device if isinstance(hf_device, torch.device) else torch.device(hf_device)
            except (TypeError, ValueError):
                pass
        get_embeddings = getattr(base, "get_input_embeddings", None)
        if callable(get_embeddings):
            emb = get_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.device
        return next(base.parameters()).device

    def _place_inputs_for_base_forward(self, batch):
        return self._move_batch_to_device(batch, self._get_base_forward_device())

    def _zero_base_grad(self, *, set_to_none: bool = True) -> None:
        """Clear base-model gradients across regular modules."""
        m = self._get_base_forward_model()
        try:
            m.zero_grad(set_to_none=set_to_none)
        except TypeError:
            m.zero_grad()
            if set_to_none:
                base = m.module if hasattr(m, "module") else m
                for p in base.parameters():
                    p.grad = None
