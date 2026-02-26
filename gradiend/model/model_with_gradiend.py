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
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Dict, Any, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import Parameter

from gradiend.util.logging import get_logger
from gradiend.model import ParamMappedGradiendModel
from gradiend.model.core import build_gradiend_from_base_model
from gradiend.model.utils import resolve_device_config_for_model, read_gradiend_context, write_gradiend_context

logger = get_logger(__name__)


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
                        "converged=False (convergent_count=%s, required: %s) for metric=%s threshold=%.4f. "
                        "Model may not have reached the convergence threshold during training.",
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
                "convergent_count=0 (required: %s) for metric=%s threshold=%.4f. "
                "Model may not have reached the convergence threshold during training.",
                min_convergent_seeds,
                convergence_metric,
                threshold if threshold is not None else 0.0,
            )
    except Exception as e:
        # Silently ignore errors reading seed_report.json (could be corrupted, etc.)
        logger.debug("Could not check convergence status from %s: %s", seed_report_path, e)


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
        gradient_creator=None,
        source: str = "factual",
        target: str = "diff",
    ):
        if not isinstance(source, str):
            raise TypeError(f"source must be str, got {type(source).__name__}")
        if source not in ("factual", "alternative", "diff"):
            raise ValueError(f"source must be one of 'factual', 'alternative', 'diff', got {source!r}")
        if not isinstance(target, str):
            raise TypeError(f"target must be str, got {type(target).__name__}")
        if target not in ("factual", "alternative", "diff"):
            raise ValueError(f"target must be one of 'factual', 'alternative', 'diff', got {target!r}")

        super().__init__()
        self.base_model = base_model
        self.gradiend = gradiend
        self._source = source
        self._target = target

        self._gradient_creator = gradient_creator or self

        if gradiend.encoder is not None:
            self.base_model_device = base_model_device or gradiend.encoder[0].linear.weight.device
        else:
            self.base_model_device = base_model_device or gradiend.device_encoder
        self.base_model.to(self.base_model_device)
        self.gradiend.to(device_encoder=device_encoder, device_decoder=device_decoder)

        # Stable parameter map for shapes + updates
        self.param_lookup = {k: v for k, v in self.base_model.named_parameters()}

        # Ensure GRADIEND param_map is always spec-dict (no list-mode in the new library design)
        self._ensure_gradiend_param_map_spec()

        self._enhancer_mask_cache = {}
        self.feature_class_encoding_direction: Optional[Dict[str, int]] = None

    def to(self, device: object) -> "ModelWithGradiend":
        """Move base_model and gradiend to the given device. Accepts str or torch.device."""
        dev = torch.device(device) if isinstance(device, str) else device
        self.base_model.to(dev)
        self.gradiend.to(dev)
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

        General form: $base\\_model + learning\\_rate * enhancer(part, feature\\_factor)$,
        where enhancer(part, feature_factor) is defined by the selected part below.

        part:
        - 'decoder'        : uses decoder(feature_factor)
        - 'decoder-weight' : uses weight vector of decoder
        - 'decoder-bias'   : uses decoder bias vector
        - 'decoder-sum'    : uses decoder (weight + bias) vector
        - 'encoder-weight' : uses encoder weights

        """
        if not isinstance(feature_factor, list):
            feature_factor = [feature_factor]

        enhanced_model = copy.deepcopy(self.base_model)

        model_device = self.base_model.device
        param_lookup = {k: v for k, v in enhanced_model.named_parameters()}

        if part == "decoder":
            enhancer = self.gradiend.decoder(torch.tensor(feature_factor, dtype=torch.float, device=model_device))
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
                p = param_lookup[param_name]
                r = spec["repr"]

                if r == "all":
                    n = p.numel()
                    chunk = enhancer[..., idx : idx + n].to(model_device)
                    chunk = chunk.unsqueeze(0) if chunk.dim() == 1 and p.dim() > 0 else chunk
                    p.add_(learning_rate * chunk.reshape(p.shape))
                    idx += n

                elif r == "mask":
                    m = spec["mask"].to(device=model_device).bool()
                    n = int(m.sum().item())
                    update_values = enhancer[idx : idx + n].to(model_device)
                    update_tensor = torch.zeros_like(m, dtype=update_values.dtype)
                    update_tensor[m] = update_values
                    p.add_(learning_rate * update_tensor)
                    idx += n

                elif r == "indices":
                    flat_idx = spec["indices"].to(device=model_device, dtype=torch.long)
                    n = int(flat_idx.numel())
                    update_values = enhancer[idx : idx + n].to(model_device)
                    update_tensor = torch.zeros(p.numel(), dtype=update_values.dtype, device=model_device)
                    update_tensor[flat_idx] = update_values
                    p.add_(learning_rate * update_tensor.reshape(p.shape))
                    idx += n

                else:
                    raise ValueError(f"Unknown param repr {r!r} for param {param_name}")

        if idx != enhancer.numel():
            raise ValueError(f"Inconsistent enhancer length vs mapping (used {idx}, enhancer has {enhancer.numel()})")

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
        other.param_lookup = {k: v for k, v in new_base.named_parameters()}
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
        Invert encoder direction by flipping encoder/decoder signs.

        This preserves reconstruction while flipping the sign of the latent feature.
        Set update_direction=True only for manual/user-driven flips.
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
    # Pruning wrapper (no list conversion here anymore)
    # ----------------------------
    def prune_gradiend(
        self,
        *,
        topk: float = None,
        threshold: float = None,
        mask: torch.Tensor = None,
        part: str = "decoder-weight",
        importance: torch.Tensor = None,
        inplace: bool = True,
        return_mask: bool = False,
    ):
        """
        Prune GRADIEND input space by selecting important input dimensions and physically reducing input_dim.

        Delegates to gradiend.prune(). Assumes gradiend.param_map is already a spec-dict (enforced in __init__).
        Selection order: mask -> threshold -> topk. When importance is provided, it is used instead of get_weight_importance(part).
        """
        kwargs = {
            "topk": topk,
            "threshold": threshold,
            "mask": mask,
            "part": part,
            "importance": importance,
            "return_mask": return_mask,
        }

        if inplace:
            res = self.gradiend.prune(**kwargs, inplace=False)
            if return_mask:
                self.gradiend, combined_mask = res
                return self, combined_mask
            self.gradiend = res
            return self

        out = copy.deepcopy(self)
        res = out.gradiend.prune(**kwargs, inplace=False)
        if return_mask:
            out.gradiend, combined_mask = res
            return out, combined_mask
        out.gradiend = res
        return out

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
    def _load_model(cls, load_directory: str, base_model_id: Optional[str] = None, gradiend_kwargs: Optional[Dict[str, Any]] = None, **kwargs) -> tuple:
        """
        Subclass hook to load the base model and modality-specific components.

        When base_model_id is set, load_directory is a GRADIEND checkpoint dir and the base model
        should be loaded from base_model_id (and gradiend_kwargs may contain e.g. tokenizer path).
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
        }


    @classmethod
    def _create_gradiend(cls, base_model: Any, load_directory: str, **kwargs) -> ParamMappedGradiendModel:
        """
        Create a new ParamMappedGradiendModel when loading a path that is not a GRADIEND checkpoint.

        Uses modality-agnostic build_gradiend_from_base_model (backbone vs head split).
        When pre_prune_config is set, uses lazy_init=True so encoder/decoder are built only after prune.
        Subclasses may override for custom behavior.
        """
        lazy_init = bool(kwargs.get("pre_prune_config") is not None)
        return build_gradiend_from_base_model(
            base_model,
            load_directory,
            params=kwargs.get("params"),
            lazy_init=lazy_init,
            **kwargs,
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
        _load_model(..., base_model_id=..., gradiend_kwargs=...) or _load_model + _create_gradiend;
        load gradiend_context.json (source/target), instantiate cls(...), restore feature_class_encoding_direction.
        Modality-specific loading is in _get_device_config, _load_model, _create_gradiend.

        Args:
            load_directory: Directory path or model identifier to load from.
            require_gradiend_model: If True, load_directory must be a GRADIEND checkpoint.
                Raises FileNotFoundError (or ValueError) if not found. Default: False.
            feature_definition: Optional FeatureLearningDefinition instance. When provided, uses its
                pair and classes attributes to set feature_class_encoding_direction on the model.
            **kwargs: Additional arguments passed to _load_model and _create_gradiend.
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
        if training_args is not None:
            gradiend_keys = (
                "params", "param_map", "trust_remote_code", "torch_dtype",
                "activation_encoder", "activation_decoder", "bias_decoder", "latent_dim",
                "encoder_decoder_same_device", "pre_prune_config",
            )
            for key in gradiend_keys:
                if hasattr(training_args, key) and key not in kwargs:
                    val = getattr(training_args, key, None)
                    if val is not None:
                        kwargs.setdefault(key, val)
        require_gradiend_model = kwargs.pop("require_gradiend_model", require_gradiend_model)
        feature_definition = kwargs.pop("feature_definition", feature_definition)
        device_config = cls._get_device_config(load_directory_str, **kwargs)
        gradiend_device_config = {k: v for k, v in device_config.items() if k != "base_model_device"}

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
                raise FileNotFoundError(
                    f"Expected GRADIEND checkpoint at {load_directory_str}, but path is not a GRADIEND directory "
                    "(missing config.json with 'architecture'). Pass require_gradiend_model=False to allow base model loading."
                )
            logger.debug(
                "Path is not a GRADIEND checkpoint -> loading as base model",
            )
            gradiend = None
            load_arg = load_directory if not isinstance(load_directory, str) else load_directory_str
            base_model, *extra = cls._load_model(load_arg, **kwargs, **device_config)
            gradiend = cls._create_gradiend(base_model, load_directory_str, **kwargs, **gradiend_device_config)

        # Source/target: from adapter_config when loading checkpoint, else kwargs
        if gradiend is not None and getattr(gradiend, "name_or_path", None) == load_directory_str:
            source, target, feature_class_encoding_direction_from_context = read_gradiend_context(load_directory_str)
        else:
            source = kwargs.get("source", "factual")
            target = kwargs.get("target", "diff")
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

        if feature_definition is not None:
            pair = getattr(feature_definition, "pair", None)
            classes = getattr(feature_definition, "classes", None) or []
            if pair and len(pair) >= 2:
                class_labels = {pair[0]: 1.0, pair[1]: -1.0}
                for c in classes:
                    if c not in class_labels:
                        class_labels[c] = 0.0
                model.set_feature_class_encoding_direction(class_labels)
        else:
            if feature_class_encoding_direction_from_context is not None:
                model.feature_class_encoding_direction = feature_class_encoding_direction_from_context

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
            "return_mask": return_mask,
        }

        if inplace:
            res = self.gradiend.prune(**kwargs, inplace=False)
            if return_mask:
                self.gradiend, combined_mask = res
                return self, combined_mask
            self.gradiend = res
            return self

        out = copy.deepcopy(self)
        res = out.gradiend.prune(**kwargs, inplace=False)
        if return_mask:
            out.gradiend, combined_mask = res
            return out, combined_mask
        out.gradiend = res
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
