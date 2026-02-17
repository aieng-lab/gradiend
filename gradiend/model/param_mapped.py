"""
GRADIEND parameter mapping-aware model.

This module defines ParamMappedGradiendModel, which extends GradiendModel with
param_map metadata, gradient IO helpers, and mapping-aware pruning.

Mapping design:
- The mapping is reconstructible from config alone (no base-model access).
- Each parameter stores its shape and selection representation.
- Supported per-parameter representations:
  - "all": full parameter selected (no mapping tensor needed)
  - "mask": boolean mask over parameter entries (dense-ish)
  - "indices": flat index list of selected entries (sparse-ish)
  - "mixed": per-parameter choice across the above, stored in one or two mapping files

Storage efficiency:
- For large parameters with tiny selection, indices are far smaller than bool masks.
- The chosen representation is decided at save time based on estimated size.

File layout (save_pretrained):
- model.safetensors (preferred if safetensors library is installed) or pytorch_model.bin   [weights]
- config.json                                                  [architecture + mapping + metadata; format_version=0]
- mapping_indices.safetensors or mapping_indices.pth           [optional]
- mapping_masks.safetensors or mapping_masks.pth               [optional]
- training.json                                                [optional run info when provided]
"""

import copy
import json
import os
import math
import hashlib
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from gradiend.model.model import GradiendModel
from gradiend.model.utils import (
    _load_tensor_dict,
    _save_tensor_dict,
    _tensor_file_name,
    _is_int32_safe,
    _bytes_per_index,
)
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


class ParamMappedGradiendModel(GradiendModel):
    """
    GRADIEND model with base-parameter mapping.

    In addition to GradiendModel (only weights), this class stores a mapping from base-model parameters
        to the GRADIEND input space. This enables:
        - extracting gradients from a base model into GRADIEND input tensor
        - accepting dict-of-parameter gradients in forward/forward_encoder (same semantics as before)
        - pruning (physically reducing input_dim) while remapping the param map consistently

        Param map representation (in-memory):
        `self.param_map` is a dict: param_name -> spec dict with:
            - "shape": tuple[int,...]   (required)
            - "repr": "all" | "mask" | "indices"
            - if repr == "mask":    "mask": torch.BoolTensor (shape == param shape)
            - if repr == "indices": "indices": 1D int tensor of flat indices in [0, numel)

        Notes:
        - repr="all" means full param selected; no mask/indices tensor needed.
        - repr="indices" avoids huge bool masks for very large params with tiny selection.
        - All mapping operations are defined by this spec; order is the insertion order of `self.param_map`.

        Saving/loading:
        - config.json includes mapping.mode ("all"|"mask"|"indices"|"mixed") and per-param entries with shapes.
        - mapping_masks.* and mapping_indices.* are written only if needed.
        - safetensors is preferred when available; otherwise torch.save/torch.load fallback is used.

        Prune:
        - prune() selects input dims via mask/threshold/topk and physically slices weights
            AND updates the mapping spec accordingly.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        param_map: Dict[str, Dict[str, Any]],
        activation_encoder: str = "tanh",
        activation_decoder: str = "id",
        bias_decoder: bool = True,
        torch_dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        device_encoder: Optional[torch.device] = None,
        device_decoder: Optional[torch.device] = None,
        **kwargs,
    ):
        """
        Initialize a GRADIEND model with a parameter mapping.

        Args:
            input_dim: Size of the GRADIEND input space (total selected gradient entries).
            latent_dim: Size of the latent bottleneck.
            param_map: Mapping spec dict keyed by parameter name. Each value must
                include "shape" and "repr" ("all" | "mask" | "indices"), and
                any selection tensor required by the repr.
            activation_encoder: Encoder activation name.
            activation_decoder: Decoder activation name.
            bias_decoder: Whether the decoder linear layer uses a bias term.
            torch_dtype: dtype used for model parameters.
            device: Optional default device for both encoder and decoder when specific
                devices are not provided.
            device_encoder: Device for encoder parameters.
            device_decoder: Device for decoder parameters.
            **kwargs: Stored in `self.kwargs` and serialized into config.json metadata
                on save.
        """
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            activation_encoder=activation_encoder,
            activation_decoder=activation_decoder,
            bias_decoder=bias_decoder,
            torch_dtype=torch_dtype,
            device=device,
            device_encoder=device_encoder,
            device_decoder=device_decoder,
            **kwargs,
        )
        self.param_map = param_map
        self._base_global_index_map: Optional[torch.Tensor] = None
        self._base_global_index_map_version: int = 0
        self._param_map_version: int = 0

    def _param_map_items(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        # dict insertion order is the mapping order
        yield from self.param_map.items()

    @property
    def param_map_hash(self) -> str:
        """
        Compute a stable hash of the current mapping spec.

        The hash includes param names, shapes, repr types, and selection tensors.
        It is suitable for cache keys and change detection.

        Returns:
            Hex digest string (MD5) of the mapping spec.
        """
        h = hashlib.md5()
        for name, spec in self._param_map_items():
            h.update(str(name).encode())
            h.update(str(tuple(spec.get("shape", ()))) .encode())
            r = spec.get("repr")
            h.update(str(r).encode())
            if r == "mask":
                m = spec["mask"].to("cpu").flatten().to(dtype=torch.uint8)
                h.update(m.numpy().tobytes())
            elif r == "indices":
                idx = spec["indices"].to("cpu").to(dtype=torch.int64)
                h.update(idx.numpy().tobytes())
        return h.hexdigest()

    def _build_base_global_index_map(self) -> torch.Tensor:
        """
        Build a base-global index map for the current input space.

        Returns:
            1D tensor of length input_dim. For each local input index, stores the
            corresponding base-global index (flattened across base-model parameters
            in param_map insertion order).
        """
        parts: List[torch.Tensor] = []
        base_offset = 0

        for param_name, spec in self._param_map_items():
            shape = tuple(spec["shape"])
            numel = int(torch.tensor(shape).prod().item())
            r = spec["repr"]

            if r == "all":
                sel_positions = torch.arange(numel, dtype=torch.long)
            elif r == "mask":
                sel_positions = spec["mask"].flatten().nonzero(as_tuple=False).flatten().to("cpu").long()
            elif r == "indices":
                sel_positions = spec["indices"].to("cpu").long()
            else:
                raise ValueError(f"Unknown param repr {r!r} for {param_name}")

            if sel_positions.numel() > 0:
                parts.append(sel_positions + base_offset)
            base_offset += numel

        if base_offset <= 0:
            return torch.empty(0, dtype=torch.long)

        mapping = torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.long)
        if mapping.numel() != self.input_dim:
            raise ValueError(
                f"Inconsistent base-global index map: expected input_dim={self.input_dim}, "
                f"got {mapping.numel()}"
            )
        return mapping

    def _get_base_global_index_map(self) -> torch.Tensor:
        """
        Return a cached base-global index map for the current input space.

        The map is rebuilt when the param_map changes (e.g., after prune).
        """
        if (
            self._base_global_index_map is not None
            and self._base_global_index_map_version == self._param_map_version
        ):
            return self._base_global_index_map

        mapping = self._build_base_global_index_map()
        self._base_global_index_map = mapping
        self._base_global_index_map_version = self._param_map_version
        return mapping

    # -----------------------------
    # gradient extraction / IO
    # -----------------------------
    def extract_gradients(self, model, return_dict: bool = False):
        """
        Extract gradients from a base model (copies).

        Returns either:
        - dict[param_name] -> gradient tensor shaped like the parameter, OR
        - a single concatenated 1D tensor in GRADIEND input space

        Args:
            model: Base model that has parameter gradients populated (after backward).
            return_dict: If True, return a dict of per-parameter gradients. If False,
                return a flattened 1D tensor in GRADIEND input space.

        Returns:
            If return_dict is True:
                Dict[param_name, grad_tensor] where each tensor matches the parameter shape.
            If return_dict is False:
                1D tensor containing only the selected entries (per param_map) concatenated
                in param_map order.

        Raises:
            RuntimeError: If any required parameter gradient is None.
        """
        param_lookup = {k: v for k, v in model.named_parameters()}

        # ensure grads exist
        missing = []
        for param_name, _spec in self._param_map_items():
            if param_lookup[param_name].grad is None:
                missing.append(param_name)
        if missing:
            raise RuntimeError(
                f"Gradients are None for parameters: {missing}. "
                "This indicates a bug in gradient computation (no backward, no_grad, requires_grad=False, ...)."
            )

        if return_dict:
            out = {}
            for param_name, _spec in self._param_map_items():
                out[param_name] = param_lookup[param_name].grad.detach().clone()
            return out

        parts: List[torch.Tensor] = []
        for param_name, spec in self._param_map_items():
            g = param_lookup[param_name].grad.detach().clone().flatten()

            r = spec["repr"]
            if r == "all":
                parts.append(g)
            elif r == "mask":
                m = spec["mask"].flatten()
                parts.append(g[m])
            elif r == "indices":
                idx = spec["indices"].to(dtype=torch.long, device=g.device)
                parts.append(g[idx])
            else:
                raise ValueError(f"Unknown param repr {r!r} for {param_name}")

        return torch.concat(parts)

    def flatten_gradient_dict(self, grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flatten a per-param gradient dict into a single 1D tensor in GRADIEND input space.
        Uses the same param_map order and selection (all/mask/indices) as in forward().

        Args:
            grad_dict: Dict of gradients keyed by parameter name with tensors shaped
                like the base model parameters.

        Returns:
            1D tensor in GRADIEND input space, concatenated in param_map order.
        """
        parts: List[torch.Tensor] = []
        for param_name, spec in self._param_map_items():
            t = grad_dict[param_name]
            flat = t.flatten()
            r = spec["repr"]
            if r == "all":
                parts.append(flat)
            elif r == "mask":
                m = spec["mask"].flatten()
                if m.device != flat.device:
                    m = m.to(flat.device)
                parts.append(flat[m])
            elif r == "indices":
                idx = spec["indices"].to(dtype=torch.long, device=flat.device)
                parts.append(flat[idx])
            else:
                raise ValueError(f"Unknown param repr {r!r} for {param_name}")
        return torch.concat(parts)

    def forward(self, x, return_encoded: bool = False):
        """
        Forward that accepts:
        - tensor: already in GRADIEND input space
        - dict: per-param gradient tensors (full tensors); selection is applied using mapping spec

        Args:
            x: Either a 1D tensor in GRADIEND input space or a dict of per-parameter
                gradient tensors.
            return_encoded: If True, also return the latent encoding.

        Returns:
            If input is a tensor:
                Same return contract as GradiendModel.forward.
            If input is a dict:
                Decoded gradients as a dict with the same keys and shapes as input
                (values filled only at selected positions), and optionally the
                encoded tensor when return_encoded is True.
        """
        orig_shapes: Dict[str, Any] = {}

        if torch.is_tensor(x):
            pass
        elif isinstance(x, dict):
            parts: List[torch.Tensor] = []
            for param_name, spec in self._param_map_items():
                t = x[param_name]
                flat = t.flatten()

                r = spec["repr"]
                if r == "all":
                    sel = flat
                    orig_shapes[param_name] = ("all", t.shape)
                elif r == "mask":
                    m = spec["mask"].flatten()
                    sel = flat[m]
                    orig_shapes[param_name] = ("mask", t.shape, spec["mask"])
                elif r == "indices":
                    idx = spec["indices"].to(dtype=torch.long, device=flat.device)
                    sel = flat[idx]
                    orig_shapes[param_name] = ("indices", t.shape, spec["indices"])
                else:
                    raise ValueError(f"Unknown param repr {r!r} for {param_name}")

                parts.append(sel)

            x = torch.concat(parts)
        else:
            raise TypeError(f"x must be a tensor or dict of gradients, got {type(x)}")

        decoded, encoded = super().forward(x, return_encoded=True)

        # reconstruct dict output if dict input
        if orig_shapes:
            out: Dict[str, torch.Tensor] = {}
            start = 0
            for param_name, info in orig_shapes.items():
                kind = info[0]
                shape = info[1]

                if kind == "all":
                    n = int(torch.tensor(shape).prod().item()) if hasattr(shape, "__iter__") else int(shape.numel())
                    out[param_name] = decoded[start : start + n].reshape(shape)
                    start += n
                elif kind == "mask":
                    _shape, mask = info[1], info[2]
                    n = int(mask.sum().item())
                    full = torch.zeros_like(mask, dtype=decoded.dtype)
                    full[mask] = decoded[start : start + n]
                    out[param_name] = full.reshape(shape)
                    start += n
                elif kind == "indices":
                    _shape, idx = info[1], info[2]
                    idx = idx.to("cpu").long()
                    n = int(idx.numel())
                    full = torch.zeros(int(torch.tensor(shape).prod().item()), dtype=decoded.dtype, device=decoded.device)
                    full[idx.to(decoded.device)] = decoded[start : start + n]
                    out[param_name] = full.reshape(shape)
                    start += n
                else:
                    raise ValueError(kind)

            decoded = out

        return (decoded, encoded) if return_encoded else decoded

    def forward_encoder(self, x):
        """
        Encoder-only forward that accepts tensor or dict input.

        Args:
            x: Either a 1D tensor in GRADIEND input space or a dict of per-parameter
                gradient tensors.

        Returns:
            Encoded tensor of shape (latent_dim,).
        """
        if isinstance(x, dict):
            parts: List[torch.Tensor] = []
            for param_name, spec in self._param_map_items():
                flat = x[param_name].flatten()
                r = spec["repr"]
                if r == "all":
                    parts.append(flat)
                elif r == "mask":
                    parts.append(flat[spec["mask"].flatten()])
                elif r == "indices":
                    idx = spec["indices"].to(dtype=torch.long, device=flat.device)
                    parts.append(flat[idx])
                else:
                    raise ValueError(f"Unknown param repr {r!r} for {param_name}")
            x = torch.concat(parts)
        elif not torch.is_tensor(x):
            raise TypeError(f"forward_encoder x must be a tensor or dict of gradients, got {type(x)}")
        return super().forward_encoder(x)

    # -----------------------------
    # prune (mapping-aware)
    # -----------------------------
    def prune(
        self,
        *,
        topk: Union[int, float, None] = None,
        threshold: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        part: str = "decoder-weight",
        importance: Optional[torch.Tensor] = None,
        inplace: bool = False,
        return_mask: bool = False,
    ):
        """
        Physically prune the model (reduce input_dim) and remap mapping spec accordingly.
        The pruning is applied based on up to three criteria: a boolean mask, an importance threshold, and/or a
        top-k selection.

        Selection order: mask -> threshold -> topk.

        Args:
            topk: int (absolute) or float in (0,1] (relative fraction among remaining dims).
            threshold: keep dims with importance >= threshold.
            mask: optional bool tensor of shape (input_dim,) in current input space.
            part: 'encoder-weight' | 'decoder-weight' | 'decoder-bias' | 'decoder-sum' (used when importance is None).
            importance: optional 1D tensor of length input_dim (e.g. from gradient mean); used instead of get_weight_importance(part) when provided.
            inplace: modify this instance if True, else return a deepcopy.
            return_mask: if True, also return final combined_mask (original input space).

        Returns:
            If return_mask is False:
                The pruned ParamMappedGradiendModel (self or a deepcopy depending on `inplace`).
            If return_mask is True:
                Tuple (model, combined_mask) where combined_mask is a bool tensor
                of shape (old_input_dim,) indicating kept dimensions in the
                original input space.
        """
        if topk is None and threshold is None and mask is None:
            raise ValueError("At least one of topk, threshold, mask must be provided.")

        m = self if inplace else copy.deepcopy(self)
        old_input_dim = int(m.input_dim)

        combined = torch.ones(old_input_dim, dtype=torch.bool)

        if mask is not None:
            if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != (old_input_dim,):
                raise ValueError(f"mask must be bool tensor with shape ({old_input_dim},)")
            combined &= mask.detach().to("cpu")

        importance_scores = None
        if threshold is not None or topk is not None:
            if importance is not None:
                importance_scores = importance.detach().to("cpu")
                if importance_scores.dim() != 1 or importance_scores.numel() != old_input_dim:
                    raise ValueError(f"importance must be 1D tensor of length {old_input_dim}, got shape {importance_scores.shape}")
            else:
                importance_scores = m.get_weight_importance(part=part).detach().to("cpu")
            if importance_scores.numel() != old_input_dim:
                raise ValueError(f"importance length {importance_scores.numel()} != input_dim {old_input_dim}")

        if threshold is not None:
            if not isinstance(threshold, (int, float)) or threshold < 0:
                raise ValueError("threshold must be a non-negative float/int")
            combined &= (importance_scores >= float(threshold))

        if topk is not None:
            if isinstance(topk, bool):
                raise TypeError("topk must be int or float, not bool")
            active = combined.nonzero(as_tuple=False).flatten()
            n_active = int(active.numel())
            if n_active == 0:
                raise ValueError("No dimensions left after mask/threshold.")

            if isinstance(topk, float):
                if not (0.0 < topk <= 1.0):
                    raise ValueError("If topk is float, it must be in (0, 1].")
                k = int(math.ceil(topk * n_active))
            elif isinstance(topk, int):
                if topk <= 0:
                    raise ValueError("If topk is int, it must be >= 1.")
                k = min(int(topk), n_active)
            else:
                raise TypeError(f"topk must be int or float, got {type(topk)}")

            if k < n_active:
                scores = importance_scores[active]
                keep_local = torch.topk(scores, k=k, largest=True, sorted=False).indices
                keep_global = active[keep_local]
                new_mask = torch.zeros_like(combined)
                new_mask[keep_global] = True
                combined = new_mask

        keep_idx = combined.nonzero(as_tuple=False).flatten().long()
        if keep_idx.numel() == 0:
            raise ValueError("Pruning would remove all dimensions (combined_mask all False).")

        # Remap param_map specs by slicing their selected positions
        new_param_map: Dict[str, Dict[str, Any]] = {}
        offset = 0
        keep_cpu = keep_idx.to("cpu")

        for param_name, spec in m._param_map_items():
            shape = tuple(spec["shape"])
            numel = int(torch.tensor(shape).prod().item())
            r = spec["repr"]

            if r == "all":
                sel_positions = torch.arange(numel, dtype=torch.long)
            elif r == "mask":
                sel_positions = spec["mask"].flatten().nonzero(as_tuple=False).flatten().to("cpu").long()
            elif r == "indices":
                sel_positions = spec["indices"].to("cpu").long()
            else:
                raise ValueError(f"Unknown param repr {r!r} for {param_name}")

            n = int(sel_positions.numel())

            in_seg = (keep_cpu >= offset) & (keep_cpu < offset + n)
            seg_keep = keep_cpu[in_seg] - offset

            if seg_keep.numel() == 0:
                new_param_map[param_name] = {
                    "shape": shape,
                    "repr": "indices",
                    "indices": torch.empty(0, dtype=torch.long),
                }
            else:
                new_sel = sel_positions[seg_keep]
                new_param_map[param_name] = {
                    "shape": shape,
                    "repr": "indices",
                    "indices": new_sel.to(dtype=torch.long),
                }

            offset += n

        if offset != old_input_dim:
            raise ValueError(f"Inconsistent mapping: expected {old_input_dim} selected total, got {offset}")

        # prune weights and set new mapping
        m = m._prune_input_dims(keep_idx, inplace=True, return_index_map=False)
        m.param_map = new_param_map
        m._param_map_version = getattr(m, "_param_map_version", 0) + 1
        m._base_global_index_map = None

        return (m, combined) if return_mask else m

    # -----------------------------
    # save/load (mapping-aware)
    # -----------------------------
    def save_pretrained(self, save_directory: str, use_safetensors: Optional[bool] = None, **kwargs):
        """
        Save weights + config + mapping.

        Mapping save strategy:
        - Always store shapes in config.
        - Choose per-param representation:
            - "all" if fully selected (k == numel)
            - else choose "indices" vs "mask" by estimated size:
                indices_size ~ k * bytes_per_index(numel)
                mask_size    ~ numel * 1 byte
          with a small safety margin to avoid flip-flopping.

        Output:
        - config.json
        - mapping_indices.(safetensors|pth) if any param uses indices
        - mapping_masks.(safetensors|pth)   if any param uses mask

        Args:
            save_directory: Folder to write model files into.
            use_safetensors: If True, require safetensors. If False, force PyTorch
                bin format. If None, prefer safetensors when available.
            **kwargs: Extra metadata to store in config.json. If "training" is
                provided, it is written to training.json instead.

        Returns:
            None.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Let core write weights + base config + optional training.json
        super().save_pretrained(save_directory, use_safetensors=use_safetensors, **kwargs)

        prefer_safetensors = (use_safetensors is not False)

        # load base config to extend mapping
        cfg_path = os.path.join(save_directory, "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Build mapping entries + tensor dicts for files
        entries = []
        idx_tensors: Dict[str, torch.Tensor] = {}
        mask_tensors: Dict[str, torch.Tensor] = {}

        # decision thresholds
        # (conservative defaults; tweak as you like)
        margin = 0.80
        for i, (param_name, spec) in enumerate(self._param_map_items()):
            shape = tuple(spec["shape"])
            numel = int(torch.tensor(shape).prod().item())

            # compute selected positions in flat space
            r = spec["repr"]
            if r == "all":
                k = numel
                sel_positions = None
            elif r == "mask":
                sel_positions = spec["mask"].flatten().nonzero(as_tuple=False).flatten().to("cpu").long()
                k = int(sel_positions.numel())
            elif r == "indices":
                sel_positions = spec["indices"].to("cpu").long()
                k = int(sel_positions.numel())
            else:
                raise ValueError(f"Unknown param repr {r!r} for {param_name}")

            # choose repr
            if k == numel:
                chosen = "all"
            else:
                bpi = _bytes_per_index(numel)
                est_indices = k * bpi
                est_mask = numel * 1
                chosen = "indices" if est_indices < est_mask * margin else "mask"

            entry: Dict[str, Any] = {
                "name": param_name,
                "shape": list(shape),
                "repr": chosen,
            }

            if chosen == "indices":
                key = f"L{i}"
                entry["key"] = key
                entry["num_selected"] = k
                if sel_positions is None:
                    sel_positions = torch.arange(numel, dtype=torch.long)
                if _is_int32_safe(numel):
                    idx_tensors[key] = sel_positions.to(dtype=torch.int32)
                else:
                    idx_tensors[key] = sel_positions.to(dtype=torch.int64)

            elif chosen == "mask":
                key = f"L{i}"
                entry["key"] = key
                entry["num_selected"] = k
                if r == "mask":
                    m = spec["mask"].to("cpu").bool()
                elif r == "indices":
                    m = torch.zeros(numel, dtype=torch.bool)
                    m[sel_positions] = True
                    m = m.reshape(shape)
                else:
                    raise RuntimeError("Unexpected repr conversion")
                mask_tensors[key] = m.to(dtype=torch.uint8)

            else:
                entry["num_selected"] = numel

            entries.append(entry)

        # global mode
        uniq = sorted({e["repr"] for e in entries})
        if uniq == ["all"]:
            mode = "all"
        elif uniq == ["mask"]:
            mode = "mask"
        elif uniq == ["indices"]:
            mode = "indices"
        else:
            mode = "mixed"

        mapping: Dict[str, Any] = {
            "mode": mode,
            "param_map": entries,
        }

        # write mapping files if needed
        if idx_tensors:
            fn = _tensor_file_name("mapping_indices", prefer_safetensors=prefer_safetensors)
            _save_tensor_dict(os.path.join(save_directory, fn), idx_tensors, prefer_safetensors=prefer_safetensors)
            mapping["indices_file"] = fn

        if mask_tensors:
            fn = _tensor_file_name("mapping_masks", prefer_safetensors=prefer_safetensors)
            _save_tensor_dict(os.path.join(save_directory, fn), mask_tensors, prefer_safetensors=prefer_safetensors)
            mapping["masks_file"] = fn

        # Save base-global index map for stable top-k comparisons
        try:
            index_map = self._get_base_global_index_map().to("cpu")
            if index_map.numel() > 0:
                total_numel = 0
                for _param_name, spec in self._param_map_items():
                    shape = tuple(spec["shape"])
                    total_numel += int(torch.tensor(shape).prod().item())
                idx_dtype = torch.int32 if _is_int32_safe(total_numel) else torch.int64
                fn = _tensor_file_name("input_index_map", prefer_safetensors=prefer_safetensors)
                _save_tensor_dict(
                    os.path.join(save_directory, fn),
                    {"base_global_index_map": index_map.to(dtype=idx_dtype)},
                    prefer_safetensors=prefer_safetensors,
                )
                mapping["input_index_map_file"] = fn
        except Exception as e:
            logger.warning("Failed to save base-global index map: %s", e)

        cfg["mapping"] = mapping

        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        device_encoder=None,
        device_decoder=None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Load weights + config + mapping.

        On load we reconstruct param_map specs. We do NOT require base model access because shapes are stored.

        Args:
            load_directory: Directory containing model files.
            device_encoder: Optional device override for encoder parameters.
            device_decoder: Optional device override for decoder parameters.
            torch_dtype: Optional dtype override. If None, uses dtype stored in config.json.

        Returns:
            Instantiated ParamMappedGradiendModel with loaded weights and mapping.
        """
        core = GradiendModel.from_pretrained(
            load_directory,
            device_encoder=device_encoder,
            device_decoder=device_decoder,
            torch_dtype=torch_dtype,
        )

        # load config
        cfg_path = os.path.join(load_directory, "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        mapping = cfg.get("mapping")
        if mapping is None:
            raise FileNotFoundError("config.json missing 'mapping' for ParamMappedGradiendModel")

        # load mapping files (optional)
        idx_dict: Dict[str, torch.Tensor] = {}
        mask_dict: Dict[str, torch.Tensor] = {}
        index_map_dict: Dict[str, torch.Tensor] = {}

        if "indices_file" in mapping:
            path = os.path.join(load_directory, mapping["indices_file"])
            idx_dict = _load_tensor_dict(path, prefer_safetensors=path.endswith(".safetensors"))

        if "masks_file" in mapping:
            path = os.path.join(load_directory, mapping["masks_file"])
            mask_dict = _load_tensor_dict(path, prefer_safetensors=path.endswith(".safetensors"))

        if "input_index_map_file" in mapping:
            path = os.path.join(load_directory, mapping["input_index_map_file"])
            index_map_dict = _load_tensor_dict(path, prefer_safetensors=path.endswith(".safetensors"))

        # reconstruct param_map spec dict (preserve order)
        param_map_spec: Dict[str, Dict[str, Any]] = {}
        for entry in mapping["param_map"]:
            name = entry["name"]
            shape = tuple(entry["shape"])
            r = entry["repr"]

            spec: Dict[str, Any] = {"shape": shape, "repr": r}

            if r == "all":
                pass
            elif r == "indices":
                key = entry["key"]
                t = idx_dict[key]
                spec["indices"] = t.to("cpu").to(dtype=torch.long)
            elif r == "mask":
                key = entry["key"]
                t = mask_dict[key]
                spec["mask"] = t.to("cpu").to(dtype=torch.bool).reshape(shape)
            else:
                raise ValueError(f"Unknown mapping repr {r!r} for param {name}")

            param_map_spec[name] = spec

        arch = cfg["architecture"]
        meta = cfg.get("metadata") or {}

        model = cls(
            input_dim=arch["input_dim"],
            latent_dim=arch["latent_dim"],
            param_map=param_map_spec,
            activation_encoder=arch.get("activation_encoder", "tanh"),
            activation_decoder=arch.get("activation_decoder", "id"),
            bias_decoder=arch.get("bias_decoder", True),
            torch_dtype=core.torch_dtype,
            device_encoder=core.device_encoder,
            device_decoder=core.device_decoder,
            **meta,
        )

        model.load_state_dict(core.state_dict())
        model.kwargs = core.kwargs
        model.name_or_path = load_directory
        if "base_global_index_map" in index_map_dict:
            model._base_global_index_map = index_map_dict["base_global_index_map"].to("cpu").to(dtype=torch.long)
            model._base_global_index_map_version = getattr(model, "_param_map_version", 0)
        return model

    def unpruned_length(self):
        """
        Compute the total number of entries in the original unpruned input space.

        This is the sum of numel of all parameters in the mapping, regardless of selection.

        Returns:
            Total number of entries in the original input space before pruning.
        """
        total = 0
        for param_name, spec in self._param_map_items():
            shape = tuple(spec["shape"])
            numel = int(torch.tensor(shape).prod().item())
            total += numel
        return total
