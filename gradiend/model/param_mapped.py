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
import bisect
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch

from gradiend.model.model import GradiendModel
from gradiend.model.utils import (
    _load_tensor_dict,
    _save_tensor_dict,
    _tensor_file_name,
    _is_int32_safe,
    _bytes_per_index, _normalize_param_name,
)
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


class _CompiledParamSelector:
    """Runtime-only selector for one parameter-map entry."""

    __slots__ = ("name", "shape", "numel", "repr", "num_selected", "selector", "_device_selector")

    def __init__(self, name: str, spec: Dict[str, Any]) -> None:
        self.name = name
        self.shape = tuple(spec["shape"])
        self.numel = int(torch.tensor(self.shape).prod().item())
        self.repr = str(spec["repr"])
        self.selector: Optional[torch.Tensor] = None
        self._device_selector: Optional[torch.Tensor] = None

        if self.repr == "all":
            self.num_selected = self.numel
        elif self.repr == "mask":
            self.selector = spec["mask"].detach().to("cpu").bool().flatten()
            self.num_selected = int(self.selector.sum().item())
            if self.num_selected == 0:
                self.repr = "empty"
                self.selector = None
        elif self.repr == "indices":
            self.selector = spec["indices"].detach().to("cpu").flatten()
            if self.selector.dtype not in (torch.int32, torch.int64):
                self.selector = self.selector.long()
            self.num_selected = int(self.selector.numel())
            if self.num_selected == 0:
                self.repr = "empty"
                self.selector = None
        else:
            raise ValueError(f"Unknown param repr {self.repr!r} for {name}")

    def selector_for(self, device: torch.device) -> Optional[torch.Tensor]:
        if self.selector is None:
            return None
        if self.selector.device == device:
            return self.selector
        target_dtype = torch.long if self.repr == "indices" else self.selector.dtype
        if (
            self._device_selector is None
            or self._device_selector.device != device
            or self._device_selector.dtype != target_dtype
        ):
            self._device_selector = self.selector.to(
                device=device,
                dtype=target_dtype,
                non_blocking=False,
            )
        return self._device_selector

    def select_flat(self, flat: torch.Tensor) -> torch.Tensor:
        if self.repr == "all":
            return flat
        if self.repr == "empty":
            return flat.new_empty(0)
        selector = self.selector_for(flat.device)
        if self.repr == "mask":
            return flat[selector]
        if self.repr == "indices":
            return flat[selector]
        raise ValueError(f"Unknown compiled repr {self.repr!r}")


def _selected_positions_from_spec(spec: Dict[str, Any]) -> torch.Tensor:
    shape = tuple(spec["shape"])
    numel = int(torch.tensor(shape).prod().item())
    r = spec["repr"]
    if r == "all":
        return torch.arange(numel, dtype=torch.long)
    if r == "mask":
        return spec["mask"].flatten().nonzero(as_tuple=False).flatten().to("cpu").long()
    if r == "indices":
        return spec["indices"].to("cpu").long().flatten()
    raise ValueError(f"Unknown param repr {r!r}")


def _make_runtime_spec_from_selected_positions(
    shape: Tuple[int, ...],
    sel_positions: torch.Tensor,
    *,
    margin: float = 0.80,
) -> Dict[str, Any]:
    """Choose an efficient in-memory mapping representation for selected positions."""
    numel = int(torch.tensor(shape).prod().item())
    sel_positions = sel_positions.to("cpu").long().flatten()
    k = int(sel_positions.numel())
    if k == numel:
        return {"shape": shape, "repr": "all"}
    if k == 0:
        dtype = torch.int32 if _is_int32_safe(numel) else torch.int64
        return {"shape": shape, "repr": "indices", "indices": torch.empty(0, dtype=dtype)}
    bpi = _bytes_per_index(numel)
    est_indices = k * bpi
    est_mask = numel
    if est_indices < est_mask * margin:
        dtype = torch.int32 if _is_int32_safe(numel) else torch.int64
        return {"shape": shape, "repr": "indices", "indices": sel_positions.to(dtype=dtype)}
    mask = torch.zeros(numel, dtype=torch.bool)
    mask[sel_positions] = True
    return {"shape": shape, "repr": "mask", "mask": mask.reshape(shape)}


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
        lazy_init: bool = False,
        **kwargs: Any,
    ) -> None:
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
            lazy_init: If True, do not create encoder/decoder weights; build on prune.
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
            lazy_init=lazy_init,
            **kwargs,
        )
        self.param_map = param_map
        self._base_global_index_map: Optional[torch.Tensor] = None
        self._base_global_index_map_version: int = 0
        self._param_map_version: int = 0
        self._compiled_param_selectors: Optional[List[_CompiledParamSelector]] = None

    def _param_map_items(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        # dict insertion order is the mapping order
        yield from self.param_map.items()

    def _invalidate_runtime_param_map_cache(self) -> None:
        self._compiled_param_selectors = None

    def _compile_param_selectors(self) -> List[_CompiledParamSelector]:
        selectors = [_CompiledParamSelector(name, spec) for name, spec in self._param_map_items()]
        selected_total = sum(selector.num_selected for selector in selectors)
        if selected_total != int(self.input_dim):
            raise ValueError(
                f"Inconsistent compiled param selectors: expected input_dim={self.input_dim}, "
                f"got selected_total={selected_total}"
            )
        return selectors

    def _get_compiled_param_selectors(self) -> List[_CompiledParamSelector]:
        if self._compiled_param_selectors is None:
            self._compiled_param_selectors = self._compile_param_selectors()
        return self._compiled_param_selectors

    def _normalize_param_map_reprs(self) -> None:
        """Normalize param_map representations without changing selected dimensions."""
        self.param_map = {
            name: _make_runtime_spec_from_selected_positions(
                tuple(spec["shape"]),
                _selected_positions_from_spec(spec),
            )
            for name, spec in self._param_map_items()
        }
        self._base_global_index_map = None
        self._invalidate_runtime_param_map_cache()

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

        for selector in self._get_compiled_param_selectors():
            numel = selector.numel
            if selector.repr == "all":
                sel_positions = torch.arange(numel, dtype=torch.long)
            elif selector.repr == "empty":
                sel_positions = torch.empty(0, dtype=torch.long)
            elif selector.repr == "mask":
                sel_positions = selector.selector.nonzero(as_tuple=False).flatten().to("cpu").long()
            elif selector.repr == "indices":
                sel_positions = selector.selector.to("cpu").long()
            else:
                raise ValueError(f"Unknown param repr {selector.repr!r} for {selector.name}")

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

    def decode_base_global_index(self, base_global_index: int) -> Dict[str, Any]:
        """
        Decode one base-global index into parameter-local coordinates.

        Args:
            base_global_index: Index in the flattened base-parameter space.

        Returns:
            Dict with parameter name, shape, flat index within that parameter, and
            coordinate tuple/list.
        """
        if isinstance(base_global_index, bool) or not isinstance(base_global_index, int):
            raise TypeError(
                f"base_global_index must be int, got {type(base_global_index).__name__}"
            )

        offsets: List[int] = []
        meta: List[Tuple[str, Tuple[int, ...], int]] = []
        base_offset = 0
        for param_name, spec in self._param_map_items():
            shape = tuple(spec["shape"])
            numel = int(torch.tensor(shape).prod().item())
            offsets.append(base_offset)
            meta.append((param_name, shape, numel))
            base_offset += numel

        if base_global_index < 0 or base_global_index >= base_offset:
            raise IndexError(
                f"base_global_index {base_global_index} out of bounds for total size {base_offset}"
            )

        param_pos = bisect.bisect_right(offsets, base_global_index) - 1
        param_name, shape, numel = meta[param_pos]
        param_offset = offsets[param_pos]
        flat_index = int(base_global_index - param_offset)
        coords = tuple(int(x) for x in torch.unravel_index(torch.tensor(flat_index), shape))

        return {
            "param_name": param_name,
            "param_shape": list(shape),
            "param_numel": numel,
            "flat_index_in_param": flat_index,
            "coords": list(coords),
            "param_row": coords[0] if len(coords) >= 1 else None,
            "param_col": coords[1] if len(coords) >= 2 else None,
        }

    def decode_base_global_indices(self, base_global_indices: List[int]) -> List[Dict[str, Any]]:
        """Vectorized convenience wrapper around decode_base_global_index()."""
        return [self.decode_base_global_index(int(idx)) for idx in base_global_indices]

    # -----------------------------
    # gradient extraction / IO
    # -----------------------------
    def extract_gradients(
        self,
        model: torch.nn.Module,
        return_dict: bool = False,
        target_device: Optional[torch.device] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract gradients from a base model (copies).

        Returns either:

        - dict[param_name] -> gradient tensor shaped like the parameter, OR
        - a single concatenated 1D tensor in GRADIEND input space

        When target_device is set, gradient chunks are moved there incrementally during
        extraction, reducing peak memory on the base model GPU (avoids holding a full
        gradient copy there in addition to the concatenated result).

        Args:
            model: Base model that has parameter gradients populated (after backward).
            return_dict: If True, return a dict of per-parameter gradients. If False,
                return a flattened 1D tensor in GRADIEND input space.
            target_device: If set, move each gradient chunk to this device before
                concatenation. Use the encoder device to avoid 2x gradient peak on
                the base model GPU.

        Returns:
            If return_dict is True:
                Dict[param_name, grad_tensor] where each tensor matches the parameter shape.
            If return_dict is False:
                1D tensor containing only the selected entries (per param_map) concatenated
                in param_map order.

        Raises:
            RuntimeError: If any required parameter gradient is None.
        """

        def _get_param_for_map_name(name: str):
            p = param_lookup.get(name)
            if p is not None:
                return p
            p = param_lookup.get(_normalize_param_name(name))
            if p is not None:
                return p
            # optional DS/DDP fallback if normalize doesn't cover all wrappers
            p = param_lookup.get(f"module.{name}")
            if p is not None:
                return p
            raise KeyError(
                f"Parameter '{name}' not found in model.named_parameters(). "
                f"Examples: {list(param_lookup.keys())[:10]}"
            )


        param_lookup = {}
        base_model = model.module if hasattr(model, "module") else model
        for n, p in base_model.named_parameters():
            param_lookup[n] = p
            param_lookup.setdefault(_normalize_param_name(n), p)

        selectors = self._get_compiled_param_selectors()
        active_selectors = [selector for selector in selectors if selector.num_selected > 0]

        # ensure grads exist
        missing = []
        for selector in active_selectors:
            p = _get_param_for_map_name(selector.name)
            if p.grad is None:
                missing.append(selector.name)
        if missing:
            raise RuntimeError(
                f"Gradients are None for parameters: {missing}. "
                "This indicates a bug in gradient computation (no backward, no_grad, requires_grad=False, ...)."
            )

        # When streaming to CPU, avoid clone on GPU (saves memory for large models).
        # Slice then move; .to(cpu) creates the copy. When staying on GPU, clone() is
        # required so factual/counterfactual copies persist (e.g. for source="diff").
        stream_to_cpu = (
            target_device is not None
            and str(target_device).split(":")[0] == "cpu"
        )

        if return_dict:
            out = {}
            empty_device = target_device
            if empty_device is None:
                for selector in active_selectors:
                    empty_device = _get_param_for_map_name(selector.name).grad.device
                    break
            empty_device = empty_device or torch.device("cpu")
            for selector in selectors:
                if selector.num_selected == 0:
                    out[selector.name] = torch.zeros(selector.shape, dtype=self.torch_dtype, device=empty_device)
                    continue
                p = _get_param_for_map_name(selector.name)
                g = p.grad.detach()
                if stream_to_cpu and g.device.type != "cpu":
                    g = g.to(target_device, non_blocking=False)
                elif not stream_to_cpu:
                    g = g.clone()
                out[selector.name] = g
            return out

        parts: List[torch.Tensor] = []
        for selector in active_selectors:
            p = _get_param_for_map_name(selector.name)
            grad = p.grad
            if stream_to_cpu and grad.device.type != "cpu":
                g = grad.detach().flatten()
            else:
                g = grad.detach().clone().flatten()

            chunk = selector.select_flat(g)
            if target_device is not None and chunk.device != target_device:
                chunk = chunk.to(target_device, non_blocking=False)
            parts.append(chunk)

        return torch.concat(parts) if parts else torch.empty(0, dtype=self.torch_dtype, device=target_device or torch.device("cpu"))

    def extract_gradients_streaming(
        self,
        model: torch.nn.Module,
        backward_fn: Callable[[], Any],
        return_dict: bool = False,
        target_device: Optional[torch.device] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Extract mapped base-model gradients while backward is running.

        Hooks collect only the entries selected by ``param_map``. On PyTorch versions
        with ``register_post_accumulate_grad_hook``, full ``p.grad`` tensors are cleared
        as soon as each parameter finishes accumulating, reducing peak memory for large
        frozen base models used only as gradient generators.

        Args:
            model: Base model whose backward pass will populate gradients.
            backward_fn: Callable that runs the base-model backward pass.
            return_dict: If True, return a per-parameter dict. Unselected entries are
                zero-filled for masked/indexed parameters.
            target_device: Optional device to move selected chunks to immediately.
        """

        if target_device is not None and not isinstance(target_device, torch.device):
            target_device = torch.device(target_device)

        base_model = model.module if hasattr(model, "module") else model
        param_lookup = {}
        for n, p in base_model.named_parameters():
            param_lookup[n] = p
            param_lookup.setdefault(_normalize_param_name(n), p)

        def _get_param_for_map_name(name: str):
            p = param_lookup.get(name)
            if p is not None:
                return p
            p = param_lookup.get(_normalize_param_name(name))
            if p is not None:
                return p
            p = param_lookup.get(f"module.{name}")
            if p is not None:
                return p
            raise KeyError(
                f"Parameter '{name}' not found in model.named_parameters(). "
                f"Examples: {list(param_lookup.keys())[:10]}"
            )

        collected: Dict[str, torch.Tensor] = {}
        handles = []
        mapped_params = []
        selectors = self._get_compiled_param_selectors()
        active_selectors = [selector for selector in selectors if selector.num_selected > 0]

        def _select_chunk(grad: torch.Tensor, selector: _CompiledParamSelector) -> torch.Tensor:
            flat = grad.detach().flatten()
            chunk = selector.select_flat(flat)

            if target_device is not None and chunk.device != target_device:
                return chunk.to(target_device, non_blocking=False)
            return chunk.clone()

        for selector in active_selectors:
            p = _get_param_for_map_name(selector.name)
            mapped_params.append(p)

            def _make_hook(name: str, param_selector: _CompiledParamSelector):
                def _hook(grad: torch.Tensor):
                    chunk = _select_chunk(grad, param_selector)
                    if name in collected:
                        collected[name] = collected[name] + chunk
                    else:
                        collected[name] = chunk
                    return grad

                return _hook

            handles.append(p.register_hook(_make_hook(selector.name, selector)))
            if hasattr(p, "register_post_accumulate_grad_hook"):
                def _clear_grad(param: torch.nn.Parameter):
                    param.grad = None

                handles.append(p.register_post_accumulate_grad_hook(_clear_grad))

        try:
            backward_fn()
        finally:
            for h in handles:
                h.remove()
            for p in mapped_params:
                p.grad = None

        missing = [selector.name for selector in active_selectors if selector.name not in collected]
        if missing:
            raise RuntimeError(
                f"Gradients were not collected for parameters: {missing}. "
                "This indicates a bug in gradient computation (no backward, no_grad, requires_grad=False, ...)."
            )

        if not return_dict:
            parts = [collected[selector.name] for selector in active_selectors]
            if not parts:
                dev = target_device or torch.device("cpu")
                return torch.empty(0, dtype=self.torch_dtype, device=dev)
            return torch.concat(parts)

        out: Dict[str, torch.Tensor] = {}
        empty_device = target_device
        if empty_device is None:
            for selector in active_selectors:
                if selector.name in collected:
                    empty_device = collected[selector.name].device
                    break
        empty_device = empty_device or torch.device("cpu")
        for selector in selectors:
            shape = selector.shape
            r = selector.repr
            if r == "empty":
                out[selector.name] = torch.zeros(shape, dtype=self.torch_dtype, device=empty_device)
                continue
            chunk = collected[selector.name]
            if r == "all":
                out[selector.name] = chunk.reshape(shape)
            elif r == "mask":
                m = selector.selector_for(chunk.device)
                full = torch.zeros(int(m.numel()), dtype=chunk.dtype, device=chunk.device)
                full[m] = chunk
                out[selector.name] = full.reshape(shape)
            elif r == "indices":
                idx = selector.selector_for(chunk.device).to(dtype=torch.long)
                full = torch.zeros(int(torch.tensor(shape).prod().item()), dtype=chunk.dtype, device=chunk.device)
                full[idx] = chunk
                out[selector.name] = full.reshape(shape)
            else:
                raise ValueError(f"Unknown param repr {r!r} for {selector.name}")
        return out

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
        for selector in self._get_compiled_param_selectors():
            if selector.num_selected == 0:
                continue
            t = grad_dict[selector.name]
            flat = t.flatten()
            parts.append(selector.select_flat(flat))
        if parts:
            return torch.concat(parts)
        dev = next(iter(grad_dict.values())).device if grad_dict else torch.device("cpu")
        return torch.empty(0, dtype=self.torch_dtype, device=dev)

    def forward(
        self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], return_encoded: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
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
            for selector in self._get_compiled_param_selectors():
                if selector.num_selected == 0:
                    orig_shapes[selector.name] = ("empty", selector.shape)
                    continue
                t = x[selector.name]
                flat = t.flatten()
                sel = selector.select_flat(flat)
                orig_shapes[selector.name] = (selector.repr, t.shape, selector)
                parts.append(sel)

            x = torch.concat(parts) if parts else torch.empty(0, dtype=self.torch_dtype, device=self.device_encoder)
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
                    selector = info[2]
                    mask = selector.selector_for(decoded.device)
                    n = selector.num_selected
                    full = torch.zeros(int(mask.numel()), dtype=decoded.dtype, device=decoded.device)
                    full[mask] = decoded[start : start + n]
                    out[param_name] = full.reshape(shape)
                    start += n
                elif kind == "indices":
                    selector = info[2]
                    idx = selector.selector_for(decoded.device).to(dtype=torch.long)
                    n = selector.num_selected
                    full = torch.zeros(int(torch.tensor(shape).prod().item()), dtype=decoded.dtype, device=decoded.device)
                    full[idx] = decoded[start : start + n]
                    out[param_name] = full.reshape(shape)
                    start += n
                elif kind == "empty":
                    out[param_name] = torch.zeros(shape, dtype=decoded.dtype, device=decoded.device)
                else:
                    raise ValueError(kind)

            decoded = out

        return (decoded, encoded) if return_encoded else decoded

    def forward_encoder(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
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
            for selector in self._get_compiled_param_selectors():
                if selector.num_selected == 0:
                    continue
                flat = x[selector.name].flatten()
                parts.append(selector.select_flat(flat))
            x = torch.concat(parts) if parts else torch.empty(0, dtype=self.torch_dtype, device=self.device_encoder)
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
        keep_idx: Optional[torch.Tensor] = None,
        keep_idx_sorted_unique: bool = False,
        inplace: bool = False,
        return_mask: bool = False,
    ) -> Union["ParamMappedGradiendModel", Tuple["ParamMappedGradiendModel", torch.Tensor]]:
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
            keep_idx: optional 1D tensor of input-space indices to keep. Bypasses dense mask/importance materialization.
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
        if keep_idx is not None and (topk is not None or threshold is not None or mask is not None or importance is not None):
            raise ValueError("keep_idx cannot be combined with topk, threshold, mask, or importance.")
        if topk is None and threshold is None and mask is None and keep_idx is None:
            raise ValueError("At least one of topk, threshold, mask must be provided.")

        # topk=1.0 (float) means no pruning (return self); topk int 1 means keep top-1 dimension
        if topk is not None and isinstance(topk, float) and topk == 1.0:
            m = self if inplace else copy.deepcopy(self)
            m._normalize_param_map_reprs()
            if return_mask:
                old_input_dim = int(m.input_dim)
                full_mask = torch.ones(old_input_dim, dtype=torch.bool)
                return m, full_mask
            return m

        m = self if inplace else copy.deepcopy(self)
        old_input_dim = int(m.input_dim)

        combined = None
        if keep_idx is not None:
            if not torch.is_tensor(keep_idx):
                raise TypeError("keep_idx must be a tensor.")
            keep_idx = keep_idx.detach().to(dtype=torch.long, device="cpu").flatten()
            if keep_idx.numel() == 0:
                raise ValueError("keep_idx must not be empty.")
            if int(keep_idx.min().item()) < 0 or int(keep_idx.max().item()) >= old_input_dim:
                raise ValueError(f"keep_idx out of bounds for input_dim={old_input_dim}")
            if not keep_idx_sorted_unique:
                keep_idx = torch.unique(keep_idx, sorted=True)
        else:
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
                n = numel
                sel_positions = None
            else:
                sel_positions = _selected_positions_from_spec(spec)
                n = int(sel_positions.numel())

            # keep_cpu is sorted, so locate each parameter segment with two scalar
            # searches instead of allocating a keep-sized boolean vector per parameter.
            lo = int(torch.searchsorted(keep_cpu, offset, right=False).item())
            hi = int(torch.searchsorted(keep_cpu, offset + n, right=False).item())
            seg_keep = keep_cpu[lo:hi] - offset

            if r == "all":
                new_sel = seg_keep
            elif seg_keep.numel() > 0:
                new_sel = sel_positions[seg_keep]
            else:
                new_sel = torch.empty(0, dtype=torch.long)
            new_param_map[param_name] = _make_runtime_spec_from_selected_positions(shape, new_sel)

            offset += n

        if offset != old_input_dim:
            raise ValueError(f"Inconsistent mapping: expected {old_input_dim} selected total, got {offset}")

        new_in = int(keep_idx.numel())

        if m.encoder is None:
            # Lazy init: build encoder/decoder at pruned size (no copy, fresh init)
            m._build_encoder_decoder(new_in)
        else:
            m = m._prune_input_dims(
                keep_idx,
                inplace=True,
                return_index_map=False,
                keep_idx_sorted_unique=keep_idx_sorted_unique,
            )

        m.param_map = new_param_map
        m._param_map_version = getattr(m, "_param_map_version", 0) + 1
        m._base_global_index_map = None
        m._invalidate_runtime_param_map_cache()

        if return_mask:
            if combined is None:
                combined = torch.zeros(old_input_dim, dtype=torch.bool)
                combined[keep_idx] = True
            return m, combined
        return m

    # -----------------------------
    # save/load (mapping-aware)
    # -----------------------------
    def save_pretrained(self, save_directory: str, use_safetensors: Optional[bool] = None, **kwargs: Any) -> None:
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

        # base_global_index_map is always reconstructible from param_map via
        # _build_base_global_index_map(); do not persist it (wastes hundreds of MB on unpruned models).

        cfg["mapping"] = mapping

        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        load_directory: str,
        device_encoder: Optional[torch.device] = None,
        device_decoder: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "ParamMappedGradiendModel":
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
        model._normalize_param_map_reprs()
        model._compiled_param_selectors = model._compile_param_selectors()
        return model

    def unpruned_length(self) -> int:
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
