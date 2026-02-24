"""
Modality-agnostic backbone vs head splitting and GRADIEND build from base model.

Uses Hugging Face convention (base_model_prefix / base_model) to identify backbone
parameters; all other parameters (prediction heads, poolers, etc.) are excluded.
"""

import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch

from gradiend.model import ParamMappedGradiendModel
from gradiend.model.utils import freeze_params_until_target
from gradiend.util import get_logger

logger = get_logger(__name__)


def get_backbone_module(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the backbone (core) submodule of a Hugging Face–style model.

    Uses base_model_prefix (e.g. "bert", "roberta", "model") when present,
    otherwise base_model, else the full model.
    """
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    if hasattr(model, "base_model"):
        return model.base_model
    return model


def split_backbone_vs_head_params(
    model: torch.nn.Module,
) -> Tuple[OrderedDict[str, torch.Tensor], List[Dict[str, Any]]]:
    """
    Split model parameters into backbone (core) and head (excluded) by identity.

    Backbone is determined by get_backbone_module(model). Any parameter that
    is not part of the backbone's parameters(recurse=True) is considered head.

    Returns:
        core: OrderedDict of (name, parameter) for backbone parameters only.
        excluded: List of dicts with name, py_id, data_ptr, shape, requires_grad, device, dtype.
    """
    backbone = get_backbone_module(model)
    backbone_param_ids = {id(p) for p in backbone.parameters(recurse=True)}

    core = OrderedDict()
    excluded: List[Dict[str, Any]] = []

    for name, p in model.named_parameters(recurse=True):
        if id(p) in backbone_param_ids:
            core[name] = p
        else:
            excluded.append({
                "name": name,
                "py_id": id(p),
                "data_ptr": int(p.data_ptr()),
                "shape": tuple(p.shape),
                "requires_grad": bool(p.requires_grad),
                "device": str(p.device),
                "dtype": str(p.dtype),
            })

    return core, excluded


def debug_param_overlap(model: torch.nn.Module) -> Dict[str, int]:
    """
    Return counts of backbone vs full model parameters (for sanity checks).
    """
    backbone = get_backbone_module(model)
    backbone_ids = {id(p) for p in backbone.parameters(recurse=True)}
    model_ids = {id(p) for _, p in model.named_parameters(recurse=True)}
    return {
        "backbone_param_count": len(backbone_ids),
        "model_param_count": len(model_ids),
        "shared_param_count": len(backbone_ids & model_ids),
    }


def _normalize_param_map_arg(param_map: Any) -> List[str]:
    """Normalize param_map to a list (unwrap single-element list of list)."""
    param_map = param_map or []
    if len(param_map) == 1 and isinstance(param_map[0], list):
        param_map = param_map[0]
    return list(param_map)


def _filter_params_by_include(
    param_lookup: OrderedDict[str, torch.Tensor],
    params: Optional[List[str]],
) -> OrderedDict[str, torch.Tensor]:
    """
    Filter param_lookup to names that match params (exact or wildcard).
    If params is None, return param_lookup unchanged.
    """
    if not params:
        return param_lookup
    keys = list(param_lookup.keys())
    matched = []
    for pattern in params:
        if pattern in param_lookup:
            matched.append(pattern)
            continue
        regex = re.compile(pattern.replace(".", r"\.").replace("*", r".*"))
        for k in keys:
            if k not in matched and regex.fullmatch(k):
                matched.append(k)
    # Preserve order of param_lookup
    return OrderedDict((k, param_lookup[k]) for k in keys if k in matched)


def build_gradiend_from_base_model(
    base_model: torch.nn.Module,
    load_directory: str,
    param_map: Optional[List[str]] = None,
    params: Optional[List[str]] = None,
    latent_dim: int = 1,
    torch_dtype: Optional[torch.dtype] = None,
    device_encoder: Optional[torch.device] = None,
    device_decoder: Optional[torch.device] = None,
    lazy_init: bool = False,
    **kwargs: Any,
) -> ParamMappedGradiendModel:
    """
    Build a ParamMappedGradiendModel from a base model (non–GRADIEND checkpoint).

    Backbone vs head is determined by Hugging Face backbone (base_model_prefix
    / base_model). Only backbone parameters are included unless params or
    param_map further restrict them.

    Args:
        base_model: The HF-style model.
        load_directory: Path/identifier for saving and base_model reference.
        param_map: Optional list of parameter names or wildcards; applied after
            backbone/params. None = use all backbone (or all matching params).
        params: Optional list of parameter names or wildcards to include.
            None = all backbone params. Applied before param_map. Enables
            future params selection processes.
        latent_dim, torch_dtype, device_encoder, device_decoder: Passed to
            ParamMappedGradiendModel.
        **kwargs: Other arguments for ParamMappedGradiendModel (excluding source/target).

    Returns:
        ParamMappedGradiendModel with spec-dict param_map and correct input_dim.
    """
    if torch_dtype is None:
        torch_dtype = torch.float32

    core, excluded = split_backbone_vs_head_params(base_model)
    overlap = debug_param_overlap(base_model)
    logger.debug(
        "Backbone vs model params: backbone=%s model=%s shared=%s",
        overlap["backbone_param_count"],
        overlap["model_param_count"],
        overlap["shared_param_count"],
    )
    if excluded:
        logger.info(
            "Excluded %d non-backbone (head) parameter(s); names: %s",
            len(excluded),
            [e["name"] for e in excluded],
        )
        for e in excluded:
            logger.debug(
                "Excluded param: name=%r py_id=%s data_ptr=%s shape=%s requires_grad=%s device=%s dtype=%s",
                e["name"],
                e["py_id"],
                e["data_ptr"],
                e["shape"],
                e["requires_grad"],
                e["device"],
                e["dtype"],
            )

    param_lookup = _filter_params_by_include(core, params)
    param_map = _normalize_param_map_arg(param_map)

    if param_map:
        if isinstance(param_map, dict):
            raise TypeError(
                "param_map must be None or list[str] (optionally with wildcards). "
                "Mask dicts are not supported in the new API; use prune() after construction."
            )
        matched_params = []
        for param_name in param_map:
            if param_name in param_lookup:
                matched_params.append(param_name)
            else:
                param_pattern = param_name.replace(".", r"\.").replace("*", r".*")
                param_regex = re.compile(param_pattern)
                for param_candidate in param_lookup:
                    if param_regex.fullmatch(param_candidate):
                        matched_params.append(param_candidate)
                        break
        param_map = list(
            sorted(matched_params, key=lambda x: list(param_lookup.keys()).index(x))
        )
    else:
        param_map = list(param_lookup.keys())

    param_map_spec: Dict[str, Dict[str, Any]] = {}
    for name in param_map:
        if name not in param_lookup:
            raise KeyError(
                f"Param {name!r} not found in backbone parameters (param_lookup)."
            )
        p = param_lookup[name]
        param_map_spec[name] = {"shape": tuple(p.shape), "repr": "all"}

    input_dim = int(sum(p.numel() for p in (param_lookup[n] for n in param_map)))

    gradiend_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in ("source", "target", "params")
    }

    gradiend = ParamMappedGradiendModel(
        input_dim,
        param_map=param_map_spec,
        latent_dim=latent_dim,
        base_model=load_directory,
        torch_dtype=torch_dtype,
        device_encoder=device_encoder,
        device_decoder=device_decoder,
        lazy_init=lazy_init,
        **gradiend_kwargs,
    )

    freeze_params_until_target(base_model, *list(param_map_spec.keys()))
    return gradiend
