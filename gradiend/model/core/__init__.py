"""
Modality-agnostic model core: backbone/head splitting and GRADIEND build from base model.
"""

from .backbone import (
    build_gradiend_from_base_model,
    debug_param_overlap,
    get_backbone_module,
    split_backbone_vs_head_params,
)

__all__ = [
    "build_gradiend_from_base_model",
    "debug_param_overlap",
    "get_backbone_module",
    "split_backbone_vs_head_params",
]
