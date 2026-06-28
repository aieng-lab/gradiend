"""Shared helpers for pre-prune mask vs trained-oracle recall experiments."""

from __future__ import annotations

from typing import Any, Set, Tuple

import torch

TOPK_EVAL = 1000
TOPK_PART = "decoder-weight"


def ref_recall_metrics(heuristic: Set[int], ref: Set[int]) -> Tuple[float, float]:
    """Recall/precision vs a fixed-size oracle top-k (``TOPK_EVAL``)."""
    if not ref:
        return 0.0, 0.0
    inter = len(heuristic & ref)
    recall = inter / TOPK_EVAL
    precision = inter / len(heuristic) if heuristic else 0.0
    return recall, precision


def topk_base_global(model: Any, *, topk: int, part: str) -> Set[int]:
    local_indices = model.gradiend.get_topk_weights(part=part, topk=topk)
    if not local_indices:
        return set()
    base_map = model.gradiend._get_base_global_index_map()
    idx_t = torch.as_tensor(local_indices, dtype=torch.long)
    return {int(x) for x in base_map[idx_t].tolist()}


def all_kept_base_global(model: Any) -> Set[int]:
    """Map every kept GRADIEND input dim to base-global indices."""
    local_indices = torch.arange(int(model.gradiend.input_dim), dtype=torch.long)
    base_map = model.gradiend._get_base_global_index_map()
    return {int(x) for x in base_map[local_indices].tolist()}
