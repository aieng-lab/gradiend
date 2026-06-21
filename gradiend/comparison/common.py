"""Shared helpers for comparison matrix builders."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


GroupingSpec = Union[str, Dict[str, str], Callable[[str], str], None]


def _validate_models(models: Dict[str, object]) -> None:
    if not isinstance(models, dict):
        raise TypeError("models must be a dict mapping labels to models")
    if len(models) < 2:
        raise ValueError("At least 2 models are required.")
    if not all(isinstance(mid, str) for mid in models.keys()):
        raise TypeError("models keys must be strings")


def _validate_topk_optional(topk: Optional[Union[int, float]]) -> None:
    if topk is None:
        return
    if isinstance(topk, bool):
        raise TypeError("topk must be int, float, or None")
    if isinstance(topk, int):
        if topk <= 0:
            raise ValueError("topk as int must be > 0")
        return
    if isinstance(topk, float):
        if not (0.0 < topk <= 1.0):
            raise ValueError("topk as float must be in (0, 1.0]")
        return
    raise TypeError("topk must be int, float, or None")


def _normalize_model_groups(models: Dict[str, object]) -> Dict[str, List[object]]:
    if not isinstance(models, dict):
        raise TypeError("models must be a dict mapping labels to models or model lists")
    if len(models) < 2:
        raise ValueError("At least 2 models are required.")
    normalized: Dict[str, List[object]] = {}
    for mid, value in models.items():
        if not isinstance(mid, str):
            raise TypeError("models keys must be strings")
        if isinstance(value, (list, tuple)):
            group = list(value)
        else:
            group = [value]
        if not group:
            raise ValueError(f"Model group {mid!r} must contain at least one model")
        normalized[mid] = group
    return normalized


def _validate_aggregate_dispersion_combo(seed_aggregate: str, dispersion: str) -> None:
    valid_aggregates = {"mean", "median", "min", "max"}
    valid_dispersion = {"none", "std", "range", "minmax"}
    if seed_aggregate not in valid_aggregates:
        raise ValueError(f"seed_aggregate must be one of {sorted(valid_aggregates)}, got {seed_aggregate!r}")
    if dispersion not in valid_dispersion:
        raise ValueError(f"dispersion must be one of {sorted(valid_dispersion)}, got {dispersion!r}")
    if dispersion == "range" and seed_aggregate in {"min", "max"}:
        raise ValueError("dispersion='range' does not make sense with seed_aggregate='min' or 'max'")


def _aggregate_seed_scores(values: Sequence[float], *, seed_aggregate: str, dispersion: str) -> Dict[str, Any]:
    scores = [float(v) for v in values]
    if not scores:
        raise ValueError("Cannot aggregate an empty score list")
    ordered = sorted(scores)
    n = len(ordered)
    if seed_aggregate == "mean":
        aggregate = float(sum(ordered) / n)
    elif seed_aggregate == "median":
        mid = n // 2
        aggregate = float(ordered[mid] if n % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid]))
    elif seed_aggregate == "min":
        aggregate = float(ordered[0])
    elif seed_aggregate == "max":
        aggregate = float(ordered[-1])
    else:
        raise ValueError(f"Unsupported seed_aggregate {seed_aggregate!r}")
    result: Dict[str, Any] = {
        "aggregate": aggregate,
        "n": n,
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "scores": scores,
    }
    if dispersion == "std":
        mean = float(sum(ordered) / n)
        result["std"] = float(math.sqrt(sum((v - mean) ** 2 for v in ordered) / n))
    elif dispersion == "range":
        result["range_half_width"] = float((ordered[-1] - ordered[0]) / 2.0)
    elif dispersion == "minmax":
        result["minmax"] = [float(ordered[0]), float(ordered[-1])]
    return result


def _rankdata_average(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        current = indexed[pos][1]
        while end < len(indexed) and indexed[end][1] == current:
            end += 1
        avg_rank = (pos + end - 1) / 2.0 + 1.0
        for idx, _ in indexed[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n == 0 or n != len(y):
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return float(num / (den_x * den_y))


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric
