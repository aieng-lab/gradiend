"""Shared helpers for comparison matrix builders."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import pandas as pd


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
            try:
                from gradiend.trainer.core.seed_models import SeedModelGroup

                if isinstance(value, SeedModelGroup):
                    group = list(value.models)
                else:
                    group = [value]
            except ImportError:
                group = [value]
        if not group:
            raise ValueError(f"Model group {mid!r} must contain at least one model")
        normalized[mid] = group
    return normalized


def encoder_dataframes_from_summary(payload: Any) -> List[pd.DataFrame]:
    """Return one or more encoder DataFrames from a cross-task encoder summary entry."""
    if not isinstance(payload, dict):
        return []
    encoder_dfs = payload.get("encoder_dfs")
    if isinstance(encoder_dfs, list):
        return [
            df for df in encoder_dfs if isinstance(df, pd.DataFrame) and not df.empty
        ]
    encoder_df = payload.get("encoder_df")
    if isinstance(encoder_df, pd.DataFrame) and not encoder_df.empty:
        return [encoder_df]
    return []


def _validate_aggregate_dispersion_combo(seed_aggregate: str, dispersion: str) -> None:
    valid_aggregates = {"mean", "median", "min", "max"}
    valid_dispersion = {"none", "std", "range", "minmax"}
    if seed_aggregate not in valid_aggregates:
        raise ValueError(f"seed_aggregate must be one of {sorted(valid_aggregates)}, got {seed_aggregate!r}")
    if dispersion not in valid_dispersion:
        raise ValueError(f"dispersion must be one of {sorted(valid_dispersion)}, got {dispersion!r}")
    if dispersion == "range" and seed_aggregate in {"min", "max"}:
        raise ValueError("dispersion='range' does not make sense with seed_aggregate='min' or 'max'")


def comparison_matrix_from_cell_stat(
    comparison_data: Dict[str, Any],
    field: str = "std",
) -> Dict[str, Any]:
    """Build a heatmap payload whose cells are a per-cell dispersion statistic."""
    cell_stats = comparison_data.get("cell_stats")
    if not isinstance(cell_stats, list):
        raise ValueError("comparison_data must contain list-valued 'cell_stats'")
    matrix: List[List[float]] = []
    for row in cell_stats:
        if not isinstance(row, list):
            raise ValueError("cell_stats must be a rectangular list of dicts")
        matrix.append(
            [
                float(stat[field])
                if isinstance(stat, dict) and isinstance(stat.get(field), (int, float))
                else float("nan")
                for stat in row
            ]
        )
    payload = dict(comparison_data)
    payload["matrix"] = matrix
    measure = str(comparison_data.get("measure") or "comparison")
    payload["measure"] = f"{measure}_{field}"
    payload.pop("cell_stats", None)
    payload.pop("row_normalized_by_diagonal", None)
    return payload


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


def _encoder_label_correlation(encoder_df: Any) -> Optional[float]:
    """Pearson correlation between encoded values and binary labels (non-neutral, label != 0)."""
    if not hasattr(encoder_df, "columns"):
        return None
    if "encoded" not in encoder_df.columns or "label" not in encoder_df.columns:
        return None
    work = encoder_df
    if "type" in work.columns:
        work = work[~work["type"].astype(str).str.contains("neutral", case=False, na=False)]
    labels = work["label"].astype(float)
    mask = labels != 0.0
    if int(mask.sum()) < 2:
        return None
    encoded = work.loc[mask, "encoded"].astype(float)
    labels = labels.loc[mask]
    if float(labels.std()) == 0.0 or float(encoded.std()) == 0.0:
        return None
    return _pearson(encoded.tolist(), labels.tolist())


def orient_encoder_df_by_label_correlation(encoder_df: Any, *, threshold: float = 0.0) -> Any:
    """Flip encoded signs when label correlation is negative (anchor-aligned frame)."""
    import pandas as pd

    if not hasattr(encoder_df, "columns") or encoder_df.empty:
        return encoder_df
    corr = _encoder_label_correlation(encoder_df)
    if corr is None or corr >= threshold:
        return encoder_df
    oriented = encoder_df.copy()
    oriented["encoded"] = -oriented["encoded"].astype(float)
    return oriented
