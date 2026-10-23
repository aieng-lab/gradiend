"""Output format conversions for text-prediction data creation."""

from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd

UNIFIED_FACTUAL = "factual"
UNIFIED_FACTUAL_CLASS = "factual_class"
UNIFIED_MASKED = "masked"
UNIFIED_SPLIT = "split"


def _to_minimal(class_dfs: Dict[str, pd.DataFrame], extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Convert per-class to minimal (masked, label, label_class, split), plus ``extra_cols`` (the alternative columns of paired targets)."""
    rows = []
    for class_id, df in class_dfs.items():
        for _, row in df.iterrows():
            rows.append({
                "masked": row["masked"],
                "label": row["label"],
                "label_class": class_id,
                "split": row["split"],
                **{c: row[c] for c in extra_cols},
            })
    return pd.DataFrame(rows)


def _to_merged(class_dfs: Dict[str, pd.DataFrame], extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Build training DataFrame (factual only): masked, split, label_class, label, feature_class_id.

    feature_class_id is the string id from TextFilterConfig (same as label_class per row). ``extra_cols``
    (the alternative columns of paired targets) are appended.
    Splits are applied per feature class in _apply_auto_split.
    """
    rows = []
    for class_id, class_df in class_dfs.items():
        if class_df is None or class_df.empty:
            continue
        for _, row in class_df.iterrows():
            rows.append(
                {
                    UNIFIED_MASKED: row[UNIFIED_MASKED],
                    UNIFIED_SPLIT: row[UNIFIED_SPLIT],
                    UNIFIED_FACTUAL_CLASS: class_id,
                    UNIFIED_FACTUAL: row["label"],
                    **{c: row[c] for c in extra_cols},
                }
            )
    df = pd.DataFrame(
        rows,
        columns=[UNIFIED_MASKED, UNIFIED_SPLIT, UNIFIED_FACTUAL_CLASS, UNIFIED_FACTUAL, *extra_cols],
    )
    df = df.rename(columns={UNIFIED_FACTUAL_CLASS: "label_class", UNIFIED_FACTUAL: "label"})
    df["feature_class_id"] = df["label_class"]
    return df


def _to_partial_merged(class_dfs: Dict[str, pd.DataFrame], extra_cols: Sequence[str] = ()) -> pd.DataFrame:
    """Build a factual-only table from whatever classes were collected before interruption."""
    rows = []
    for class_id, df in class_dfs.items():
        for _, row in df.iterrows():
            rows.append({
                "masked": row["masked"],
                "split": row.get("split", "train"),
                "label_class": class_id,
                "label": row["label"],
                "feature_class_id": class_id,
                **{c: row[c] for c in extra_cols},
            })
    return pd.DataFrame(rows, columns=["masked", "split", "label_class", "label", "feature_class_id", *extra_cols])

__all__ = ["_to_minimal", "_to_merged", "_to_partial_merged"]
