"""Output format conversions for text-prediction data creation."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from gradiend.trainer.core.unified_data import per_class_dict_to_unified
from gradiend.trainer.core.unified_schema import UNIFIED_FACTUAL, UNIFIED_FACTUAL_CLASS


def _to_minimal(class_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Convert per-class to minimal (masked, label, label_class, split)."""
    rows = []
    for class_id, df in class_dfs.items():
        for _, row in df.iterrows():
            rows.append({
                "masked": row["masked"],
                "label": row["label"],
                "label_class": class_id,
                "split": row["split"],
            })
    return pd.DataFrame(rows)


def _to_merged(class_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build training DataFrame (factual only): masked, split, label_class, label, feature_class_id.

    feature_class_id is the string id from TextFilterConfig (same as label_class per row).
    Splits are applied per feature class in _apply_auto_split.
    """
    df = per_class_dict_to_unified(
        class_dfs,
        classes=list(class_dfs.keys()),
        masked_col="masked",
        split_col="split",
        use_class_names_as_columns=True,
    )
    # Map unified schema to merged column names expected by callers
    df = df.rename(columns={UNIFIED_FACTUAL_CLASS: "label_class", UNIFIED_FACTUAL: "label"})
    df["feature_class_id"] = df["label_class"]
    return df


def _to_partial_merged(class_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
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
            })
    return pd.DataFrame(rows, columns=["masked", "split", "label_class", "label", "feature_class_id"])

__all__ = ["_to_minimal", "_to_merged", "_to_partial_merged"]
