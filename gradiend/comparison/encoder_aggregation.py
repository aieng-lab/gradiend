"""Aggregation of per-seed encoder evaluation DataFrames."""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

from gradiend.comparison.common import orient_encoder_df_by_label_correlation


def encoder_probe_key_from_row(row: pd.Series) -> Optional[Tuple[str, ...]]:
    """Stable probe identity for one encoder output row."""
    masked = row.get("masked")
    factual_token = row.get("factual_token")
    alternative_token = row.get("alternative_token")
    if (
        masked is not None
        and not pd.isna(masked)
        and factual_token is not None
        and not pd.isna(factual_token)
        and alternative_token is not None
        and not pd.isna(alternative_token)
    ):
        return (str(masked), str(factual_token), str(alternative_token))
    factual_id = row.get("factual_id")
    if factual_id is None or (isinstance(factual_id, float) and pd.isna(factual_id)):
        factual_id = row.get("source_id")
    alternative_id = row.get("alternative_id")
    if alternative_id is None or (isinstance(alternative_id, float) and pd.isna(alternative_id)):
        alternative_id = row.get("counterfactual_id")
    if alternative_id is None or (isinstance(alternative_id, float) and pd.isna(alternative_id)):
        alternative_id = row.get("target_id")
    if (
        masked is not None
        and not pd.isna(masked)
        and factual_id is not None
        and not pd.isna(factual_id)
        and alternative_id is not None
        and not pd.isna(alternative_id)
        and str(factual_id) != str(alternative_id)
    ):
        return (str(masked), str(factual_id), str(alternative_id))
    return None


def encoder_probe_keys(encoder_df: pd.DataFrame) -> set[Tuple[str, ...]]:
    keys: set[Tuple[str, ...]] = set()
    for _, row in encoder_df.iterrows():
        key = encoder_probe_key_from_row(row)
        if key is not None:
            keys.add(key)
    return keys


def aggregate_encoder_dataframes(
    frames: Sequence[Any],
    *,
    orient: bool = True,
) -> pd.DataFrame:
    """Mean-aggregate encoded values across seed-level encoder DataFrames.

    Rows are matched by :func:`encoder_probe_key_from_row`. Non-numeric columns
    are taken from the first seed frame in each probe group.
    """
    valid: List[pd.DataFrame] = []
    for frame in frames:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        work = orient_encoder_df_by_label_correlation(frame) if orient else frame.copy()
        valid.append(work)
    if not valid:
        return pd.DataFrame()
    if len(valid) == 1:
        return valid[0].copy()

    tagged: List[pd.DataFrame] = []
    for seed_index, frame in enumerate(valid):
        work = frame.copy()
        work["_probe_key"] = work.apply(
            lambda row: encoder_probe_key_from_row(row),
            axis=1,
        )
        work = work[work["_probe_key"].notna()].copy()
        if work.empty:
            continue
        work["_seed_index"] = seed_index
        tagged.append(work)
    if not tagged:
        return pd.DataFrame()

    combined = pd.concat(tagged, ignore_index=True)
    if combined.empty or "encoded" not in combined.columns:
        return pd.DataFrame()

    group_cols = ["_probe_key"]
    numeric = (
        combined.groupby(group_cols, dropna=False)["encoded"]
        .mean()
        .reset_index()
    )
    template = (
        combined.sort_values("_seed_index")
        .drop_duplicates(subset=group_cols, keep="first")
        .drop(columns=["encoded", "_seed_index"], errors="ignore")
    )
    merged = template.merge(numeric, on="_probe_key", how="inner")
    merged = merged.drop(columns=["_probe_key"], errors="ignore")
    return merged.reset_index(drop=True)


def aggregate_encoder_dataframes_registry(values: Sequence[Any]) -> pd.DataFrame:
    """Seed-aggregator entry point for :mod:`gradiend.trainer.core.multi_seed`."""
    return aggregate_encoder_dataframes(values, orient=True)
