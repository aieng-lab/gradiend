"""Validation helpers for dataframe split assignment."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from gradiend.data.core.split_group_key import SplitGroupKey, apply_split_group_key
from gradiend.data.core.split_ratios import min_vocabulary_keys_for_split_ratios


def validate_vocabulary_group_split_coverage(
    df: pd.DataFrame,
    group_col: str,
    split_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    group_key: SplitGroupKey = None,
    class_id: Optional[str] = None,
) -> None:
    """Raise when vocabulary-held-out split assignment leaves a requested bucket empty.

    Args:
        df: DataFrame after split assignment.
        group_col: Column whose canonical values define split groups.
        split_col: Column containing assigned split names.
        train_ratio: Requested train split fraction.
        val_ratio: Requested validation split fraction.
        test_ratio: Requested test split fraction.
        group_key: Optional callable or sequence of callables applied to
            ``group_col`` values before counting distinct vocabulary keys.
        class_id: Optional class id included in error messages.

    Raises:
        ValueError: If too few distinct group keys exist for the requested
            non-empty buckets, or if no group was assigned to a requested bucket.

    Returns:
        ``None``. The function validates by raising on failure.
    """
    if df is None or df.empty or group_col not in df.columns or split_col not in df.columns:
        return

    prefix = f"class {class_id!r}: " if class_id is not None else ""
    canonical_keys = {
        apply_split_group_key(value, group_key)
        for value in df[group_col].dropna().astype(str).tolist()
    }
    n_keys = len(canonical_keys)
    min_keys = min_vocabulary_keys_for_split_ratios(train_ratio, val_ratio, test_ratio)

    if min_keys > 0 and n_keys < min_keys:
        raise ValueError(
            f"{prefix}vocabulary-held-out split requires at least {min_keys} distinct "
            f"{group_col!r} value(s) for ratios train={train_ratio}, validation={val_ratio}, "
            f"test={test_ratio}, but only {n_keys} found ({canonical_keys}). "
            "Add more target tokens to the dataset or use fewer non-zero split buckets."
        )

    splits_present = set(df[split_col].dropna().astype(str).tolist())
    ratio_by_split = (
        ("train", train_ratio),
        ("validation", val_ratio),
        ("test", test_ratio),
    )
    missing = [
        split_name
        for split_name, ratio in ratio_by_split
        if ratio > 0 and split_name not in splits_present
    ]
    if missing:
        present = sorted(splits_present) if splits_present else ["<none>"]
        raise ValueError(
            f"{prefix}vocabulary-held-out split requested bucket(s) {missing!r}, but no "
            f"{group_col!r} groups were assigned there (distinct groups={n_keys}, "
            f"present buckets={present}). "
            "Add more target tokens or adjust split ratios."
        )


def validate_row_split_coverage(
    df: pd.DataFrame,
    split_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    class_id: Optional[str] = None,
) -> None:
    """Raise when random row splitting leaves a requested bucket empty.

    Args:
        df: DataFrame after split assignment.
        split_col: Column containing assigned split names.
        train_ratio: Requested train split fraction.
        val_ratio: Requested validation split fraction.
        test_ratio: Requested test split fraction.
        class_id: Optional class id included in error messages.

    Raises:
        ValueError: If enough rows exist to populate all requested buckets, but
            at least one requested bucket is absent.

    Returns:
        ``None``. The function validates by raising on failure.
    """
    if df is None or df.empty or split_col not in df.columns:
        return
    min_rows = min_vocabulary_keys_for_split_ratios(train_ratio, val_ratio, test_ratio)
    if len(df) < min_rows:
        return
    prefix = f"class {class_id!r}: " if class_id is not None else ""
    splits_present = set(df[split_col].dropna().astype(str).tolist())
    ratio_by_split = (
        ("train", train_ratio),
        ("validation", val_ratio),
        ("test", test_ratio),
    )
    missing = [
        split_name
        for split_name, ratio in ratio_by_split
        if ratio > 0 and split_name not in splits_present
    ]
    if missing:
        raise ValueError(
            f"{prefix}split requested bucket(s) {missing!r}, but no rows were assigned "
            f"(n_rows={len(df)}, present buckets={sorted(splits_present) or ['<none>']}). "
            "Use more data or adjust split ratios."
        )


__all__ = ["validate_vocabulary_group_split_coverage", "validate_row_split_coverage"]
