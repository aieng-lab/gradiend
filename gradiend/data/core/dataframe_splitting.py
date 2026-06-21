"""Generic train/validation/test dataframe splitting."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from gradiend.data.core.split_group_key import SplitGroupKey, apply_split_group_key
from gradiend.data.core.split_validation import (
    validate_row_split_coverage,
    validate_vocabulary_group_split_coverage,
)


def split_dataframe(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    *,
    split_col: str = "split",
    min_rows: int = 0,
) -> pd.DataFrame:
    """Assign train/validation/test splits to rows by random shuffle.

    Args:
        df: DataFrame to split. The input is not modified.
        train_ratio: Fraction assigned to ``"train"``.
        val_ratio: Fraction assigned to ``"validation"``.
        test_ratio: Fraction assigned to ``"test"``.
        seed: Random seed used for shuffling.
        split_col: Output column name for split labels.
        min_rows: If greater than zero, raise when ``len(df)`` is smaller.

    Returns:
        A shuffled copy of ``df`` with ``split_col`` assigned.

    Raises:
        ValueError: If ratios do not sum to ``1.0``, ``min_rows`` is not met, or
            enough rows exist but a requested split bucket is absent.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio, val_ratio, test_ratio must sum to 1.0")
    if min_rows > 0 and len(df) < min_rows:
        raise ValueError(
            f"Data split requires at least {min_rows} rows (got {len(df)}). "
            "Use more data or set min_rows=0 to disable."
        )
    if len(df) == 0:
        df = df.copy()
        df[split_col] = []
        return df

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = max(0, int(round(n * train_ratio)))
    n_val = max(0, int(round(n * val_ratio)))
    n_test = n - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n - n_train
    splits = ["train"] * n_train + ["validation"] * n_val + ["test"] * n_test
    df = df.copy()
    df[split_col] = splits[:n]
    validate_row_split_coverage(df, split_col, train_ratio, val_ratio, test_ratio)
    return df


def split_dataframe_by_group_key(
    df: pd.DataFrame,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    *,
    split_col: str = "split",
    min_rows: int = 0,
    group_key: SplitGroupKey = None,
    class_id: Optional[str] = None,
) -> pd.DataFrame:
    """Assign splits by canonical group key so no group spans split buckets.

    Args:
        df: DataFrame to split. The input is not modified.
        group_col: Column whose canonical values define groups.
        train_ratio: Fraction of groups assigned to ``"train"``.
        val_ratio: Fraction of groups assigned to ``"validation"``.
        test_ratio: Fraction of groups assigned to ``"test"``.
        seed: Random seed used for shuffling distinct group keys.
        split_col: Output column name for split labels.
        min_rows: If greater than zero, raise when ``len(df)`` is smaller.
        group_key: Optional callable or sequence of callables applied to
            ``group_col`` values before grouping.
        class_id: Optional class id included in validation error messages.

    Returns:
        A copy of ``df`` with ``split_col`` assigned.

    Raises:
        ValueError: If ``group_col`` is missing, ratios do not sum to ``1.0``,
            ``min_rows`` is not met, too few distinct groups exist, or a
            requested split bucket is absent.
    """
    if group_col not in df.columns:
        raise ValueError(f"split group column {group_col!r} not in DataFrame")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio, val_ratio, test_ratio must sum to 1.0")
    if min_rows > 0 and len(df) < min_rows:
        raise ValueError(
            f"Data split requires at least {min_rows} rows (got {len(df)}). "
            "Use more data or set min_rows=0 to disable."
        )
    if len(df) == 0:
        out = df.copy()
        out[split_col] = []
        return out

    keys = (
        df[group_col]
        .dropna()
        .map(lambda v: apply_split_group_key(v, group_key))
        .drop_duplicates()
        .sample(frac=1, random_state=seed)
        .tolist()
    )
    n_keys = len(keys)
    if n_keys == 0:
        out = df.copy()
        out[split_col] = "train"
        return out

    n_train = max(0, int(round(n_keys * train_ratio)))
    n_val = max(0, int(round(n_keys * val_ratio)))
    n_test = n_keys - n_train - n_val
    if n_test < 0:
        n_test = 0
        n_val = n_keys - n_train

    splits_needed = sum(1 for ratio in (train_ratio, val_ratio, test_ratio) if ratio > 0)
    if n_keys >= splits_needed:
        counts = {"train": n_train, "validation": n_val, "test": n_test}
        ratio_by_split = (
            ("train", train_ratio),
            ("validation", val_ratio),
            ("test", test_ratio),
        )
        for split_name, ratio in ratio_by_split:
            if ratio <= 0 or counts[split_name] > 0:
                continue
            donor = max(counts, key=lambda name: counts[name])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[split_name] = 1
        n_train, n_val, n_test = counts["train"], counts["validation"], counts["test"]

    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train : n_train + n_val])

    def _assign_split(key: str) -> str:
        if key in train_keys:
            return "train"
        if key in val_keys:
            return "validation"
        return "test"

    out = df.copy()
    out["_split_group_key"] = out[group_col].map(lambda v: apply_split_group_key(v, group_key))
    out[split_col] = out["_split_group_key"].map(_assign_split)
    out = out.drop(columns=["_split_group_key"])
    validate_vocabulary_group_split_coverage(
        out,
        group_col,
        split_col,
        train_ratio,
        val_ratio,
        test_ratio,
        group_key=group_key,
        class_id=class_id,
    )
    return out


def split_dataframe_per_group(
    class_dfs: Dict[str, pd.DataFrame],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    *,
    split_col: str = "split",
    min_rows_per_group: int = 0,
    split_group_col: Optional[str] = None,
    split_group_key: SplitGroupKey = None,
) -> None:
    """Assign train/validation/test splits to each class DataFrame in place.

    Args:
        class_dfs: Mapping from class id to DataFrame. Values are replaced with
            split-assigned DataFrames.
        train_ratio: Fraction assigned to ``"train"``.
        val_ratio: Fraction assigned to ``"validation"``.
        test_ratio: Fraction assigned to ``"test"``.
        seed: Random seed used for shuffling.
        split_col: Output column name for split labels.
        min_rows_per_group: If greater than zero, raise when a non-empty class
            DataFrame has fewer rows.
        split_group_col: Optional column used for vocabulary-held-out grouping
            within each class. When ``None``, rows are split randomly.
        split_group_key: Optional callable or sequence of callables applied to
            ``split_group_col`` values before grouping.

    Raises:
        ValueError: If ratios do not sum to ``1.0``, any non-empty group is too
            small, or a delegated split validation fails.

    Returns:
        ``None``. ``class_dfs`` is modified in place.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio, val_ratio, test_ratio must sum to 1.0")
    if min_rows_per_group > 0:
        too_small = [
            (cid, len(df))
            for cid, df in class_dfs.items()
            if len(df) > 0 and len(df) < min_rows_per_group
        ]
        if too_small:
            raise ValueError(
                f"Data split requires at least {min_rows_per_group} rows per class. "
                f"Classes with too few rows: {too_small}. "
                "Use more base data, increase max_size_per_class, or set min_rows_per_group=0 to disable."
            )
    for class_id in list(class_dfs.keys()):
        df = class_dfs[class_id]
        if len(df) == 0:
            continue
        if split_group_col is not None:
            class_dfs[class_id] = split_dataframe_by_group_key(
                df,
                split_group_col,
                train_ratio,
                val_ratio,
                test_ratio,
                seed,
                split_col=split_col,
                min_rows=0,
                group_key=split_group_key,
                class_id=str(class_id),
            )
        else:
            class_dfs[class_id] = split_dataframe(
                df,
                train_ratio,
                val_ratio,
                test_ratio,
                seed,
                split_col=split_col,
                min_rows=0,
            )


__all__ = ["split_dataframe", "split_dataframe_by_group_key", "split_dataframe_per_group"]
