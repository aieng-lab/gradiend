"""DataFrame balancing and balanced capping helpers."""

from __future__ import annotations

from typing import Optional, Union

import pandas as pd


def balance_dataframe(
    df: pd.DataFrame,
    label_col: str,
    balance: Union[bool, str],
    seed: int,
) -> pd.DataFrame:
    """Balance a classification DataFrame by label.

    Args:
        df: DataFrame to balance. The input is not modified when balancing is
            performed.
        label_col: Column containing class labels.
        balance: ``"strict"`` undersamples every class to the smallest class
            size; ``"try"`` oversamples minority classes with replacement to
            match the largest class; ``False`` disables balancing. Other values
            leave the DataFrame unchanged.
        seed: Random seed used for shuffling and sampling.

    Returns:
        A balanced/shuffled DataFrame, or ``df`` unchanged when balancing is
        disabled, ``label_col`` is missing, fewer than two labels exist, or the
        mode is unsupported.
    """
    if balance is False or label_col not in df.columns:
        return df
    groups = df.groupby(label_col, sort=False)
    class_ids = list(groups.groups.keys())
    if len(class_ids) < 2:
        return df
    shuffled = {
        cid: g.sample(frac=1, random_state=seed).reset_index(drop=True)
        for cid, g in groups
    }
    sizes = {cid: len(shuffled[cid]) for cid in class_ids}
    if balance == "strict":
        cap = min(sizes.values())
        out = pd.concat(
            [shuffled[cid].iloc[:cap] for cid in class_ids],
            ignore_index=True,
        )
        return out.sample(frac=1, random_state=seed).reset_index(drop=True)
    if balance == "try":
        target = max(sizes.values())
        out_dfs = []
        for cid in class_ids:
            g = shuffled[cid]
            n = len(g)
            if n >= target:
                out_dfs.append(g.iloc[:target])
            else:
                extra = g.sample(n=target - n, replace=True, random_state=seed)
                out_dfs.append(pd.concat([g, extra], ignore_index=True))
        out = pd.concat(out_dfs, ignore_index=True)
        return out.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def balance_dataframe_per_target(
    df: pd.DataFrame,
    target_col: str = "label",
    balance: Union[bool, str] = "strict",
    seed: int = 42,
) -> pd.DataFrame:
    """Undersample rows so each target value has the same count.

    Args:
        df: DataFrame to balance.
        target_col: Column containing target tokens/values.
        balance: ``"strict"`` enables undersampling; ``False`` disables
            balancing. Other string values are rejected.
        seed: Random seed used for sampling/shuffling.

    Returns:
        A balanced DataFrame, or ``df`` unchanged when balancing is disabled,
        the target column is missing, or the frame is empty.

    Raises:
        ValueError: If ``balance`` is a value other than ``False`` or
            ``"strict"``.
    """
    if balance is False or target_col not in df.columns or df.empty:
        return df
    if balance != "strict":
        raise ValueError(
            f"balance_dataframe_per_target only supports balance='strict'; got {balance!r}"
        )
    groups = df.groupby(target_col, sort=False)
    sizes = {tok: len(g) for tok, g in groups}
    if not sizes:
        return df
    cap = min(sizes.values())
    if cap <= 0:
        return df
    parts = [
        g.sample(n=cap, random_state=seed).reset_index(drop=True)
        for _, g in groups
    ]
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def balance_dataframe_per_target_with_floor(
    df: pd.DataFrame,
    target_col: str = "label",
    *,
    min_rows_per_target: int = 1,
    max_size: Optional[int] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Balance target-value rows while enforcing a minimum count with replacement.

    Args:
        df: DataFrame to balance.
        target_col: Column containing target tokens/values.
        min_rows_per_target: Minimum rows requested for each target value.
        max_size: Optional global cap. If too small to satisfy the requested
            floor for every target, the largest feasible per-target count is
            used.
        seed: Random seed used for sampling/shuffling.

    Returns:
        A balanced DataFrame, or ``df`` unchanged when ``target_col`` is missing
        or the frame is empty.

    Raises:
        ValueError: If ``min_rows_per_target`` is less than ``1``.
    """
    if target_col not in df.columns or df.empty:
        return df
    if min_rows_per_target < 1:
        raise ValueError(f"min_rows_per_target must be >= 1, got {min_rows_per_target}")
    groups = [(tok, g.reset_index(drop=True)) for tok, g in df.groupby(target_col, sort=False)]
    if not groups:
        return df
    min_observed = min(len(g) for _, g in groups)
    target_count = max(min_observed, min_rows_per_target)
    if max_size is not None:
        feasible = max_size // len(groups)
        if feasible <= 0:
            feasible = 1
        target_count = min(target_count, feasible)
    parts = []
    for offset, (_, g) in enumerate(groups):
        if len(g) >= target_count:
            part = g.sample(n=target_count, random_state=seed + offset).reset_index(drop=True)
        else:
            extra = g.sample(
                n=target_count - len(g),
                replace=True,
                random_state=seed + offset,
            ).reset_index(drop=True)
            part = pd.concat([g, extra], ignore_index=True)
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def cap_dataframe_balanced(
    df: pd.DataFrame,
    max_size: int,
    label_col: str,
    seed: int,
) -> pd.DataFrame:
    """Cap a DataFrame while keeping class counts as equal as possible.

    Args:
        df: DataFrame to cap.
        max_size: Maximum number of rows to keep.
        label_col: Column containing class labels.
        seed: Random seed used for per-class shuffling and final shuffle.

    Returns:
        A capped/shuffled DataFrame, or ``df`` unchanged when ``label_col`` is
        missing or ``len(df) <= max_size``.
    """
    if label_col not in df.columns or len(df) <= max_size:
        return df
    groups = df.groupby(label_col, sort=False)
    class_ids = list(groups.groups.keys())
    n_classes = len(class_ids)
    if n_classes < 1:
        return df
    per_class = max_size // n_classes
    remainder = max_size % n_classes
    out_dfs = []
    for i, cid in enumerate(class_ids):
        g = groups.get_group(cid).sample(frac=1, random_state=seed + i).reset_index(drop=True)
        take = per_class + (1 if i < remainder else 0)
        take = min(take, len(g))
        out_dfs.append(g.iloc[:take])
    out = pd.concat(out_dfs, ignore_index=True)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


__all__ = [
    "balance_dataframe",
    "balance_dataframe_per_target",
    "balance_dataframe_per_target_with_floor",
    "cap_dataframe_balanced",
]
