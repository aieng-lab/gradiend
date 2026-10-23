"""Unified-schema split helpers for text-prediction data."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from gradiend.data.core.dataframe_splitting import split_dataframe_by_group_key
from gradiend.data.core.split_group_key import SplitGroupKey, apply_split_group_key

UNIFIED_FACTUAL_CLASS = "factual_class"
UNIFIED_SPLIT = "split"
UNIFIED_FACTUAL = "factual"
UNIFIED_ALTERNATIVE = "alternative"
UNIFIED_ALTERNATIVE_CLASS = "alternative_class"


def _apply_factual_casing(factual: str, alternative: str) -> str:
    if not factual or not alternative:
        return alternative
    if factual.islower():
        return alternative.lower()
    if factual.isupper():
        return alternative.upper()
    if len(factual) >= 1 and factual[0].isupper() and (len(factual) == 1 or factual[1:].islower()):
        return alternative.title()
    return alternative


def resplit_unified_dataframe(
    df: pd.DataFrame,
    *,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_key: SplitGroupKey = None,
    per_feature_class: bool = True,
    feature_class_col: str = UNIFIED_FACTUAL_CLASS,
    split_col: str = UNIFIED_SPLIT,
    align_alternatives_with_split_vocab: bool = False,
    balanced_cycle_index: int | None = None,
    balanced_cycle_length: int | None = None,
) -> pd.DataFrame:
    """Assign vocabulary-held-out splits to a unified prediction DataFrame.

    If every alternative is the other word of an explicit pair (paired targets), the rows are split by pair across all
    classes, so both words of a pair share a split, and the alternatives are kept as they are.

    Args:
        df: Unified DataFrame to resplit. The input is not modified.
        group_col: Column whose canonical values define held-out vocabulary
            groups, commonly ``"factual"``.
        train_ratio: Fraction of groups assigned to ``"train"``.
        val_ratio: Fraction of groups assigned to ``"validation"``.
        test_ratio: Fraction of groups assigned to ``"test"``.
        seed: Random seed used for shuffling group keys.
        group_key: Optional callable or sequence of callables applied to
            ``group_col`` values before grouping.
        per_feature_class: If ``True``, split groups independently within each
            ``feature_class_col`` value. If ``False``, split across all rows.
        feature_class_col: Column containing the factual feature class.
        split_col: Output column name for split labels.
        align_alternatives_with_split_vocab: If ``True``, replace alternatives
            with tokens from the same split and alternative class so held-out
            factual vocabulary does not re-enter as alternatives.
        balanced_cycle_index: Optional seed index for balanced multi-seed
            split rotation.
        balanced_cycle_length: Number of seed slots in the balanced rotation.

    Returns:
        A copy of ``df`` with split assignments, and optionally aligned
        alternatives.

    Raises:
        ValueError: If required columns are missing or delegated split
            validation fails.
    """
    if group_col not in df.columns:
        raise ValueError(f"resplit group column {group_col!r} not in DataFrame")
    common = dict(
        group_col=group_col, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed,
        group_key=group_key, per_feature_class=per_feature_class, feature_class_col=feature_class_col,
        split_col=split_col, align_alternatives_with_split_vocab=align_alternatives_with_split_vocab,
        balanced_cycle_index=balanced_cycle_index, balanced_cycle_length=balanced_cycle_length,
    )
    pair_group = _explicit_pair_groups(df, group_key, feature_class_col)
    if not pair_group.notna().all():
        return _resplit_by_vocabulary(df, **common)
    return split_dataframe_by_group_key(
        df.assign(_pair_group=pair_group), "_pair_group", train_ratio, val_ratio, test_ratio, seed,
        split_col=split_col, group_key=group_key, balanced_cycle_index=balanced_cycle_index,
        balanced_cycle_length=balanced_cycle_length,
    ).drop(columns="_pair_group")


def _explicit_pair_groups(df: pd.DataFrame, group_key: SplitGroupKey, feature_class_col: str) -> pd.Series:
    """Group id per row whose alternative is the other word of an explicit pair, ``None`` for all other rows.

    A row is an explicit pair if its factual word always has this one alternative within the alternative class, no
    other word has it, and the alternative has no other alternative back. Sampled alternatives do not satisfy this.
    The id joins all words of a pair (for three or more classes too), so the words of a pair share a split. The caller
    treats the data as paired only if every row is.
    """
    needed = {feature_class_col, UNIFIED_ALTERNATIVE_CLASS, UNIFIED_FACTUAL, UNIFIED_ALTERNATIVE}
    if df.empty or not needed.issubset(df.columns):
        return pd.Series(None, index=df.index, dtype=object)
    rows = list(zip(
        df[feature_class_col].astype(str),
        df[UNIFIED_FACTUAL].astype(str).map(lambda v: apply_split_group_key(v, group_key)),
        df[UNIFIED_ALTERNATIVE_CLASS].astype(str),
        df[UNIFIED_ALTERNATIVE].astype(str).map(lambda v: apply_split_group_key(v, group_key)),
    ))
    alternatives: Dict[Tuple[str, str, str], set] = {}
    sources: Dict[Tuple[str, str, str], set] = {}
    for cls, word, alt_cls, alt in rows:
        alternatives.setdefault((cls, word, alt_cls), set()).add(alt)
        sources.setdefault((cls, alt_cls, alt), set()).add(word)

    def is_pair(cls: str, word: str, alt_cls: str, alt: str) -> bool:
        back = alternatives.get((alt_cls, alt, cls))
        one_to_one = alternatives[(cls, word, alt_cls)] == {alt} and sources[(cls, alt_cls, alt)] == {word}
        return one_to_one and (back is None or back == {word})

    parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def find(node: Tuple[str, str]) -> Tuple[str, str]:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    flags = [is_pair(*row) for row in rows]
    for (cls, word, alt_cls, alt), flag in zip(rows, flags):
        if flag:
            parent[find((cls, word))] = find((alt_cls, alt))
    return pd.Series(
        ["|".join(find((cls, word))) if flag else None for (cls, word, _, _), flag in zip(rows, flags)],
        index=df.index, dtype=object,
    )


def _resplit_by_vocabulary(
    df: pd.DataFrame,
    *,
    group_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_key: SplitGroupKey,
    per_feature_class: bool,
    feature_class_col: str,
    split_col: str,
    align_alternatives_with_split_vocab: bool,
    balanced_cycle_index: Optional[int],
    balanced_cycle_length: Optional[int],
) -> pd.DataFrame:
    """Vocabulary-held-out split by ``group_col``, with optionally aligned alternatives."""
    if per_feature_class:
        if feature_class_col not in df.columns:
            raise ValueError(f"feature class column {feature_class_col!r} not in DataFrame")
        parts = []
        for class_id, grp in df.groupby(feature_class_col, sort=False):
            parts.append(
                split_dataframe_by_group_key(
                    grp,
                    group_col,
                    train_ratio,
                    val_ratio,
                    test_ratio,
                    seed,
                    split_col=split_col,
                    group_key=group_key,
                    class_id=str(class_id),
                    balanced_cycle_index=balanced_cycle_index,
                    balanced_cycle_length=balanced_cycle_length,
                )
            )
        if not parts:
            out = df.copy()
        else:
            out = pd.concat(parts, ignore_index=True)
        if align_alternatives_with_split_vocab:
            out = align_unified_alternatives_with_split_vocab(
                out,
                group_col=group_col,
                split_col=split_col,
                group_key=group_key,
                feature_class_col=feature_class_col,
            )
        return out
    out = split_dataframe_by_group_key(
        df,
        group_col,
        train_ratio,
        val_ratio,
        test_ratio,
        seed,
        split_col=split_col,
        group_key=group_key,
        balanced_cycle_index=balanced_cycle_index,
        balanced_cycle_length=balanced_cycle_length,
    )
    if align_alternatives_with_split_vocab:
        out = align_unified_alternatives_with_split_vocab(
            out,
            group_col=group_col,
            split_col=split_col,
            group_key=group_key,
            feature_class_col=feature_class_col,
        )
    return out


def align_unified_alternatives_with_split_vocab(
    df: pd.DataFrame,
    *,
    group_col: str = UNIFIED_FACTUAL,
    token_col: str = UNIFIED_FACTUAL,
    split_col: str = UNIFIED_SPLIT,
    group_key: SplitGroupKey = None,
    feature_class_col: str = UNIFIED_FACTUAL_CLASS,
    alternative_col: str = UNIFIED_ALTERNATIVE,
    alternative_class_col: str = UNIFIED_ALTERNATIVE_CLASS,
) -> pd.DataFrame:
    """Replace alternatives with tokens from the same split and alternative class.

    Args:
        df: Unified DataFrame after split assignment. The input is not modified.
        group_col: Column used for canonical vocabulary uniqueness.
        token_col: Column containing factual tokens used as replacement source.
        split_col: Column containing split labels.
        group_key: Optional callable or sequence of callables used to canonicalize
            vocabulary values.
        feature_class_col: Column containing the factual feature class.
        alternative_col: Column to replace.
        alternative_class_col: Column containing the alternative feature class.

    Returns:
        A copy of ``df`` with ``alternative_col`` replaced where same-split
        candidates exist. If required columns are absent or the frame is empty,
        returns an unchanged copy.
    """
    required = {group_col, token_col, split_col, feature_class_col, alternative_col, alternative_class_col}
    if df is None or df.empty or not required.issubset(df.columns):
        return df.copy()

    out = df.copy()
    candidates: Dict[Tuple[str, str], List[str]] = {}
    for _, row in out.iterrows():
        split = str(row[split_col])
        cls = str(row[feature_class_col])
        value = row[token_col]
        if pd.isna(value):
            continue
        candidates.setdefault((split, cls), []).append(str(value))

    for key, values in list(candidates.items()):
        seen_keys: set[str] = set()
        unique_values: List[str] = []
        for value in values:
            canonical = apply_split_group_key(value, group_key)
            if canonical in seen_keys:
                continue
            seen_keys.add(canonical)
            unique_values.append(value)
        candidates[key] = unique_values

    counters: Dict[Tuple[str, str], int] = {}
    replacements: List[str] = []
    for _, row in out.iterrows():
        split = str(row[split_col])
        alt_cls = str(row[alternative_class_col])
        key = (split, alt_cls)
        choices = candidates.get(key)
        if not choices:
            replacements.append(str(row[alternative_col]))
            continue
        idx = counters.get(key, 0)
        counters[key] = idx + 1
        replacement = choices[idx % len(choices)]
        replacements.append(_apply_factual_casing(str(row[token_col]), replacement))

    out[alternative_col] = replacements
    return out


__all__ = ["resplit_unified_dataframe", "align_unified_alternatives_with_split_vocab"]
