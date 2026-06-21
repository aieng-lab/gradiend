"""
Resolve encoder evaluation split arguments (single split, list, or "all").
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional, Union

from gradiend.util import normalize_split_name

EncoderSplit = Union[str, Sequence[str]]

_STANDARD_SPLITS = ("train", "validation", "test")


def order_split_names(splits: Sequence[str]) -> List[str]:
    """Return split names in canonical order: train, validation, test."""
    normalized = [normalize_split_name(str(s)) for s in splits]
    avail = set(normalized)
    ordered = [s for s in _STANDARD_SPLITS if s in avail]
    extras = sorted(avail - set(_STANDARD_SPLITS))
    return ordered + extras


def resolve_encoder_splits(
    split: EncoderSplit,
    available: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Normalize encoder split selection to a list of split names.

    Args:
        split: A single split name, ``"all"``, or a sequence of split names.
        available: Optional splits present in the dataset (used when split is ``"all"``).

    Returns:
        List of normalized split names (train / validation / test).
    """
    if isinstance(split, (list, tuple)):
        if len(split) == 0:
            raise ValueError("split sequence must not be empty")
        return order_split_names(split)

    token = str(split).strip()
    if token.lower() == "all":
        if available is not None:
            avail = {normalize_split_name(str(s)) for s in available}
            ordered = order_split_names(avail)
            return ordered or sorted(avail)
        return list(_STANDARD_SPLITS)
    return [normalize_split_name(token)]


def encoder_split_cache_key(
    split: EncoderSplit,
    available: Optional[Sequence[str]] = None,
) -> Union[str, List[str]]:
    """Cache key component for encoder artifacts."""
    resolved = resolve_encoder_splits(split, available=available)
    if len(resolved) == 1:
        return resolved[0]
    return resolved
