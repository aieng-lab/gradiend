"""Helpers for canonical split-group keys.

Vocabulary-held-out splits often need to keep spelling or casing variants in
the same bucket. A split-group key is an optional callable, or sequence of
callables, applied to the raw grouping value before split assignment.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Optional, Union

SplitGroupKey = Optional[Union[Callable[[Any], Any], Sequence[Callable[[Any], Any]]]]


def normalize_split_group_key(group_key: SplitGroupKey) -> SplitGroupKey:
    """Validate and return a split-group key specification.

    Args:
        group_key: ``None``, a callable, or a sequence of callables. Strings and
            bytes are rejected because they are sequences but not meaningful
            callable pipelines.

    Returns:
        The original ``group_key`` when valid.

    Raises:
        TypeError: If ``group_key`` is not ``None``, a callable, or a sequence
            of callables.
    """
    if group_key is None:
        return None
    if callable(group_key):
        return group_key
    if isinstance(group_key, (str, bytes)):
        raise TypeError("split_group_key must be a callable or a sequence of callables, not a string")
    if isinstance(group_key, Sequence):
        for idx, func in enumerate(group_key):
            if not callable(func):
                raise TypeError(
                    "split_group_key sequence entries must be callable; "
                    f"entry {idx} has type {type(func).__name__}"
                )
        return group_key
    raise TypeError(
        "split_group_key must be None, a callable, or a sequence of callables; "
        f"got {type(group_key).__name__}"
    )


def apply_split_group_key(value: Any, group_key: SplitGroupKey = None) -> Any:
    """Apply a split-group key specification to a value.

    Args:
        value: Raw value to canonicalize for split grouping.
        group_key: ``None``, a callable, or a sequence of callables. Sequences
            are applied left to right.

    Returns:
        ``value`` unchanged when ``group_key`` is ``None``; otherwise the
        callable or callable pipeline result.

    Raises:
        TypeError: If ``group_key`` is invalid.
    """
    normalized = normalize_split_group_key(group_key)
    if normalized is None:
        return value
    if callable(normalized):
        return normalized(value)
    out = value
    for func in normalized:
        out = func(out)
    return out


__all__ = [
    "SplitGroupKey",
    "apply_split_group_key",
    "normalize_split_group_key",
]
