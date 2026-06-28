"""Small helpers for consistent deprecation warnings."""

from __future__ import annotations

import warnings
from typing import Optional


def warn_deprecated_annot_fmt(*, fmt: Optional[str], annot_fmt: Optional[str], stacklevel: int = 2) -> None:
    if fmt is not None and annot_fmt is None:
        warnings.warn(
            "The fmt argument is deprecated; use annot_fmt instead.",
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )


def resolve_include_other_classes(
    *,
    include_other_classes: Optional[bool],
    use_all_transitions: bool,
    default: bool = False,
    stacklevel: int = 2,
) -> bool:
    """Merge deprecated ``use_all_transitions`` into ``include_other_classes``."""
    if use_all_transitions:
        warnings.warn(
            "use_all_transitions is deprecated; use include_other_classes instead "
            "(TrainingArguments.include_other_classes or evaluate_encoder(include_other_classes=True)).",
            DeprecationWarning,
            stacklevel=stacklevel + 1,
        )
    if include_other_classes is None:
        include_other_classes = default
    return bool(include_other_classes) or bool(use_all_transitions)
