"""Tokenizer helpers shared by dataset construction and visualization."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple


def tokenize_with_offsets(tokenizer: Any, text: str, **kwargs: Any) -> Tuple[Any, Optional[Any]]:
    """Tokenize ``text`` and also return its character offsets when the tokenizer supports them.

    Returns ``(encoded, offsets)``. ``offsets`` is ``None`` for tokenizers without offset
    support. Such tokenizers fail in different ways: slow Hugging Face tokenizers raise
    ``NotImplementedError``, minimal tokenizers may reject the keyword with ``TypeError`` and
    test stubs may silently ignore it. All three cases are handled here so callers can simply
    test ``offsets is None``.
    """
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, **kwargs)
    except (TypeError, NotImplementedError):
        return tokenizer(text, **kwargs), None
    offsets = encoded.get("offset_mapping") if hasattr(encoded, "get") else None
    return encoded, offsets


def offset_pairs(offsets: Any) -> List[Tuple[int, int]]:
    """Convert an offset mapping (list or ``(1, n, 2)`` tensor) to ``[(start, end), ...]``."""
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]  # drop the batch dimension
    return [(int(start), int(end)) for start, end in offsets]


__all__ = ["offset_pairs", "tokenize_with_offsets"]
