"""Token-position helpers that hold for LEFT and RIGHT padding.

Tokenizers pad on different sides (Gemma pads left; GPT-2, Pythia, Llama and Qwen pad
right). ``attention_mask.sum() - 1`` is the last real token only for right padding, and
silently addresses the pad block otherwise. Anything that needs "the last real token"
must come through here, so one implementation carries the padding-side contract.
"""

from __future__ import annotations

from typing import Optional

import torch


def _as_2d_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(attention_mask) or attention_mask.ndim != 2:
        raise ValueError(
            "attention_mask must be a 2D (batch, seq) tensor, got "
            f"{tuple(attention_mask.shape) if torch.is_tensor(attention_mask) else type(attention_mask).__name__}."
        )
    return attention_mask.ne(0)


def last_real_token_positions(attention_mask: torch.Tensor, *, allow_empty: bool = False) -> torch.Tensor:
    """Index of the last non-padding token of every row, shape ``(batch,)``.

    Valid for left, right and mixed padding. A row without any real token raises
    unless ``allow_empty`` is set, in which case it maps to position 0.
    """
    valid = _as_2d_mask(attention_mask)
    positions = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0).expand_as(valid)
    last = positions.masked_fill(~valid, -1).max(dim=1).values
    empty = last < 0
    if empty.any():
        if not allow_empty:
            raise ValueError("Cannot resolve the last real token of an empty (all-padding) row.")
        last = last.clamp_min(0)
    return last


def first_real_token_positions(attention_mask: torch.Tensor, *, allow_empty: bool = False) -> torch.Tensor:
    """Index of the first non-padding token of every row (the pad-block width under left padding)."""
    valid = _as_2d_mask(attention_mask)
    size = valid.shape[1]
    positions = torch.arange(size, device=valid.device).unsqueeze(0).expand_as(valid)
    first = positions.masked_fill(~valid, size).min(dim=1).values
    empty = first >= size
    if empty.any():
        if not allow_empty:
            raise ValueError("Cannot resolve the first real token of an empty (all-padding) row.")
        first = first.clamp_max(size - 1)
    return first


def assert_labels_on_real_tokens(
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    *,
    where: str,
    ignore_index: int = -100,
    allow_empty_rows: bool = False,
) -> None:
    """Item contract: a supervised position must never sit on padding.

    ``allow_empty_rows`` exempts rows that have no real token at all: the target is the first
    word of the text and the tokenizer adds no BOS, so nothing precedes it and the row is
    all padding by construction (race/religion sentences that start with the class word).
    Such a row has no sentence-dependent signal to protect; a row that HAS real tokens is
    still checked.

    A label on a pad position still yields a finite loss, so nothing downstream fails;
    the gradient just stops depending on the sentence. Raise at the source instead.
    """
    if attention_mask is None:
        return
    labels = labels.reshape(1, -1) if labels.ndim == 1 else labels
    attention_mask = attention_mask.reshape(1, -1) if attention_mask.ndim == 1 else attention_mask
    bad = labels.ne(ignore_index) & attention_mask.eq(0)
    if allow_empty_rows:
        bad = bad & attention_mask.ne(0).any(dim=-1, keepdim=True)
    if bool(bad.any()):
        rows = sorted({int(r) for r in bad.nonzero()[:, 0].tolist()})
        raise ValueError(
            f"{where}: supervised label(s) fall on padding positions (rows {rows[:5]}). "
            "This happens when a position is computed as attention_mask.sum() - 1 under LEFT "
            "padding (e.g. Gemma tokenizers); use gradiend.util.positions.last_real_token_positions."
        )
