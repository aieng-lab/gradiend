"""Shared error-message hints for prediction_objective validation (no heavy imports)."""

from __future__ import annotations

_SEQ2SEQ_OBJECTIVE_DESCRIPTIONS: tuple[tuple[str, str], ...] = (
    ("seq2seq_decoder", "single-token decoder"),
    ("seq2seq_decoder_sequence_cloze", "[MASK] multi-token decoder cloze"),
    ("seq2seq_encoder_mlm", "encoder-side MLM"),
)


def format_seq2seq_objective_hint(*, prefix: str = "For encoder-decoder (seq2seq) models, use") -> str:
    """Human-readable list of all seq2seq prediction_objective values."""
    parts = [f"'{name}' ({desc})" for name, desc in _SEQ2SEQ_OBJECTIVE_DESCRIPTIONS]
    return f"{prefix} " + ", ".join(parts) + "."
