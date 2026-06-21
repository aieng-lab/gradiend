"""GRADIEND backbone selection for encoder-decoder (seq2seq) models."""

from __future__ import annotations

from typing import Iterator, Tuple

import torch
import torch.nn as nn

from gradiend.model.utils import is_seq2seq_model
from gradiend.util import get_logger

logger = get_logger(__name__)

# Canonical names — aligned with ``prediction_objective`` (seq2seq_decoder / seq2seq_encoder_mlm).
SEQ2SEQ_DECODER = "seq2seq_decoder"
SEQ2SEQ_ENCODER_MLM = "seq2seq_encoder_mlm"
SUPPORTED_SEQ2SEQ_GRADIENT_MODES = frozenset({SEQ2SEQ_DECODER, SEQ2SEQ_ENCODER_MLM})
DEFAULT_SEQ2SEQ_GRADIENT_MODE = SEQ2SEQ_ENCODER_MLM


class Seq2SeqEncoderMLMBackbone(nn.Module):
    """View over encoder + shared embeddings + lm_head (no decoder stack)."""

    def __init__(self, parent: nn.Module) -> None:
        super().__init__()
        for idx, (name, module) in enumerate(_encoder_mlm_submodules(parent)):
            safe_name = name.replace(".", "_")
            if hasattr(self, safe_name):
                safe_name = f"{safe_name}_{idx}"
            self.add_module(safe_name, module)


def _encoder_mlm_submodules(parent: nn.Module) -> Iterator[Tuple[str, nn.Module]]:
    """Yield submodules that participate in encoder-side MLM gradients."""
    seen: set[int] = set()

    def add(name: str, module: nn.Module | None):
        if module is None or not isinstance(module, nn.Module):
            return
        mid = id(module)
        if mid in seen:
            return
        seen.add(mid)
        yield name, module

    for attr in ("shared", "encoder", "lm_head"):
        if hasattr(parent, attr):
            yield from add(attr, getattr(parent, attr))

    inner = getattr(parent, "model", None)
    if inner is not None:
        for attr in ("shared", "encoder", "embed_tokens", "embed_positions"):
            if hasattr(inner, attr):
                yield from add(f"model.{attr}", getattr(inner, attr))


def normalize_seq2seq_mode(mode: str) -> str:
    """Validate ``seq2seq_decoder`` or ``seq2seq_encoder_mlm``."""
    canonical = mode.strip()
    if canonical not in SUPPORTED_SEQ2SEQ_GRADIENT_MODES:
        raise ValueError(
            f"Unsupported seq2seq mode={mode!r}. "
            f"Supported: {sorted(SUPPORTED_SEQ2SEQ_GRADIENT_MODES)}"
        )
    return canonical


def resolve_seq2seq_mode_from_kwargs(kwargs: dict) -> str | None:
    """
    Resolve seq2seq backbone mode from ``prediction_objective``.

    Returns None when the caller is not building a seq2seq model.
    """
    objective = str(kwargs.get("prediction_objective", "auto") or "auto").strip()

    if objective in SUPPORTED_SEQ2SEQ_GRADIENT_MODES:
        return normalize_seq2seq_mode(objective)
    if objective == "seq2seq_decoder_sequence_cloze":
        return SEQ2SEQ_DECODER
    if objective == "auto":
        return DEFAULT_SEQ2SEQ_GRADIENT_MODE
    return None


def configure_seq2seq_gradiend_backbone(model: nn.Module, mode: str = DEFAULT_SEQ2SEQ_GRADIENT_MODE) -> str:
    """
    Configure which base-model parameters GRADIEND tracks for a seq2seq model.

    seq2seq_encoder_mlm: encoder + shared + lm_head only.
    seq2seq_decoder: full seq2seq backbone (encoder + decoder + lm_head).
    """
    if not is_seq2seq_model(model):
        return mode

    mode = normalize_seq2seq_mode(mode)
    model._gradiend_seq2seq_gradient_mode = mode  # type: ignore[attr-defined]

    if mode == SEQ2SEQ_ENCODER_MLM:
        backbone = Seq2SeqEncoderMLMBackbone(model)
        model.get_gradiend_backbone_module = lambda: backbone  # type: ignore[attr-defined]
        n_params = sum(p.numel() for p in backbone.parameters())
        logger.info(
            "prediction_objective=%s: GRADIEND uses encoder/shared/lm_head only (%s weights).",
            mode,
            f"{n_params:,}",
        )
    else:
        if hasattr(model, "get_gradiend_backbone_module"):
            delattr(model, "get_gradiend_backbone_module")
        logger.info("prediction_objective=%s: GRADIEND uses the full seq2seq backbone.", mode)

    return mode


def resolve_seq2seq_gradient_mode(model: nn.Module, requested: str | None = None) -> str | None:
    """Return effective seq2seq mode stored on the model, or None for non-seq2seq models."""
    if not is_seq2seq_model(model):
        return None
    if requested is not None:
        return normalize_seq2seq_mode(requested)
    stored = getattr(model, "_gradiend_seq2seq_gradient_mode", DEFAULT_SEQ2SEQ_GRADIENT_MODE)
    return normalize_seq2seq_mode(str(stored))
