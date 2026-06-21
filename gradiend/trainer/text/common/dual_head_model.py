"""
Shared-backbone dual-head wrapper for encoder-based sequence classification + LM.

Used so decoder evaluation can compute LMS on the same (possibly GRADIEND-modified)
encoder as classification, without loading a separate LM or copying state dicts.
Supports BERT/DistilBERT/RoBERTa-style encoder + MLM head; decoder-only uses
the same CLM model for both (no separate wrapper needed).
"""

from __future__ import annotations

import copy
import types
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import MaskedLMOutput

from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def _copy_module(module: nn.Module) -> nn.Module:
    """Deep copy an nn.Module (state and structure)."""
    return copy.deepcopy(module)


def _make_mlm_head_module(lm_model: nn.Module, base_model_prefix: str) -> nn.Module:
    """
    Build a single nn.Module that takes sequence_output [B, L, H] and returns logits [B, L, V].

    Handles HuggingFace MLM head layouts:
    - BERT/RoBERTa: .cls (BertOnlyMLMHead / RobertaLMHead) with forward(sequence_output).
    - DistilBERT: .vocab_transform, .vocab_layer_norm, .vocab_projector (no single head module).
    """
    # BERT / RoBERTa: head is a single module with forward(sequence_output)
    if hasattr(lm_model, "cls") and callable(getattr(lm_model.cls, "forward", None)):
        return _copy_module(lm_model.cls)

    # DistilBERT-style: vocab_transform, vocab_layer_norm, vocab_projector
    if hasattr(lm_model, "vocab_transform") and hasattr(lm_model, "vocab_projector"):
        vocab_transform = lm_model.vocab_transform
        vocab_layer_norm = getattr(lm_model, "vocab_layer_norm", None)
        vocab_projector = lm_model.vocab_projector

        class _DistilBertMLMHead(nn.Module):
            def __init__(self, vocab_transform, vocab_layer_norm, vocab_projector):
                super().__init__()
                self.vocab_transform = vocab_transform
                self.vocab_layer_norm = vocab_layer_norm
                self.vocab_projector = vocab_projector

            def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
                x = self.vocab_transform(sequence_output)
                x = F.gelu(x)
                if self.vocab_layer_norm is not None:
                    x = self.vocab_layer_norm(x)
                return self.vocab_projector(x)

        return _DistilBertMLMHead(
            _copy_module(vocab_transform),
            _copy_module(vocab_layer_norm) if vocab_layer_norm is not None else None,
            _copy_module(vocab_projector),
        )

    # Fallback: try lm_head (e.g. some models)
    if hasattr(lm_model, "lm_head") and callable(getattr(lm_model.lm_head, "forward", None)):
        return _copy_module(lm_model.lm_head)

    raise ValueError(
        "Could not extract MLM head from LM model. "
        "Expected BERT-style .cls, DistilBERT-style .vocab_*, or .lm_head."
    )


def attach_lm_head_to_classification_model(
    cls_model: nn.Module,
    lm_model: nn.Module,
    base_model_prefix: Optional[str] = None,
) -> nn.Module:
    """
    Attach the LM model's MLM head to the classification model and add forward_lm.

    The classification model is mutated in place: it gets a new submodule `lm_head`
    (a copy of the LM head so the LM model can be dropped) and a method
    `forward_lm` that runs encoder + lm_head and returns an object with .logits.

    This allows a single model to serve both classification (forward) and LMS
    (forward_lm). When GRADIEND does deepcopy(base_model), the copy has one
    encoder and both heads; decoder evaluation uses that copy for both tasks.

    Args:
        cls_model: AutoModelForSequenceClassification (or similar).
        lm_model: AutoModelForMaskedLM (or similar) from the same base.
        base_model_prefix: e.g. "bert", "distilbert". Inferred from cls_model if None.

    Returns:
        cls_model (mutated) with .lm_head and .forward_lm.
    """
    prefix = base_model_prefix or getattr(cls_model, "base_model_prefix", None)
    if not prefix or not hasattr(cls_model, prefix):
        raise ValueError(
            "Classification model has no base_model_prefix or matching attribute; "
            "cannot attach LM head."
        )
    if not hasattr(lm_model, prefix):
        raise ValueError(
            f"LM model has no attribute {prefix!r}; cannot share backbone."
        )

    # Build LM head module (may be a reference to lm_model's head or a new module)
    lm_head = _make_mlm_head_module(lm_model, prefix)
    # Ensure we own a copy so we can drop lm_model
    if not isinstance(lm_head, nn.Module):
        raise ValueError("_make_mlm_head_module did not return an nn.Module")
    # Copy state into a new module so we don't hold references to lm_model
    lm_head = _copy_module(lm_head)
    setattr(cls_model, "lm_head", lm_head)

    def forward_lm(
        self: nn.Module,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        base = getattr(self, prefix)
        encoder_out = base(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = (
            encoder_out[0]
            if isinstance(encoder_out, (list, tuple))
            else getattr(encoder_out, "last_hidden_state", encoder_out)
        )
        logits = self.lm_head(sequence_output)
        return MaskedLMOutput(logits=logits)

    cls_model.forward_lm = types.MethodType(forward_lm, cls_model)  # type: ignore[attr-defined]
    return cls_model


def build_dual_head_sequence_model(
    cls_model: nn.Module,
    lm_model: nn.Module,
    base_model_prefix: Optional[str] = None,
) -> nn.Module:
    """
    Build a dual-head model: same encoder as cls_model, classification head + LM head.

    Mutates cls_model in place by attaching the LM head and forward_lm, then
    returns it. The returned model is the same object as cls_model.
    """
    return attach_lm_head_to_classification_model(
        cls_model, lm_model, base_model_prefix=base_model_prefix
    )


def try_build_dual_head_from_base_path(
    cls_model: nn.Module,
    base_path: str,
    trust_remote_code: bool = False,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    If the base path provides an MLM model, attach its head to cls_model and return it.

    Otherwise return cls_model unchanged. Use this when loading the classification
    model from a base (e.g. distilbert-base-uncased) so that decoder LMS can use
    forward_lm on the same model.
    """
    from gradiend.trainer.text.common.loading import AutoModelForLM

    try:
        lm_model = AutoModelForLM.from_pretrained(
            base_path, trust_remote_code=trust_remote_code
        )
    except Exception as e:
        logger.debug(
            "Could not load LM from %s for dual-head; LMS will use fallback path: %s",
            base_path, e,
        )
        return cls_model

    prefix = getattr(cls_model, "base_model_prefix", None)
    if not prefix or not hasattr(cls_model, prefix):
        return cls_model
    if not hasattr(lm_model, prefix):
        return cls_model

    if device is not None:
        lm_model = lm_model.to(device)

    try:
        build_dual_head_sequence_model(cls_model, lm_model, base_model_prefix=prefix)
        logger.info(
            "Attached LM head from %s to classification model (forward_lm available).",
            base_path,
        )
    except Exception as e:
        logger.debug("Could not attach LM head: %s", e)

    return cls_model
