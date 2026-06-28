"""CLM perplexity must not crash on badly rewritten models (e.g. GPT-2 decoder grid)."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from gradiend.trainer.text.common.lm_eval import evaluate_clm_perplexity


def test_evaluate_clm_perplexity_handles_extreme_loss_without_overflow():
    """Very high NLL yields inf perplexity and lms=0 instead of OverflowError."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0

    def _encode(sentences, **kwargs):
        n = len(sentences)
        return {
            "input_ids": torch.ones((n, 3), dtype=torch.long),
            "attention_mask": torch.ones((n, 3), dtype=torch.long),
        }

    tokenizer.side_effect = _encode

    model = MagicMock()
    model.device = torch.device("cpu")
    model.eval = MagicMock()

    # Logits strongly disfavor the true token (id=1) -> huge NLL, exp() would overflow.
    bad_logits = torch.zeros((1, 3, 2))
    bad_logits[:, :, 1] = -2000.0

    def _forward(input_ids, attention_mask):
        out = MagicMock()
        out.logits = bad_logits
        return out

    model.side_effect = _forward

    result = evaluate_clm_perplexity(model, tokenizer, ["hello world"], batch_size=1)
    assert result["perplexity"] == float("inf")
    assert result["lms"] == 0.0
