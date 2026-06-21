"""Tests for seq2seq language-model evaluation (LMS / perplexity)."""

import pytest
import torch

from gradiend.trainer.text.common.lm_eval import compute_lms, evaluate_seq2seq_perplexity


class _TinySeq2Seq(torch.nn.Module):
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        batch, seq = input_ids.shape
        vocab = self.vocab_size
        logits = torch.zeros(batch, seq, vocab)
        loss = None
        if labels is not None:
            active = (labels != -100).sum().float()
            loss = torch.tensor(0.5) if active > 0 else None
        return type("Out", (), {"logits": logits, "loss": loss})()


class _TinyT5Tokenizer:
    name_or_path = "t5-small"
    mask_token = None
    mask_token_id = None
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1 + (hash(text) % 10)]

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
        if isinstance(texts, str):
            texts = [texts]
        ids = [self.encode(t) + [0] * 3 for t in texts]
        max_len = max(len(row) for row in ids)
        padded = [row + [0] * (max_len - len(row)) for row in ids]
        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_compute_lms_routes_seq2seq_to_seq2seq_perplexity():
    model = _TinySeq2Seq()
    tok = _TinyT5Tokenizer()
    result = compute_lms(model, tok, ["hello world", "another sentence"], batch_size=2)
    assert "lms" in result
    assert "perplexity" in result
    assert result["total_tokens"] > 0


def test_evaluate_seq2seq_perplexity_handles_ignore_tokens():
    model = _TinySeq2Seq()
    tok = _TinyT5Tokenizer()
    result = evaluate_seq2seq_perplexity(
        model,
        tok,
        ["hello world"],
        ignore=["hello"],
    )
    assert result["total_tokens"] >= 0


@pytest.mark.slow(reason="Loads t5-small and computes real seq2seq LMS/perplexity.")
@pytest.mark.integration
def test_compute_lms_t5_small():
    pytest.importorskip("transformers")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    tok = AutoTokenizer.from_pretrained("t5-small")
    model.eval()
    result = compute_lms(
        model,
        tok,
        ["The cat sat on the mat.", "Dogs run in the park."],
        batch_size=2,
    )
    assert 0 < result["lms"] < 1
    assert result["perplexity"] > 1
