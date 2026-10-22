"""Causal-LM perplexity / LMS must score the logits at position t against token t+1.

The decoder-only LMS (``compute_lms`` -> ``evaluate_clm_perplexity``) previously compared each position's logits
with the token at that same position, i.e. it measured the probability of repeating the current token rather than the
language-model loss. These tests pin the causal shift, padding-side and batch-size invariance, ignore-token handling,
and agreement with Hugging Face's own causal-LM loss.
"""

import math

import pytest
import torch

from gradiend.trainer.text.common.lm_eval import (
    _clm_lms_from_avg_neg_log_likelihood,
    compute_lms,
    evaluate_clm_perplexity,
)

TEXTS = ["3 5 7 9 11", "2 4 6", "1 8 13 5", "12 3"]
N_TARGETS = 4 + 2 + 3 + 1  # every token except the first of each text


class _Tok:
    """Whitespace tokenizer over integer strings, with configurable padding side."""

    def __init__(self, padding_side="right", pad_id=0):
        self.padding_side = padding_side
        self.pad_token_id = pad_id

    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split()]

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, **_):
        seqs = [self.encode(t) for t in texts]
        n = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            pad = [self.pad_token_id] * (n - len(s))
            if self.padding_side == "left":
                ids.append(pad + s)
                mask.append([0] * len(pad) + [1] * len(s))
            else:
                ids.append(s + pad)
                mask.append([1] * len(s) + [0] * len(pad))
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class _Fake(torch.nn.Module):
    def __init__(self, vocab=16):
        super().__init__()
        self.vocab = vocab
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.dummy.device


class _NextTokenOracle(_Fake):
    """Puts almost all probability on the true NEXT token at every position."""

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab)
        logits.scatter_(2, torch.roll(input_ids, shifts=-1, dims=1).unsqueeze(-1), 30.0)
        return type("Out", (), {"logits": logits})()


class _CopyOracle(_Fake):
    """Puts almost all probability on the CURRENT token (what the unshifted loss rewarded)."""

    def forward(self, input_ids, attention_mask=None):
        b, t = input_ids.shape
        logits = torch.zeros(b, t, self.vocab)
        logits.scatter_(2, input_ids.unsqueeze(-1), 30.0)
        return type("Out", (), {"logits": logits})()


class _TokenOnly(_Fake):
    """Logits depend only on the token at the position, so padding offsets cannot matter."""

    def forward(self, input_ids, attention_mask=None):
        logits = torch.sin(input_ids.unsqueeze(-1).float() * (1 + torch.arange(self.vocab).float()))
        return type("Out", (), {"logits": logits})()


@pytest.mark.parametrize("side", ["right", "left"])
def test_next_token_predictor_scores_near_perfect(side):
    res = evaluate_clm_perplexity(_NextTokenOracle(), _Tok(side), TEXTS, batch_size=4)
    assert res["total_tokens"] == N_TARGETS
    assert res["perplexity"] < 1.01
    assert res["lms"] > 0.99


@pytest.mark.parametrize("side", ["right", "left"])
def test_current_token_copier_is_not_rewarded(side):
    """The unshifted bug gave a copier near-zero NLL; with the shift it must be heavily penalised."""
    res = evaluate_clm_perplexity(_CopyOracle(), _Tok(side), TEXTS, batch_size=4)
    assert res["total_log_likelihood"] / res["total_tokens"] > 20.0
    assert res["lms"] < 0.05


def test_padding_side_does_not_change_the_score():
    right = evaluate_clm_perplexity(_TokenOnly(), _Tok("right"), TEXTS, batch_size=4)
    left = evaluate_clm_perplexity(_TokenOnly(), _Tok("left"), TEXTS, batch_size=4)
    assert right["total_tokens"] == left["total_tokens"] == N_TARGETS
    assert math.isclose(right["total_log_likelihood"], left["total_log_likelihood"], rel_tol=1e-6)


@pytest.mark.parametrize("side", ["right", "left"])
def test_batch_size_does_not_change_the_score(side):
    full = evaluate_clm_perplexity(_TokenOnly(), _Tok(side), TEXTS, batch_size=4)
    single = evaluate_clm_perplexity(_TokenOnly(), _Tok(side), TEXTS, batch_size=1)
    assert full["total_tokens"] == single["total_tokens"] == N_TARGETS
    assert math.isclose(full["total_log_likelihood"], single["total_log_likelihood"], rel_tol=1e-6)


def test_ignored_tokens_are_excluded_as_targets_only():
    """An ignored token is never a prediction target, but still serves as context."""
    res = evaluate_clm_perplexity(_NextTokenOracle(), _Tok(), ["3 5 7 9 11"], batch_size=1, ignore=["7"])
    assert res["total_tokens"] == 3  # targets 5, 9, 11
    assert res["perplexity"] < 1.01


def test_single_token_text_has_no_target():
    res = evaluate_clm_perplexity(_NextTokenOracle(), _Tok(), ["7"], batch_size=1)
    assert res["total_tokens"] == 0
    assert res["lms"] == 0.0


def test_lms_is_monotone_decreasing_in_nll():
    vals = [_clm_lms_from_avg_neg_log_likelihood(x) for x in (0.0, 0.5, 2.0, 4.5, 9.0)]
    assert vals[0] == 1.0
    assert all(a > b for a, b in zip(vals, vals[1:]))


def _tiny_gpt2():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(vocab_size=32, n_positions=32, n_embd=16, n_layer=2, n_head=2)
    torch.manual_seed(0)
    return transformers.GPT2LMHeadModel(cfg).eval()


def test_matches_huggingface_causal_lm_loss_on_unpadded_text():
    model = _tiny_gpt2()
    ids = torch.tensor([[3, 5, 7, 9, 11, 4]])
    with torch.no_grad():
        hf_nll = model(input_ids=ids, labels=ids).loss.item()
    res = evaluate_clm_perplexity(model, _Tok(), ["3 5 7 9 11 4"], batch_size=1)
    assert res["total_tokens"] == 5
    assert math.isclose(res["total_log_likelihood"] / res["total_tokens"], hf_nll, rel_tol=1e-5)


def test_right_padded_batch_matches_token_weighted_per_sentence_hf_loss():
    model = _tiny_gpt2()
    total, count = 0.0, 0
    for t in TEXTS:
        ids = torch.tensor([[int(x) for x in t.split()]])
        with torch.no_grad():
            nll = model(input_ids=ids, labels=ids).loss.item()
        total += nll * (ids.shape[1] - 1)
        count += ids.shape[1] - 1
    res = evaluate_clm_perplexity(model, _Tok("right"), TEXTS, batch_size=4)
    assert res["total_tokens"] == count
    assert math.isclose(res["total_log_likelihood"], total, rel_tol=1e-4)


def test_compute_lms_dispatches_decoder_only_models_to_the_shifted_loss(monkeypatch):
    """compute_lms (the entry point used by causal/decoder evaluation) must use the shifted CLM loss."""
    import gradiend.trainer.text.common.lm_eval as lm_eval

    monkeypatch.setattr(lm_eval, "is_seq2seq_model", lambda _x: False)
    monkeypatch.setattr(lm_eval, "is_decoder_only_model", lambda _x: True)
    good = compute_lms(_NextTokenOracle(), _Tok(), TEXTS, batch_size=2)
    copier = compute_lms(_CopyOracle(), _Tok(), TEXTS, batch_size=2)
    assert good["lms"] > 0.99
    assert copier["lms"] < 0.05
