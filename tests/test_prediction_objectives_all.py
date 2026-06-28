"""Coverage tests for every explicit prediction_objective value."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from gradiend.model.core.seq2seq_backbone import SEQ2SEQ_ENCODER_MLM
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.decoder_eval_utils import evaluate_probability_shift_score
from gradiend.trainer.text.prediction.prediction_objective import (
    SUPPORTED_PREDICTION_OBJECTIVES,
    resolve_prediction_objective,
    validate_prediction_objective_for_model,
)


class _BertTokenizerStub:
    name_or_path = "bert-base-cased"
    mask_token = "[MASK]"
    mask_token_id = 103

    @property
    def __class__(self):
        return type("BertTokenizer", (), {"__name__": "BertTokenizer"})


class _GptTokenizerStub:
    name_or_path = "gpt2"
    mask_token = None
    mask_token_id = None

    @property
    def __class__(self):
        return type("GPT2Tokenizer", (), {"__name__": "GPT2Tokenizer"})


class _T5TokenizerStub:
    name_or_path = "t5-small"
    mask_token = None
    mask_token_id = None

    @property
    def __class__(self):
        return type("T5Tokenizer", (), {"__name__": "T5Tokenizer"})


class _TrainerStub:
    experiment_dir = None

    def __init__(self, prediction_objective: str = "auto"):
        self._training_args = TrainingArguments(
            prediction_objective=prediction_objective,
            experiment_dir=None,
        )


class _Seq2SeqEncoderMlmModelStub(torch.nn.Module):
    name_or_path = "t5-small"

    def __init__(self):
        super().__init__()
        self._gradiend_seq2seq_gradient_mode = SEQ2SEQ_ENCODER_MLM

    @property
    def device(self):
        return torch.device("cpu")

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        vocab = 32
        batch, seq = input_ids.shape
        logits = torch.zeros(batch, seq, vocab)
        logits[..., 5] = 10.0  # token id 5 -> high prob for class A
        logits[..., 7] = 10.0  # token id 7 -> high prob for class B
        return type("Output", (), {"logits": logits})()


class _TinyTokenizerForSeq2seqEval:
    name_or_path = "t5-small"
    mask_token = None
    mask_token_id = None

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=512):
        del padding, truncation, max_length
        if isinstance(texts, str):
            texts = [texts]
        ids = [[1, 2, 3] for _ in texts]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
        }

    def tokenize(self, text):
        return [str(text)]

    def convert_tokens_to_ids(self, token):
        return {"he": 5, "they": 7}.get(str(token), 0)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


EXPLICIT_OBJECTIVES = sorted(SUPPORTED_PREDICTION_OBJECTIVES - {"auto"})

# objective -> {bert, gpt2, t5} expected to validate without error
_COMPAT_OK = {
    "mlm_mask_token": {"bert"},
    "clm_next_token": {"gpt2"},
    "clm_mlm_head": {"gpt2"},
    "clm_sequence_cloze": {"gpt2"},
    "seq2seq_decoder": {"t5"},
    "seq2seq_decoder_sequence_cloze": {"t5"},
    "seq2seq_encoder_mlm": {"t5"},
}

_STUBS = {
    "bert": _BertTokenizerStub(),
    "gpt2": _GptTokenizerStub(),
    "t5": _T5TokenizerStub(),
}


@pytest.mark.parametrize("objective", EXPLICIT_OBJECTIVES)
@pytest.mark.parametrize("model_kind", ["bert", "gpt2", "t5"])
def test_validate_prediction_objective_matrix(objective, model_kind):
    stub = _STUBS[model_kind]
    should_pass = model_kind in _COMPAT_OK.get(objective, set())
    if should_pass:
        validate_prediction_objective_for_model(objective, stub)
    else:
        with pytest.raises(ValueError):
            validate_prediction_objective_for_model(objective, stub)


@pytest.mark.parametrize(
    ("model_kind", "expected"),
    [
        ("bert", "mlm_mask_token"),
        ("gpt2", "clm_next_token"),
        ("t5", "seq2seq_encoder_mlm"),
    ],
)
def test_resolve_prediction_objective_auto(model_kind, expected):
    objective = resolve_prediction_objective(_TrainerStub("auto"), _STUBS[model_kind])
    assert objective.name == expected


def test_evaluate_probability_shift_seq2seq_encoder_mlm_does_not_use_mask_token(monkeypatch):
    df = pd.DataFrame(
        [
            {"masked": "The chef [MASK] the soup.", "label_class": "3SG"},
            {"masked": "They [MASK] quickly.", "label_class": "3PL"},
        ]
    )
    model = _Seq2SeqEncoderMlmModelStub()
    tokenizer = _TinyTokenizerForSeq2seqEval()

    def _fake_seq2seq_mlm_probs_at_mask(m, tok, texts, device):
        del m, tok, device
        vocab = 32
        rows = []
        for _ in texts:
            probs = torch.zeros(vocab)
            probs[5] = 0.9
            probs[7] = 0.1
            rows.append(probs)
        return torch.stack(rows, dim=0)

    monkeypatch.setattr(
        "gradiend.trainer.text.prediction.decoder_eval_utils.seq2seq_mlm_probs_at_mask",
        _fake_seq2seq_mlm_probs_at_mask,
    )

    result = evaluate_probability_shift_score(
        model,
        tokenizer,
        targets={"3SG": ["he"], "3PL": ["they"]},
        eval_data_df=df,
        objective="seq2seq_encoder_mlm",
    )

    assert "3SG" in result
    assert "3PL" in result
    assert result["3SG"]["3SG"] > 0.5
    assert result["3PL"]["3PL"] >= 0.0


def test_evaluate_probability_shift_auto_seq2seq_encoder_mlm_does_not_use_mask_token(monkeypatch):
    df = pd.DataFrame([{"masked": "The chef [MASK] the soup.", "label_class": "3SG"}])
    model = _Seq2SeqEncoderMlmModelStub()
    tokenizer = _TinyTokenizerForSeq2seqEval()

    def _fake_seq2seq_mlm_probs_at_mask(m, tok, texts, device):
        del m, tok, texts, device
        probs = torch.zeros(1, 32)
        probs[0, 5] = 0.9
        probs[0, 7] = 0.1
        return probs

    monkeypatch.setattr(
        "gradiend.trainer.text.prediction.decoder_eval_utils.seq2seq_mlm_probs_at_mask",
        _fake_seq2seq_mlm_probs_at_mask,
    )

    result = evaluate_probability_shift_score(
        model,
        tokenizer,
        targets={"3SG": ["he"], "3PL": ["they"]},
        eval_data_df=df,
        objective="auto",
    )

    assert result["3SG"]["3SG"] > 0.5


@pytest.mark.slow(reason="Loads t5-small and runs seq2seq probability-shift integration.")
@pytest.mark.integration
def test_evaluate_probability_shift_seq2seq_encoder_mlm_on_t5_small():
    pytest.importorskip("transformers")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model._gradiend_seq2seq_gradient_mode = SEQ2SEQ_ENCODER_MLM
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    df = pd.DataFrame(
        [
            {
                "masked": "The chef [MASK] the soup.",
                "label_class": "3SG",
            }
        ]
    )

    result = evaluate_probability_shift_score(
        model,
        tokenizer,
        targets={"3SG": ["tasted"], "3PL": ["ignored"]},
        eval_data_df=df,
        objective="seq2seq_encoder_mlm",
    )

    assert isinstance(result, dict)
    assert "3SG" in result
    assert "3SG" in result["3SG"]
