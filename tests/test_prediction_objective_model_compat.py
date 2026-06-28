"""Tests that explicit prediction_objective values match the base model architecture."""

import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.objective_hints import format_seq2seq_objective_hint
from gradiend.trainer.text.prediction.prediction_objective import (
    resolve_prediction_objective,
    validate_prediction_objective_for_model,
)


class _TrainerStub:
    experiment_dir = None

    def __init__(self, prediction_objective: str):
        self._training_args = TrainingArguments(
            prediction_objective=prediction_objective,
            experiment_dir=None,
        )


class _BertTokenizerStub:
    name_or_path = "bert-base-cased"
    mask_token = "[MASK]"
    mask_token_id = 103

    @property
    def __class__(self):
        return type("BertTokenizer", (), {"__name__": "BertTokenizer"})


class _T5TokenizerStub:
    name_or_path = "t5-small"
    mask_token = None
    mask_token_id = None

    @property
    def __class__(self):
        return type("T5Tokenizer", (), {"__name__": "T5Tokenizer"})


class _GptTokenizerStub:
    name_or_path = "gpt2"
    mask_token = None
    mask_token_id = None

    @property
    def __class__(self):
        return type("GPT2Tokenizer", (), {"__name__": "GPT2Tokenizer"})


def test_format_seq2seq_objective_hint_lists_all_three():
    hint = format_seq2seq_objective_hint()
    assert "seq2seq_decoder" in hint
    assert "seq2seq_decoder_sequence_cloze" in hint
    assert "seq2seq_encoder_mlm" in hint


def test_seq2seq_decoder_sequence_cloze_rejects_bert():
    with pytest.raises(ValueError, match="encoder-decoder"):
        validate_prediction_objective_for_model(
            "seq2seq_decoder_sequence_cloze",
            _BertTokenizerStub(),
        )


def test_seq2seq_decoder_sequence_cloze_accepts_t5():
    validate_prediction_objective_for_model(
        "seq2seq_decoder_sequence_cloze",
        _T5TokenizerStub(),
    )


def test_clm_sequence_cloze_rejects_bert():
    with pytest.raises(ValueError, match="decoder-only"):
        validate_prediction_objective_for_model(
            "clm_sequence_cloze",
            _BertTokenizerStub(),
        )


def test_mlm_mask_token_rejects_t5():
    with pytest.raises(ValueError, match="seq2seq_encoder_mlm"):
        validate_prediction_objective_for_model(
            "mlm_mask_token",
            _T5TokenizerStub(),
        )


def test_clm_next_token_rejects_t5():
    with pytest.raises(ValueError, match="seq2seq_encoder_mlm.*seq2seq_decoder"):
        validate_prediction_objective_for_model(
            "clm_next_token",
            _T5TokenizerStub(),
        )


def test_resolve_prediction_objective_validates_with_tokenizer():
    trainer = _TrainerStub("seq2seq_decoder_sequence_cloze")
    with pytest.raises(ValueError, match="encoder-decoder"):
        resolve_prediction_objective(trainer, _BertTokenizerStub())


def test_resolve_prediction_objective_auto_does_not_validate_bert():
    trainer = _TrainerStub("auto")
    objective = resolve_prediction_objective(trainer, _BertTokenizerStub())
    assert objective.name == "mlm_mask_token"


def test_resolve_prediction_objective_auto_seq2seq_uses_encoder_mlm():
    trainer = _TrainerStub("auto")
    objective = resolve_prediction_objective(trainer, _T5TokenizerStub())
    assert objective.name == "seq2seq_encoder_mlm"


@pytest.mark.slow(reason="Loads bert-base-cased tokenizer from Hugging Face cache.")
@pytest.mark.integration
def test_trainer_prediction_objective_rejects_bert_with_seq2seq_objective():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    from gradiend import TextPredictionTrainer, TrainingArguments

    trainer = TextPredictionTrainer(
        model="bert-base-cased",
        target_classes=["3SG", "3PL"],
        args=TrainingArguments(
            prediction_objective="seq2seq_decoder_sequence_cloze",
            experiment_dir=None,
        ),
    )
    tok = AutoTokenizer.from_pretrained("bert-base-cased")
    with pytest.raises(ValueError, match="encoder-decoder"):
        trainer._prediction_objective(tok)
