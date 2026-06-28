"""Tests for seq2seq_decoder_sequence_cloze (multi-token decoder training)."""

import pandas as pd
import pytest
import torch

from gradiend.model.core.seq2seq_backbone import (
    SEQ2SEQ_ENCODER_MLM,
    resolve_seq2seq_mode_from_kwargs,
)
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset
from gradiend.trainer.text.prediction.prediction_objective import SUPPORTED_PREDICTION_OBJECTIVES
from gradiend.trainer.text.prediction.seq2seq import (
    SEQ2SEQ_DECODER_SEQUENCE_CLOZE,
    create_seq2seq_decoder_item,
    create_seq2seq_decoder_sequence_item,
    create_seq2seq_mlm_item,
    score_seq2seq_continuation_logprob,
)


class _TinyT5SpanTokenizer:
    name_or_path = "t5-small"
    mask_token = None
    mask_token_id = None
    pad_token_id = 0
    unk_token_id = 2
    model_max_length = 32

    _vocab = {
        "<pad>": 0,
        "<unk>": 2,
        "<extra_id_0>": 32099,
        "<extra_id_1>": 32098,
        "he": 10,
        "quickly": 11,
        "The": 12,
        "chef": 13,
        "soup": 14,
    }

    def _tokenize(self, text):
        tokens = []
        for part in str(text).replace(".", " ").split():
            tokens.append(part)
        return tokens

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_tensors=None,
        max_length=None,
        truncation=False,
        padding=False,
    ):
        del add_special_tokens
        if isinstance(text, list):
            rows = [self._encode_one(t) for t in text]
        else:
            rows = [self._encode_one(text)]
        if max_length is not None and truncation:
            rows = [row[:max_length] for row in rows]
        if return_tensors == "pt":
            width = max(len(row) for row in rows)
            if padding == "max_length" and max_length is not None:
                width = max_length
            padded = [row + [self.pad_token_id] * (width - len(row)) for row in rows]
            input_ids = torch.tensor(padded, dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": (input_ids != self.pad_token_id).long(),
            }
        return {"input_ids": rows[0] if not isinstance(text, list) else rows}

    def _encode_one(self, text):
        return [self._vocab.get(tok, self.unk_token_id) for tok in self._tokenize(text)]

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(str(token), self.unk_token_id)

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        inverse = {v: k for k, v in self._vocab.items()}
        return " ".join(inverse.get(int(i), "<unk>") for i in ids)


def test_prediction_objective_includes_seq2seq_decoder_sequence_cloze():
    assert SEQ2SEQ_DECODER_SEQUENCE_CLOZE in SUPPORTED_PREDICTION_OBJECTIVES


def test_resolve_seq2seq_mode_maps_sequence_cloze_to_decoder_backbone():
    mode = resolve_seq2seq_mode_from_kwargs({"prediction_objective": SEQ2SEQ_DECODER_SEQUENCE_CLOZE})
    assert mode == "seq2seq_decoder"


def test_sequence_cloze_uses_natural_t5_span_target():
    tok = _TinyT5SpanTokenizer()
    item = create_seq2seq_decoder_sequence_item("The chef [MASK] soup.", "he", tok)
    assert item["labels"].tolist() == [[32099, 10, 32098]]


def test_single_token_seq2seq_decoder_keeps_continuation_only_target():
    tok = _TinyT5SpanTokenizer()
    item = create_seq2seq_decoder_item("The chef [MASK] soup.", "he", tok)
    assert item["labels"].tolist() == [[10]]


@pytest.mark.slow(reason="Loads t5-small and checks real seq2seq decoder labels/loss.")
@pytest.mark.integration
def test_create_seq2seq_decoder_sequence_item_multi_token():
    pytest.importorskip("transformers")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    template = "The chef tasted the soup, then [MASK] added pepper."
    item = create_seq2seq_decoder_sequence_item(template, "he", tok, base_model=model, rhs_window=2)
    assert "input_ids" in item and "labels" in item
    assert item["labels"].shape[-1] >= 1
    outputs = model(**item)
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


@pytest.mark.slow(reason="Uses the real t5-small tokenizer for multi-token rejection behavior.")
@pytest.mark.integration
def test_create_seq2seq_decoder_item_rejects_multi_token_without_sequence_cloze():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("t5-small")
    template = "The chef tasted the soup, then [MASK] added pepper."
    with pytest.raises(ValueError, match="seq2seq_encoder_mlm.*seq2seq_decoder"):
        create_seq2seq_decoder_item(template, "he quickly", tok)


@pytest.mark.slow(reason="Uses the real t5-small tokenizer across seq2seq dataset objective routes.")
@pytest.mark.integration
def test_dataset_routes_encoder_mlm_and_sequence_cloze():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("t5-small")
    df = pd.DataFrame(
        [
            {
                "masked": "The chef [MASK] the soup.",
                "factual": "tasted",
                "alternative": "ignored",
                "factual_id": "A",
                "alternative_id": "B",
                "label": "A",
                "feature_class_id": 0,
            }
        ]
    )

    mlm_ds = TextTrainingDataset(
        df,
        tok,
        batch_size=1,
        is_seq2seq_model=True,
        prediction_objective=SEQ2SEQ_ENCODER_MLM,
    )
    mlm_item = mlm_ds._create_item("The chef <extra_id_0> the soup.", "tasted")
    assert mlm_item["labels"].ndim == 1
    assert (mlm_item["labels"] != -100).sum() >= 1
    assert "decoder_input_ids" not in mlm_item

    seq_ds = TextTrainingDataset(
        df,
        tok,
        batch_size=1,
        is_seq2seq_model=True,
        prediction_objective=SEQ2SEQ_DECODER_SEQUENCE_CLOZE,
        rhs_window=-1,
    )
    seq_item = seq_ds._create_item("The chef [MASK] the soup.", "tasted")
    assert seq_item["labels"].ndim == 1
    assert (seq_item["labels"] != -100).all()

    entry = seq_ds[0]
    assert "[MASK]" in entry["input_text"]


@pytest.mark.slow(reason="Loads t5-small and scores a real seq2seq continuation.")
@pytest.mark.integration
def test_score_seq2seq_continuation_logprob_finite():
    pytest.importorskip("transformers")
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    model.eval()
    logp = score_seq2seq_continuation_logprob(
        model,
        tok,
        prefix="The chef tasted the soup, then ",
        continuation="he",
        device=torch.device("cpu"),
        rhs=" added pepper.",
    )
    assert logp > float("-inf")
