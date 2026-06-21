"""Tests for encoder-decoder (seq2seq) encoder-side MLM support."""

import pytest
import torch

from gradiend.model.utils import is_seq2seq_model, prediction_eval_kind
from gradiend.trainer.text.common.loading import AutoModelForLM
from gradiend.trainer.text.prediction.dataset import create_masked_pair_from_text
from gradiend.trainer.text.prediction.seq2seq import (
    create_seq2seq_mlm_item,
    mask_placeholder_for_tokenizer,
    seq2seq_encoder_mlm_loss,
    seq2seq_mask_token_id,
    tokenize_prediction_label,
)


def test_is_seq2seq_model_heuristic():
    class T5Tok:
        name_or_path = "t5-small"

    T5Tok.__name__ = "T5Tokenizer"
    assert is_seq2seq_model(T5Tok()) is True

    class BertTok:
        name_or_path = "bert-base-uncased"

    BertTok.__name__ = "BertTokenizer"
    assert is_seq2seq_model(BertTok()) is False


def test_mask_placeholder_uses_t5_sentinel():
    class Tok:
        mask_token = None

    assert mask_placeholder_for_tokenizer("foo [MASK] bar", Tok()) == "foo <extra_id_0> bar"


@pytest.mark.slow(reason="Uses the real t5-small SentencePiece tokenizer.")
def test_tokenize_prediction_label_prefers_leading_space_for_sentencepiece():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("t5-small")
    plain = tok("he", add_special_tokens=False)["input_ids"]
    spaced = tok(" he", add_special_tokens=False)["input_ids"]
    resolved = tokenize_prediction_label(tok, "he")
    if len(plain) > 1 and len(spaced) == 1:
        assert resolved == spaced
    else:
        assert len(resolved) >= 1


def test_seq2seq_mask_token_id_falls_back_to_t5_sentinel():
    class Tok:
        mask_token_id = None

        def convert_tokens_to_ids(self, token):
            assert token == "<extra_id_0>"
            return 32099

    assert seq2seq_mask_token_id(Tok()) == 32099


def test_seq2seq_mlm_rejects_empty_prediction_label():
    class Tok:
        mask_token = None
        mask_token_id = None
        pad_token_id = 0
        model_max_length = 16

        def __call__(
            self,
            text,
            add_special_tokens=False,
            return_tensors=None,
            max_length=None,
            truncation=False,
            padding=False,
        ):
            del add_special_tokens, truncation
            ids = [] if not str(text).strip() else [10]
            if return_tensors == "pt":
                width = max_length if padding == "max_length" and max_length is not None else max(1, len(ids))
                padded = ids[:width] + [self.pad_token_id] * max(0, width - len(ids))
                input_ids = torch.tensor([padded], dtype=torch.long)
                return {
                    "input_ids": input_ids,
                    "attention_mask": (input_ids != self.pad_token_id).long(),
                }
            return {"input_ids": ids}

        def convert_tokens_to_ids(self, token):
            return 32099 if token == "<extra_id_0>" else 2

    with pytest.raises(ValueError, match="Could not tokenize prediction label"):
        create_seq2seq_mlm_item("The cat sat on [MASK].", " ", Tok())


def test_create_masked_pair_skips_standalone_sentencepiece_marker():
    class Tok:
        def tokenize(self, text):
            del text
            return ["▁", "word"]

        def convert_tokens_to_string(self, tokens):
            return " ".join(tokens)

    pair = create_masked_pair_from_text(
        "ignored",
        Tok(),
        is_decoder_only_model=False,
        mask_token=None,
        min_prefix_tokens=0,
    )

    assert pair == ("▁[MASK]", "word")


@pytest.mark.slow(reason="Loads t5-small and runs a real encoder-side MLM loss.")
@pytest.mark.integration
def test_load_t5_small_encoder_mlm():
    model = AutoModelForLM.from_pretrained("t5-small")
    assert getattr(model.config, "is_encoder_decoder", False)
    assert prediction_eval_kind(model) == "seq2seq_decoder"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("t5-small")
    item = create_seq2seq_mlm_item("The cat sat on the [MASK].", "mat", tok, base_model=model)
    assert "input_ids" in item and "labels" in item
    mask_id = seq2seq_mask_token_id(tok)
    assert (item["labels"] == -100).sum() > 0
    assert (item["labels"] == mask_id).sum() == 0
    assert (item["labels"] != -100).sum() == 1
    loss = seq2seq_encoder_mlm_loss(model, item)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
