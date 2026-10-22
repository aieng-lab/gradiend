"""Decoder-only training items must label the last REAL token under either padding side.

Gemma tokenizers (2/3, all sizes) pad on the LEFT.  ``_create_item`` used to write
the label at ``attention_mask.sum() - 1``, which is the last real token only for
right padding; under left padding it landed inside the pad block, so the loss --
and therefore every gradient / encoder input -- no longer depended on the sentence
(all sentences of a class produced bit-identical GRADIEND inputs).
"""

import pandas as pd
import pytest
import torch

from gradiend.trainer.text.prediction.dataset import TextTrainingDataset
from tests.testing_mocks import MockTokenizer


class PaddedSideTokenizer(MockTokenizer):
    """MockTokenizer whose padding side is configurable."""

    def __init__(self, padding_side="right"):
        super().__init__()
        self.padding_side = padding_side

    def __call__(self, text, *args, **kwargs):
        out = super().__call__(text, *args, **kwargs)
        if self.padding_side != "left" or kwargs.get("return_tensors") != "pt":
            return out
        ids, mask = out["input_ids"], out["attention_mask"]
        n_real = mask.sum(dim=1)
        size = ids.size(1)
        rolled_ids = torch.stack([torch.roll(r, int(size - n), 0) for r, n in zip(ids, n_real)])
        rolled_mask = torch.stack([torch.roll(r, int(size - n), 0) for r, n in zip(mask, n_real)])
        out["input_ids"], out["attention_mask"] = rolled_ids, rolled_mask
        return out


def _dataset(tokenizer):
    data = pd.DataFrame({
        "masked": ["token_5 token_6 [MASK]"],
        "factual": ["token_9"],
        "alternative": ["token_8"],
        "factual_class": ["c1"],
        "alternative_class": ["c2"],
        "factual_id": [1],
        "alternative_id": [2],
        "label": ["positive"],
        "feature_class_id": [1],
        "feature_pole": ["pos"],
    })
    return TextTrainingDataset(
        data=data, tokenizer=tokenizer, batch_size=1, is_decoder_only_model=True
    )


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("prefix", ["token_5", "token_5 token_6 token_7 token_11"])
def test_label_is_on_last_real_token_for_both_padding_sides(side, prefix):
    tok = PaddedSideTokenizer(side)
    item = _dataset(tok)._create_item(prefix, "token_9")
    labels, mask = item["labels"], item["attention_mask"]
    labeled = (labels != -100).nonzero().flatten().tolist()
    last_real = mask.nonzero().flatten()[-1].item()
    assert labeled == [last_real], f"padding_side={side}: label at {labeled}, last real token at {last_real}"
    assert mask[labeled[0]].item() == 1, "label must never sit on a padding position"
    assert labels[labeled[0]].item() == tok.vocab["token_9"]


def test_left_and_right_padding_label_the_same_token():
    """Same sentence, different padding side: the labelled *token* is identical."""
    prefix = "token_5 token_6 token_7"
    got = {}
    for side in ("right", "left"):
        item = _dataset(PaddedSideTokenizer(side))._create_item(prefix, "token_9")
        pos = (item["labels"] != -100).nonzero().flatten().item()
        got[side] = item["input_ids"][pos].item()
    assert got["right"] == got["left"]


class NoBosTokenizer(PaddedSideTokenizer):
    """Like GPT-2 / Pythia / Qwen: no BOS is added, so an empty text tokenizes to an all-padding row."""

    def __call__(self, text, *args, **kwargs):
        if str(text).strip():
            return super().__call__(text, *args, **kwargs)
        out = super().__call__("token_5", *args, **kwargs)
        out["attention_mask"] = torch.zeros_like(out["attention_mask"])
        out["input_ids"] = torch.zeros_like(out["input_ids"])
        return out


@pytest.mark.parametrize("side", ["right", "left"])
def test_a_target_that_starts_the_text_is_kept_not_rejected(side):
    """race/religion: '[MASK] Beauty is a 2015 album' has an EMPTY prefix and no BOS to stand in for it.

    Such a row is all padding by construction. It must not trip the item contract (that crashed every
    race/religion training run) and must be built exactly as before: label on the last position.
    """
    tok = NoBosTokenizer(side)
    item = _dataset(tok)._create_item("", "token_9")
    labels, mask = item["labels"], item["attention_mask"]
    assert mask.sum().item() == 0, "premise: an empty prefix yields an all-padding row"
    assert (labels != -100).nonzero().flatten().tolist() == [labels.numel() - 1]
    assert labels[-1].item() == tok.vocab["token_9"]


@pytest.mark.parametrize("side", ["right", "left"])
def test_a_row_with_real_tokens_is_still_checked(side):
    """The exemption is only for rows without any real token."""
    from gradiend.util.positions import assert_labels_on_real_tokens

    labels = torch.full((2, 4), -100)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 0, 0]])
    labels[1, 3] = 5  # the empty row: allowed
    assert_labels_on_real_tokens(labels, mask, where="t", allow_empty_rows=True)
    labels[0, 3] = 7  # a row with real tokens and a label on padding: still an error
    with pytest.raises(ValueError, match="fall on padding"):
        assert_labels_on_real_tokens(labels, mask, where="t", allow_empty_rows=True)


def test_the_empty_row_exemption_is_opt_in():
    from gradiend.util.positions import assert_labels_on_real_tokens

    labels = torch.tensor([[-100, -100, 5]])
    mask = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="fall on padding"):
        assert_labels_on_real_tokens(labels, mask, where="t")
    assert_labels_on_real_tokens(labels, mask, where="t", allow_empty_rows=True)
