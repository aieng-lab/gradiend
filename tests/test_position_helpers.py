"""Unit tests for the "last/first real token" helpers (left- and right-padding)."""

import pytest
import torch

from gradiend.trainer.text.prediction import dataset as dataset_module
from gradiend.util.positions import (
    assert_labels_on_real_tokens,
    first_real_token_positions,
    last_real_token_positions,
)
from tests.test_decoder_only_padding_side import PaddedSideTokenizer, _dataset


def _batch(side, lengths, width=None):
    """Sequences of unique token ids, padded on ``side``; returns (input_ids, attention_mask, last_ids)."""
    width = width or max(lengths)
    ids = torch.zeros(len(lengths), width, dtype=torch.long)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    last_ids = []
    for row, n in enumerate(lengths):
        tokens = torch.arange(1, n + 1) + 100 * (row + 1)  # non-zero, unique per row
        last_ids.append(int(tokens[-1]))
        sl = slice(width - n, width) if side == "left" else slice(0, n)
        ids[row, sl], mask[row, sl] = tokens, 1
    return ids, mask, torch.tensor(last_ids)


LENGTHS = [3, 7, 1, 5, 7]


@pytest.mark.parametrize("side", ["left", "right"])
def test_last_real_token_positions_for_both_padding_sides(side):
    ids, mask, last_ids = _batch(side, LENGTHS)
    pos = last_real_token_positions(mask)
    assert ids[torch.arange(len(LENGTHS)), pos].tolist() == last_ids.tolist()


def test_first_real_token_positions_is_the_left_pad_width():
    _, mask, _ = _batch("left", LENGTHS)
    assert first_real_token_positions(mask).tolist() == [max(LENGTHS) - n for n in LENGTHS]
    _, mask, _ = _batch("right", LENGTHS)
    assert first_real_token_positions(mask).tolist() == [0] * len(LENGTHS)


def test_empty_rows_raise_unless_allowed():
    mask = torch.tensor([[0, 0, 0], [0, 1, 1]])
    with pytest.raises(ValueError, match="empty"):
        last_real_token_positions(mask)
    assert last_real_token_positions(mask, allow_empty=True).tolist() == [0, 2]


def test_non_2d_mask_is_rejected():
    with pytest.raises(ValueError, match="2D"):
        last_real_token_positions(torch.ones(4))


def test_contract_accepts_labels_on_real_tokens_and_rejects_padding():
    mask = torch.tensor([[0, 0, 1, 1]])
    ok = torch.tensor([[-100, -100, -100, 7]])
    assert_labels_on_real_tokens(ok, mask, where="test")
    with pytest.raises(ValueError, match="padding"):
        assert_labels_on_real_tokens(torch.tensor([[-100, 7, -100, -100]]), mask, where="test")
    assert_labels_on_real_tokens(torch.tensor([7]), None, where="no mask is nothing to check")


def test_dataset_raises_at_the_source_if_a_position_is_computed_the_old_way(monkeypatch):
    """The guard fires in ``_create_item`` -- not three layers later as constant gradients."""

    def sum_minus_one(attention_mask, *, allow_empty=False):  # the old, right-padding-only formula
        return attention_mask.sum(dim=1) - 1

    monkeypatch.setattr(dataset_module, "last_real_token_positions", sum_minus_one)
    with pytest.raises(ValueError, match="fall on padding"):
        _dataset(PaddedSideTokenizer("left"))._create_item("token_5 token_6 token_7", "token_9")
    # right padding is unaffected by that formula
    _dataset(PaddedSideTokenizer("right"))._create_item("token_5 token_6 token_7", "token_9")
