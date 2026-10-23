"""A long masked-LM row keeps its mask: a window around it replaces the naive truncation that cut the mask off."""

import re

import pytest
import torch

from gradiend.trainer.text.prediction.dataset import TextBatchedDataset, _window_around_masks


class _WordTokenizer:
    """Whitespace tokenizer with character offsets and the arguments of a Hugging Face tokenizer (no download)."""

    mask_token = "[MASK]"
    CLS, SEP, PAD = 100, 101, 0

    def __init__(self, offsets=True):
        self.vocab = {"[MASK]": 1}
        self.offsets = offsets

    @property
    def mask_token_id(self):
        return self.vocab["[MASK]"]

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, 0)

    def __call__(self, text, add_special_tokens=True, truncation=False, max_length=None, padding=None,
                 return_tensors=None, return_offsets_mapping=False, **kwargs):
        spans = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        ids = [self.vocab.setdefault(word, len(self.vocab) + 1) for word, _, _ in spans]
        encoded = {"input_ids": ids}
        if add_special_tokens:
            ids = [self.CLS] + ids + [self.SEP]
        if truncation and max_length:
            ids = ids[:max_length]
        attention = [1] * len(ids)
        if padding == "max_length" and max_length:
            attention += [0] * (max_length - len(ids))
            ids += [self.PAD] * (max_length - len(ids))
        encoded = {"input_ids": ids, "attention_mask": attention}
        if return_tensors == "pt":
            encoded = {key: torch.tensor([value]) for key, value in encoded.items()}
        if return_offsets_mapping:
            if not self.offsets:
                raise NotImplementedError("slow tokenizer")
            encoded["offset_mapping"] = [(start, end) for _, start, end in spans]
        return encoded


def _text(mask_at, n=40):
    words = [f"w{i}" for i in range(n)]
    words[mask_at] = "[MASK]"
    return " ".join(words)


def _window(text, tokenizer=None, max_length=10):
    return _window_around_masks(tokenizer or _WordTokenizer(), text, mask_token="[MASK]", max_length=max_length)


def test_the_mask_stays_with_context_on_both_sides():
    words = _window(_text(25)).split()
    assert "[MASK]" in words
    assert len(words) == 8  # max_length 10 minus two special tokens
    assert 0 < words.index("[MASK]") < len(words) - 1


def test_text_that_fits_is_unchanged():
    short = _text(3, n=6)
    assert _window(short) == short


def test_a_mask_near_the_start_spends_the_room_on_the_right():
    words = _window(_text(1)).split()
    assert words[:2] == ["w0", "[MASK]"] and len(words) == 8


def test_a_mask_near_the_end_spends_the_room_on_the_left():
    words = _window(_text(38)).split()
    assert words[-2:] == ["[MASK]", "w39"] and len(words) == 8


def test_all_masks_of_a_span_are_kept():
    words = [f"w{i}" for i in range(40)]
    words[20:23] = ["[MASK]"] * 3
    result = _window(" ".join(words)).split()
    assert result.count("[MASK]") == 3 and len(result) == 8


def test_tokenizer_without_offsets_returns_the_text_unchanged():
    text = _text(25)
    assert _window(text, _WordTokenizer(offsets=False)) == text


def test_text_without_a_mask_is_unchanged():
    text = " ".join(f"w{i}" for i in range(40))
    assert _window(text) == text


def _dataset(tokenizer, max_length=10):
    dataset = object.__new__(TextBatchedDataset)
    dataset.tokenizer = tokenizer
    dataset.max_length = max_length
    dataset.mask_token = tokenizer.mask_token
    dataset.mask_token_id = tokenizer.mask_token_id
    dataset.mask_placeholder = "[MASK]"
    dataset.is_decoder_only_model = False
    return dataset


def test_a_long_row_is_trained_on_the_window_around_its_mask():
    tokenizer = _WordTokenizer()
    item = _dataset(tokenizer)._create_item(_text(25), "target")
    labels = item["labels"]
    assert item["input_ids"].shape == (10,)
    positions = (labels != -100).nonzero().flatten().tolist()
    assert len(positions) == 1
    assert item["input_ids"][positions[0]].item() == tokenizer.mask_token_id  # the label sits on the mask
    assert labels[positions[0]].item() == tokenizer.vocab["target"]


def test_a_short_row_is_unchanged():
    tokenizer = _WordTokenizer()
    item = _dataset(tokenizer)._create_item(_text(3, n=6), "target")
    assert (item["labels"] != -100).sum().item() == 1
    assert item["attention_mask"].sum().item() == 8  # six words and two special tokens


def test_a_row_whose_mask_cannot_be_kept_fails_instead_of_training_on_nothing():
    dataset = _dataset(_WordTokenizer(offsets=False))
    with pytest.raises(ValueError, match="no mask position"):
        dataset._create_item(_text(25), "target")
