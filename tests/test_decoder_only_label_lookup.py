"""Decoder-only training items look up their label token id without rebuilding the tokenizer vocabulary.

Hugging Face fast tokenizers build a new dict on every ``tokenizer.vocab`` access (about 40 ms for a 150k vocabulary).
Reading it for every training item made decoder-only training CPU-bound; the dataset reads it once.
"""

import torch

from gradiend.trainer.text.prediction.dataset import TextBatchedDataset

TOKENIZER_OFFSET = 1000  # ids from tokenizer(...) differ from vocabulary ids, so the two lookup paths can be told apart
UNKNOWN_FIRST_ID = TOKENIZER_OFFSET


class _FastTokenizerLike:
    """Whitespace tokenizer whose ``vocab`` property builds a new dict per access, as Hugging Face fast tokenizers do."""

    def __init__(self, words):
        self._ids = {word: index for index, word in enumerate(words, start=1)}
        self.vocab_builds = 0

    @property
    def vocab(self):
        self.vocab_builds += 1
        return dict(self._ids)

    def __call__(self, text, add_special_tokens=True, return_tensors=None, truncation=False, max_length=None, padding=None):
        ids = [TOKENIZER_OFFSET + self._ids.get(word, 0) for word in text.split()]
        if max_length is not None:
            ids = ids[:max_length]
        encoded = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if padding == "max_length" and max_length is not None:
            pad = max_length - len(ids)
            encoded = {"input_ids": ids + [0] * pad, "attention_mask": [1] * len(ids) + [0] * pad}
        if return_tensors == "pt":
            return {key: torch.tensor([value]) for key, value in encoded.items()}
        return encoded


class _TokenizerWithoutVocab:
    """A tokenizer that has no ``vocab`` attribute at all."""

    def __call__(self, text, add_special_tokens=True, return_tensors=None, truncation=False, max_length=None, padding=None):
        return _FastTokenizerLike(["a", "b"])(text, add_special_tokens, return_tensors, truncation, max_length, padding)


def _dataset(tokenizer):
    dataset = object.__new__(TextBatchedDataset)
    dataset.tokenizer = tokenizer
    dataset.max_length = 8
    dataset.mask_token = None
    dataset.mask_token_id = None
    dataset.mask_placeholder = "[MASK]"
    dataset.is_decoder_only_model = True
    return dataset


def _supervised_id(item):
    labels = item["labels"]
    positions = (labels != -100).nonzero().flatten().tolist()
    assert len(positions) == 1
    return int(labels[positions[0]])


def test_the_vocabulary_is_read_once_per_dataset_not_once_per_item():
    tokenizer = _FastTokenizerLike(["a", "b", "she"])
    dataset = _dataset(tokenizer)
    for _ in range(25):
        dataset._create_item("a b [MASK]", "she")
    assert tokenizer.vocab_builds == 1


def test_a_label_in_the_vocabulary_is_supervised_with_its_vocabulary_id():
    tokenizer = _FastTokenizerLike(["a", "b", "she"])
    supervised = _supervised_id(_dataset(tokenizer)._create_item("a b [MASK]", "she"))
    assert supervised == tokenizer._ids["she"]
    assert supervised != tokenizer("she")["input_ids"][0]  # the tokenizer would have given another id


def test_a_label_outside_the_vocabulary_uses_the_first_id_of_the_tokenized_label():
    tokenizer = _FastTokenizerLike(["a", "b", "she"])
    assert _supervised_id(_dataset(tokenizer)._create_item("a b [MASK]", "zzz")) == UNKNOWN_FIRST_ID


def test_a_tokenizer_without_a_vocabulary_attribute_uses_the_tokenized_label():
    item = _dataset(_TokenizerWithoutVocab())._create_item("a b [MASK]", "she")
    assert _supervised_id(item) == UNKNOWN_FIRST_ID
