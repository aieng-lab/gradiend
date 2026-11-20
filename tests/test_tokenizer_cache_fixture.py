"""Tests for the session-wide Hub tokenizer cache fixture in conftest.py."""

from transformers import AutoTokenizer


def test_hub_id_returns_independent_copies():
    first = AutoTokenizer.from_pretrained("gpt2")
    second = AutoTokenizer.from_pretrained("gpt2")
    assert first is not second
    first.add_special_tokens({"pad_token": "<my_pad>"})
    third = AutoTokenizer.from_pretrained("gpt2")
    assert third.pad_token != "<my_pad>"
    assert len(third) == len(second)


def test_local_directory_is_not_cached(tmp_path):
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.save_pretrained(str(tmp_path))
    loaded = AutoTokenizer.from_pretrained(str(tmp_path))
    assert loaded.pad_token is None

    tok.add_special_tokens({"pad_token": "<local_pad>"})
    tok.save_pretrained(str(tmp_path))
    reloaded = AutoTokenizer.from_pretrained(str(tmp_path))
    assert reloaded.pad_token == "<local_pad>"


def test_classmethod_still_binds_cls():
    assert AutoTokenizer.from_pretrained("gpt2").__class__.__name__.startswith("GPT2")
