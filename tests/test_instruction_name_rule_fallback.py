"""The Instruct name rule is a hint: without a chat format a model is read as plain text (no network)."""
import logging
from types import SimpleNamespace

import pytest

from gradiend.trainer.text.common import loading
from gradiend.trainer.text.common.loading import AutoTokenizerForLM, InstructTokenizerWrapper

TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}{% if add_generation_prompt %}A:{% endif %}"


class _FakeTokenizer:
    def __init__(self, chat_template=None):
        self.chat_template = chat_template
        self.name_or_path = "fake"


def _load(monkeypatch, name, tokenizer, **kwargs):
    monkeypatch.setattr(loading, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
    return AutoTokenizerForLM.from_pretrained(name, **kwargs)


@pytest.mark.parametrize("name", ["Helsinki-NLP/opus-mt-en-it", "dbmdz/bert-base-it", "org/model-chat"])
def test_matching_name_without_chat_format_is_plain_text_with_one_warning(monkeypatch, caplog, name):
    tok = _FakeTokenizer()
    with caplog.at_level(logging.WARNING, logger=loading.logger.name):
        out = _load(monkeypatch, name, tok)
    assert out is tok
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert name in warnings[0].getMessage() and "instruction_model" in warnings[0].getMessage()


def test_matching_name_with_chat_template_is_chat_wrapped(monkeypatch):
    out = _load(monkeypatch, "org/x-instruct", _FakeTokenizer(TEMPLATE))
    assert isinstance(out, InstructTokenizerWrapper)


def test_explicit_true_without_chat_format_still_raises(monkeypatch):
    with pytest.raises(ValueError, match="No chat format"):
        _load(monkeypatch, "Helsinki-NLP/opus-mt-en-it", _FakeTokenizer(), instruction_model=True)


def test_explicit_false_is_plain_even_with_template(monkeypatch):
    tok = _FakeTokenizer(TEMPLATE)
    assert _load(monkeypatch, "org/x-instruct", tok, instruction_model=False) is tok


def test_non_matching_name_is_plain_without_warning(monkeypatch, caplog):
    tok = _FakeTokenizer(TEMPLATE)
    with caplog.at_level(logging.WARNING, logger=loading.logger.name):
        assert _load(monkeypatch, "org/base-model", tok) is tok
    assert not caplog.records
