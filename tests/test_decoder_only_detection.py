"""``is_decoder_only_model`` reads the model config, not names or the tokenizer's mask token.

Two real failures this guards against:

* ``create_training_data`` passed the GRADIEND wrapper (class ``TextPredictionModelWithGradiend``)
  to the classifier, which fell back to "``Model`` in the class name -> decoder-only": every
  masked-LM (BERT, ...) was trained cut off at ``[MASK]`` with the label on the last token.
* Gemma-3 is a causal LM whose tokenizer ships an unused mask token, so "mask token present ->
  masked LM" sent it down the wrong path.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

import gradiend.model.utils as model_utils
from gradiend.model.utils import decoder_only_from_config, is_decoder_only_model
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from tests.testing_mocks import MockTokenizer


def _config(model_type, architectures=None, **extra):
    return SimpleNamespace(model_type=model_type, architectures=architectures, **extra)


BERT = _config("bert", ["BertForMaskedLM"])
ROBERTA = _config("roberta", ["RobertaForMaskedLM"])
DEBERTA_NO_ARCH = _config("deberta-v2")
GPT2 = _config("gpt2", ["GPT2LMHeadModel"])
LLAMA = _config("llama", ["LlamaForCausalLM"])
QWEN_NO_ARCH = _config("qwen3")
GEMMA3 = _config("gemma3", ["Gemma3ForConditionalGeneration"], text_config=_config("gemma3_text"))
GEMMA3_TEXT_ONLY = _config("gemma3_text", ["Gemma3ForCausalLM"])
T5 = _config("t5", ["T5ForConditionalGeneration"], is_encoder_decoder=True)
MULTIMODAL_UNKNOWN_ARCH = _config("some_vlm", None, text_config=_config("llama"))


@pytest.mark.parametrize(
    "config, expected",
    [
        (BERT, False), (ROBERTA, False), (DEBERTA_NO_ARCH, False),
        (GPT2, True), (LLAMA, True), (QWEN_NO_ARCH, True),
        (GEMMA3, True), (GEMMA3_TEXT_ONLY, True),
        (T5, False),
        (MULTIMODAL_UNKNOWN_ARCH, True),
    ],
)
def test_decoder_only_from_config(config, expected):
    assert decoder_only_from_config(config) is expected


def test_unknown_config_gives_no_verdict():
    assert decoder_only_from_config(None) is None
    assert decoder_only_from_config(_config("brand_new_arch")) is None


class TextPredictionModelWithGradiend:
    """Stand-in for the GRADIEND wrapper: no mask token, "Model" in its name, wraps a HF model."""

    def __init__(self, config, tokenizer=None):
        self.base_model = SimpleNamespace(config=config)
        self.tokenizer = tokenizer or MockTokenizer()


@pytest.mark.parametrize("config, expected", [(BERT, False), (ROBERTA, False), (GPT2, True), (GEMMA3, True), (T5, False)])
def test_wrapper_is_classified_by_the_wrapped_model(config, expected):
    assert is_decoder_only_model(TextPredictionModelWithGradiend(config)) is expected


def test_huggingface_model_object_uses_its_config():
    assert is_decoder_only_model(SimpleNamespace(config=BERT)) is False
    assert is_decoder_only_model(SimpleNamespace(config=GPT2)) is True


def test_class_name_of_a_model_without_config():
    class BertForMaskedLM:
        pass

    class GPT2LMHeadModel:
        pass

    assert is_decoder_only_model(BertForMaskedLM()) is False
    assert is_decoder_only_model(GPT2LMHeadModel()) is True


def test_tokenizer_resolves_its_model_config(monkeypatch):
    """A Gemma-3 tokenizer has a mask token yet its model is causal: the config decides."""
    gemma_tokenizer = SimpleNamespace(name_or_path="google/gemma-3-4b-pt", mask_token="<mask>", mask_token_id=4)
    monkeypatch.setattr(model_utils, "_is_hf_tokenizer", lambda obj: True)
    seen = []
    monkeypatch.setattr(model_utils, "_config_for_name", lambda name: seen.append(name) or GEMMA3)
    assert is_decoder_only_model(gemma_tokenizer) is True
    assert seen == ["google/gemma-3-4b-pt"]

    bert_tokenizer = SimpleNamespace(name_or_path="bert-base-cased", mask_token="[MASK]", mask_token_id=103)
    monkeypatch.setattr(model_utils, "_config_for_name", lambda name: BERT)
    assert is_decoder_only_model(bert_tokenizer) is False


def test_mask_token_is_only_the_last_resort(monkeypatch):
    """No config obtainable (offline, test doubles): fall back to "no mask token -> decoder-only"."""
    monkeypatch.setattr(model_utils, "_is_hf_tokenizer", lambda obj: True)
    monkeypatch.setattr(model_utils, "_config_for_name", lambda name: None)
    assert is_decoder_only_model(SimpleNamespace(name_or_path="x/y", mask_token="[MASK]", mask_token_id=1)) is False
    assert is_decoder_only_model(SimpleNamespace(name_or_path="x/y", mask_token=None, mask_token_id=None)) is True
    assert is_decoder_only_model(MockTokenizer()) is False  # not a HF tokenizer: never looked up


def test_encoder_decoder_is_never_decoder_only():
    assert is_decoder_only_model(TextPredictionModelWithGradiend(T5)) is False


@pytest.mark.parametrize("config, expected_decoder_only", [(BERT, False), (GPT2, True)])
def test_create_training_data_follows_the_wrapped_model(config, expected_decoder_only):
    """The regression: a BERT wrapper must yield masked-LM items (full sentence, [MASK] kept)."""
    df = pd.DataFrame([
        {"masked": "the nurse said that [MASK] was tired today", "split": "train", "label_class": "3SG", "label": "he"},
        {"masked": "the nurses said that [MASK] were tired today", "split": "train", "label_class": "3PL", "label": "they"},
    ])
    trainer = TextPredictionTrainer(
        model="bert-base-uncased",
        config=TextPredictionConfig(data=df, target_classes=["3SG", "3PL"]),
    )
    wrapper = TextPredictionModelWithGradiend(config)
    training_data = trainer.create_training_data(wrapper, split="train", batch_size=1)
    item = training_data[0]
    assert training_data.is_decoder_only_model is expected_decoder_only
    if expected_decoder_only:
        assert item["input_text"] == "the nurse said that " or item["input_text"] == "the nurses said that "
    else:
        assert item["input_text"] == item["template"], "masked LM must see the full sentence with [MASK]"
