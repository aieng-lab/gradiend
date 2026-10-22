from unittest.mock import ANY, Mock

import pytest

from gradiend.trainer.text.common.loading import (
    AutoModelForLM,
    _CatastrophicWeightLoad,
    _is_gemma3_multimodal_text_checkpoint,
    _load_causal_lm_for_config,
    _missing_weight_fraction,
    _raise_if_catastrophic_load,
)
from gradiend.trainer.text.common.model_base import _auto_base_model_device_map


def test_auto_model_for_lm_explains_unknown_transformers_architecture(monkeypatch):
    from gradiend.trainer.text.common import loading

    mock_config = Mock()
    mock_config.is_encoder_decoder = False
    monkeypatch.setattr(
        loading.AutoConfig,
        "from_pretrained",
        Mock(return_value=mock_config),
    )
    monkeypatch.setattr(
        loading.AutoModelForMaskedLM,
        "from_pretrained",
        Mock(side_effect=ValueError("not a masked LM")),
    )
    monkeypatch.setattr(
        loading.AutoModelForCausalLM,
        "from_pretrained",
        Mock(
            side_effect=ValueError(
                "The checkpoint you are trying to load has model type `gemma4` "
                "but Transformers does not recognize this architecture."
            )
        ),
    )

    with pytest.raises(ValueError, match="Transformers .* does not recognize"):
        AutoModelForLM.from_pretrained("google/gemma-4-31B", trust_remote_code=True)


def test_auto_base_model_device_map_for_large_model_names(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert _auto_base_model_device_map("Qwen/Qwen3-32B", None) == "auto"
    assert _auto_base_model_device_map("meta-llama/Llama-3.2-3B", None) is None
    assert _auto_base_model_device_map("Qwen/Qwen3-32B", False) is False
    assert _auto_base_model_device_map("Qwen/Qwen3-32B", "balanced") == "balanced"


def test_auto_base_model_device_map_requires_more_than_three_gpus(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    assert _auto_base_model_device_map("Qwen/Qwen3-32B", None) is None


def test_is_gemma3_multimodal_text_checkpoint():
    multimodal = Mock(model_type="gemma3", vision_config=Mock())
    text_only = Mock(model_type="gemma3", vision_config=None)
    other = Mock(model_type="llama", vision_config=Mock())

    assert _is_gemma3_multimodal_text_checkpoint(multimodal) is True
    assert _is_gemma3_multimodal_text_checkpoint(text_only) is False
    assert _is_gemma3_multimodal_text_checkpoint(other) is False


def _model_with_n_params(n, named=None):
    model = Mock()
    model.state_dict = Mock(return_value={f"p{i}": None for i in range(n)})
    return model


def test_load_causal_lm_for_config_uses_gemma3_text_only_class(monkeypatch):
    from gradiend.trainer.text.common import loading

    config = Mock(model_type="gemma3", vision_config=Mock())
    gemma_model = _model_with_n_params(10)
    gemma_cls = Mock(from_pretrained=Mock(return_value=(gemma_model, {"missing_keys": []})))
    monkeypatch.setattr(loading, "_Gemma3ForCausalLM", gemma_cls)

    result = _load_causal_lm_for_config(
        "google/gemma-3-27b-pt",
        config,
        {"trust_remote_code": False, "dtype": "auto"},
    )

    assert result is gemma_model
    gemma_cls.from_pretrained.assert_called_once_with(
        "google/gemma-3-27b-pt",
        token=ANY,
        output_loading_info=True,
        trust_remote_code=False,
        dtype="auto",
    )


def test_load_causal_lm_for_config_falls_back_to_auto_model_when_gemma3_class_cannot_load(monkeypatch):
    """Real cluster failure: Gemma3ForCausalLM itself reports 100% of weights missing for
    google/gemma-3-4b-pt AND google/gemma-3-27b-pt on the installed Transformers build --
    its key layout does not match these checkpoints at all, so it is not merely "vision
    tower omitted", it is a silent no-op. Falling back to plain AutoModelForCausalLM (the
    same call CAA/SAE already use successfully on these exact checkpoints) must still
    produce a working model instead of a hard failure."""
    from gradiend.trainer.text.common import loading

    config = Mock(model_type="gemma3", vision_config=Mock())
    broken_model = _model_with_n_params(10)
    gemma_cls = Mock(from_pretrained=Mock(return_value=(broken_model, {"missing_keys": [f"p{i}" for i in range(10)]})))
    monkeypatch.setattr(loading, "_Gemma3ForCausalLM", gemma_cls)
    good_model = _model_with_n_params(10)
    auto_causal = Mock(from_pretrained=Mock(return_value=(good_model, {"missing_keys": []})))
    monkeypatch.setattr(loading, "AutoModelForCausalLM", auto_causal)

    result = _load_causal_lm_for_config("google/gemma-3-27b-pt", config, {})

    assert result is good_model
    auto_causal.from_pretrained.assert_called_once()


def test_load_causal_lm_for_config_raises_when_both_gemma3_and_auto_causal_fail(monkeypatch):
    """If even the plain AutoModelForCausalLM fallback is catastrophic, still fail loudly --
    the fallback must not paper over a genuinely broken/mismatched checkpoint."""
    from gradiend.trainer.text.common import loading

    config = Mock(model_type="gemma3", vision_config=Mock())
    broken_gemma3 = _model_with_n_params(10)
    gemma_cls = Mock(from_pretrained=Mock(return_value=(broken_gemma3, {"missing_keys": [f"p{i}" for i in range(10)]})))
    monkeypatch.setattr(loading, "_Gemma3ForCausalLM", gemma_cls)
    broken_auto = _model_with_n_params(10)
    auto_causal = Mock(from_pretrained=Mock(return_value=(broken_auto, {"missing_keys": [f"p{i}" for i in range(10)]})))
    monkeypatch.setattr(loading, "AutoModelForCausalLM", auto_causal)

    with pytest.raises(_CatastrophicWeightLoad):
        _load_causal_lm_for_config("google/gemma-3-27b-pt", config, {})


class TestCatastrophicWeightLoadDetection:
    """Regression coverage for the real bug: AutoModelForMaskedLM.from_pretrained did not
    raise for a Gemma 3 4B/27B checkpoint -- it returned a model with EVERY weight
    (embeddings, every layer, lm_head) reported "missing", so every trainer-based method
    (GRADIEND/ACTIEND/AGIEND/CGA/CAGA) silently trained against a randomly initialized
    backbone. CAA/SAE were unaffected (they load via a different, direct AutoModelForCausalLM
    call in the study, never through AutoModelForLM)."""

    def test_missing_fraction(self):
        model = _model_with_n_params(10)
        assert _missing_weight_fraction(model, {"missing_keys": []}) == 0.0
        assert _missing_weight_fraction(model, {"missing_keys": [f"p{i}" for i in range(5)]}) == 0.5
        assert _missing_weight_fraction(model, {"missing_keys": [f"p{i}" for i in range(10)]}) == 1.0
        assert _missing_weight_fraction(model, {}) == 0.0

    def test_a_legitimate_head_only_mismatch_is_not_catastrophic(self):
        """A handful of missing keys (e.g. the MLM head, absent from a causal checkpoint) is normal."""
        model = _model_with_n_params(100)
        _raise_if_catastrophic_load("some/model", model, {"missing_keys": ["cls.predictions.bias"]})

    def test_every_weight_missing_raises(self):
        model = _model_with_n_params(10)
        with pytest.raises(_CatastrophicWeightLoad, match="100%"):
            _raise_if_catastrophic_load("google/gemma-3-4b-pt", model, {"missing_keys": [f"p{i}" for i in range(10)]})

    def test_the_real_bug_no_longer_silently_succeeds(self, monkeypatch):
        """The exact failure mode found on a real cluster run: AutoModelForMaskedLM.from_pretrained
        returns without raising, but loading_info shows a catastrophic load. AutoModelForLM must
        fall through to the causal-LM path instead of accepting the garbage model."""
        from gradiend.trainer.text.common import loading

        mock_config = Mock(model_type="gemma3", vision_config=Mock(), is_encoder_decoder=False)
        monkeypatch.setattr(loading.AutoConfig, "from_pretrained", Mock(return_value=mock_config))
        garbage_model = _model_with_n_params(20)
        monkeypatch.setattr(
            loading.AutoModelForMaskedLM, "from_pretrained",
            Mock(return_value=(garbage_model, {"missing_keys": [f"p{i}" for i in range(20)]})),
        )
        real_model = _model_with_n_params(20)
        real_model.parameters = lambda: iter([])
        monkeypatch.setattr(loading, "_load_causal_lm_for_config", Mock(return_value=real_model))

        model = AutoModelForLM.from_pretrained("google/gemma-3-4b-pt", trust_remote_code=False)

        assert model is real_model, "the catastrophically-loaded MaskedLM model must not be returned"
        loading._load_causal_lm_for_config.assert_called_once()

    def test_a_genuine_masked_lm_load_is_still_accepted(self, monkeypatch):
        """No false positive: a real, correctly-loaded MaskedLM checkpoint must not be rejected."""
        from gradiend.trainer.text.common import loading

        mock_config = Mock(model_type="bert", vision_config=None, is_encoder_decoder=False)
        monkeypatch.setattr(loading.AutoConfig, "from_pretrained", Mock(return_value=mock_config))
        good_model = _model_with_n_params(20)
        good_model.parameters = lambda: iter([])
        monkeypatch.setattr(
            loading.AutoModelForMaskedLM, "from_pretrained",
            Mock(return_value=(good_model, {"missing_keys": []})),
        )
        causal_fallback = Mock()
        monkeypatch.setattr(loading, "_load_causal_lm_for_config", causal_fallback)

        model = AutoModelForLM.from_pretrained("bert-base-uncased")

        assert model is good_model
        causal_fallback.assert_not_called()

    def test_seq2seq_branch_is_also_checked(self, monkeypatch):
        from gradiend.trainer.text.common import loading

        mock_config = Mock(is_encoder_decoder=True)
        monkeypatch.setattr(loading.AutoConfig, "from_pretrained", Mock(return_value=mock_config))
        broken = _model_with_n_params(10)
        monkeypatch.setattr(
            loading.AutoModelForSeq2SeqLM, "from_pretrained",
            Mock(return_value=(broken, {"missing_keys": [f"p{i}" for i in range(10)]})),
        )

        with pytest.raises(_CatastrophicWeightLoad):
            AutoModelForLM.from_pretrained("some/seq2seq-model")


def test_auto_model_for_lm_prefers_gemma3_causal_for_multimodal_gemma3(monkeypatch):
    from gradiend.trainer.text.common import loading

    mock_config = Mock(model_type="gemma3", vision_config=Mock(), is_encoder_decoder=False)
    monkeypatch.setattr(
        loading.AutoConfig,
        "from_pretrained",
        Mock(return_value=mock_config),
    )
    monkeypatch.setattr(
        loading.AutoModelForMaskedLM,
        "from_pretrained",
        Mock(side_effect=ValueError("not a masked LM")),
    )
    expected_model = Mock()
    expected_model.parameters = lambda: iter([])
    monkeypatch.setattr(
        loading,
        "_load_causal_lm_for_config",
        Mock(return_value=expected_model),
    )

    model = AutoModelForLM.from_pretrained("google/gemma-3-27b-pt", trust_remote_code=False)
    assert model is expected_model
    loading._load_causal_lm_for_config.assert_called_once()


def test_failed_gemma3_load_is_freed_before_the_fallback_loads(monkeypatch):
    """gemma-3-27b on one 80 GiB A100: the failed Gemma3ForCausalLM copy used to stay alive
    (caller local + exception traceback) while AutoModelForCausalLM loaded, so accelerate
    offloaded part of the REAL model to the CPU. The failed copy must be gone first."""
    import gc
    import weakref

    from gradiend.trainer.text.common import loading

    config = Mock(model_type="gemma3", vision_config=Mock())
    refs = {}

    def broken_load(*args, **kwargs):
        broken = _model_with_n_params(10)
        refs["broken"] = weakref.ref(broken)
        return broken, {"missing_keys": [f"p{i}" for i in range(10)]}

    alive_at_fallback = []
    good_model = _model_with_n_params(10)

    def good_load(*args, **kwargs):
        gc.collect()
        alive_at_fallback.append(refs["broken"]() is not None)
        return good_model, {"missing_keys": []}

    monkeypatch.setattr(loading, "_Gemma3ForCausalLM", Mock(from_pretrained=Mock(side_effect=broken_load)))
    monkeypatch.setattr(loading, "AutoModelForCausalLM", Mock(from_pretrained=Mock(side_effect=good_load)))

    assert _load_causal_lm_for_config("google/gemma-3-27b-pt", config, {}) is good_model
    assert alive_at_fallback == [False]


def test_failed_masked_lm_attempt_is_freed_before_the_causal_fallback(monkeypatch):
    import gc
    import weakref

    from gradiend.trainer.text.common import loading

    from types import SimpleNamespace

    config = SimpleNamespace(is_encoder_decoder=False, model_type="llama", vision_config=None)
    monkeypatch.setattr(loading.AutoConfig, "from_pretrained", Mock(return_value=config))
    refs = {}

    def broken_masked(*args, **kwargs):
        broken = _model_with_n_params(10)
        refs["broken"] = weakref.ref(broken)
        return broken, {"missing_keys": [f"p{i}" for i in range(10)]}

    alive_at_fallback = []
    good_model = _model_with_n_params(10)

    def causal_fallback(*args, **kwargs):
        gc.collect()
        alive_at_fallback.append(refs["broken"]() is not None)
        return good_model

    monkeypatch.setattr(loading.AutoModelForMaskedLM, "from_pretrained", Mock(side_effect=broken_masked))
    monkeypatch.setattr(loading, "_load_causal_lm_for_config", Mock(side_effect=causal_fallback))
    monkeypatch.setattr(loading, "set_requires_grad_true", lambda model: None)

    assert AutoModelForLM.from_pretrained("google/gemma-3-27b-pt") is good_model
    assert alive_at_fallback == [False]
