from unittest.mock import ANY, Mock

import pytest

from gradiend.trainer.text.common.loading import (
    AutoModelForLM,
    _is_gemma3_multimodal_text_checkpoint,
    _load_causal_lm_for_config,
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


def test_load_causal_lm_for_config_uses_gemma3_text_only_class(monkeypatch):
    from gradiend.trainer.text.common import loading

    config = Mock(model_type="gemma3", vision_config=Mock())
    gemma_model = Mock()
    gemma_cls = Mock(from_pretrained=Mock(return_value=gemma_model))
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
        trust_remote_code=False,
        dtype="auto",
    )


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
