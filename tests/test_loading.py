from unittest.mock import Mock

import pytest

from gradiend.trainer.text.common.loading import AutoModelForLM
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
