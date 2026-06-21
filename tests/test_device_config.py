import pytest
import torch
import pandas as pd

from gradiend.model.model_with_gradiend import ModelWithGradiend
from gradiend.model.utils import get_hf_device_map, resolve_device_config_for_model
from gradiend.trainer.text.prediction.decoder_only_mlm import train_mlm_head
from gradiend.util import format_count


def test_format_count_uses_grouped_thousands():
    assert format_count(31982849024) == "31,982,849,024"
    assert format_count(torch.tensor(16)) == "16"


def test_visible_but_unusable_cuda_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    with pytest.raises(RuntimeError, match="CUDA devices are visible"):
        resolve_device_config_for_model()


def test_visible_but_unusable_cuda_allows_explicit_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)

    device_encoder, device_decoder, device_base_model, requires_multiple_gpus = (
        resolve_device_config_for_model(device="cpu")
    )

    assert device_encoder == torch.device("cpu")
    assert device_decoder == torch.device("cpu")
    assert device_base_model == torch.device("cpu")
    assert requires_multiple_gpus is False


def test_explicit_cuda_raises_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="PyTorch cannot initialize CUDA"):
        resolve_device_config_for_model(device="cuda:0")


def test_decoder_only_mlm_head_raises_on_visible_but_unusable_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    train_df = pd.DataFrame({"masked": ["[MASK]"], "label": ["der"]})

    with pytest.raises(RuntimeError, match="CUDA devices are visible"):
        train_mlm_head("unused-model", train_df, "unused-output")


def test_nested_hf_device_map_is_detected():
    class Base:
        hf_device_map = {"model.embed_tokens": 0, "model.layers.0": 1}

    class Decoder:
        base_model = Base()

    class Wrapped:
        decoder = Decoder()

    assert get_hf_device_map(Wrapped()) == Base.hf_device_map


def test_model_with_gradiend_does_not_move_nested_sharded_base_model():
    class Base:
        hf_device_map = {"model.embed_tokens": 0, "model.layers.0": 1}

        def to(self, *_args, **_kwargs):
            raise AssertionError("Sharded base model must not be moved with .to().")

        def named_parameters(self):
            return iter(())

    class WrappedBase:
        decoder = Base()

        def to(self, *_args, **_kwargs):
            raise AssertionError("Wrapper around sharded base model must not be moved with .to().")

        def named_parameters(self):
            return iter(())

    class Gradiend:
        encoder = None
        device_encoder = torch.device("cpu")
        param_map = []

        def to(self, *_, **__):
            return self

    class ConcreteModelWithGradiend(ModelWithGradiend):
        def _save_model(self, save_directory, **kwargs):
            pass

        @classmethod
        def _load_model(cls, load_directory, **kwargs):
            return None

        def create_gradients(self, factual, counterfactual=None, **kwargs):
            return {}

        def _ensure_gradiend_param_map_spec(self):
            pass

        def _sync_base_requires_grad_to_param_map(self):
            pass

    model = ConcreteModelWithGradiend(WrappedBase(), Gradiend(), base_model_device=torch.device("cpu"))

    assert model.base_model_is_sharded
    assert model.base_model_device is None


def test_place_inputs_follows_actual_base_model_device():
    """Forward inputs must match where base weights live, not a stale base_model_device."""

    class FakeEmb(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(10, 4))

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = FakeEmb()

        def get_input_embeddings(self):
            return self.embeddings

        def forward(self, **kwargs):
            return kwargs

    class Gradiend:
        encoder = None
        device_encoder = torch.device("cuda:0")
        param_map = []

        def to(self, *_, **__):
            return self

    class ConcreteModelWithGradiend(ModelWithGradiend):
        def _save_model(self, save_directory, **kwargs):
            pass

        @classmethod
        def _load_model(cls, load_directory, **kwargs):
            return None

        def create_gradients(self, factual, counterfactual=None, **kwargs):
            return {}

        def _ensure_gradiend_param_map_spec(self):
            pass

        def _sync_base_requires_grad_to_param_map(self):
            pass

    model = ConcreteModelWithGradiend(Base(), Gradiend(), base_model_device=torch.device("cpu"))
    model.base_model_device = torch.device("cuda:0")

    placed = model._place_inputs_for_base_forward({"input_ids": torch.tensor([1, 2, 3])})
    assert placed["input_ids"].device.type == "cpu"
