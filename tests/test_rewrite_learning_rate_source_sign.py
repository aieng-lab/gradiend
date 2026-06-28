"""
Regression tests: rewrite learning rate is source-independent.

Contract (do not break):
- Decoder evaluation stores and plots the nominal feature_factor and learning_rate.
- rewrite_base_model(..., feature_factor=ff, learning_rate=lr) applies the same signed update
  for source="factual", "alternative", and "diff" (orientation lives in feature_factor).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import pandas as pd

from gradiend.evaluator.decoder import DecoderEvaluator
from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel
from gradiend.model.model_with_gradiend import effective_rewrite_learning_rate
from gradiend import TrainingArguments
from tests.conftest import MockTokenizer, bind_trainer_cache_resolver


class _TinyParamModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))


class _TinyMWG(ModelWithGradiend):
    """Minimal ModelWithGradiend for rewrite sign tests."""

    def create_gradients(self, *args, **kwargs):
        raise NotImplementedError

    def _save_model(self, save_directory, **kwargs):
        raise NotImplementedError

    @classmethod
    def _load_model(cls, *args, **kwargs):
        raise NotImplementedError


def _make_tiny_mwg(source: str) -> _TinyMWG:
    torch.manual_seed(0)
    base = _TinyParamModel()
    gradiend = ParamMappedGradiendModel(
        input_dim=4,
        latent_dim=1,
        param_map={"weight": {"shape": (2, 2), "repr": "all"}},
    )
    with torch.no_grad():
        gradiend.decoder[0].linear.weight.fill_(1.0)
        gradiend.decoder[0].linear.bias.fill_(0.5)
    return _TinyMWG(base, gradiend, source=source, target="diff")


def _rewrite_delta(model: _TinyMWG, nominal_lr: float, nominal_ff: float = 1.0) -> torch.Tensor:
    before = model.base_model.weight.detach().clone()
    rewritten = model.rewrite_base_model(
        learning_rate=nominal_lr, feature_factor=nominal_ff, part="decoder"
    )
    return rewritten.weight.detach() - before


@pytest.mark.parametrize(
    "source,nominal,expected",
    [
        ("factual", 0.1, 0.1),
        ("diff", 0.05, 0.05),
        ("alternative", 0.1, 0.1),
    ],
)
def test_effective_rewrite_learning_rate_parametrized(source, nominal, expected):
    assert effective_rewrite_learning_rate(nominal, source) == expected


@pytest.mark.parametrize(
    "bad_source",
    ["counterfactual", "cf", None, "", "factual "],
)
def test_effective_rewrite_learning_rate_rejects_invalid_source(bad_source):
    with pytest.raises((ValueError, TypeError), match="source must be"):
        effective_rewrite_learning_rate(0.1, bad_source)


def test_rewrite_base_model_same_update_for_all_sources():
    nominal_lr = 0.05
    nominal_ff = 1.0
    factual = _make_tiny_mwg("factual")
    alternative = _make_tiny_mwg("alternative")

    delta_f = _rewrite_delta(factual, nominal_lr, nominal_ff)
    delta_a = _rewrite_delta(alternative, nominal_lr, nominal_ff)

    assert not torch.allclose(delta_f, torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(delta_f, delta_a, atol=1e-6)


def test_rewrite_base_model_calls_effective_rewrite_learning_rate_helper():
    model = _make_tiny_mwg("alternative")
    with patch(
        "gradiend.model.model_with_gradiend.effective_rewrite_learning_rate",
        wraps=effective_rewrite_learning_rate,
    ) as spy:
        model.rewrite_base_model(learning_rate=0.07, feature_factor=1.0, part="decoder")

    spy.assert_called_once_with(0.07, "alternative")


def test_rewrite_base_model_feature_factor_not_negated_for_alternative_source():
    model = _make_tiny_mwg("alternative")
    ff = 1.0
    delta_pos = _rewrite_delta(model, 0.08, ff)
    delta_neg_ff = _rewrite_delta(model, 0.08, -ff)
    assert not torch.allclose(delta_pos, delta_neg_ff, atol=1e-6)


class _DecoderTrainerForSignTest:
    target_classes = ["3SG", "3PL"]

    def __init__(self, model: _TinyMWG):
        self._training_args = MagicMock()
        self._training_args.use_cache = False
        self.experiment_dir = None
        self.run_id = None
        self._model = model
        self.config = MagicMock()
        self.config.decoder_eval_prob_on_other_class = True
        bind_trainer_cache_resolver(self)

    def _default_from_training_args(self, value, name, fallback=None):
        if value is not None:
            return value
        return getattr(self._training_args, name, fallback)

    def get_model(self):
        return self._model

    def get_target_feature_classes(self):
        return self.target_classes

    def _model_for_decoder_eval(self, model_with_gradiend):
        return model_with_gradiend

    def _resolve_decoder_eval_targets(self, training_like_df=None):
        return ({"3SG": ["he"], "3PL": ["they"]}, False)

    def _get_decoder_eval_dataframe(self, tokenizer, **kwargs):
        training_like_df = pd.DataFrame([
            {"masked": "[MASK] a", "label_class": "3PL", "label": "they", "alternative_id": "3SG"},
        ])
        neutral_df = pd.DataFrame([{"text": "neutral"}])
        return training_like_df, neutral_df

    def evaluate_base_model(self, base_model, tokenizer, **kwargs):
        return {
            "lms": {"lms": 0.99},
            "probs": {"3SG": 0.5},
            "probs_by_dataset": {"3SG": {"3SG": 0.5, "3PL": 0.5}},
        }


def test_decoder_eval_applies_nominal_lr_for_alternative_source():
    model = _make_tiny_mwg("alternative")
    model.tokenizer = MockTokenizer()
    model.name_or_path = "tiny"
    model.feature_class_encoding_direction = {"3SG": 1.0, "3PL": -1.0}

    trainer = _DecoderTrainerForSignTest(model)
    evaluator = DecoderEvaluator()

    with patch(
        "gradiend.model.model_with_gradiend.effective_rewrite_learning_rate",
        wraps=effective_rewrite_learning_rate,
    ) as spy:
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            feature_factors=[1.0],
            lrs=[0.01, 0.1],
            plot=False,
        )

    assert result["3SG"]["learning_rate"] > 0
    for _args, _kwargs in spy.call_args_list:
        nominal = _args[0]
        source = _args[1]
        assert nominal > 0
        assert source == "alternative"
        assert effective_rewrite_learning_rate(nominal, source) == nominal


def test_trainer_rewrite_base_model_forwards_nominal_lr_to_model():
    from tests.test_trainer_model import MockTrainerForTest

    trainer = MockTrainerForTest(model="mock-base", args=TrainingArguments(experiment_dir=None))
    mock_model = MagicMock()
    mock_model.rewrite_base_model = MagicMock(return_value=MagicMock())
    trainer._model_instance = mock_model

    decoder_results = {
        "3SG": {"feature_factor": -1.0, "learning_rate": 0.05, "value": 0.8},
        "grid": {},
    }
    trainer.rewrite_base_model(decoder_results=decoder_results, target_class="3SG")

    mock_model.rewrite_base_model.assert_called_once_with(
        learning_rate=0.05,
        feature_factor=-1.0,
    )
