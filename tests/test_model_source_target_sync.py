"""TrainingArguments.source/target must be stored on ModelWithGradiend."""

from __future__ import annotations

from gradiend.evaluator.decoder import derive_default_feature_factor
from gradiend.model._source_target import (
    feature_factor_from_encoding_direction,
    resolve_model_source,
    sync_model_source_target_from_training_args,
)
from tests.conftest import MockTokenizer


class _Args:
    source = "alternative"
    target = "diff"


class _Model:
    def __init__(self, source: str = "factual"):
        self._source = source
        self._target = "diff"
        self.feature_class_encoding_direction = {"3SG": 1.0, "3PL": -1.0}
        self.tokenizer = MockTokenizer()

    @property
    def source(self):
        return self._source

    @property
    def target(self):
        return self._target


class _Trainer:
    def __init__(self, model: _Model, args: _Args | None = None):
        self._model = model
        self._training_args = args

    def get_model(self):
        return self._model


def test_sync_model_source_target_from_training_args():
    model = _Model()
    sync_model_source_target_from_training_args(model, _Args(), log_mismatch=False)
    assert model.source == "alternative"
    assert model.target == "diff"


def test_resolve_model_source_prefers_model_over_training_args():
    model = _Model("factual")
    trainer = _Trainer(model, _Args())
    assert resolve_model_source(model, trainer) == "factual"


def test_resolve_model_source_falls_back_to_training_args():
    bare = type("Bare", (), {})()
    trainer = _Trainer(_Model())
    trainer._training_args = _Args()
    assert resolve_model_source(bare, trainer) == "alternative"


def test_derive_uses_training_args_when_model_has_no_source():
    bare = type("M", (), {
        "feature_class_encoding_direction": {"3SG": 1.0, "3PL": -1.0},
        "tokenizer": MockTokenizer(),
    })()
    trainer = _Trainer(_Model())
    trainer._training_args = _Args()
    assert derive_default_feature_factor(trainer, bare, class_name="3SG") == 1.0


def test_factual_and_alternative_feature_factor_signs():
    """Documented contract in gradiend.model._source_target."""
    assert feature_factor_from_encoding_direction(1.0, "factual") == -1.0
    assert feature_factor_from_encoding_direction(1.0, "alternative") == 1.0
    assert feature_factor_from_encoding_direction(1.0, "diff") == -1.0


def test_encoding_view_sign_for_source():
    from gradiend.model._source_target import encoding_view_sign_for_source

    assert encoding_view_sign_for_source("factual", "factual") == 1.0
    assert encoding_view_sign_for_source("alternative", "factual") == -1.0
    assert encoding_view_sign_for_source("factual", "counterfactual") == -1.0
    assert encoding_view_sign_for_source("alternative", "counterfactual") == 1.0
    assert encoding_view_sign_for_source("alternative", "transition") == 1.0
