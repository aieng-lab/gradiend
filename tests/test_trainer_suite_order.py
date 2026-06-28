import os
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.multi_seed import is_multi_seed_view
from gradiend.trainer.trainer import Trainer
from gradiend.trainer.suite import PositiveFeatureDefinition, PositiveTrainerSuite, SuitePairDefinition, TrainerSuite
from tests.test_multi_seed_view import _local_temp
from tests.test_trainer_model import MockTrainerForTest


class _ConcreteTrainerSuite(TrainerSuite):
    def _resolve_pair_definitions(self):
        return []


class _PairSuite(TrainerSuite):
    def _resolve_pair_definitions(self):
        return [
            SuitePairDefinition(target_classes=("good", "bad"), child_id="good__bad"),
            SuitePairDefinition(target_classes=("happy", "sad"), child_id="happy__sad"),
        ]


class _RecordingTrainer(Trainer):
    calls = []

    def __init__(self, *args, **kwargs):
        self.__class__.calls.append(kwargs)
        self._target_classes = kwargs.get("target_classes")
        self._all_classes = kwargs.get("all_classes")
        self._base_model_path = kwargs.get("model")

    def create_training_data(self, *args, **kwargs):
        raise NotImplementedError

    def create_gradient_training_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def _get_decoder_eval_dataframe(self, *args, **kwargs):
        raise NotImplementedError

    def _get_decoder_eval_targets(self):
        raise NotImplementedError

    def evaluate_base_model(self, *args, **kwargs):
        raise NotImplementedError

    def _analyze_encoder(self, *args, **kwargs):
        raise NotImplementedError


def _make_suite():
    suite = object.__new__(PositiveTrainerSuite)
    suite.label_mapping = {
        "alpha_id": "Alpha",
        "beta_id": "Beta",
        "gamma_id": "Gamma",
        "delta_id": "Delta",
    }
    suite.pair_by_id = {
        "gamma_id": ("gamma", "non_gamma"),
        "alpha_id": ("alpha", "non_alpha"),
        "delta_id": ("delta", "non_delta"),
        "beta_id": ("beta", "non_beta"),
    }
    suite.pair_definitions = {
        child_id: SuitePairDefinition(target_classes=pair, child_id=child_id)
        for child_id, pair in suite.pair_by_id.items()
    }
    suite.trainers = {child_id: object() for child_id in suite.pair_by_id}
    return suite


def test_child_trainers_receive_suite_wide_all_classes():
    _RecordingTrainer.calls = []

    _PairSuite(
        _RecordingTrainer,
        model="bert-base-uncased",
        target_classes=["good", "bad", "happy", "sad"],
    )

    assert [call["target_classes"] for call in _RecordingTrainer.calls] == [
        ["good", "bad"],
        ["happy", "sad"],
    ]
    assert [call["all_classes"] for call in _RecordingTrainer.calls] == [
        ["good", "bad", "happy", "sad"],
        ["good", "bad", "happy", "sad"],
    ]


def test_positive_suite_child_training_args_include_other_classes_for_eval():
    _RecordingTrainer.calls = []

    PositiveTrainerSuite(
        _RecordingTrainer,
        model="bert-base-uncased",
        target_classes=["good", "bad", "happy", "sad"],
        positive_feature_definitions=[
            PositiveFeatureDefinition("good", "bad"),
            PositiveFeatureDefinition("happy", "sad"),
        ],
        args=TrainingArguments(include_other_classes=False),
    )

    assert [call["target_classes"] for call in _RecordingTrainer.calls] == [
        ["good", "bad"],
        ["happy", "sad"],
    ]
    assert all(call["args"].include_other_classes is True for call in _RecordingTrainer.calls)


def test_plot_similarity_heatmap_respects_explicit_order(monkeypatch):
    suite = _make_suite()

    captured_get_models = {}

    def _fake_get_models(**kwargs):
        captured_get_models["kwargs"] = kwargs
        return {suite.label_mapping[k]: object() for k in suite.pair_by_id}

    monkeypatch.setattr(suite, "get_models", _fake_get_models)

    captured = {}

    def _fake_plot(models, **kwargs):
        captured["order"] = kwargs.get("order")
        return {"model_ids": list(models.keys())}

    monkeypatch.setattr("gradiend.trainer.suite.plot_similarity_heatmap", _fake_plot)

    explicit_order = [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
    ]
    suite.plot_similarity_heatmap(order=explicit_order)

    assert captured["order"] == explicit_order
    assert captured_get_models["kwargs"]["gradiend_only"] is True


def test_plot_similarity_heatmap_preserves_input_order_when_no_operator_pattern(monkeypatch):
    suite = object.__new__(PositiveTrainerSuite)
    suite.label_mapping = {"a": "A", "b": "B"}
    suite.pair_by_id = {
        "b": ("fem_nom", "masc_nom"),
        "a": ("fem_acc", "masc_acc"),
    }
    suite.pair_definitions = {
        child_id: SuitePairDefinition(target_classes=pair, child_id=child_id)
        for child_id, pair in suite.pair_by_id.items()
    }
    suite.trainers = {child_id: object() for child_id in suite.pair_by_id}

    monkeypatch.setattr(
        suite,
        "get_models",
        lambda **kwargs: {"A": object(), "B": object()},
    )

    captured = {}

    def _fake_plot(models, **kwargs):
        captured["order"] = kwargs.get("order")
        return {"model_ids": list(models.keys())}

    monkeypatch.setattr("gradiend.trainer.suite.plot_similarity_heatmap", _fake_plot)

    suite.plot_similarity_heatmap()

    assert captured["order"] == "input"


class _FakeTrainer:
    def __init__(self, base_model_path, resolved_model_path):
        self.base_model_path = base_model_path
        self._resolved_model_path = resolved_model_path

    def resolve_model_path(self, model):
        return self._resolved_model_path


def test_validate_shared_model_compatibility_accepts_same_resolved_path():
    suite = object.__new__(_ConcreteTrainerSuite)
    suite.trainers = {
        "a": _FakeTrainer("bert-base", "shared-head"),
        "b": _FakeTrainer("bert-base", "shared-head"),
    }
    suite._shared_model_key = None

    suite._validate_shared_model_compatibility()

    assert suite._shared_model_key == "shared-head"


def test_validate_shared_model_compatibility_rejects_mismatched_resolved_path():
    suite = object.__new__(_ConcreteTrainerSuite)
    suite.trainers = {
        "a": _FakeTrainer("bert-base", "head-a"),
        "b": _FakeTrainer("bert-base", "head-b"),
    }
    suite._shared_model_key = None

    with pytest.raises(ValueError, match="same base/head model path"):
        suite._validate_shared_model_compatibility()


class _CacheHitTrainer:
    def __init__(self):
        self.run_id = "cached-child"
        self.base_model_path = "bert-base"
        self._last_train_used_cache = False
        self.get_model_called = False

    def resolve_model_path(self, model):
        return "shared-head"

    def train(self, *args, **kwargs):
        self._last_train_used_cache = True
        return self

    def get_model(self):
        self.get_model_called = True
        return object()


def test_call_train_skips_release_and_model_load_for_cached_children(monkeypatch):
    suite = object.__new__(_ConcreteTrainerSuite)
    trainer = _CacheHitTrainer()
    suite.trainers = {"cached": trainer}
    suite._shared_base_model = None
    suite._shared_tokenizer = None
    suite._shared_model_key = "shared-head"
    suite.retain_models_in_memory = False
    suite._models = {}

    release_calls = []
    monkeypatch.setattr(
        suite,
        "_release_model_refs",
        lambda **kwargs: release_calls.append(kwargs),
    )

    result = suite.call("train")

    assert result["cached"] is trainer
    assert trainer.get_model_called is False
    assert release_calls == []


class _SimilarityTrainer:
    target_classes = ("a", "b")
    model_path = "saved-model"

    def __init__(self):
        self.get_model_called = False

    def get_model(self, **kwargs):
        self.get_model_called = True
        return object()


def test_get_models_gradiend_only_avoids_full_model_load(monkeypatch):
    suite = object.__new__(_ConcreteTrainerSuite)
    trainer = _SimilarityTrainer()
    suite.trainers = {"child": trainer}
    suite.label_mapping = {}
    suite._models = {}
    suite._shared_base_model = None
    suite._shared_tokenizer = None
    suite.retain_models_in_memory = True
    suite.model_device = "cpu"

    loaded = object()
    monkeypatch.setattr(
        "gradiend.trainer.suite.base.models_for_comparison",
        lambda trainer, **kwargs: (loaded, None, None),
    )

    models = suite.get_models(gradiend_only=True)

    assert models == {"child": loaded}
    assert trainer.get_model_called is False


def test_call_train_wraps_stability_trainers_for_analysis():
    temp_dir = _local_temp("suite_train_wrap")
    try:
        trainer = MockTrainerForTest(
            model=os.path.join(temp_dir, "model"),
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        trainer._last_train_used_cache = False

        suite = object.__new__(_ConcreteTrainerSuite)
        suite.trainers = {"child": trainer}
        suite._shared_base_model = None
        suite._shared_tokenizer = None
        suite._shared_model_key = "shared-head"
        suite.retain_models_in_memory = False
        suite._models = {}

        with patch.object(trainer, "get_model", return_value=SimpleNamespace(base_model=None)):
            with patch.object(trainer, "train", return_value=trainer):
                suite.call("train")

        assert is_multi_seed_view(suite.trainers["child"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_call_train_does_not_wrap_when_training_used_cache():
    temp_dir = _local_temp("suite_train_cache_no_wrap")
    try:
        trainer = MockTrainerForTest(
            model=os.path.join(temp_dir, "model"),
            args=TrainingArguments(experiment_dir=temp_dir, analyze_seed_stability=True),
        )
        trainer._last_train_used_cache = True

        suite = object.__new__(_ConcreteTrainerSuite)
        suite.trainers = {"child": trainer}
        suite._shared_base_model = None
        suite._shared_tokenizer = None
        suite._shared_model_key = "shared-head"
        suite.retain_models_in_memory = False
        suite._models = {}

        with patch.object(trainer, "train", return_value=trainer):
            suite.call("train")

        assert suite.trainers["child"] is trainer
        assert not is_multi_seed_view(suite.trainers["child"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
