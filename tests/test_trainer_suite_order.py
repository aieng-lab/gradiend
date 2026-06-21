import pytest

from gradiend.trainer.suite import PositiveTrainerSuite, SuitePairDefinition, TrainerSuite


class _ConcreteTrainerSuite(TrainerSuite):
    def _resolve_pair_definitions(self):
        return []


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


def test_plot_similarity_heatmap_respects_explicit_order(monkeypatch):
    suite = _make_suite()

    monkeypatch.setattr(
        suite,
        "get_models",
        lambda **kwargs: {suite.label_mapping[k]: object() for k in suite.pair_by_id},
    )

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
