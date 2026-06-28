"""Tests for TrainerCollection."""

import pytest

from gradiend.trainer.suite import TrainerCollection


def test_trainer_collection_keys_by_run_id():
    class DummyTrainer:
        def __init__(self, run_id: str):
            self.run_id = run_id

    a = DummyTrainer("run_a")
    b = DummyTrainer("run_b")
    collection = TrainerCollection(a, b)

    assert list(collection.keys()) == ["run_a", "run_b"]
    assert collection.get_trainer("run_a") is a


def test_trainer_collection_rejects_missing_run_id():
    class DummyTrainer:
        run_id = None

    with pytest.raises(ValueError, match="non-empty run_id"):
        TrainerCollection(DummyTrainer())


def test_trainer_collection_rejects_duplicate_run_id():
    class DummyTrainer:
        def __init__(self, run_id: str):
            self.run_id = run_id

    with pytest.raises(ValueError, match="Duplicate trainer id 'dup'"):
        TrainerCollection(DummyTrainer("dup"), DummyTrainer("dup"))


def test_trainer_collection_merge_trainers():
    class DummyTrainer:
        def __init__(self, run_id: str):
            self.run_id = run_id

    left = TrainerCollection(DummyTrainer("a"))
    right = TrainerCollection(DummyTrainer("b"))
    collection = TrainerCollection.merge(left, right)

    assert list(collection.keys()) == ["a", "b"]


def test_trainer_collection_unloads_after_uncached_train_when_not_retaining():
    class DummyTrainer:
        def __init__(self):
            self.run_id = "child"
            self.trained_with_use_cache = None
            self._last_train_used_cache = False
            self.unloaded = False

        def train(self, *, use_cache=True):
            self.trained_with_use_cache = use_cache

        def unload_model(self):
            self.unloaded = True

    trainer = DummyTrainer()
    collection = TrainerCollection(trainer, retain_models_in_memory=False)

    collection.train(use_cache=False)

    assert trainer.trained_with_use_cache is False
    assert trainer.unloaded is True
