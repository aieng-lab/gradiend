"""Tests for training cache policy (use_cache='only_convergent')."""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.cache_policy import (
    USE_CACHE_ONLY_CONVERGENT,
    coerce_artifact_use_cache,
    is_unconditional_training_cache,
    should_reuse_seed_training_cache,
    should_reuse_training_cache,
)
from gradiend.trainer.core.feature_definition import FeatureLearningDefinition
from gradiend.util.paths import should_use_cached

_WORKSPACE_TMP = os.path.join(os.path.dirname(__file__), "..", ".pytest_tmp_use_cache_tests")


def _temp_dir(name: str) -> str:
    root = os.path.abspath(_WORKSPACE_TMP)
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"{name}_", dir=root)


def _write_min_model(model_dir: str, *, converged: bool, convergent_count: int = 1) -> None:
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"architecture": {"input_dim": 4}}, handle)
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as handle:
        handle.write(b"")
    with open(os.path.join(model_dir, "training.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "convergence_info": {
                    "converged": converged,
                    "convergent_count": convergent_count,
                    "min_convergent_seeds": 2,
                }
            },
            handle,
        )


def _artifact_cache_stub(use_cache):
    stub = SimpleNamespace(training_args=TrainingArguments(use_cache=use_cache))
    stub._get_training_arg = FeatureLearningDefinition._get_training_arg.__get__(stub)
    stub._default_from_training_args = FeatureLearningDefinition._default_from_training_args.__get__(stub)
    stub._resolve_artifact_use_cache = FeatureLearningDefinition._resolve_artifact_use_cache.__get__(stub)
    return stub


def test_only_convergent_string_is_truthy_in_python():
    assert bool(USE_CACHE_ONLY_CONVERGENT) is True


def test_is_unconditional_training_cache_rejects_only_convergent():
    assert is_unconditional_training_cache(True) is True
    assert is_unconditional_training_cache(False) is False
    assert is_unconditional_training_cache(USE_CACHE_ONLY_CONVERGENT) is False


def test_coerce_artifact_use_cache_only_convergent_is_false():
    assert coerce_artifact_use_cache(True) is True
    assert coerce_artifact_use_cache(False) is False
    assert coerce_artifact_use_cache(None) is False
    assert coerce_artifact_use_cache(USE_CACHE_ONLY_CONVERGENT) is False


def test_truthy_only_convergent_must_not_enable_artifact_cache():
    """Regression: `if use_cache:` would wrongly treat the string as enabled."""
    assert bool(USE_CACHE_ONLY_CONVERGENT) is True
    assert coerce_artifact_use_cache(USE_CACHE_ONLY_CONVERGENT) is False


def test_should_use_cached_does_not_treat_only_convergent_as_true():
    temp = _temp_dir("should_use_cached")
    cache_file = os.path.join(temp, "cache.csv")
    with open(cache_file, "w", encoding="utf-8") as handle:
        handle.write("x")
    assert should_use_cached(cache_file, USE_CACHE_ONLY_CONVERGENT) is False
    assert should_use_cached(cache_file, True) is True


def test_training_arguments_accepts_only_convergent_use_cache():
    args = TrainingArguments(use_cache="only_convergent")
    assert args.use_cache == "only_convergent"


def test_only_convergent_reuses_convergent_checkpoint():
    temp = _temp_dir("convergent_ckpt")
    model_dir = os.path.join(temp, "model")
    _write_min_model(model_dir, converged=True, convergent_count=2)
    assert should_reuse_training_cache("only_convergent", model_dir, min_convergent_seeds=2)


def test_only_convergent_rejects_non_convergent_checkpoint():
    temp = _temp_dir("nonconvergent_ckpt")
    model_dir = os.path.join(temp, "model")
    _write_min_model(model_dir, converged=False, convergent_count=0)
    assert not should_reuse_training_cache("only_convergent", model_dir, min_convergent_seeds=2)


def test_only_convergent_does_not_match_unconditional_bool_true_semantics():
    temp = _temp_dir("bool_vs_string")
    model_dir = os.path.join(temp, "model")
    _write_min_model(model_dir, converged=False, convergent_count=0)
    assert should_reuse_training_cache(True, model_dir, min_convergent_seeds=2)
    assert not should_reuse_training_cache(USE_CACHE_ONLY_CONVERGENT, model_dir, min_convergent_seeds=2)


def test_only_convergent_seed_cache():
    temp = _temp_dir("seed_cache")
    seed_dir = os.path.join(temp, "seed_0")
    _write_min_model(seed_dir, converged=True, convergent_count=1)
    assert should_reuse_seed_training_cache("only_convergent", seed_dir)


def test_bool_use_cache_still_works():
    temp = _temp_dir("bool_cache")
    model_dir = os.path.join(temp, "model")
    _write_min_model(model_dir, converged=False, convergent_count=0)
    assert should_reuse_training_cache(True, model_dir, min_convergent_seeds=2)
    assert not should_reuse_training_cache(False, model_dir, min_convergent_seeds=2)


def test_invalid_use_cache_raises():
    with pytest.raises(ValueError, match="use_cache"):
        TrainingArguments(use_cache="sometimes")


def test_resolve_artifact_use_cache_from_training_args():
    stub = _artifact_cache_stub("only_convergent")
    assert stub._resolve_artifact_use_cache() is False
    assert stub._resolve_artifact_use_cache(True) is True


def test_text_prediction_trainer_resolves_only_convergent_for_encoder_cache():
    stub = _artifact_cache_stub("only_convergent")
    assert stub._resolve_artifact_use_cache(None, fallback=False) is False


def test_annotate_data_does_not_use_cache_when_only_convergent():
    from gradiend.trainer.core.annotation import TrainerAnnotationMixin

    temp = _temp_dir("annotate")

    class _AnnotTrainer(TrainerAnnotationMixin):
        target_classes = ["a", "b"]
        experiment_dir = temp
        training_args = TrainingArguments(use_cache="only_convergent")

        def _get_training_arg(self, name: str):
            return getattr(self.training_args, name, None)

        def _default_from_training_args(self, value, name, fallback=None):
            return FeatureLearningDefinition._default_from_training_args(self, value, name, fallback)

        def _resolve_artifact_use_cache(self, value=None, *, fallback=False):
            return FeatureLearningDefinition._resolve_artifact_use_cache(self, value, fallback=fallback)

        def _load_base_annotation_model(self):
            raise AssertionError("cache hit should not load model")

    trainer = _AnnotTrainer()
    csv_path = os.path.join(temp, "annotated.csv")
    json_path = os.path.join(temp, "annotated.json")
    with open(csv_path, "w", encoding="utf-8") as handle:
        handle.write("x")
    with open(json_path, "w", encoding="utf-8") as handle:
        handle.write("{}")

    with patch(
        "gradiend.trainer.core.annotation.resolve_annotated_data_csv_path",
        return_value=csv_path,
    ), patch(
        "gradiend.trainer.core.annotation.resolve_annotated_data_json_path",
        return_value=json_path,
    ):
        with pytest.raises(AssertionError, match="cache hit"):
            trainer.annotate_data()
