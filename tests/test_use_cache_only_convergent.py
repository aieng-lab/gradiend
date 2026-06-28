"""Tests for training cache policy (use_cache='only_convergent')."""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.cache_policy import (
    USE_CACHE_ALWAYS,
    USE_CACHE_ONLY_CONVERGENT,
    STALE_PRUNED_INPUT_DIM_THRESHOLD,
    build_training_cache_fingerprint,
    checkpoint_matches_training_fingerprint,
    coerce_artifact_use_cache,
    is_stale_pruned_checkpoint,
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


def _write_min_model(
    model_dir: str,
    *,
    converged: bool,
    convergent_count: int = 1,
    include_mean_convergence: bool = True,
) -> None:
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
                    **(
                        {
                            "convergent_mean_by_class_threshold": 0.5,
                            "convergent_min_target_class_abs_mean": 0.7,
                        }
                        if include_mean_convergence
                        else {}
                    ),
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


def test_is_unconditional_training_cache():
    assert is_unconditional_training_cache(USE_CACHE_ALWAYS) is True
    assert is_unconditional_training_cache(True) is False
    assert is_unconditional_training_cache(False) is False
    assert is_unconditional_training_cache(USE_CACHE_ONLY_CONVERGENT) is False


def test_coerce_artifact_use_cache_only_convergent_reuses_related_artifacts():
    assert coerce_artifact_use_cache(True) is True
    assert coerce_artifact_use_cache(False) is False
    assert coerce_artifact_use_cache(None) is False
    assert coerce_artifact_use_cache(USE_CACHE_ALWAYS) is True
    assert coerce_artifact_use_cache(USE_CACHE_ONLY_CONVERGENT) is True


def test_only_convergent_enables_artifact_cache_via_explicit_policy():
    """Regression: only_convergent must reuse artifacts from accepted convergent runs."""
    assert bool(USE_CACHE_ONLY_CONVERGENT) is True
    assert coerce_artifact_use_cache(USE_CACHE_ONLY_CONVERGENT) is True


def test_should_use_cached_treats_only_convergent_as_artifact_cache_enabled():
    temp = _temp_dir("should_use_cached")
    cache_file = os.path.join(temp, "cache.csv")
    with open(cache_file, "w", encoding="utf-8") as handle:
        handle.write("x")
    assert should_use_cached(cache_file, USE_CACHE_ONLY_CONVERGENT) is True
    assert should_use_cached(cache_file, True) is True


def test_training_arguments_accepts_only_convergent_use_cache():
    args = TrainingArguments(use_cache="only_convergent")
    assert args.use_cache == "only_convergent"


def test_training_arguments_accepts_always_use_cache():
    args = TrainingArguments(use_cache="always")
    assert args.use_cache == "always"


def test_only_convergent_reuses_convergent_checkpoint():
    temp = _temp_dir("convergent_ckpt")
    model_dir = os.path.join(temp, "model")
    _write_min_model(model_dir, converged=True, convergent_count=2)
    assert should_reuse_training_cache("only_convergent", model_dir, min_convergent_seeds=2)


def test_only_convergent_rejects_legacy_correlation_only_checkpoint_with_mean_threshold():
    temp = _temp_dir("legacy_corr_only")
    model_dir = os.path.join(temp, "model")
    _write_min_model(model_dir, converged=True, convergent_count=2, include_mean_convergence=False)
    args = TrainingArguments(use_cache="only_convergent")

    assert not should_reuse_training_cache(
        "only_convergent",
        model_dir,
        min_convergent_seeds=2,
        training_args=args,
    )


def test_only_convergent_accepts_checkpoint_with_mean_by_class_training_json_fallback():
    temp = _temp_dir("mean_by_class_fallback")
    model_dir = os.path.join(temp, "model")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"architecture": {"input_dim": 4}}, handle)
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as handle:
        handle.write(b"")
    with open(os.path.join(model_dir, "training.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "training_stats": {
                    "mean_by_class": {
                        500: {
                            "1.0": 0.8,
                            "-1.0": -0.7,
                        }
                    }
                },
                "best_score_checkpoint": {"global_step": 500},
                "convergence_info": {
                    "converged": True,
                    "convergent_count": 1,
                    "min_convergent_seeds": 1,
                },
            },
            handle,
        )
    args = TrainingArguments(use_cache="only_convergent")

    assert should_reuse_training_cache(
        "only_convergent",
        model_dir,
        min_convergent_seeds=1,
        training_args=args,
    )


def test_only_convergent_rejects_high_pooled_abs_mean_with_weak_per_class_means():
    temp = _temp_dir("pooled_abs_only")
    model_dir = os.path.join(temp, "model")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"architecture": {"input_dim": 4}}, handle)
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as handle:
        handle.write(b"")
    with open(os.path.join(model_dir, "training.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "abs_mean_by_type": {"training": 0.7},
                "training_stats": {
                    "mean_by_class": {
                        500: {
                            "1.0": 1.0,
                            "-1.0": -0.3,
                        }
                    }
                },
                "best_score_checkpoint": {"global_step": 500},
                "convergence_info": {
                    "converged": True,
                    "convergent_count": 1,
                    "min_convergent_seeds": 1,
                    "convergent_mean_by_class_threshold": 0.5,
                    "convergent_min_target_class_abs_mean": 0.7,
                },
            },
            handle,
        )
    args = TrainingArguments(use_cache="only_convergent")

    assert not should_reuse_training_cache(
        "only_convergent",
        model_dir,
        min_convergent_seeds=1,
        training_args=args,
    )


def test_only_convergent_rejects_non_convergent_checkpoint():
    temp = _temp_dir("nonconvergent_ckpt")
    model_dir = os.path.join(temp, "model")
    _write_min_model(model_dir, converged=False, convergent_count=0)
    assert not should_reuse_training_cache("only_convergent", model_dir, min_convergent_seeds=2)


def test_only_convergent_does_not_match_unconditional_bool_true_semantics():
    temp = _temp_dir("bool_vs_string")
    model_dir = os.path.join(temp, "model")
    _write_min_model(model_dir, converged=False, convergent_count=0)
    assert should_reuse_training_cache(USE_CACHE_ALWAYS, model_dir, min_convergent_seeds=2)
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
    assert stub._resolve_artifact_use_cache() is True
    assert stub._resolve_artifact_use_cache(True) is True


def test_text_prediction_trainer_resolves_only_convergent_for_encoder_cache():
    stub = _artifact_cache_stub("only_convergent")
    assert stub._resolve_artifact_use_cache(None, fallback=False) is True


def test_annotate_data_uses_cache_when_only_convergent():
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
        result = trainer.annotate_data()

    assert result is not None


def test_stale_pruned_checkpoint_rejected_with_training_args():
    from gradiend.trainer import PrePruneConfig

    temp = _temp_dir("stale_pruned")
    model_dir = os.path.join(temp, "model")
    _write_min_model(model_dir, converged=True, convergent_count=2)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"architecture": {"input_dim": 108_490_752}}, handle)

    args = TrainingArguments(
        use_cache="only_convergent",
        pre_prune_config=PrePruneConfig(topk=0.1, n_samples=8, source="alternative"),
    )
    assert is_stale_pruned_checkpoint(model_dir, training_args=args)
    assert not should_reuse_training_cache(
        "only_convergent",
        model_dir,
        min_convergent_seeds=2,
        training_args=args,
    )


def test_pruned_checkpoint_with_matching_fingerprint_is_reused():
    from gradiend.trainer import PrePruneConfig

    temp = _temp_dir("matching_fp")
    model_dir = os.path.join(temp, "model")
    pre_cfg = PrePruneConfig(topk=0.1, n_samples=8, source="alternative")
    args = TrainingArguments(use_cache="only_convergent", pre_prune_config=pre_cfg)
    fingerprint = build_training_cache_fingerprint(args)
    fingerprint["gradiend_input_dim"] = 10_849_075

    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"architecture": {"input_dim": 10_849_075}}, handle)
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as handle:
        handle.write(b"")
    with open(os.path.join(model_dir, "training.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_fingerprint": fingerprint,
                "convergence_info": {
                    "converged": True,
                    "convergent_count": 2,
                    "min_convergent_seeds": 2,
                    "convergent_mean_by_class_threshold": 0.5,
                    "convergent_min_target_class_abs_mean": 0.7,
                },
            },
            handle,
        )

    assert checkpoint_matches_training_fingerprint(model_dir, args)
    assert should_reuse_training_cache(
        "only_convergent",
        model_dir,
        min_convergent_seeds=2,
        training_args=args,
    )


def test_seed_cache_rejects_stale_pruned_checkpoint():
    from gradiend.trainer import PrePruneConfig

    temp = _temp_dir("stale_seed")
    seed_dir = os.path.join(temp, "seed_0")
    _write_min_model(seed_dir, converged=True, convergent_count=1)
    with open(os.path.join(seed_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"architecture": {"input_dim": 108_490_752}}, handle)

    args = TrainingArguments(
        use_cache="only_convergent",
        pre_prune_config=PrePruneConfig(topk=0.1, n_samples=8, source="alternative"),
    )
    assert not should_reuse_seed_training_cache("only_convergent", seed_dir, training_args=args)


def test_always_reuses_despite_fingerprint_mismatch():
    from gradiend.trainer import PrePruneConfig

    temp = _temp_dir("always_fp_mismatch")
    model_dir = os.path.join(temp, "model")
    _write_min_model(model_dir, converged=False, convergent_count=0)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"architecture": {"input_dim": 108_490_752}}, handle)

    args = TrainingArguments(
        use_cache="always",
        pre_prune_config=PrePruneConfig(topk=0.1, n_samples=8, source="alternative"),
    )
    assert not should_reuse_training_cache(True, model_dir, training_args=args)
    assert should_reuse_training_cache(USE_CACHE_ALWAYS, model_dir, training_args=args)
    assert should_reuse_seed_training_cache(USE_CACHE_ALWAYS, model_dir, training_args=args)
