"""Comprehensive tests for TrainingArguments.use_cache policy modes."""

from __future__ import annotations

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
    build_training_cache_fingerprint,
    coerce_artifact_use_cache,
    normalize_use_cache,
    should_reuse_seed_training_cache,
    should_reuse_training_cache,
)
from gradiend.trainer.core.feature_definition import FeatureLearningDefinition
from gradiend.util.paths import has_saved_model, should_use_cached

_WORKSPACE_TMP = os.path.join(os.path.dirname(__file__), "..", ".pytest_tmp_use_cache_policy")


def _temp_dir(name: str) -> str:
    root = os.path.abspath(_WORKSPACE_TMP)
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"{name}_", dir=root)


def _write_saved_checkpoint(
    model_dir: str,
    *,
    input_dim: int = 64,
    converged: bool = False,
    convergent_count: int = 0,
    cache_fingerprint: dict | None = None,
) -> None:
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"architecture": {"input_dim": input_dim}}, handle)
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as handle:
        handle.write(b"")
    training: dict = {
        "convergence_info": {
            "converged": converged,
            "convergent_count": convergent_count,
            "min_convergent_seeds": 2,
            "convergent_mean_by_class_threshold": 0.5,
            "convergent_min_target_class_abs_mean": 0.7,
        },
    }
    if cache_fingerprint is not None:
        training["cache_fingerprint"] = cache_fingerprint
    with open(os.path.join(model_dir, "training.json"), "w", encoding="utf-8") as handle:
        json.dump(training, handle)


def _mismatched_source_args(*, use_cache) -> TrainingArguments:
    return TrainingArguments(use_cache=use_cache, source="alternative", target="diff")


def _matching_source_args(*, use_cache) -> tuple[TrainingArguments, dict]:
    args = TrainingArguments(use_cache=use_cache, source="alternative", target="diff")
    fingerprint = build_training_cache_fingerprint(args)
    fingerprint["gradiend_input_dim"] = 64
    return args, fingerprint


@pytest.mark.parametrize(
    ("use_cache", "fp_matches", "converged", "expected"),
    [
        (False, True, True, False),
        (False, False, True, False),
        (USE_CACHE_ALWAYS, True, True, True),
        (USE_CACHE_ALWAYS, True, False, True),
        (USE_CACHE_ALWAYS, False, True, True),
        (USE_CACHE_ALWAYS, False, False, True),
        (True, True, True, True),
        (True, True, False, True),
        (True, False, True, False),
        (True, False, False, False),
        (USE_CACHE_ONLY_CONVERGENT, True, True, True),
        (USE_CACHE_ONLY_CONVERGENT, True, False, False),
        (USE_CACHE_ONLY_CONVERGENT, False, True, False),
        (USE_CACHE_ONLY_CONVERGENT, False, False, False),
    ],
)
def test_should_reuse_training_cache_policy_matrix(use_cache, fp_matches, converged, expected):
    temp = _temp_dir("matrix")
    model_dir = os.path.join(temp, "model")

    if fp_matches:
        args, fingerprint = _matching_source_args(use_cache=use_cache)
        _write_saved_checkpoint(
            model_dir,
            input_dim=64,
            converged=converged,
            convergent_count=2 if converged else 0,
            cache_fingerprint=fingerprint,
        )
    else:
        args = _mismatched_source_args(use_cache=use_cache)
        _write_saved_checkpoint(
            model_dir,
            input_dim=64,
            converged=converged,
            convergent_count=2 if converged else 0,
            cache_fingerprint={"source": "factual", "target": "diff"},
        )

    assert has_saved_model(model_dir)
    assert (
        should_reuse_training_cache(
            use_cache,
            model_dir,
            min_convergent_seeds=2,
            training_args=args,
        )
        is expected
    )


@pytest.mark.parametrize(
    ("use_cache", "fp_matches", "converged", "expected"),
    [
        (USE_CACHE_ALWAYS, False, False, True),
        (True, False, False, False),
        (True, True, False, True),
        (USE_CACHE_ONLY_CONVERGENT, True, False, False),
        (USE_CACHE_ONLY_CONVERGENT, True, True, True),
    ],
)
def test_should_reuse_seed_training_cache_policy_matrix(use_cache, fp_matches, converged, expected):
    temp = _temp_dir("seed_matrix")
    seed_dir = os.path.join(temp, "seed_0")

    if fp_matches:
        args, fingerprint = _matching_source_args(use_cache=use_cache)
        _write_saved_checkpoint(
            seed_dir,
            input_dim=64,
            converged=converged,
            convergent_count=1 if converged else 0,
            cache_fingerprint=fingerprint,
        )
    else:
        args = _mismatched_source_args(use_cache=use_cache)
        _write_saved_checkpoint(
            seed_dir,
            input_dim=64,
            converged=converged,
            convergent_count=1 if converged else 0,
            cache_fingerprint={"source": "factual", "target": "diff"},
        )

    assert (
        should_reuse_seed_training_cache(use_cache, seed_dir, training_args=args)
        is expected
    )


@pytest.mark.parametrize("use_cache", [True, USE_CACHE_ALWAYS, USE_CACHE_ONLY_CONVERGENT])
def test_all_cache_modes_reject_missing_checkpoint(use_cache):
    temp = _temp_dir("missing")
    model_dir = os.path.join(temp, "model")
    args = _mismatched_source_args(use_cache=use_cache)
    assert not should_reuse_training_cache(use_cache, model_dir, training_args=args)
    assert not should_reuse_seed_training_cache(use_cache, model_dir, training_args=args)


def test_true_without_training_args_skips_fingerprint_check():
    temp = _temp_dir("no_args")
    model_dir = os.path.join(temp, "model")
    _write_saved_checkpoint(
        model_dir,
        input_dim=108_490_752,
        converged=False,
        cache_fingerprint={"source": "factual"},
    )
    assert should_reuse_training_cache(True, model_dir, training_args=None)
    assert should_reuse_training_cache(USE_CACHE_ALWAYS, model_dir, training_args=None)


def test_normalize_use_cache_accepts_all_modes():
    assert normalize_use_cache(False) is False
    assert normalize_use_cache(True) is True
    assert normalize_use_cache(USE_CACHE_ALWAYS) == USE_CACHE_ALWAYS
    assert normalize_use_cache(USE_CACHE_ONLY_CONVERGENT) == USE_CACHE_ONLY_CONVERGENT


def test_normalize_use_cache_rejects_unknown_values():
    with pytest.raises(ValueError, match="use_cache"):
        normalize_use_cache("sometimes")


def test_should_use_cached_enables_disk_cache_for_always():
    temp = _temp_dir("disk_cache")
    cache_file = os.path.join(temp, "artifact.csv")
    with open(cache_file, "w", encoding="utf-8") as handle:
        handle.write("x")
    assert should_use_cached(cache_file, USE_CACHE_ALWAYS) is True
    assert should_use_cached(cache_file, False) is False


def test_coerce_artifact_use_cache_maps_string_modes_to_true():
    assert coerce_artifact_use_cache(USE_CACHE_ALWAYS) is True
    assert coerce_artifact_use_cache(USE_CACHE_ONLY_CONVERGENT) is True
    assert coerce_artifact_use_cache(True) is True
    assert coerce_artifact_use_cache(False) is False


def test_resolve_artifact_use_cache_from_training_args_always():
    stub = SimpleNamespace(training_args=TrainingArguments(use_cache="always"))
    stub._get_training_arg = FeatureLearningDefinition._get_training_arg.__get__(stub)
    stub._default_from_training_args = FeatureLearningDefinition._default_from_training_args.__get__(stub)
    stub._resolve_artifact_use_cache = FeatureLearningDefinition._resolve_artifact_use_cache.__get__(stub)
    assert stub._resolve_artifact_use_cache() is True
    assert stub._resolve_artifact_use_cache(False) is False


def test_annotate_data_uses_cache_when_always():
    from gradiend.trainer.core.annotation import TrainerAnnotationMixin

    temp = _temp_dir("annotate_always")

    class _AnnotTrainer(TrainerAnnotationMixin):
        target_classes = ["a", "b"]
        experiment_dir = temp
        training_args = TrainingArguments(use_cache="always")

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
        assert trainer.annotate_data() is not None


class TestTrainerUseCacheIntegration:
    """End-to-end train() behavior for use_cache policy modes."""

    @staticmethod
    def _trainer_with_mismatched_checkpoint(temp_dir: str, *, use_cache):
        from tests.test_trainer_model import MockTrainerForTest

        output_dir = os.path.join(temp_dir, "model")
        _write_saved_checkpoint(
            output_dir,
            input_dim=64,
            converged=False,
            cache_fingerprint={"source": "factual", "target": "diff"},
        )
        args = TrainingArguments(
            experiment_dir=temp_dir,
            output_dir=output_dir,
            use_cache=use_cache,
            source="alternative",
            target="diff",
            max_seeds=1,
            max_steps=2,
        )
        return MockTrainerForTest(model="mock-base", args=args), output_dir

    def test_train_always_skips_despite_fingerprint_mismatch(self, temp_dir):
        trainer, output_dir = self._trainer_with_mismatched_checkpoint(temp_dir, use_cache="always")

        from tests.test_trainer_model import MockTrainerForTest

        with patch.object(MockTrainerForTest, "_train") as mock_train:
            result = trainer.train()

        assert result is trainer
        assert trainer._last_train_used_cache is True
        assert trainer._model_arg == output_dir
        mock_train.assert_not_called()

    def test_train_true_retrains_on_fingerprint_mismatch(self, temp_dir):
        trainer, output_dir = self._trainer_with_mismatched_checkpoint(temp_dir, use_cache=True)

        from tests.test_trainer_model import MockTrainerForTest

        with patch.object(MockTrainerForTest, "_train", return_value=output_dir) as mock_train:
            result = trainer.train()

        assert result is trainer
        assert trainer._last_train_used_cache is False
        mock_train.assert_called_once()

    def test_train_true_skips_when_fingerprint_matches(self, temp_dir):
        from tests.test_trainer_model import MockTrainerForTest

        output_dir = os.path.join(temp_dir, "model")
        args, fingerprint = _matching_source_args(use_cache=True)
        _write_saved_checkpoint(
            output_dir,
            input_dim=64,
            converged=False,
            cache_fingerprint=fingerprint,
        )
        args = TrainingArguments(
            experiment_dir=temp_dir,
            output_dir=output_dir,
            use_cache=True,
            source="alternative",
            target="diff",
            max_seeds=1,
            max_steps=2,
        )
        trainer = MockTrainerForTest(model="mock-base", args=args)

        with patch.object(MockTrainerForTest, "_train") as mock_train:
            result = trainer.train()

        assert result is trainer
        assert trainer._last_train_used_cache is True
        mock_train.assert_not_called()

    def test_train_only_convergent_retrains_non_convergent_even_with_matching_fingerprint(self, temp_dir):
        from tests.test_trainer_model import MockTrainerForTest

        output_dir = os.path.join(temp_dir, "model")
        _, fingerprint = _matching_source_args(use_cache="only_convergent")
        _write_saved_checkpoint(
            output_dir,
            input_dim=64,
            converged=False,
            convergent_count=0,
            cache_fingerprint=fingerprint,
        )
        args = TrainingArguments(
            experiment_dir=temp_dir,
            output_dir=output_dir,
            use_cache="only_convergent",
            source="alternative",
            target="diff",
            min_convergent_seeds=1,
            max_seeds=1,
            max_steps=2,
        )
        trainer = MockTrainerForTest(model="mock-base", args=args)

        with patch.object(MockTrainerForTest, "_train", return_value=output_dir) as mock_train:
            result = trainer.train()

        assert result is trainer
        assert trainer._last_train_used_cache is False
        mock_train.assert_called_once()

    def test_multi_seed_always_reuses_cached_seed_only(self, temp_dir):
        from tests.test_trainer_model import MockTrainerForTest

        exp_dir = temp_dir
        seed_0_dir = os.path.join(exp_dir, "seeds", "seed_0")
        _write_saved_checkpoint(
            seed_0_dir,
            input_dim=64,
            converged=False,
            cache_fingerprint={"source": "factual", "target": "diff"},
        )
        args = TrainingArguments(
            experiment_dir=exp_dir,
            use_cache="always",
            source="alternative",
            target="diff",
            max_seeds=2,
            min_convergent_seeds=1,
            seed=0,
            max_steps=2,
        )
        trainer = MockTrainerForTest(model="mock-base", args=args)

        def _fake_train(output_dir, **kwargs):
            return output_dir

        with patch.object(MockTrainerForTest, "_train", side_effect=_fake_train) as mock_train:
            trainer.train()

        assert mock_train.call_count == 1
        trained_seed_dir = mock_train.call_args.kwargs.get("output_dir") or mock_train.call_args[0][0]
        assert os.path.basename(trained_seed_dir) == "seed_1"

    def test_multi_seed_true_retrains_all_seeds_on_fingerprint_mismatch(self, temp_dir):
        from tests.test_trainer_model import MockTrainerForTest

        exp_dir = temp_dir
        seed_0_dir = os.path.join(exp_dir, "seeds", "seed_0")
        _write_saved_checkpoint(
            seed_0_dir,
            input_dim=64,
            converged=True,
            convergent_count=1,
            cache_fingerprint={"source": "factual", "target": "diff"},
        )
        args = TrainingArguments(
            experiment_dir=exp_dir,
            use_cache=True,
            source="alternative",
            target="diff",
            max_seeds=2,
            min_convergent_seeds=1,
            seed=0,
            max_steps=2,
        )
        trainer = MockTrainerForTest(model="mock-base", args=args)

        with patch.object(MockTrainerForTest, "_train", side_effect=lambda output_dir, **kwargs: output_dir) as mock_train:
            trainer.train()

        assert mock_train.call_count == 2
