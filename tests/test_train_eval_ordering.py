"""
Tests for unusual train/eval orderings (e.g. evaluate before train).
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.training import train
from gradiend.trainer.core.stats import load_training_stats
from tests.test_trainer_model import MockTrainerForTest
from tests.test_training_loop import MockModelWithGradiend


class TestEvalBeforeTrain:
    """Encoder/decoder evaluation and get_model before train() should behave predictably."""

    def test_get_model_before_train_creates_model_from_base(self, temp_dir):
        trainer = MockTrainerForTest(model="mock-base", args=TrainingArguments(experiment_dir=temp_dir))
        created = MagicMock()
        created.name_or_path = "mock-base"

        with patch(
            "gradiend.trainer.trainer.FeatureLearningDefinition.create_model_with_gradiend",
            return_value=created,
        ) as mock_create:
            got = trainer.get_model()

        assert got is created
        assert mock_create.called
        assert trainer._model_instance is created

    def test_evaluate_encoder_before_train_uses_base_model(self, temp_dir):
        trainer = MockTrainerForTest(model="mock-base", args=TrainingArguments(experiment_dir=temp_dir))
        untrained = MagicMock()
        untrained.name_or_path = "mock-base"

        with patch.object(trainer, "get_model", return_value=untrained) as mock_get_model:
            result = trainer.evaluate_encoder(split="test", use_cache=False)

        mock_get_model.assert_called()
        assert "correlation" in result

    def test_evaluate_decoder_before_train_uses_prepared_model(self, temp_dir):
        trainer = MockTrainerForTest(model="mock-base", args=TrainingArguments(experiment_dir=temp_dir))
        untrained = MagicMock()
        untrained.name_or_path = "mock-base"
        decoder_result = {"summary": {"x": {"feature_factor": 1.0, "learning_rate": 1e-4, "value": 0.5}}, "grid": {}}

        with patch.object(trainer, "_prepare_model_for_evaluation", return_value=untrained) as mock_prepare:
            with patch.object(trainer.evaluator, "evaluate_decoder", return_value=decoder_result) as mock_eval:
                result = trainer.evaluate_decoder(use_cache=False)

        mock_prepare.assert_called_once()
        mock_eval.assert_called_once()
        assert result is decoder_result

    def test_rewrite_base_model_before_train_requires_decoder_results(self, temp_dir):
        trainer = MockTrainerForTest(
            model="mock-base",
            args=TrainingArguments(experiment_dir=None),
        )
        trainer._model_instance = MagicMock()

        with pytest.raises(ValueError, match="decoder_results is required"):
            trainer.rewrite_base_model(target_class="x")

    def test_pre_prune_then_evaluate_encoder_before_train(self, temp_dir):
        trainer = MockTrainerForTest(
            model="mock-base",
            args=TrainingArguments(
                experiment_dir=temp_dir,
                pre_prune_config=__import__(
                    "gradiend.trainer.core.pruning", fromlist=["PrePruneConfig"]
                ).PrePruneConfig(n_samples=2, topk=0.5),
            ),
        )
        pruned = MagicMock()
        pruned.name_or_path = "mock-base"

        with patch.object(trainer, "pre_prune", return_value=pruned) as mock_pre_prune:
            trainer.pre_prune(inplace=True)
            with patch.object(trainer, "get_model", return_value=pruned):
                result = trainer.evaluate_encoder(split="test", use_cache=False)

        mock_pre_prune.assert_called_once()
        assert "correlation" in result

    def test_pre_prune_does_not_raise_missing_prepare_method(self, temp_dir):
        """Regression: pre_prune() must call _prepare_model_for_pre_prune_if_needed."""
        from gradiend.trainer.core.pruning import PrePruneConfig

        trainer = MockTrainerForTest(
            model="mock-base",
            args=TrainingArguments(
                experiment_dir=temp_dir,
                pre_prune_config=PrePruneConfig(n_samples=2, topk=0.5),
            ),
        )
        model = MagicMock()
        model.gradiend = MagicMock()
        model.gradiend._lazy_init = True

        with patch.object(trainer, "get_model", return_value=model):
            with patch("gradiend.trainer.trainer.pre_prune_with_cache", return_value=model):
                with patch.object(trainer, "create_training_data", return_value=MagicMock()):
                    trainer.pre_prune(inplace=True)

        assert trainer._model_instance is model

    def test_prepare_model_for_pre_prune_recreates_eager_cached_model(self, temp_dir):
        from gradiend.trainer.core.pruning import PrePruneConfig

        trainer = MockTrainerForTest(
            model="mock-base",
            args=TrainingArguments(
                experiment_dir=temp_dir,
                pre_prune_config=PrePruneConfig(n_samples=2, topk=0.5),
            ),
        )
        eager_model = MagicMock()
        eager_model.gradiend = MagicMock()
        eager_model.gradiend._lazy_init = False
        lazy_model = MagicMock()
        lazy_model.gradiend = MagicMock()
        lazy_model.gradiend._lazy_init = True

        with patch(
            "gradiend.trainer.trainer.create_model_with_gradiend",
            return_value=lazy_model,
        ) as mock_create:
            result = trainer._prepare_model_for_pre_prune_if_needed(
                eager_model,
                trainer._training_args,
            )

        mock_create.assert_called_once()
        assert result is lazy_model


class TestInitialEvalOnlyTraining:
    """Training loop edge cases around step-0 evaluation."""

    @staticmethod
    def _make_dataloader(num_samples: int = 4):
        from torch.utils.data import DataLoader
        from gradiend.trainer.core.dataset import GradientTrainingDataset
        from tests.test_training_loop import MockTrainingData
        import torch

        training_data = MockTrainingData(
            [
                {
                    "factual": torch.randn(10),
                    "alternative": torch.randn(10),
                    "label": 1.0,
                }
            ]
            * num_samples
        )

        def gradient_creator(inputs):
            return torch.randn(100)

        dataset = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff",
        )
        return DataLoader(dataset, batch_size=1)

    def test_initial_eval_runs_before_first_training_step(self, temp_dir, set_seed):
        set_seed(42)
        model = MockModelWithGradiend()
        eval_calls = []

        def evaluate_fn(config=None, training_stats=None, **kwargs):
            step = training_stats.get("global_step", -1)
            eval_calls.append(step)
            return {"correlation": 0.25}

        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=1,
            eval_steps=100,
            num_train_epochs=1,
            train_batch_size=1,
            evaluate_fn=evaluate_fn,
            do_eval=True,
            convergent_score_threshold=None,
        )

        train(model_with_gradiend=model, data=self._make_dataloader(), training_args=training_args)

        assert eval_calls[0] == 0
        stats = load_training_stats(temp_dir)
        scores = stats["training_stats"]["scores"]
        assert 0 in {int(k) for k in scores.keys()}


class TestEncoderNormsStepDict:
    def test_encoder_norms_saved_as_step_dict(self, temp_dir, set_seed):
        set_seed(42)
        model = MockModelWithGradiend()
        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=3,
            eval_steps=1000,
            num_train_epochs=1,
            train_batch_size=1,
            do_eval=False,
            convergent_score_threshold=None,
        )

        train(
            model_with_gradiend=model,
            data=TestInitialEvalOnlyTraining._make_dataloader(),
            training_args=training_args,
        )

        stats = load_training_stats(temp_dir)
        norms = stats["training_stats"]["encoder_norms"]
        assert isinstance(norms, dict)
        assert norms
        assert all(isinstance(k, str) for k in norms.keys())
        assert {int(k) for k in norms.keys()} == {1, 2, 3}

    def test_load_training_stats_converts_legacy_encoder_norms_list(self, temp_dir):
        import json

        path = os.path.join(temp_dir, "training.json")
        legacy = {
            "training_stats": {
                "encoder_norms": [0.5, 0.6, 0.7],
                "decoder_norms": [1.0, 1.1],
            },
            "best_score_checkpoint": {"global_step": 3},
            "training_args": {},
        }
        with open(path, "w") as f:
            json.dump(legacy, f)

        stats = load_training_stats(temp_dir)
        assert stats["training_stats"]["encoder_norms"] == {"1": 0.5, "2": 0.6, "3": 0.7}
        assert stats["training_stats"]["decoder_norms"] == {"1": 1.0, "2": 1.1}
