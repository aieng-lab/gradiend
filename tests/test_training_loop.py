"""
Tests for training loop (modality-independent).

Tests basic training flow, seed handling, and parameter passing/overwriting.
"""

import os
import tempfile
from contextlib import contextmanager
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import pytest
import torch
import torch.nn as nn

from gradiend.trainer.core.training import format_non_convergence_error, train
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.dataset import GradientTrainingDataset
from gradiend.trainer.core.stats import (
    _best_checkpoint_step_is_after_initial,
    _best_step_min_target_class_abs_mean,
    _best_step_target_class_mean_product,
    load_training_stats,
)
from gradiend.model import GradiendModel
from gradiend.trainer.trainer import Trainer
from tests.conftest import SimpleMockModel, MockTokenizer, set_seed


class MockModelWithGradiend:
    """Mock ModelWithGradiend for training tests."""
    
    def __init__(self):
        self.gradiend = GradiendModel(input_dim=100, latent_dim=1)
        self.base_model = SimpleMockModel()
        self.name_or_path = "mock-model"
    
    def __len__(self):
        return 100
    
    def parameters(self, recurse=True):
        return self.gradiend.parameters(recurse=recurse)
    
    def to(self, dtype=None):
        if dtype is not None:
            self.base_model.dtype = dtype
        return self
    
    def train(self):
        return self
    
    def eval(self):
        return self
    
    def save_pretrained(self, save_directory, **kwargs):
        """Create output dir so training loop and tests can assume path exists."""
        os.makedirs(save_directory, exist_ok=True)

    @contextmanager
    def exclusive_base_gradient_access(self):
        yield


class MockTrainingData:
    """Mock training dataset."""
    
    def __init__(self, items, batch_size=1):
        self.items = items
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


class TestNonConvergenceErrorMessage:
    def test_formatter_includes_gradiend_identity_and_seed_details(self):
        args = TrainingArguments(
            fail_on_non_convergence=True,
            min_convergent_seeds=2,
            convergent_metric="correlation",
            convergent_score_threshold=0.8,
            output_dir="runs/example/gradiend_a",
        )
        message = format_non_convergence_error(
            actual=1,
            min_required=2,
            training_args=args,
            run_id="sentiment_positive_negative",
            model="bert-base-cased",
            pair=("positive", "negative"),
            seed_report=[
                {
                    "seed": 42,
                    "converged": True,
                    "convergence_metric_value": 0.83,
                    "best_checkpoint_global_step": 100,
                    "output_dir": "runs/example/seeds/seed_42",
                },
                {
                    "seed": 43,
                    "converged": False,
                    "convergence_metric_value": 0.73,
                    "best_checkpoint_global_step": 100,
                    "output_dir": "runs/example/seeds/seed_43",
                },
            ],
        )

        assert "GRADIEND 'sentiment_positive_negative'" in message
        assert "model: bert-base-cased" in message
        assert "target pair: ('positive', 'negative')" in message
        assert "convergence metric: correlation (threshold=0.8)" in message
        assert "seed 43: converged=False, value=0.73" in message

    def test_trainer_non_convergence_failure_uses_informative_message(self):
        args = TrainingArguments(
            fail_on_non_convergence=True,
            min_convergent_seeds=2,
            convergent_metric="correlation",
            convergent_score_threshold=0.8,
        )
        with pytest.raises(RuntimeError) as excinfo:
            Trainer._maybe_fail_on_non_convergence(
                args,
                convergent_count=1,
                min_convergent=2,
                run_id="gender_de_der_die",
                model="google-bert/bert-base-multilingual-cased",
                pair=("der", "die"),
                output_dir="runs/gender_de_der_die",
                seed_report=[
                    {
                        "seed": 7,
                        "converged": False,
                        "convergence_metric_value": 0.73262,
                        "best_checkpoint_global_step": 150,
                    }
                ],
                convergence_metric="correlation",
                threshold=0.8,
            )

        message = str(excinfo.value)
        assert "GRADIEND 'gender_de_der_die'" in message
        assert "only 1 seed(s) converged, but 2 are required" in message
        assert "seed 7: converged=False, value=0.73262" in message

    def test_core_train_does_not_fail_inner_seed_when_multi_seed_not_exhausted(self, temp_dir, set_seed):
        set_seed(42)
        model = MockModelWithGradiend()
        dataloader = TestFinalStepEvaluation._make_dataloader(num_samples=4)
        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=1,
            eval_steps=1,
            num_train_epochs=1,
            train_batch_size=1,
            evaluate_fn=lambda **kwargs: {"correlation": 0.75},
            convergent_score_threshold=0.5,
            fail_on_non_convergence=True,
            max_seeds=10,
            min_convergent_seeds=2,
        )

        train(
            model_with_gradiend=model,
            data=dataloader,
            training_args=training_args,
        )

        stats = load_training_stats(temp_dir)
        assert stats["convergence_info"]["convergent_count"] == 0
        assert stats["convergence_info"]["min_convergent_seeds"] == 2


class TestTrainingLoop:
    """Test basic training loop functionality."""

    def test_gradient_dataset_separates_base_and_gradiend_batching(self):
        """Internal base batches should become stacked GRADIEND batches."""
        training_data = MockTrainingData(
            [
                {
                    "factual": {"input_ids": torch.arange(2 + (i % 2))},
                    "alternative": {"input_ids": torch.arange(3 + (i % 2))},
                    "label": float(i),
                }
                for i in range(8)
            ],
            batch_size=2,
        )

        seen_base_shapes = []

        def gradient_creator(inputs):
            seen_base_shapes.append(tuple(inputs["input_ids"].shape))
            return torch.ones(100)

        dataset = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff",
        )

        from torch.utils.data import DataLoader

        batch = next(iter(DataLoader(dataset, batch_size=2)))

        assert batch["source"].shape == (2, 100)
        assert batch["target"].shape == (2, 100)
        assert "factual" not in batch
        assert "alternative" not in batch
        assert len(seen_base_shapes) == 4  # factual+alternative for two GRADIEND rows
        assert all(shape[0] == 2 for shape in seen_base_shapes)

    def test_precomputed_training_dataset_matches_wrapped_dataset(self):
        """Precomputed wrapper should yield the same rows in the same order."""
        from gradiend.trainer.core.dataset import PreComputedTrainingDataset

        class NumberDataset:
            def __len__(self):
                return 5

            def __getitem__(self, index):
                return {
                    "source": torch.tensor([float(index)]),
                    "target": torch.tensor([float(-index)]),
                }

        wrapped = PreComputedTrainingDataset(NumberDataset(), buffer_size=2)
        rows = list(wrapped)

        assert len(wrapped) == 5
        assert [float(row["source"].item()) for row in rows] == [0, 1, 2, 3, 4]
        assert [float(row["target"].item()) for row in rows] == [0, -1, -2, -3, -4]

        from torch.utils.data import DataLoader

        batch = next(iter(DataLoader(wrapped, batch_size=2)))
        assert batch["source"].shape == (2, 1)
        assert batch["target"].shape == (2, 1)
    
    def test_train_basic_flow(self, temp_dir, set_seed):
        """Test that basic training flow works."""
        set_seed(42)
        
        model = MockModelWithGradiend()
        
        # Create simple training data
        training_data = MockTrainingData([
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": 1.0
            }
        ] * 10)
        
        def gradient_creator(inputs):
            # Handle both single inputs and batched inputs
            if isinstance(inputs, dict):
                # Batched input - return per-item gradients that will be stacked
                # For simplicity, return a single gradient per item
                return torch.randn(100)
            else:
                # Single input
                return torch.randn(100)
        
        dataset = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )
        
        from torch.utils.data import DataLoader
        # Use batch_size=1 to avoid batching complexity in tests
        dataloader = DataLoader(dataset, batch_size=1)
        
        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=5,  # Use max_steps instead of max_steps
            learning_rate=1e-3,
            train_batch_size=1  # Match DataLoader batch_size
        )
        
        # Train
        output_path = train(
            model_with_gradiend=model,
            data=dataloader,
            training_args=training_args
        )
        
        # Should return output path
        assert isinstance(output_path, str)
        assert os.path.exists(output_path)
    
    def test_train_parameter_overwriting(self, temp_dir, set_seed):
        """Test that parameters passed to train() override TrainingArguments."""
        set_seed(42)
        
        model = MockModelWithGradiend()
        
        training_data = MockTrainingData([
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": 1.0
            }
        ] * 10)
        
        def gradient_creator(inputs):
            return torch.randn(100)
        
        dataset = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=1)
        
        # Create TrainingArguments with default values
        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=1,
            learning_rate=1e-4,  # Default learning rate
            train_batch_size=1
        )
        
        # Override learning_rate via kwargs
        output_path = train(
            model_with_gradiend=model,
            data=dataloader,
            training_args=training_args,
            learning_rate=1e-3  # Override
        )

        stats = load_training_stats(temp_dir)
        assert stats["training_args"]["learning_rate"] == pytest.approx(1e-3)
        assert isinstance(output_path, str)
    
    def test_train_seed_handling(self, temp_dir, set_seed):
        """Same seed should yield identical per-step loss traces."""
        def _run_once(output_subdir: str):
            set_seed(42)
            model = MockModelWithGradiend()
            training_data = MockTrainingData([
                {
                    "factual": torch.randn(10),
                    "alternative": torch.randn(10),
                    "label": 1.0
                }
            ] * 5)

            def gradient_creator(inputs):
                return torch.randn(100)

            dataset = GradientTrainingDataset(
                training_data=training_data,
                gradient_creator=gradient_creator,
                source="factual",
                target="diff"
            )

            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=1)

            training_args = TrainingArguments(
                output_dir=os.path.join(temp_dir, output_subdir),
                max_steps=10,
                seed=42,
                train_batch_size=1
            )

            train(
                model_with_gradiend=model,
                data=dataloader,
                training_args=training_args
            )
            return load_training_stats(training_args.output_dir)["losses"]

        losses_a = _run_once("seed_a")
        losses_b = _run_once("seed_b")
        assert losses_a == losses_b
    
    def test_train_with_callbacks(self, temp_dir, set_seed):
        """Test that training works with custom callbacks."""
        set_seed(42)
        
        model = MockModelWithGradiend()
        
        training_data = MockTrainingData([
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": 1.0
            }
        ] * 10)
        
        def gradient_creator(inputs):
            return torch.randn(100)
        
        dataset = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=1)
        
        from gradiend.trainer.core.callbacks import LoggingCallback

        logged_steps = []

        class RecordingLoggingCallback(LoggingCallback):
            def on_step_end(self, *args, **kwargs):
                logged_steps.append(kwargs.get("step"))
                return super().on_step_end(*args, **kwargs)

        custom_callback = RecordingLoggingCallback(n_loss_report=10)
        
        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=1,
            train_batch_size=1
        )
        
        output_path = train(
            model_with_gradiend=model,
            data=dataloader,
            training_args=training_args,
            callbacks=[custom_callback]
        )

        assert isinstance(output_path, str)
        assert logged_steps, "Custom callback should be invoked during training"
    
    def test_train_empty_dataloader_raises_error(self, temp_dir):
        """Test that training raises error for empty dataloader."""
        model = MockModelWithGradiend()
        
        from torch.utils.data import DataLoader
        empty_dataloader = DataLoader([], batch_size=2)
        
        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=10,
            train_batch_size=1
        )
        
        with pytest.raises(ValueError, match="empty"):
            train(
                model_with_gradiend=model,
                data=empty_dataloader,
                training_args=training_args
            )
    
    def test_train_output_dir_creation(self, temp_dir, set_seed):
        """Test that output directory is created during training."""
        set_seed(42)
        
        model = MockModelWithGradiend()
        
        training_data = MockTrainingData([
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": 1.0
            }
        ] * 10)
        
        def gradient_creator(inputs):
            return torch.randn(100)
        
        dataset = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=1)
        
        output_subdir = os.path.join(temp_dir, "train_output")
        training_args = TrainingArguments(
            output_dir=output_subdir,
            max_steps=1,
            train_batch_size=1
        )
        
        output_path = train(
            model_with_gradiend=model,
            data=dataloader,
            training_args=training_args
        )
        
        # Output directory should exist
        assert os.path.exists(output_subdir)
        assert isinstance(output_path, str)
        assert os.path.exists(output_path)
    
    def test_train_multiple_parameter_overrides(self, temp_dir, set_seed):
        """Test that multiple parameters can be overridden."""
        set_seed(42)
        
        model = MockModelWithGradiend()
        
        training_data = MockTrainingData([
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": 1.0
            }
        ] * 10)
        
        def gradient_creator(inputs):
            return torch.randn(100)
        
        dataset = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=1)
        
        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=10,
            learning_rate=1e-4,
            train_batch_size=1
        )
        
        # Override multiple parameters
        output_path = train(
            model_with_gradiend=model,
            data=dataloader,
            training_args=training_args,
            learning_rate=1e-3,  # Override 1
            max_steps=20  # Override 2
        )

        stats = load_training_stats(temp_dir)
        assert stats["training_args"]["learning_rate"] == pytest.approx(1e-3)
        assert stats["training_args"]["max_steps"] == 20
        assert isinstance(output_path, str)
    
    def test_train_invalid_parameter_raises_error(self, temp_dir):
        """Test that invalid parameters raise ValueError."""
        model = MockModelWithGradiend()
        
        training_data = MockTrainingData([
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": 1.0
            }
        ] * 10)
        
        def gradient_creator(inputs):
            return torch.randn(100)
        
        dataset = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=1)
        
        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=10,
            train_batch_size=1
        )
        
        # Invalid parameter should raise ValueError
        with pytest.raises(ValueError, match="Invalid training argument"):
            train(
                model_with_gradiend=model,
                data=dataloader,
                training_args=training_args,
                invalid_param=123  # Invalid parameter
            )


class TestConvergenceCriteria:
    """Test helper logic used by convergence checks."""

    def test_best_checkpoint_step_must_be_after_initial(self):
        assert _best_checkpoint_step_is_after_initial({"global_step": 1}) is True
        assert _best_checkpoint_step_is_after_initial({"global_step": "2"}) is True
        assert _best_checkpoint_step_is_after_initial({"global_step": 0}) is False
        assert _best_checkpoint_step_is_after_initial({"global_step": "0"}) is False
        assert _best_checkpoint_step_is_after_initial({"global_step": None}) is False

    def test_target_class_mean_product_ignores_identity_class(self):
        training_stats = {
            "mean_by_class": {
                25: {
                    -1.0: -0.6,
                    0.0: 0.02,
                    1.0: 0.8,
                }
            }
        }
        best_score_checkpoint = {"global_step": 25}

        product = _best_step_target_class_mean_product(training_stats, best_score_checkpoint)

        assert product is not None
        assert product < 0

    def test_target_class_mean_product_is_positive_when_signs_match(self):
        training_stats = {
            "mean_by_class": {
                50: {
                    "-1.0": 0.4,
                    "1.0": 0.9,
                }
            }
        }
        best_score_checkpoint = {"global_step": 50}

        product = _best_step_target_class_mean_product(training_stats, best_score_checkpoint)

        assert product == pytest.approx(0.36)

    def test_target_class_mean_product_is_none_without_exactly_two_targets(self):
        training_stats = {
            "mean_by_class": {
                10: {
                    0.0: 0.01,
                    1.0: 0.7,
                }
            }
        }
        best_score_checkpoint = {"global_step": 10}

        product = _best_step_target_class_mean_product(training_stats, best_score_checkpoint)

        assert product is None

    def test_min_target_class_abs_mean_requires_each_class_above_threshold(self):
        training_stats = {
            "mean_by_class": {
                500: {
                    1.0: 1.0,
                    -1.0: -0.3,
                }
            }
        }
        best_score_checkpoint = {"global_step": 500}

        min_abs = _best_step_min_target_class_abs_mean(training_stats, best_score_checkpoint)

        assert min_abs == pytest.approx(0.3)

    def test_min_target_class_abs_mean_ignores_neutral_class(self):
        training_stats = {
            "mean_by_class": {
                25: {
                    -1.0: -0.6,
                    0.0: 0.02,
                    1.0: 0.8,
                }
            }
        }
        best_score_checkpoint = {"global_step": 25}

        min_abs = _best_step_min_target_class_abs_mean(training_stats, best_score_checkpoint)

        assert min_abs == pytest.approx(0.6)


class TestFinalStepEvaluation:
    """Ensure the last training step is always evaluated."""

    @staticmethod
    def _score_steps(scores):
        return {int(k) for k in scores.keys()}

    @staticmethod
    def _make_dataloader(num_samples: int = 200):
        from torch.utils.data import DataLoader
        from gradiend.trainer.core.dataset import GradientTrainingDataset

        training_data = MockTrainingData([
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": 1.0,
            }
        ] * num_samples)

        def gradient_creator(inputs):
            return torch.randn(100)

        dataset = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff",
        )
        return DataLoader(dataset, batch_size=1)

    @pytest.mark.parametrize(
        "max_steps,expected_final_step",
        [
            (500, 500),  # last step aligns with eval_steps=100 (user scenario)
            (503, 503),  # last step does not align with eval interval
            (497, 497),
        ],
    )
    def test_train_evaluates_at_final_step(self, temp_dir, set_seed, max_steps, expected_final_step):
        """Final global_step must appear in training scores after train() completes."""
        set_seed(42)
        model = MockModelWithGradiend()
        dataloader = self._make_dataloader()

        eval_calls = []

        def evaluate_fn(config=None, training_stats=None, **kwargs):
            step = training_stats.get("global_step", -1)
            eval_calls.append(step)
            return {"correlation": 0.5 + step * 0.001}

        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=max_steps,
            eval_steps=100,
            num_train_epochs=3,
            train_batch_size=1,
            evaluate_fn=evaluate_fn,
            do_eval=True,
            convergent_score_threshold=None,
        )

        train(
            model_with_gradiend=model,
            data=dataloader,
            training_args=training_args,
        )

        stats = load_training_stats(temp_dir)
        scores = stats["training_stats"]["scores"]
        score_steps = self._score_steps(scores)
        assert expected_final_step in score_steps, (
            f"Expected evaluation at final step {expected_final_step}, "
            f"got eval calls at {eval_calls}, scores keys {list(scores.keys())}"
        )
        assert eval_calls[-1] == expected_final_step

    def test_train_retries_final_eval_when_last_step_eval_fails(self, temp_dir, set_seed):
        """If the last-step eval fails during training, run a final eval after the loop."""
        set_seed(42)
        model = MockModelWithGradiend()
        dataloader = self._make_dataloader(num_samples=20)
        attempts_at_seven = 0

        def evaluate_fn(config=None, training_stats=None, **kwargs):
            step = training_stats.get("global_step", -1)
            nonlocal attempts_at_seven
            if step == 7:
                attempts_at_seven += 1
                if attempts_at_seven == 1:
                    return None
            return {"correlation": 0.75}

        training_args = TrainingArguments(
            output_dir=temp_dir,
            max_steps=7,
            eval_steps=100,
            num_train_epochs=1,
            train_batch_size=1,
            evaluate_fn=evaluate_fn,
            do_eval=True,
            convergent_score_threshold=None,
        )

        train(
            model_with_gradiend=model,
            data=dataloader,
            training_args=training_args,
        )

        stats = load_training_stats(temp_dir)
        scores = stats["training_stats"]["scores"]
        assert 7 in self._score_steps(scores)
        assert scores.get(7, scores.get("7")) == pytest.approx(0.75)
        assert attempts_at_seven == 2
