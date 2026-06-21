"""
Tests for training callbacks (modality-independent).

Tests EarlyStoppingCallback, EvaluationCallback, CheckpointCallback,
LoggingCallback, and NormalizationCallback.
"""

import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import pytest
import torch

from gradiend.trainer.core.callbacks import (
    TrainingCallback,
    EvaluationCallback,
    NormalizationCallback,
    CheckpointCallback,
    LoggingCallback,
)


class TestTrainingCallback:
    """Test base TrainingCallback class."""
    
    def test_callback_lifecycle(self):
        """Test that callback lifecycle methods can be called."""
        callback = TrainingCallback()
        
        # All methods should be callable without errors
        callback.on_train_begin({})
        callback.on_epoch_begin(0, {})
        callback.on_step_begin(0, {})
        callback.on_step_end(0, 0.5, None, {})
        callback.on_epoch_end(0, None, {})
        callback.on_train_end({})


class TestEvaluationCallback:
    """Test EvaluationCallback."""
    
    def test_evaluation_callback_creation(self):
        """Test EvaluationCallback can be created."""
        evaluate_fn = MagicMock(return_value={"correlation": 0.8})
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=100, do_eval=True)
        
        assert callback.evaluate_fn == evaluate_fn
        assert callback.n_evaluation == 100
        assert callback.do_eval is True
        assert callback.last_eval_step == -1
    
    def test_evaluation_callback_skips_when_disabled(self):
        """Test that evaluation is skipped when do_eval=False."""
        evaluate_fn = MagicMock()
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=100, do_eval=False)
        
        result = callback.on_step_end(step=100, loss=0.5, model=None, config={}, training_stats={})
        
        assert result is None
        evaluate_fn.assert_not_called()
    
    def test_evaluation_callback_evaluates_at_intervals(self):
        """Test that evaluation happens at specified intervals."""
        eval_result = {"correlation": 0.8, "mean_by_class": 0.7}
        evaluate_fn = MagicMock(return_value=eval_result)
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=50, do_eval=True)
        
        model = MagicMock()
        model.base_model.training = True
        
        # Should evaluate at step 0
        result = callback.on_step_end(step=0, loss=0.5, model=model, config={}, training_stats={})
        assert result == eval_result
        assert evaluate_fn.call_count == 1
        
        # Should not evaluate at step 25
        result = callback.on_step_end(step=25, loss=0.5, model=model, config={}, training_stats={})
        assert result is None
        assert evaluate_fn.call_count == 1
        
        # Should evaluate at step 50
        result = callback.on_step_end(step=50, loss=0.5, model=model, config={}, training_stats={})
        assert result == eval_result
        assert evaluate_fn.call_count == 2
    
    def test_evaluation_callback_evaluates_at_last_iteration(self):
        """Test that evaluation happens at last iteration."""
        eval_result = {"correlation": 0.8}
        evaluate_fn = MagicMock(return_value=eval_result)
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=100, do_eval=True)
        
        model = MagicMock()
        model.base_model.training = True
        
        # Should evaluate at last iteration even if not at interval
        result = callback.on_step_end(
            step=75, loss=0.5, model=model, config={}, training_stats={},
            last_iteration=True
        )
        assert result == eval_result
        evaluate_fn.assert_called_once()

    def test_evaluation_callback_does_not_mark_step_on_failed_eval(self):
        """Failed evaluations must not mark the step as done (allows retry)."""
        evaluate_fn = MagicMock(return_value=None)
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=100, do_eval=True)

        model = MagicMock()
        model.base_model.training = True

        result = callback.on_step_end(
            step=75, loss=0.5, model=model, config={}, training_stats={},
            last_iteration=True,
        )

        assert result is None
        assert callback.last_eval_step == -1
        evaluate_fn.assert_called_once()

    def test_evaluation_callback_evaluates_on_last_iteration_when_n_evaluation_zero(self):
        """last_iteration should still trigger eval when periodic interval is disabled."""
        eval_result = {"correlation": 0.8}
        evaluate_fn = MagicMock(return_value=eval_result)
        callback = EvaluationCallback(evaluate_fn=evaluate_fn, n_evaluation=0, do_eval=True)

        model = MagicMock()
        model.base_model.training = True

        result = callback.on_step_end(
            step=12, loss=0.5, model=model, config={}, training_stats={},
            last_iteration=True,
        )

        assert result == eval_result
        assert callback.last_eval_step == 12
        evaluate_fn.assert_called_once()


class TestNormalizationCallback:
    """Test NormalizationCallback."""
    
    def test_normalization_callback_creation(self):
        """Test NormalizationCallback can be created."""
        callback = NormalizationCallback()
        assert callback is not None
    
    def test_normalization_callback_inverts_on_negative_correlation(self):
        """Test that normalization inverts encoding when correlation < -0.6."""
        callback = NormalizationCallback()

        model = MagicMock()
        model.gradiend.latent_dim = 1
        model.invert_encoding = MagicMock()

        eval_result = {"correlation": -0.7, "mean_by_class": {1.0: 0.5, -1.0: -0.5}}
        training_stats = {"scores": {100: -0.7}, "mean_by_class": {100: eval_result["mean_by_class"]}}

        callback.on_step_end(
            step=100, loss=0.5, model=model, config={}, training_stats=training_stats,
            eval_result=eval_result,
        )

        model.invert_encoding.assert_called_once_with(update_direction=False)
        assert eval_result["correlation"] == pytest.approx(0.7)
        assert training_stats["correlation"] == pytest.approx(0.7)
        assert training_stats["scores"][100] == pytest.approx(0.7)
        assert eval_result["mean_by_class"][1.0] == pytest.approx(-0.5)

    def test_normalization_callback_no_inversion_on_positive_correlation(self):
        """Test that normalization doesn't invert when correlation >= -0.6."""
        callback = NormalizationCallback()

        model = MagicMock()
        model.gradiend.latent_dim = 1
        model.invert_encoding = MagicMock()

        eval_result = {"correlation": 0.5, "mean_by_class": {1.0: 0.5, -1.0: -0.5}}

        callback.on_step_end(
            step=100, loss=0.5, model=model, config={}, training_stats={},
            eval_result=eval_result,
        )

        model.invert_encoding.assert_not_called()
        assert eval_result["correlation"] == pytest.approx(0.5)

    def test_normalization_callback_no_inversion_when_correlation_mildly_negative(self):
        """Correlation below zero but above the inversion threshold must not flip."""
        callback = NormalizationCallback()

        model = MagicMock()
        model.gradiend.latent_dim = 1
        model.invert_encoding = MagicMock()

        eval_result = {"correlation": -0.55, "mean_by_class": {1.0: 0.5, -1.0: -0.5}}

        callback.on_step_end(
            step=100, loss=0.5, model=model, config={}, training_stats={},
            eval_result=eval_result,
        )

        model.invert_encoding.assert_not_called()
        assert eval_result["correlation"] == pytest.approx(-0.55)


class TestCheckpointCallback:
    """Test CheckpointCallback."""
    
    def test_checkpoint_callback_creation(self, temp_dir):
        """Test CheckpointCallback can be created."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=True,
            checkpoint_interval=100,
            use_loss_for_best=False
        )
        
        assert callback.output == temp_dir
        assert callback.keep_only_best is True
        assert callback.checkpoint_step == 0  # checkpoints=False means no periodic checkpoints
        assert callback.use_loss_for_best is False
    
    def test_checkpoint_callback_saves_best_model(self, temp_dir):
        """Test that checkpoint saves best model based on correlation."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=True,
            use_loss_for_best=False
        )
        
        model = MagicMock()
        model.save_pretrained = MagicMock()
        
        config = {}
        training_stats = {}
        
        # First step with correlation 0.5
        training_stats['correlation'] = 0.5
        callback.on_step_end(
            step=50, loss=0.5, model=model, config=config, training_stats=training_stats
        )
        
        # Should save as best (first one, but only if step > 1)
        # Step 50 > 1, so should save
        assert model.save_pretrained.call_count >= 1
        assert callback.best_score == 0.5
        
        # Second step with better correlation
        training_stats['correlation'] = 0.8
        callback.on_step_end(
            step=100, loss=0.5, model=model, config=config, training_stats=training_stats
        )
        
        # Should save again (better metric)
        assert model.save_pretrained.call_count >= 2
        assert callback.best_score == 0.8
        
        save_count_after_best = model.save_pretrained.call_count

        # Third step with worse correlation
        training_stats['correlation'] = 0.6
        callback.on_step_end(
            step=150, loss=0.5, model=model, config=config, training_stats=training_stats
        )

        assert callback.best_score == 0.8
        assert model.save_pretrained.call_count == save_count_after_best

    def test_checkpoint_callback_tracks_but_does_not_save_step_zero_best_model(self, temp_dir):
        """Initial step-0 best metrics must not become the selected checkpoint."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=True,
            use_loss_for_best=False
        )

        model = MagicMock()
        model.save_pretrained = MagicMock()

        training_stats = {"correlation": 0.9}

        callback.on_step_end(
            step=0, loss=0.0, model=model, config={}, training_stats=training_stats
        )

        model.save_pretrained.assert_not_called()
        assert callback.best_step == 0
        assert callback.best_score == 0.9
    
    def test_checkpoint_callback_saves_periodic_checkpoints(self, temp_dir):
        """Test that checkpoint saves periodic checkpoints when checkpoints=True."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=True,  # Enable periodic checkpoints
            checkpoint_interval=50,
            keep_only_best=False,
            use_loss_for_best=False
        )
        
        model = MagicMock()
        model.save_pretrained = MagicMock()
        
        config = {}
        training_stats = {'correlation': 0.5}
        
        # Should save at every checkpoint interval
        for step in [50, 100, 150]:
            callback.on_step_end(
                step=step, loss=0.5, model=model, config=config, training_stats=training_stats
            )
        
        # Should save periodic checkpoints (3 times) plus best model saves
        assert model.save_pretrained.call_count >= 3
    
    def test_checkpoint_callback_saves_at_epoch_end(self, temp_dir):
        """Test that checkpoint saves at epoch end."""
        callback = CheckpointCallback(
            output=temp_dir,
            checkpoints=False,
            keep_only_best=False,
            use_loss_for_best=False
        )
        
        model = MagicMock()
        model.save_pretrained = MagicMock()
        
        config = {}
        training_stats = {}
        losses = [0.5, 0.4, 0.3]
        
        # Should save at epoch end (callback calls model.save_pretrained, not model.gradiend)
        callback.on_epoch_end(epoch=0, model=model, config=config, training_stats=training_stats, losses=losses)
        
        assert model.save_pretrained.call_count == 1


class TestLoggingCallback:
    """Test LoggingCallback."""
    
    def test_logging_callback_creation(self):
        """Test LoggingCallback can be created."""
        callback = LoggingCallback(n_loss_report=100, loss_only=False)
        
        assert callback.n_loss_report == 100
        assert callback.loss_only is False
    
    def test_logging_callback_logs_loss(self):
        """Test that logging callback emits a step log at report intervals."""
        callback = LoggingCallback(n_loss_report=50, loss_only=True)
        training_stats = {}

        with patch("gradiend.trainer.core.callbacks.logger") as mock_logger:
            callback.on_step_end(
                step=50, loss=0.5, model=None, config={}, training_stats=training_stats,
                last_losses=[0.5, 0.4, 0.3],
            )

        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        assert message.startswith("Step 50,")
        assert "Correlation: N/A" in message

    def test_logging_callback_logs_metrics(self):
        """Test that logging callback records correlation and marks new best runs."""
        callback = LoggingCallback(n_loss_report=50, loss_only=False)
        training_stats = {"correlation": 0.8}

        with patch("gradiend.trainer.core.callbacks.logger") as mock_logger:
            callback.on_step_end(
                step=50, loss=0.5, model=None, config={}, training_stats=training_stats,
                eval_result={"correlation": 0.8},
                last_losses=[0.5, 0.4, 0.3],
            )

        mock_logger.info.assert_called_once()
        message = mock_logger.info.call_args[0][0]
        assert "Step 50," in message
        assert "Correlation: 0.8000" in message
        assert "(new best)" in message


# Note: EarlyStoppingCallback is not yet implemented in the codebase
# Tests will be added when it is implemented
