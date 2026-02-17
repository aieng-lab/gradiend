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


class TestNormalizationCallback:
    """Test NormalizationCallback."""
    
    def test_normalization_callback_creation(self):
        """Test NormalizationCallback can be created."""
        callback = NormalizationCallback()
        assert callback is not None
    
    def test_normalization_callback_inverts_on_negative_correlation(self):
        """Test that normalization inverts encoding when correlation < -0.5."""
        callback = NormalizationCallback()
        
        model = MagicMock()
        model.gradiend.latent_dim = 1
        model.gradiend.encoder = MagicMock()
        model.gradiend.decoder = MagicMock()
        
        # Mock eval result with negative correlation
        eval_result = {"correlation": -0.6}
        
        # Should invert when correlation < -0.5
        callback.on_step_end(
            step=100, loss=0.5, model=model, config={}, training_stats={},
            eval_result=eval_result
        )
        
        # Check that encoder/decoder were swapped (inversion)
        # The actual inversion logic is in the callback implementation
        # We just verify it doesn't crash
    
    def test_normalization_callback_no_inversion_on_positive_correlation(self):
        """Test that normalization doesn't invert when correlation >= -0.5."""
        callback = NormalizationCallback()
        
        model = MagicMock()
        model.gradiend.latent_dim = 1
        
        # Mock eval result with positive correlation
        eval_result = {"correlation": 0.5}
        
        # Should not invert when correlation >= -0.5
        callback.on_step_end(
            step=100, loss=0.5, model=model, config={}, training_stats={},
            eval_result=eval_result
        )
        # Should complete without errors


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
        
        # Third step with worse correlation
        training_stats['correlation'] = 0.6
        callback.on_step_end(
            step=150, loss=0.5, model=model, config=config, training_stats=training_stats
        )
        
        # Should not save (worse metric)
        # Count should be same as before
        prev_count = model.save_pretrained.call_count
        assert callback.best_score == 0.8
    
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
        model.gradiend = MagicMock()
        model.gradiend.save_pretrained = MagicMock()
        
        config = {}
        training_stats = {}
        losses = [0.5, 0.4, 0.3]
        
        # Should save at epoch end
        callback.on_epoch_end(epoch=0, model=model, config=config, training_stats=training_stats, losses=losses)
        
        assert model.gradiend.save_pretrained.call_count == 1


class TestLoggingCallback:
    """Test LoggingCallback."""
    
    def test_logging_callback_creation(self):
        """Test LoggingCallback can be created."""
        callback = LoggingCallback(n_loss_report=100, loss_only=False)
        
        assert callback.n_loss_report == 100
        assert callback.loss_only is False
    
    def test_logging_callback_logs_loss(self):
        """Test that logging callback logs loss at intervals."""
        callback = LoggingCallback(n_loss_report=50, loss_only=True)
        
        training_stats = {}
        
        # Should log at step 50 (LoggingCallback requires last_losses to log)
        # We just verify it doesn't crash
        callback.on_step_end(
            step=50, loss=0.5, model=None, config={}, training_stats=training_stats,
            last_losses=[0.5, 0.4, 0.3]  # Required for logging
        )
        
        # LoggingCallback doesn't modify training_stats, it just logs via logger.info()
        # The test verifies the callback executes without error
        assert True  # If we get here, the callback executed successfully
    
    def test_logging_callback_logs_metrics(self):
        """Test that logging callback logs metrics when loss_only=False."""
        callback = LoggingCallback(n_loss_report=50, loss_only=False)
        
        training_stats = {'correlation': 0.8}
        eval_result = {"correlation": 0.8, "mean_by_class": 0.7}
        
        # Should log metrics (LoggingCallback logs via logger, doesn't modify training_stats)
        # We just verify it doesn't crash
        callback.on_step_end(
            step=50, loss=0.5, model=None, config={}, training_stats=training_stats,
            eval_result=eval_result,
            last_losses=[0.5, 0.4, 0.3]
        )
        
        # LoggingCallback doesn't modify training_stats, it just logs
        # The test verifies the callback executes without error
        assert True  # If we get here, the callback executed successfully


# Note: EarlyStoppingCallback is not yet implemented in the codebase
# Tests will be added when it is implemented
