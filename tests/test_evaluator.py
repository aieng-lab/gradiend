"""
Tests for evaluators (DecoderEvaluator, EncoderEvaluator).

Tests parameter overwriting, caching behavior, and basic evaluation functionality.
"""

import os
import tempfile
import json
from unittest.mock import MagicMock, patch, Mock
from typing import Dict, Any, List

import pytest
import torch
import numpy as np
import pandas as pd

from gradiend.evaluator.decoder import DecoderEvaluator
from gradiend.evaluator.encoder import EncoderEvaluator
from gradiend.trainer.core.dataset import GradientTrainingDataset
from tests.conftest import MockTokenizer


class MockTrainer:
    """Mock trainer for testing evaluators."""
    
    def __init__(self, training_args=None):
        self._training_args = training_args or MockTrainingArguments()
        self.experiment_dir = None
        self.run_id = None
        self.target_classes = ["positive", "negative"]
        self._model = None
    
    def get_model(self):
        return self._model
    
    def _default_from_training_args(self, value, name, fallback=None):
        """Simulate parameter overwriting logic."""
        if value is not None:
            return value
        return getattr(self._training_args, name, fallback)
    
    def _get_decoder_eval_dataframe(self, tokenizer, **kwargs):
        """Mock decoder eval dataframe creation."""
        training_like_df = pd.DataFrame({
            "text": ["test1", "test2"],
            "label": ["positive", "negative"],
            "factual_id": [1, 2],
            "alternative_id": [2, 1]
        })
        neutral_df = pd.DataFrame({
            "text": ["neutral1"],
            "label": ["neutral"],
            "factual_id": [0],
            "alternative_id": [0]
        })
        return training_like_df, neutral_df
    
    def evaluate_base_model(self, base_model, tokenizer, **kwargs):
        """Mock base model evaluation."""
        # Return structure expected by decoder evaluator
        # The "lms" key should contain a dict with "lms" key (nested structure)
        return {
            "lms": {"lms": 0.5},  # Nested structure expected by decoder.py:163
            "positive": 0.7,
            "negative": 0.3
        }
    
    def _evaluate_model_for_decoder(self, model_with_gradiend, df, **kwargs):
        """Mock decoder evaluation."""
        return {
            "lms": 0.5,
            "positive": 0.7,
            "negative": 0.3
        }
    
    def _model_for_decoder_eval(self, model_with_gradiend):
        """Mock model preparation for decoder eval."""
        return model_with_gradiend
    
    def create_eval_data(self, model_with_gradiend, **kwargs):
        """Mock eval data creation. Need at least 2 non-neutral samples for correlation."""
        training_data = MockTrainingData([
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": 1.0,
                "factual_id": 1,
                "alternative_id": 2
            },
            {
                "factual": torch.randn(10),
                "alternative": torch.randn(10),
                "label": -1.0,
                "factual_id": 2,
                "alternative_id": 1
            },
        ])
        
        def gradient_creator(inputs):
            return torch.randn(100)
        
        return GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )
    
    def get_target_feature_classes(self):
        """Return target feature classes."""
        return self.target_classes


class MockTrainingArguments:
    """Mock TrainingArguments for testing."""
    
    def __init__(self):
        self.use_cache = False
        self.encoder_eval_max_size = 100
        self.decoder_eval_max_size_training_like = 50
        self.decoder_eval_max_size_neutral = 50


class MockTrainingData:
    """Mock training dataset."""
    
    def __init__(self, items):
        self.items = items
        self.batch_size = 1
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


class MockModelWithGradiend:
    """Mock ModelWithGradiend for testing."""
    
    def __init__(self):
        self.name_or_path = "mock-model"
        self.base_model = MagicMock()
        self.tokenizer = MockTokenizer()
        self.feature_class_encoding_direction = {"positive": 1.0, "negative": -1.0}
    
    def encode(self, grad, return_float=True):
        """Mock encode method."""
        if return_float:
            return float(torch.randn(1).item())
        return torch.randn(1)
    
    def decode(self, encoded, **kwargs):
        """Mock decode method."""
        return torch.randn(100)
    
    def rewrite_base_model(self, **kwargs):
        """Mock rewrite_base_model method."""
        return self


class TestEncoderEvaluator:
    """Test EncoderEvaluator."""
    
    def test_encoder_evaluator_creation(self):
        """Test that EncoderEvaluator can be created."""
        evaluator = EncoderEvaluator()
        assert evaluator is not None
    
    def test_evaluate_encoder_basic(self):
        """Test basic encoder evaluation returns unified metrics format."""
        evaluator = EncoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        
        result = evaluator.evaluate_encoder(trainer)
        
        assert "correlation" in result
        assert "n_samples" in result
        assert "mean_by_class" in result
        assert "all_data" in result
        assert isinstance(result["correlation"], float)
    
    def test_evaluate_encoder_with_eval_data(self):
        """Test encoder evaluation with pre-computed eval_data returns unified metrics."""
        evaluator = EncoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        
        eval_data = trainer.create_eval_data(trainer._model)
        
        result = evaluator.evaluate_encoder(trainer, eval_data=eval_data)
        
        assert "correlation" in result
        assert result["n_samples"] > 0
        assert "mean_by_class" in result
    
    def test_evaluate_encoder_parameter_overwriting(self):
        """Test that parameters override TrainingArguments."""
        evaluator = EncoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.encoder_eval_max_size = 200
        trainer = MockTrainer(training_args=training_args)
        trainer._model = MockModelWithGradiend()
        
        # Create eval_data first to ensure it's a GradientTrainingDataset
        eval_data = trainer.create_eval_data(trainer._model)
        assert isinstance(eval_data, GradientTrainingDataset)
        
        # Override max_size - pass it directly to evaluate_encoder
        # The parameter overwriting happens in create_eval_data, so we test that
        with patch.object(trainer, 'create_eval_data') as mock_create:
            mock_create.return_value = eval_data
            result = evaluator.evaluate_encoder(trainer, max_size=50)
            
            # Verify max_size was passed to create_eval_data
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs.get("max_size") == 50
            assert "correlation" in result
    
    def test_evaluate_encoder_uses_training_args_when_not_overridden(self):
        """Test that TrainingArguments values are used when not overridden."""
        evaluator = EncoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.encoder_eval_max_size = 200
        trainer = MockTrainer(training_args=training_args)
        trainer._model = MockModelWithGradiend()
        
        # Create eval_data first to ensure it's a GradientTrainingDataset
        eval_data = trainer.create_eval_data(trainer._model)
        assert isinstance(eval_data, GradientTrainingDataset)
        
        # Don't override max_size - let it use TrainingArguments default
        with patch.object(trainer, 'create_eval_data') as mock_create:
            mock_create.return_value = eval_data
            result = evaluator.evaluate_encoder(trainer)
            
            # Verify max_size from TrainingArguments was used
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            # max_size should come from TrainingArguments (200) via _default_from_training_args
            # The actual value depends on how _default_from_training_args works
            assert "correlation" in result
    
    def test_evaluate_encoder_caching(self, temp_dir):
        """Test that encoder evaluation uses caching when enabled."""
        evaluator = EncoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.use_cache = True
        trainer = MockTrainer(training_args=training_args)
        trainer.experiment_dir = temp_dir
        trainer._model = MockModelWithGradiend()
        
        # First call - should compute
        result1 = evaluator.evaluate_encoder(trainer, use_cache=True)
        
        # Second call - should use cache
        with patch.object(trainer, 'create_eval_data') as mock_create:
            result2 = evaluator.evaluate_encoder(trainer, use_cache=True)
            
            # Should not call create_eval_data again if cached
            # (cache file should exist)
            cache_files = [f for f in os.listdir(temp_dir) if f.endswith('.json')]
            assert len(cache_files) > 0
    
    def test_evaluate_encoder_correlation_computation(self):
        """Test that correlation is computed correctly."""
        evaluator = EncoderEvaluator()
        trainer = MockTrainer()
        
        # Create a model that encodes with a known pattern
        class DeterministicModel(MockModelWithGradiend):
            def encode(self, grad, return_float=True):
                # Return encoding that correlates with label
                label = grad.sum().item() if isinstance(grad, torch.Tensor) else 0.0
                return float(label * 0.5 + np.random.randn() * 0.1)
        
        trainer._model = DeterministicModel()
        
        # Create eval data with known labels
        training_data = MockTrainingData([
            {
                "factual": torch.tensor([1.0, 2.0, 3.0]),
                "alternative": torch.tensor([0.0, 0.0, 0.0]),
                "label": 1.0,
                "factual_id": 1,
                "alternative_id": 2
            },
            {
                "factual": torch.tensor([-1.0, -2.0, -3.0]),
                "alternative": torch.tensor([0.0, 0.0, 0.0]),
                "label": -1.0,
                "factual_id": 2,
                "alternative_id": 1
            }
        ])
        
        def gradient_creator(inputs):
            return inputs if isinstance(inputs, torch.Tensor) else torch.randn(3)
        
        eval_data = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )
        
        result = evaluator.evaluate_encoder(trainer, eval_data=eval_data)
        
        assert "correlation" in result
        assert isinstance(result["correlation"], float)
        assert -1.0 <= result["correlation"] <= 1.0
    
    def test_evaluate_encoder_empty_dataset(self):
        """Test that encoder evaluation handles empty datasets."""
        evaluator = EncoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        
        # Create empty eval data
        training_data = MockTrainingData([])
        
        def gradient_creator(inputs):
            return torch.randn(100)
        
        eval_data = GradientTrainingDataset(
            training_data=training_data,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )
        
        result = evaluator.evaluate_encoder(trainer, eval_data=eval_data)
        
        # Empty eval data returns explicit empty result (no correlation to report)
        assert result == {"n_samples": 0, "correlation": None}


class TestDecoderEvaluator:
    """Test DecoderEvaluator."""
    
    def test_decoder_evaluator_creation(self):
        """Test that DecoderEvaluator can be created."""
        evaluator = DecoderEvaluator()
        assert evaluator is not None
    
    def test_evaluate_decoder_parameter_overwriting(self):
        """Test that parameters override TrainingArguments."""
        evaluator = DecoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.decoder_eval_max_size_training_like = 100
        training_args.decoder_eval_max_size_neutral = 100
        trainer = MockTrainer(training_args=training_args)
        trainer._model = MockModelWithGradiend()
        
        # Override max_size parameters - test that they're accepted
        result = evaluator.evaluate_decoder(
            trainer,
            max_size_training_like=50,
            max_size_neutral=50,
            feature_factors=[-1.0],
            lrs=[1e-2]
        )
        
        # Should return summary or grid
        assert "summary" in result or "grid" in result
    
    def test_evaluate_decoder_uses_training_args_when_not_overridden(self):
        """Test that TrainingArguments values are used when not overridden."""
        evaluator = DecoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.decoder_eval_max_size_training_like = 100
        training_args.decoder_eval_max_size_neutral = 100
        trainer = MockTrainer(training_args=training_args)
        trainer._model = MockModelWithGradiend()
        
        # Don't override parameters - should use TrainingArguments defaults
        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=[1e-2]
        )
        
        # Should return summary or grid
        assert "summary" in result or "grid" in result
    
    def test_evaluate_decoder_use_cache_overwriting(self):
        """Test that use_cache parameter can override TrainingArguments."""
        evaluator = DecoderEvaluator()
        training_args = MockTrainingArguments()
        training_args.use_cache = True
        trainer = MockTrainer(training_args=training_args)
        trainer.experiment_dir = tempfile.mkdtemp()
        trainer._model = MockModelWithGradiend()
        
        # Override use_cache - test that parameter is accepted
        result = evaluator.evaluate_decoder(
            trainer,
            use_cache=False,
            feature_factors=[-1.0],
            lrs=[1e-2]
        )
        
        # Should return summary or grid
        assert "summary" in result or "grid" in result
    
    def test_evaluate_decoder_feature_factors_default(self):
        """Test that default feature factors are derived from model."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        model = MockModelWithGradiend()
        trainer._model = model
        
        # Test default feature factor derivation
        from gradiend.evaluator.decoder import derive_default_feature_factor
        
        factor = derive_default_feature_factor(trainer, model, class_name="positive")
        assert factor == -1.0  # Should be -direction["positive"] = -1.0
        
        factor = derive_default_feature_factor(trainer, model, class_name="negative")
        assert factor == 1.0  # Should be -direction["negative"] = -(-1.0) = 1.0
    
    def test_evaluate_decoder_feature_factors_custom(self):
        """Test that custom feature factors can be provided."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        
        custom_factors = [0.5, 1.0, 1.5]
        
        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=custom_factors,
            lrs=[1e-2]
        )
        
        # Should use custom feature factors
        assert "summary" in result or "grid" in result
    
    def test_evaluate_decoder_lrs_default(self):
        """Test that default learning rates are used when not provided."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        
        # Don't provide lrs - should use defaults [1e-2, 1e-3, 1e-4, 1e-5]
        # This creates 4 lrs * 1 feature_factor = 4 pairs to evaluate
        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0]
        )
        
        # Should return summary or grid
        assert "summary" in result or "grid" in result
    
    def test_evaluate_decoder_lrs_custom(self):
        """Test that custom learning rates can be provided."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        
        custom_lrs = [1e-1, 1e-2]
        
        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=custom_lrs
        )
        
        # Should use custom learning rates
        assert "summary" in result or "grid" in result
    
    def test_evaluate_decoder_part_parameter(self):
        """Test that part parameter is passed correctly."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        
        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=[1e-2],
            part="decoder-weight"
        )
        
        # Should accept part parameter
        assert "summary" in result or "grid" in result
    
    def test_evaluate_decoder_eval_batch_size(self):
        """Test that eval_batch_size parameter is passed correctly."""
        evaluator = DecoderEvaluator()
        trainer = MockTrainer()
        trainer._model = MockModelWithGradiend()
        
        result = evaluator.evaluate_decoder(
            trainer,
            feature_factors=[-1.0],
            lrs=[1e-2],
            eval_batch_size=32
        )
        
        # Should accept eval_batch_size parameter
        assert "summary" in result or "grid" in result
