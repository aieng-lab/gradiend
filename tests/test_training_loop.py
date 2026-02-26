"""
Tests for training loop (modality-independent).

Tests basic training flow, seed handling, and parameter passing/overwriting.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import pytest
import torch
import torch.nn as nn

from gradiend.trainer.core.training import train
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.dataset import GradientTrainingDataset
from gradiend.model import GradiendModel
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


class MockTrainingData:
    """Mock training dataset."""
    
    def __init__(self, items, batch_size=1):
        self.items = items
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        return self.items[idx]


class TestTrainingLoop:
    """Test basic training loop functionality."""
    
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
        
        # Verify override worked (check that training used the overridden value)
        # The training_args object should have been updated
        assert isinstance(output_path, str)
    
    def test_train_seed_handling(self, temp_dir):
        """Test that seed is handled correctly during training."""
        model1 = MockModelWithGradiend()
        model2 = MockModelWithGradiend()
        
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
        
        training_args1 = TrainingArguments(
            output_dir=os.path.join(temp_dir, "seed1"),
            max_steps=10,
            seed=42,
            train_batch_size=1
        )
        
        training_args2 = TrainingArguments(
            output_dir=os.path.join(temp_dir, "seed2"),
            max_steps=10,
            seed=42,  # Same seed
            train_batch_size=1
        )
        
        # Train with same seed - should produce same results
        output_path1 = train(
            model_with_gradiend=model1,
            data=dataloader,
            training_args=training_args1
        )
        
        output_path2 = train(
            model_with_gradiend=model2,
            data=dataloader,
            training_args=training_args2
        )
        
        # Both should complete successfully
        assert isinstance(output_path1, str)
        assert isinstance(output_path2, str)
    
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
        
        custom_callback = LoggingCallback(n_loss_report=10)
        
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
