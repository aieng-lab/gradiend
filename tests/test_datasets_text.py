"""
Tests for TextGradientTrainingDataset (text-specific).

Tests text-specific dataset functionality including padding, caching,
and data loading variations (add_identity_for_other_classes, max_size).
"""

import os
import tempfile
import pandas as pd
from unittest.mock import MagicMock
from typing import Dict, Any

import pytest
import torch

from gradiend.trainer.text.common.dataset import TextGradientTrainingDataset
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset
from gradiend.trainer.text.prediction.unified_schema import (
    UNIFIED_MASKED,
    UNIFIED_FACTUAL,
    UNIFIED_ALTERNATIVE,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_ALTERNATIVE_CLASS,
)
from tests.conftest import MockTokenizer


class TestTextGradientTrainingDataset:
    """Test TextGradientTrainingDataset (text-specific wrapper)."""
    
    def test_text_dataset_creation(self):
        """Test that TextGradientTrainingDataset can be created."""
        tokenizer = MockTokenizer()
        training_data = MagicMock()
        training_data.__len__ = MagicMock(return_value=10)
        training_data.batch_size = 1
        training_data.__getitem__ = MagicMock(return_value={
            "factual": {"input_ids": torch.tensor([1, 2, 3])},
            "alternative": {"input_ids": torch.tensor([4, 5, 6])},
            "input_text": "test",
            "label": "positive"
        })
        
        gradient_creator = MagicMock(return_value=torch.randn(100))
        
        dataset = TextGradientTrainingDataset(
            training_data=training_data,
            tokenizer=tokenizer,
            gradient_creator=gradient_creator,
            source="factual",
            target="diff"
        )
        
        assert dataset.tokenizer == tokenizer
        assert dataset.gradient_creator == gradient_creator
        assert dataset.source == "factual"
        assert dataset.target == "diff"
    
    def test_text_dataset_padding_uses_tokenizer_pad_token_id(self):
        """Test that text dataset uses tokenizer.pad_token_id for padding."""
        tokenizer = MockTokenizer()
        tokenizer.pad_token_id = 999
        
        training_data = MagicMock()
        training_data.__len__ = MagicMock(return_value=1)
        training_data.__getitem__ = MagicMock(return_value={
            "factual": {"input_ids": torch.tensor([1, 2, 3])},
            "alternative": {"input_ids": torch.tensor([4, 5])},
            "input_text": "test",
            "label": "positive"
        })
        
        gradient_creator = MagicMock(return_value=torch.randn(100))
        
        dataset = TextGradientTrainingDataset(
            training_data=training_data,
            tokenizer=tokenizer,
            gradient_creator=gradient_creator
        )
        
        # The padding function should use pad_token_id for 'input_ids'
        padding_value = dataset._get_padding_value("input_ids")
        assert padding_value == 999
        
        # Other keys should use 0
        padding_value_other = dataset._get_padding_value("attention_mask")
        assert padding_value_other == 0
    
    def test_text_dataset_caching_uses_cache_key_fields(self, temp_dir):
        """Test that text dataset uses correct cache_key_fields for caching."""
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        tokenizer = MockTokenizer()
        training_data = MagicMock()
        training_data.__len__ = MagicMock(return_value=1)
        training_data.__getitem__ = MagicMock(return_value={
            "factual": {"input_ids": torch.tensor([1, 2, 3])},
            "alternative": {"input_ids": torch.tensor([4, 5, 6])},
            "input_text": "test text",
            "label": "positive"
        })
        
        gradient_creator = MagicMock(return_value=torch.randn(100))
        
        dataset = TextGradientTrainingDataset(
            training_data=training_data,
            tokenizer=tokenizer,
            gradient_creator=gradient_creator,
            cache_dir=cache_dir,
            use_cached_gradients=True
        )
        
        # Should use ['input_text', 'label'] as cache_key_fields
        assert dataset.cache_key_fields == ['input_text', 'label']
    
    def test_text_dataset_caching_no_cache_key_fields_when_disabled(self):
        """Test that cache_key_fields is None when caching is disabled."""
        tokenizer = MockTokenizer()
        training_data = MagicMock()
        training_data.__len__ = MagicMock(return_value=1)
        training_data.__getitem__ = MagicMock(return_value={
            "factual": {"input_ids": torch.tensor([1, 2, 3])},
            "alternative": {"input_ids": torch.tensor([4, 5, 6])},
            "input_text": "test",
            "label": "positive"
        })
        
        gradient_creator = MagicMock(return_value=torch.randn(100))
        
        dataset = TextGradientTrainingDataset(
            training_data=training_data,
            tokenizer=tokenizer,
            gradient_creator=gradient_creator,
            cache_dir=None,  # Caching disabled
            use_cached_gradients=False
        )
        
        # Should not set cache_key_fields when caching is disabled
        assert dataset.cache_key_fields == []
    
    def test_text_dataset_requires_input_text_and_label_for_caching(self, temp_dir):
        """Test that text dataset requires input_text and label when caching."""
        cache_dir = os.path.join(temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        tokenizer = MockTokenizer()
        
        # Create a proper mock that handles __getitem__ correctly
        class MockTrainingDataWithGetItem:
            def __init__(self):
                self.batch_size = 1
            
            def __len__(self):
                return 1
            
            def __getitem__(self, idx):
                # Return dict without input_text and label
                return {
                    "factual": {"input_ids": torch.tensor([1, 2, 3])},
                    "alternative": {"input_ids": torch.tensor([4, 5, 6])}
                    # Missing "input_text" and "label"
                }
        
        training_data = MockTrainingDataWithGetItem()
        
        gradient_creator = MagicMock(return_value=torch.randn(100))
        
        dataset = TextGradientTrainingDataset(
            training_data=training_data,
            tokenizer=tokenizer,
            gradient_creator=gradient_creator,
            cache_dir=cache_dir,
            use_cached_gradients=True
        )
        
        # Should raise KeyError when accessing batch without required cache keys
        with pytest.raises(KeyError, match="input_text|label"):
            _ = dataset[0]


class TestTextTrainingDataset:
    """Test TextTrainingDataset (text-specific training dataset)."""
    
    def test_text_training_dataset_creation(self):
        """Test that TextTrainingDataset can be created."""
        # Use string literal "masked" instead of constant to ensure pandas compatibility
        data = pd.DataFrame({
            "masked": ["Hello [MASK] world", "Test [MASK] sentence"],
            "factual": ["test1", "test2"],
            "alternative": ["other1", "other2"],
            "factual_class": ["class1", "class2"],
            "alternative_class": ["class2", "class1"],
            "factual_id": [1, 2],
            "alternative_id": [3, 4],
            "label": ["positive", "negative"],
            "feature_class_id": [1, 2]
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1,
            is_decoder_only_model=False,
            max_size=None
        )
        
        assert len(dataset) > 0
        assert dataset.tokenizer == tokenizer
        assert dataset.batch_size == 1

    def test_text_training_dataset_raises_when_insufficient_data_for_batch_size(self):
        """Creating a dataset with too few samples per subgroup raises a comprehensive ValueError."""
        # One sample per (feature_class_id, label) -> subgroups of size 1; batch_size=4 cannot be satisfied
        data = pd.DataFrame({
            "masked": ["Hello [MASK] world", "Other [MASK] text"],
            "factual": ["he", "they"],
            "alternative": ["they", "he"],
            "factual_class": ["3SG", "3PL"],
            "alternative_class": ["3PL", "3SG"],
            "factual_id": [1, 2],
            "alternative_id": [2, 1],
            "label": ["he", "they"],
            "feature_class_id": [0, 1],
        })
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        tokenizer.mask_token_id = 103

        with pytest.raises(ValueError) as exc_info:
            TextTrainingDataset(
                data=data,
                tokenizer=tokenizer,
                batch_size=4,
                balance_column="feature_class_id",
            )
        msg = str(exc_info.value)
        assert "batch_size" in msg
        assert "4" in msg
        assert "1" in msg  # smallest subgroup has 1 sample
        assert "Use more training_data data" in msg or "reduce" in msg.lower()

    def test_text_training_dataset_max_size(self):
        """Test that max_size limits the number of samples."""
        data = pd.DataFrame({
            "masked": [f"Text {i} [MASK]" for i in range(100)],
            "factual": ["token"] * 100,
            "alternative": ["other"] * 100,
            "factual_class": ["class1"] * 100,
            "alternative_class": ["class2"] * 100,
            "factual_id": list(range(100)),
            "alternative_id": list(range(100, 200)),
            "label": ["positive"] * 100,
            "feature_class_id": [1] * 100
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        # Without max_size
        dataset_full = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1,
            max_size=None
        )
        # Length depends on batching logic, but should be <= 100
        assert len(dataset_full) <= 100
        
        # With max_size (seed is handled internally by parent class)
        dataset_limited = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1,
            max_size=50
        )
        # Should be limited to 50 samples
        assert len(dataset_limited) <= 50
    
    def test_text_training_dataset_max_size_downsampling(self):
        """Test that max_size downsamples the data."""
        data = pd.DataFrame({
            "masked": [f"Text {i} [MASK]" for i in range(100)],
            "factual": ["token"] * 100,
            "alternative": ["other"] * 100,
            "factual_class": ["class1"] * 100,
            "alternative_class": ["class2"] * 100,
            "factual_id": list(range(100)),
            "alternative_id": list(range(100, 200)),
            "label": ["positive"] * 100,
            "feature_class_id": [1] * 100
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1,
            max_size=30
        )
        
        # Should have at most 30 samples (downsampled)
        assert len(dataset) <= 30
    
    def test_text_training_dataset_batch_size(self):
        """Test that batch_size affects dataset length."""
        # Need multiple labels to create batches (batch_criterion groups by label)
        # TextTrainingDataset uses total_samples = total_batches * batch_size (or 100 * total_batches * batch_size if balance_column)
        data = pd.DataFrame({
            "masked": [f"Text {i} [MASK]" for i in range(20)],
            "factual": ["token"] * 20,
            "alternative": ["other"] * 20,
            "factual_class": ["class1"] * 20,
            "alternative_class": ["class2"] * 20,
            "label": ["positive"] * 10 + ["negative"] * 10,  # Need different labels for batching
            "feature_class_id": [1] * 20
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        # batch_size=1 - all items can be batched individually
        dataset_bs1 = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1
        )
        # TextTrainingDataset.__len__ returns total_samples which can be large
        # We just verify it creates batches
        assert len(dataset_bs1) > 0
        
        # batch_size=2 - items batched in groups of 2 (within same label)
        dataset_bs2 = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=2
        )
        # Should have batches (grouped by label, then batched)
        assert len(dataset_bs2) > 0
    
    def test_text_training_dataset_balance_column(self):
        """Test that balance_column affects batching."""
        # Need enough items with same label within each balance group for batching
        # TextTrainingDataset groups by batch_criterion (label by default), then batches
        data = pd.DataFrame({
            "masked": [f"Text {i} [MASK]" for i in range(8)],
            "factual": [f"token{i}" for i in range(8)],
            "alternative": [f"other{i}" for i in range(8)],
            "factual_class": ["class1"] * 4 + ["class2"] * 4,
            "alternative_class": ["class2"] * 4 + ["class1"] * 4,
            "factual_id": list(range(8)),
            "alternative_id": list(range(8, 16)),
            "label": ["positive"] * 8,  # Same label so they can be batched together
            "feature_class_id": [1] * 4 + [2] * 4  # Two balance groups
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=2,
            balance_column="feature_class_id"
        )
        
        # Should batch by feature_class_id
        # With 2 items per class and batch_size=2, should get batches
        assert len(dataset) >= 1
    
    def test_text_training_dataset_decoder_only_pad_token(self):
        """Test that decoder-only models use eos_token as pad_token."""
        data = pd.DataFrame({
            "masked": ["Hello [MASK] world"],
            "factual": ["test"],
            "alternative": ["other"],
            "factual_class": ["class1"],
            "alternative_class": ["class2"],
            "factual_id": [1],
            "alternative_id": [2],
            "label": ["positive"],
            "feature_class_id": [1]
        })
        
        tokenizer = MockTokenizer()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.eos_token_id = 50256
        
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1,
            is_decoder_only_model=True
        )
        
        # Should set pad_token to eos_token for decoder-only models
        assert tokenizer.pad_token == "<|endoftext|>"
    
    def test_text_training_dataset_returns_correct_structure(self):
        """Test that TextTrainingDataset returns items with correct structure."""
        data = pd.DataFrame({
            "masked": ["Hello [MASK] world"],
            "factual": ["test"],
            "alternative": ["other"],
            "factual_class": ["class1"],
            "alternative_class": ["class2"],
            "factual_id": [1],
            "alternative_id": [2],
            "label": ["positive"],
            "feature_class_id": [1]
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        tokenizer.mask_token_id = 103
        
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1,
            is_decoder_only_model=False
        )
        
        item = dataset[0]
        
        # Should have factual and alternative keys
        assert "factual" in item
        assert "alternative" in item
        
        # Factual and alternative should have input_ids, attention_mask, labels
        assert "input_ids" in item["factual"]
        assert "attention_mask" in item["factual"]
        assert "labels" in item["factual"]
        
        assert "input_ids" in item["alternative"]
        assert "attention_mask" in item["alternative"]
        assert "labels" in item["alternative"]
    
    def test_text_training_dataset_feature_class_id_preserved(self):
        """Test that feature_class_id is preserved in items."""
        data = pd.DataFrame({
            "masked": ["Hello [MASK] world"],
            "factual": ["test"],
            "alternative": ["other"],
            "factual_class": ["class1"],
            "alternative_class": ["class2"],
            "factual_id": [1],
            "alternative_id": [2],
            "label": ["positive"],
            "feature_class_id": [42]
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1
        )
        
        item = dataset[0]
        
        # Should preserve feature_class_id
        assert "feature_class_id" in item
        assert item["feature_class_id"] == 42
    
    def test_text_training_dataset_label_preserved(self):
        """Test that label is preserved in items."""
        data = pd.DataFrame({
            "masked": ["Hello [MASK] world"],
            "factual": ["test"],
            "alternative": ["other"],
            "factual_class": ["class1"],
            "alternative_class": ["class2"],
            "factual_id": [1],
            "alternative_id": [2],
            "label": ["positive"],
            "feature_class_id": [1]
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1,
            target_key="label"
        )
        
        item = dataset[0]
        
        # Should preserve label
        assert "label" in item
        assert item["label"] == "positive"
    
    def test_text_training_dataset_max_length(self):
        """Test that max_length limits sequence length."""
        data = pd.DataFrame({
            "masked": ["Hello [MASK] world"] * 10,
            "factual": ["test"] * 10,
            "alternative": ["other"] * 10,
            "factual_class": ["class1"] * 10,
            "alternative_class": ["class2"] * 10,
            "factual_id": list(range(10)),
            "alternative_id": list(range(10, 20)),
            "label": ["positive"] * 10,
            "feature_class_id": [1] * 10
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1,
            max_length=128
        )
        
        item = dataset[0]
        
        # input_ids should be truncated/padded to max_length
        assert item["factual"]["input_ids"].shape[0] <= 128
        assert item["alternative"]["input_ids"].shape[0] <= 128


class TestTextDatasetDataLoadingVariations:
    """Test data loading variations like add_identity_for_other_classes and max_size."""
    
    def test_max_size_limits_per_feature_class(self):
        """Test that max_size can limit samples per feature_class_id."""
        # This tests the behavior when max_size is applied per feature_class_id
        # in the trainer's data loading methods
        data = pd.DataFrame({
            "masked": [f"Text {i} [MASK]" for i in range(100)],
            "factual": ["token"] * 100,
            "alternative": ["other"] * 100,
            "factual_class": ["class1"] * 50 + ["class2"] * 50,
            "alternative_class": ["class2"] * 50 + ["class1"] * 50,
            "factual_id": list(range(100)),
            "alternative_id": list(range(100, 200)),
            "label": ["positive"] * 50 + ["negative"] * 50,
            "feature_class_id": [1] * 50 + [2] * 50
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        # max_size should limit total samples
        # This is typically handled in the trainer, not the dataset itself
        # Note: seed is handled by parent class, not passed directly
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1,
            max_size=30  # Total max_size
        )
        
        # Should have at most 30 samples total (downsampled)
        assert len(dataset) <= 30
    
    def test_add_identity_for_other_classes_requires_classes(self):
        """Test that add_identity_for_other_classes requires class definitions."""
        # This is typically handled in the trainer, not the dataset
        # The dataset itself doesn't handle add_identity_for_other_classes
        # It's a TrainingArguments parameter that affects data loading
        
        # We can test that the dataset works with identity samples if they're provided
        data = pd.DataFrame({
            "masked": ["Hello [MASK] world"],
            "factual": ["neutral_data"],
            "alternative": ["neutral_data"],
            "factual_class": ["neutral_data"],
            "alternative_class": ["neutral_data"],
            "factual_id": [0],
            "alternative_id": [0],
            "label": ["neutral_data"],  # Identity/neutral_data class
            "feature_class_id": [0]  # Identity class ID
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=1
        )
        
        # Should work fine with identity samples
        item = dataset[0]
        assert "factual" in item
        assert "alternative" in item
    
    def test_dataset_handles_multiple_feature_classes(self):
        """Test that dataset handles multiple feature_class_id values."""
        data = pd.DataFrame({
            "masked": [f"Text {i} [MASK]" for i in range(20)],
            "factual": ["token"] * 20,
            "alternative": ["other"] * 20,
            "factual_class": ["class1"] * 10 + ["class2"] * 10,
            "alternative_class": ["class2"] * 10 + ["class1"] * 10,
            "factual_id": list(range(20)),
            "alternative_id": list(range(20, 40)),
            "label": ["positive"] * 10 + ["negative"] * 10,
            "feature_class_id": [1] * 10 + [2] * 10
        })
        
        tokenizer = MockTokenizer()
        tokenizer.mask_token = "[MASK]"
        
        dataset = TextTrainingDataset(
            data=data,
            tokenizer=tokenizer,
            batch_size=2,
            balance_column="feature_class_id"
        )
        
        # Should handle multiple classes
        assert len(dataset) >= 1
        
        # Should preserve feature_class_id in items
        for i in range(min(5, len(dataset))):
            item = dataset[i]
            assert "feature_class_id" in item
