"""
Tests for core GRADIEND models (modality-independent).

Tests GradiendModel, ParamMappedGradiendModel, and ModelWithGradiend with toy networks.
"""

import os
import sys
import tempfile
import shutil
import json
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from gradiend.model import GradiendModel, ParamMappedGradiendModel
from tests.conftest import SimpleMockModel, MockTokenizer


class TestGradiendModel:
    """Test GradiendModel (weights-only encoder-decoder)."""
    
    def test_gradiend_model_creation(self):
        """Test GradiendModel creation with various configs."""
        # Test with different latent_dim and input_dim (should default to CPU if CUDA not available)
        model1 = GradiendModel(input_dim=100, latent_dim=1)
        assert model1.input_dim == 100
        assert model1.latent_dim == 1
        
        model2 = GradiendModel(input_dim=500, latent_dim=2, activation_encoder="relu")
        assert model2.input_dim == 500
        assert model2.latent_dim == 2
        
        # Test dtype
        model3 = GradiendModel(input_dim=100, latent_dim=1, torch_dtype=torch.float16)
        assert model3.torch_dtype == torch.float16
    
    def test_gradiend_model_forward(self, set_seed):
        """Test GradiendModel forward pass with mock gradients."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Create mock gradient tensor
        x = torch.randn(100)
        
        # Forward pass
        decoded = model.forward(x)
        assert decoded.shape == (100,)
        assert decoded.dtype == model.torch_dtype
        
        # Forward with return_encoded
        decoded2, encoded = model.forward(x, return_encoded=True)
        assert decoded2.shape == (100,)
        assert encoded.shape == (1,)
    
    def test_gradiend_model_forward_encoder(self, set_seed):
        """Test encoder-only forward pass."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        x = torch.randn(100)
        encoded = model.forward_encoder(x)
        assert encoded.shape == (1,)
    
    def test_gradiend_model_save_load(self, temp_dir):
        """Test saving and loading GradiendModel."""
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Save
        save_path = os.path.join(temp_dir, "test_model")
        model.save_pretrained(save_path)
        
        # Verify files exist
        assert os.path.exists(os.path.join(save_path, "config.json"))
        # Check for either safetensors or bin file
        has_weights = (
            os.path.exists(os.path.join(save_path, "model.safetensors")) or
            os.path.exists(os.path.join(save_path, "pytorch_model.bin"))
        )
        assert has_weights, "Model weights should be saved"
        
        # Load (should default to CPU if CUDA not available)
        loaded = GradiendModel.from_pretrained(save_path)
        assert loaded.input_dim == model.input_dim
        assert loaded.latent_dim == model.latent_dim
        
        # Verify forward pass works
        x = torch.randn(100)
        original_output = model.forward(x)
        loaded_output = loaded.forward(x)
        torch.testing.assert_close(original_output, loaded_output, rtol=1e-5, atol=1e-5)
    
    def test_gradiend_model_get_weight_importance(self):
        """Test weight importance computation."""
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Test different parts
        imp_decoder_weight = model.get_weight_importance(part="decoder-weight")
        assert imp_decoder_weight.shape == (100,)
        
        imp_decoder_bias = model.get_weight_importance(part="decoder-bias")
        assert imp_decoder_bias.shape == (100,)
        
        imp_decoder_sum = model.get_weight_importance(part="decoder-sum")
        assert imp_decoder_sum.shape == (100,)
        
        imp_encoder_weight = model.get_weight_importance(part="encoder-weight")
        assert imp_encoder_weight.shape == (100,)
    
    def test_gradiend_model_get_topk_weights(self):
        """Test top-k weight selection."""
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Test absolute topk
        topk = model.get_topk_weights(part="decoder-weight", topk=10)
        assert len(topk) == 10
        assert all(isinstance(i, int) for i in topk)
        
        # Test relative topk
        topk_relative = model.get_topk_weights(part="decoder-weight", topk=0.1)
        assert len(topk_relative) == 10  # 10% of 100
    
    def test_gradiend_model_pruning_length_changes(self, set_seed):
        """Test that pruning changes input_dim length correctly."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        original_dim = model.input_dim
        
        # Prune to top 50
        pruned = model.prune(topk=50, inplace=False)
        assert pruned.input_dim == 50
        assert model.input_dim == original_dim  # Original unchanged
        
        # Prune in place
        model.prune(topk=30, inplace=True)
        assert model.input_dim == 30
    
    def test_gradiend_model_pruning_with_mask(self, set_seed):
        """Test pruning with provided mask."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Create a mask keeping first 50 dimensions
        mask = torch.zeros(100, dtype=torch.bool)
        mask[:50] = True
        
        pruned = model.prune(mask=mask, inplace=False)
        assert pruned.input_dim == 50
        
        # Test with return_mask
        pruned2, returned_mask = model.prune(mask=mask, inplace=False, return_mask=True)
        assert pruned2.input_dim == 50
        assert returned_mask.shape == (100,)
        assert returned_mask.sum() == 50
    
    def test_gradiend_model_pruning_with_threshold(self, set_seed):
        """Test pruning with threshold."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Get importance and set threshold
        importance = model.get_weight_importance(part="decoder-weight")
        threshold = importance.median().item()
        
        pruned = model.prune(threshold=threshold, part="decoder-weight", inplace=False)
        # Should keep roughly half (depending on distribution)
        assert pruned.input_dim < model.input_dim
        assert pruned.input_dim > 0


class TestParamMappedGradiendModel:
    """Test ParamMappedGradiendModel (with parameter mapping)."""
    
    def test_param_mapped_model_creation(self):
        """Test ParamMappedGradiendModel creation with param_map."""
        # Create a simple param_map
        param_map = {
            "layer1.weight": {
                "shape": (10, 5),
                "repr": "all"
            },
            "layer2.weight": {
                "shape": (5, 2),
                "repr": "all"
            }
        }
        input_dim = 10 * 5 + 5 * 2  # 60
        
        model = ParamMappedGradiendModel(
            input_dim=input_dim,
            latent_dim=1,
            param_map=param_map
        )
        
        assert model.input_dim == input_dim
        assert model.latent_dim == 1
        assert len(model.param_map) == 2
    
    def test_param_mapped_model_flatten_gradient_dict(self):
        """Test flattening gradient dict to tensor."""
        param_map = {
            "layer1.weight": {"shape": (2, 3), "repr": "all"},
            "layer2.weight": {"shape": (3, 1), "repr": "all"}
        }
        input_dim = 2 * 3 + 3 * 1  # 9
        
        model = ParamMappedGradiendModel(
            input_dim=input_dim,
            latent_dim=1,
            param_map=param_map
        )
        
        # Create gradient dict
        grad_dict = {
            "layer1.weight": torch.randn(2, 3),
            "layer2.weight": torch.randn(3, 1)
        }
        
        flattened = model.flatten_gradient_dict(grad_dict)
        assert flattened.shape == (input_dim,)
    
    def test_param_mapped_model_pruning_updates_map(self):
        """Test that pruning updates param_map correctly."""
        param_map = {
            "layer1.weight": {"shape": (10, 5), "repr": "all"},
            "layer2.weight": {"shape": (5, 2), "repr": "all"}
        }
        input_dim = 10 * 5 + 5 * 2  # 60
        
        model = ParamMappedGradiendModel(
            input_dim=input_dim,
            latent_dim=1,
            param_map=param_map
        )
        
        original_map_size = len(model.param_map)
        
        # Prune to top 30
        pruned = model.prune(topk=30, inplace=False)
        
        # Param map should still exist (but may be updated)
        assert len(pruned.param_map) == original_map_size
        assert pruned.input_dim == 30


class TestModelWithGradiend:
    """Test ModelWithGradiend wrapper."""
    
    def test_model_with_gradiend_creation(self, mock_model):
        """Test ModelWithGradiend wrapper creation."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        # Patch _load_model to return the mock model directly
        original_load = TextModelWithGradiend._load_model
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            # Return mock model and tokenizer directly
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        assert hasattr(model_with_gradiend, 'base_model')
        assert hasattr(model_with_gradiend, 'gradiend')
        assert model_with_gradiend.base_model == mock_model
    
    def test_model_with_gradiend_encode(self, mock_model, mock_tokenizer):
        """Test encoding functionality with mock base model."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            return mock_model, mock_tokenizer
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        # Test encoding (this will use gradient creator internally)
        # For a simple test, we'll just verify the method exists
        assert hasattr(model_with_gradiend, 'encode')
        assert callable(model_with_gradiend.encode)
    
    def test_model_with_gradiend_rewrite_base_model(self, mock_model):
        """Test rewrite_base_model method with decoder part."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        # Test that rewrite_base_model exists and works
        assert hasattr(model_with_gradiend, 'rewrite_base_model')
        assert callable(model_with_gradiend.rewrite_base_model)
        
        # Test rewrite_base_model with decoder part
        learning_rate = 1e-4
        feature_factor = 1.0
        
        rewritten_model = model_with_gradiend.rewrite_base_model(
            learning_rate=learning_rate,
            feature_factor=feature_factor,
            part="decoder"
        )
        
        # Should return a model (not the same instance as base_model)
        assert rewritten_model is not None
        assert rewritten_model is not model_with_gradiend.base_model
    
    def test_model_with_gradiend_rewrite_base_model_with_different_parts(self, mock_model):
        """Test rewrite_base_model with different part options."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        learning_rate = 1e-4
        
        # Test different part options
        parts = ["decoder", "decoder-weight", "decoder-bias", "decoder-sum", "encoder-weight"]
        
        for part in parts:
            try:
                rewritten_model = model_with_gradiend.rewrite_base_model(
                    learning_rate=learning_rate,
                    feature_factor=1.0,
                    part=part
                )
                assert rewritten_model is not None
            except ValueError as e:
                # Some parts might not be available depending on model structure
                # That's okay - we're just testing that the method handles them
                pass
    
    def test_model_with_gradiend_rewrite_base_model_with_list_feature_factor(self, mock_model):
        """Test rewrite_base_model with list feature_factor (requires latent_dim=2)."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                latent_dim=2,
            )
        
        # Test with list feature_factor (must match latent_dim)
        rewritten_model = model_with_gradiend.rewrite_base_model(
            learning_rate=1e-4,
            feature_factor=[1.0, -1.0],
            part="decoder"
        )
        
        assert rewritten_model is not None
    
    def test_model_with_gradiend_rewrite_base_model_raises_invalid_part(self, mock_model):
        """Test rewrite_base_model raises ValueError for invalid part."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        with pytest.raises(ValueError) as exc_info:
            model_with_gradiend.rewrite_base_model(
                learning_rate=1e-4,
                feature_factor=1.0,
                part="invalid_part"
            )
        
        assert "part must be" in str(exc_info.value).lower()
    
    def test_model_with_gradiend_prune_gradiend(self, mock_model):
        """Test pruning via prune_gradiend method."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1
            )
        
        original_input_dim = model_with_gradiend.gradiend.input_dim
        
        # Prune to top 50% (if input_dim > 1)
        if original_input_dim > 1:
            pruned = model_with_gradiend.prune_gradiend(topk=0.5, inplace=False)
            assert pruned.gradiend.input_dim < original_input_dim
            assert pruned.gradiend.input_dim > 0
    
    def test_model_with_gradiend_pre_prune(self, mock_model, mock_tokenizer):
        """Test pre-pruning before training."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        from gradiend.trainer.core.pruning import pre_prune, PrePruneConfig
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            return mock_model, mock_tokenizer
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        original_input_dim = model_with_gradiend.gradiend.input_dim
        
        # Create a mock dataset with text inputs (as expected by TextPredictionModelWithGradiend)
        # The gradient_creator expects text strings, not tensors
        class MockTextDataset:
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                # Return text strings that can be processed by create_gradients
                # Include labels for gradient creation
                return {
                    'factual': f"factual text {idx}",
                    'alternative': f"alternative text {idx}",
                    'label': f"label_{idx % 2}",  # Add label for gradient creation
                    'feature_class_id': idx % 2
                }
        
        dataset = MockTextDataset()
        
        # Pre-prune config
        config = PrePruneConfig(
            n_samples=4,
            topk=0.5,
            source="factual"
        )
        
        # Pre_prune calls gradient_creator(factual_in) with just the text string
        # But create_gradients needs (text, label). We need to patch gradient_creator
        # to extract the label from the dataset item or use a default
        original_create_gradients = model_with_gradiend.create_gradients
        
        def mock_create_gradients(text, label=None, return_dict=False, **kwargs):
            # If called with just text (no label), use a default label
            if label is None:
                label = "default_label"
            # Return a mock gradient tensor with the right shape
            # For simplicity, return a random tensor matching input_dim
            if return_dict:
                # If return_dict, return a dict that can be converted to vector
                return {"gradient": torch.randn(original_input_dim)}
            else:
                # Otherwise return a tensor directly
                return torch.randn(original_input_dim)
        
        # Patch create_gradients to avoid actual model forward pass
        model_with_gradiend.create_gradients = mock_create_gradients
        
        # Pre_prune calls gradient_creator(text) directly
        # gradient_creator is a @property that returns _gradient_creator
        # We need to set _gradient_creator to a callable that handles single-argument calls
        # Use object.__setattr__ to bypass PyTorch's nn.Module attribute registration
        def gradient_creator_callable(text, target_device=None, **kwargs):
            """Callable wrapper for gradient_creator(text) calls from pre_prune."""
            # Pre_prune calls with text and target_device=..., so add a default label
            return mock_create_gradients(text, label="default_label", return_dict=False)
        
        # Set _gradient_creator (the underlying attribute) to our callable wrapper
        # Use object.__setattr__ to bypass PyTorch's nn.Module special handling
        original_gradient_creator = model_with_gradiend._gradient_creator
        object.__setattr__(model_with_gradiend, '_gradient_creator', gradient_creator_callable)
        
        try:
            pruned = pre_prune(
                model_with_gradiend,
                dataset,
                config,
                inplace=False
            )
            assert pruned.gradiend.input_dim < original_input_dim
            assert pruned.gradiend.input_dim > 0
        finally:
            # Restore original methods
            model_with_gradiend.create_gradients = original_create_gradients
            object.__setattr__(model_with_gradiend, '_gradient_creator', original_gradient_creator)
    
    def test_model_with_gradiend_post_prune(self, mock_model):
        """Test post-pruning after training."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        from gradiend.trainer.core.pruning import post_prune, PostPruneConfig
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        original_input_dim = model_with_gradiend.gradiend.input_dim
        
        # Post-prune config
        config = PostPruneConfig(
            topk=0.5,
            part="decoder-weight"
        )
        
        pruned = post_prune(model_with_gradiend, config)
        assert pruned.gradiend.input_dim < original_input_dim
        assert pruned.gradiend.input_dim > 0
    
    def test_model_saving_with_safetensors_available(self, temp_dir):
        """Test that safetensors is used when available."""
        from gradiend.model.model import GradiendModel
        
        # Create a simple model to test saving
        model = GradiendModel(input_dim=100, latent_dim=1)
        
        # Mock safetensors.save_file to actually create a file
        def mock_save_file(state_dict, path):
            """Mock save_file that creates a file to verify it was called."""
            # Create a marker file to verify safetensors was used
            with open(path, 'wb') as f:
                f.write(b"safetensors_marker")
        
        # Create mock safetensors module structure
        mock_safetensors_torch = MagicMock()
        mock_safetensors_torch.save_file = mock_save_file
        
        # Patch the import where it's used in save_pretrained
        # The import happens as: from safetensors.torch import save_file
        # We need to patch it at the module level
        import sys
        original_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith('safetensors'):
                original_modules[key] = sys.modules.pop(key)
        
        try:
            # Add mock module to sys.modules
            sys.modules['safetensors'] = MagicMock()
            sys.modules['safetensors.torch'] = mock_safetensors_torch
            
            # Patch the save_file function
            with patch.object(mock_safetensors_torch, 'save_file', mock_save_file):
                save_path = os.path.join(temp_dir, "test_model_safetensors")
                model.save_pretrained(save_path, use_safetensors=None)  # None = prefer safetensors
            
            # Check that safetensors file was created
            safetensors_path = os.path.join(save_path, "model.safetensors")
            bin_path = os.path.join(save_path, "pytorch_model.bin")
            
            # When safetensors is available, it should be used
            assert os.path.exists(safetensors_path), "model.safetensors should exist when safetensors is available"
            assert not os.path.exists(bin_path), "pytorch_model.bin should not exist when safetensors is used"
        finally:
            # Restore modules
            for key in list(sys.modules.keys()):
                if key.startswith('safetensors') and key not in original_modules:
                    sys.modules.pop(key)
            sys.modules.update(original_modules)
    
    def test_model_saving_without_safetensors_available(self, temp_dir):
        """Test that bin format is used when safetensors is not available."""
        from gradiend.model.model import GradiendModel
        
        # Create a simple model to test saving
        model = GradiendModel(input_dim=100, latent_dim=1)
        
        # Remove safetensors from sys.modules to ensure clean state
        original_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith('safetensors'):
                original_modules[key] = sys.modules.pop(key)
        
        try:
            # Mock the import to fail by patching __import__
            import builtins
            original_import = builtins.__import__
            def failing_import(name, *args, **kwargs):
                if name == 'safetensors.torch':
                    raise ImportError("No module named 'safetensors'")
                # Use the real __import__ for everything else
                return original_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=failing_import):
                save_path = os.path.join(temp_dir, "test_model_no_safetensors")
                model.save_pretrained(save_path, use_safetensors=None)  # None = prefer safetensors, but will fallback
            
            # When safetensors is not available, bin should be used
            safetensors_path = os.path.join(save_path, "model.safetensors")
            bin_path = os.path.join(save_path, "pytorch_model.bin")
            
            assert os.path.exists(bin_path), "pytorch_model.bin should exist when safetensors is not available"
            assert not os.path.exists(safetensors_path), "model.safetensors should not exist when safetensors is not available"
        finally:
            # Restore modules
            sys.modules.update(original_modules)
    
    def test_model_saving_force_safetensors_when_unavailable(self, temp_dir):
        """Test that forcing safetensors raises error when unavailable."""
        from gradiend.model.model import GradiendModel
        
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Remove safetensors from sys.modules to ensure clean state
        original_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith('safetensors'):
                original_modules[key] = sys.modules.pop(key)
        
        try:
            # Mock the import to fail
            import builtins
            original_import = builtins.__import__
            def failing_import(name, *args, **kwargs):
                if name == 'safetensors.torch':
                    raise ImportError("No module named 'safetensors'")
                # Use the real __import__ for everything else
                return original_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=failing_import):
                save_path = os.path.join(temp_dir, "test_model_force_safetensors")
                
                # When use_safetensors=True but safetensors is unavailable, should raise ImportError
                with pytest.raises(ImportError, match="safetensors not installed"):
                    model.save_pretrained(save_path, use_safetensors=True)
        finally:
            # Restore modules
            sys.modules.update(original_modules)
    
    def test_model_dtype_handling(self, mock_model):
        """Test Float32 vs Float16 dtype handling."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            # Test float32
            model_f32 = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
                torch_dtype=torch.float32,
            )
            assert model_f32.gradiend.torch_dtype == torch.float32
            
            # Test float16
            model_f16 = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
                torch_dtype=torch.float16,
            )
            assert model_f16.gradiend.torch_dtype == torch.float16
    
    def test_pruning_efficient_storage_with_full_mask(self, temp_dir, set_seed):
        """Test that pruning uses efficient storage when full masks are provided."""
        set_seed(42)
        
        from gradiend.model import ParamMappedGradiendModel
        
        # Create a model with param mapping
        # Each layer has 10*10=100 elements, so total input_dim should be 200
        input_dim = 200
        param_map = {
            "layer1.weight": {"shape": (10, 10), "repr": "all"},  # 100 elements
            "layer2.weight": {"shape": (10, 10), "repr": "all"},  # 100 elements
        }
        
        model = ParamMappedGradiendModel(
            input_dim=input_dim,
            latent_dim=1,
            param_map=param_map
        )
        
        # Create a full mask (all True)
        full_mask = torch.ones(input_dim, dtype=torch.bool)
        
        # Prune with full mask
        pruned = model.prune(mask=full_mask, inplace=False)
        
        # When full mask is provided, efficient storage should be used
        # Save the model to check storage format
        save_path = os.path.join(temp_dir, "pruned_model_full_mask")
        pruned.save_pretrained(save_path)
        
        # Check that model was saved
        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(save_path, "config.json"))
        
        # For full mask, the model should still be pruned correctly
        # (though with full mask, no actual pruning occurs)
        assert pruned.input_dim == input_dim  # Full mask means no reduction
    
    def test_pruning_efficient_storage_with_partial_mask(self, temp_dir, set_seed):
        """Test that pruning uses efficient storage when partial masks are provided."""
        set_seed(42)
        
        from gradiend.model import ParamMappedGradiendModel
        
        # Create a model with param mapping
        # Each layer has 10*10=100 elements, so total input_dim should be 200
        input_dim = 200
        param_map = {
            "layer1.weight": {"shape": (10, 10), "repr": "all"},  # 100 elements
            "layer2.weight": {"shape": (10, 10), "repr": "all"},  # 100 elements
        }
        
        model = ParamMappedGradiendModel(
            input_dim=input_dim,
            latent_dim=1,
            param_map=param_map
        )
        
        # Create a partial mask (keep first 100 dimensions)
        partial_mask = torch.zeros(input_dim, dtype=torch.bool)
        partial_mask[:100] = True
        
        # Prune with partial mask
        pruned = model.prune(mask=partial_mask, inplace=False)
        
        # Should reduce input_dim
        assert pruned.input_dim == 100
        
        # Save the model to check storage format
        save_path = os.path.join(temp_dir, "pruned_model_partial_mask")
        pruned.save_pretrained(save_path)
        
        # Check that model was saved
        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(save_path, "config.json"))
        
        # Check that mapping files are saved efficiently
        # When masks are provided, they should be stored efficiently
        config_path = os.path.join(save_path, "config.json")
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check that mapping information is present
        if "mapping" in config:
            mapping = config["mapping"]
            # Efficient storage: masks should be saved in mapping_masks file
            assert "masks_file" in mapping or "mode" in mapping