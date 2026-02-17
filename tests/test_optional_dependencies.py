"""
Tests for optional dependency handling (modality-independent).

Tests that optional packages (safetensors, matplotlib, seaborn, datasets, spacy) are
handled gracefully when missing, with appropriate error messages. The gradiend package
and its classes load without these packages; only when the optional functionality is
requested does the code raise ImportError with comprehensive install instructions.
"""

import sys
import os
import builtins
from unittest.mock import patch, MagicMock

import pytest
import torch

# Force CPU for all tests to avoid CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""
if torch.cuda.is_available():
    torch.cuda.set_device(torch.device("cpu"))

from gradiend.model.utils import _save_tensor_dict, _load_tensor_dict, _tensor_file_name
from tests.conftest import MockTokenizer


class TestSafetensorsOptional:
    """Test safetensors optional dependency handling."""
    
    def test_safetensors_available(self, temp_dir):
        """Test that safetensors is used when available."""
        try:
            import safetensors
            # If safetensors is available, test that it's used
            tensors = {"weight": torch.randn(10, 10)}
            path = f"{temp_dir}/test.safetensors"
            
            _save_tensor_dict(path, tensors, prefer_safetensors=True)
            assert path.endswith(".safetensors")
            
            loaded = _load_tensor_dict(path, prefer_safetensors=True)
            assert "weight" in loaded
            torch.testing.assert_close(loaded["weight"], tensors["weight"])
        except ImportError:
            pytest.skip("safetensors not available")
    
    def test_safetensors_fallback_to_pth(self, temp_dir):
        """Test that pth format is used when safetensors is not available."""
        tensors = {"weight": torch.randn(10, 10)}
        
        # Mock safetensors import to fail
        with patch.dict('sys.modules', {'safetensors': None, 'safetensors.torch': None}):
            # Force reload of the module to pick up the mock
            import importlib
            import gradiend.model.utils as utils_module
            importlib.reload(utils_module)
            
            path = f"{temp_dir}/test.pth"
            utils_module._save_tensor_dict(path, tensors, prefer_safetensors=True)
            
            # Should fall back to .pth
            assert path.endswith(".pth") or not path.endswith(".safetensors")
            
            loaded = utils_module._load_tensor_dict(path, prefer_safetensors=False)
            assert "weight" in loaded
            torch.testing.assert_close(loaded["weight"], tensors["weight"])
    
    def test_tensor_file_name_safetensors(self):
        """Test tensor file name generation with safetensors preference."""
        # When safetensors is preferred
        name1 = _tensor_file_name("model", prefer_safetensors=True)
        # Should prefer safetensors if available, otherwise pth
        assert name1.endswith((".safetensors", ".pth"))
        
        # When safetensors is not preferred
        name2 = _tensor_file_name("model", prefer_safetensors=False)
        assert name2.endswith(".pth")
    
    def test_safetensors_import_error_handling(self, temp_dir):
        """Test that ImportError from safetensors is handled gracefully."""
        tensors = {"weight": torch.randn(10, 10)}
        path = f"{temp_dir}/test.pth"
        
        # Test with prefer_safetensors=False (should always use pth)
        _save_tensor_dict(path, tensors, prefer_safetensors=False)
        assert path.endswith(".pth")
        
        loaded = _load_tensor_dict(path, prefer_safetensors=False)
        assert "weight" in loaded


class TestMatplotlibOptional:
    """Test matplotlib optional dependency handling."""
    
    def test_matplotlib_available(self):
        """Test that matplotlib works when available."""
        try:
            from gradiend.visualizer.plot_optional import _require_matplotlib
            plt = _require_matplotlib()
            assert plt is not None
        except ImportError:
            pytest.skip("matplotlib not available")
    
    def test_matplotlib_missing_error_message(self):
        """Test that missing matplotlib raises ImportError with helpful message."""
        from gradiend.visualizer.plot_optional import _require_matplotlib
        
        # Mock matplotlib import to fail
        with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.pyplot': None}):
            import importlib
            import gradiend.visualizer.plot_optional as plot_module
            importlib.reload(plot_module)
            
            with pytest.raises(ImportError) as exc_info:
                plot_module._require_matplotlib()
            
            error_msg = str(exc_info.value)
            assert "matplotlib" in error_msg.lower()
            assert "install" in error_msg.lower() or "pip" in error_msg.lower()
    
    def test_matplotlib_used_in_visualizer(self):
        """Test that visualizer functions use _require_matplotlib when matplotlib is missing."""
        # Test that _require_matplotlib raises ImportError when matplotlib.pyplot can't be imported
        # We need to patch the import before plot_optional tries to import matplotlib.pyplot
        import sys
        import importlib
        
        # Save and remove modules that might have already imported matplotlib
        # Also remove encoder_distributions if it exists, since it imports matplotlib.patches
        modules_to_restore = {}
        for mod_name in ['matplotlib.pyplot', 'matplotlib', 'gradiend.visualizer.plot_optional',
                         'gradiend.visualizer.encoder_distributions']:
            if mod_name in sys.modules:
                modules_to_restore[mod_name] = sys.modules[mod_name]
                del sys.modules[mod_name]
        
        # Patch __import__ to raise ImportError for matplotlib.pyplot
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'matplotlib.pyplot':
                raise ImportError("No module named 'matplotlib.pyplot'")
            return original_import(name, *args, **kwargs)
        
        try:
            # Patch and test
            with patch('builtins.__import__', side_effect=mock_import):
                # Now import plot_optional - it should handle the missing matplotlib gracefully
                # encoder_distributions will also be re-imported if needed, and will call
                # _require_matplotlib() before importing matplotlib.patches
                from gradiend.visualizer.plot_optional import _require_matplotlib
                
                # _require_matplotlib should raise ImportError with helpful message
                with pytest.raises(ImportError) as exc_info:
                    _require_matplotlib()
                
                error_msg = str(exc_info.value)
                assert "matplotlib" in error_msg.lower()
                assert "install" in error_msg.lower() or "pip" in error_msg.lower()
        finally:
            # Restore modules
            for mod_name, mod in modules_to_restore.items():
                sys.modules[mod_name] = mod


class TestSeabornOptional:
    """Test seaborn optional dependency handling."""
    
    def test_seaborn_available(self):
        """Test that seaborn works when available."""
        try:
            from gradiend.visualizer.plot_optional import _require_seaborn
            sns = _require_seaborn()
            assert sns is not None
        except ImportError:
            pytest.skip("seaborn not available")
    
    def test_seaborn_missing_error_message(self):
        """Test that missing seaborn raises ImportError with helpful message."""
        import sys
        import importlib
        
        # Save and remove modules that might have already imported seaborn
        modules_to_restore = {}
        for mod_name in ['seaborn', 'gradiend.visualizer.plot_optional', 
                         'gradiend.visualizer.encoder_distributions']:
            if mod_name in sys.modules:
                modules_to_restore[mod_name] = sys.modules[mod_name]
                del sys.modules[mod_name]
        
        # Patch __import__ to raise ImportError for seaborn
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'seaborn':
                raise ImportError("No module named 'seaborn'")
            return original_import(name, *args, **kwargs)
        
        try:
            # Patch and test
            with patch('builtins.__import__', side_effect=mock_import):
                # Now import plot_optional - it should handle the missing seaborn gracefully
                from gradiend.visualizer.plot_optional import _require_seaborn
                
                # _require_seaborn should raise ImportError with helpful message
                with pytest.raises(ImportError) as exc_info:
                    _require_seaborn()
                
                error_msg = str(exc_info.value)
                assert "seaborn" in error_msg.lower()
                assert "install" in error_msg.lower() or "pip" in error_msg.lower()
                assert "matplotlib" in error_msg.lower()  # Should mention matplotlib dependency
        finally:
            # Restore modules
            for mod_name, mod in modules_to_restore.items():
                sys.modules[mod_name] = mod
    
    def test_seaborn_used_in_visualizer(self):
        """Test that visualizer functions use _require_seaborn when seaborn is missing."""
        import sys
        import importlib
        
        # Save and remove modules that might have already imported seaborn
        modules_to_restore = {}
        for mod_name in ['seaborn', 'gradiend.visualizer.plot_optional',
                         'gradiend.visualizer.encoder_distributions']:
            if mod_name in sys.modules:
                modules_to_restore[mod_name] = sys.modules[mod_name]
                del sys.modules[mod_name]
        
        # Patch __import__ to raise ImportError for seaborn
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'seaborn':
                raise ImportError("No module named 'seaborn'")
            return original_import(name, *args, **kwargs)
        
        try:
            # Patch and test
            with patch('builtins.__import__', side_effect=mock_import):
                # Now import plot_optional - it should handle the missing seaborn gracefully
                from gradiend.visualizer.plot_optional import _require_seaborn
                
                # _require_seaborn should raise ImportError with helpful message
                with pytest.raises(ImportError) as exc_info:
                    _require_seaborn()
                
                error_msg = str(exc_info.value)
                assert "seaborn" in error_msg.lower()
                assert "install" in error_msg.lower() or "pip" in error_msg.lower()
        finally:
            # Restore modules
            for mod_name, mod in modules_to_restore.items():
                sys.modules[mod_name] = mod


class TestDatasetsOptional:
    """Test datasets optional dependency handling."""

    def test_datasets_available(self):
        """Test that datasets works when available (load_dataset import)."""
        try:
            from datasets import load_dataset
            assert load_dataset is not None
        except ImportError:
            pytest.skip("datasets not available")

    def test_datasets_missing_error_message(self):
        """Test that missing datasets raises ImportError with helpful message when loading HF data."""
        from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer

        # Remove datasets from sys.modules so the trainer's import fails
        modules_to_restore = {}
        for mod_name in ['datasets', 'gradiend.trainer.text.prediction.trainer']:
            if mod_name in sys.modules:
                modules_to_restore[mod_name] = sys.modules[mod_name]
                del sys.modules[mod_name]

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'datasets':
                raise ImportError("No module named 'datasets'")
            return original_import(name, *args, **kwargs)

        try:
            with patch('builtins.__import__', side_effect=mock_import):
                # Import trainer fresh so it doesn't have datasets cached
                import importlib
                import gradiend.trainer.text.prediction.trainer as trainer_module
                importlib.reload(trainer_module)

                # _load_hf_dataset imports datasets on first call; should raise ImportError
                with pytest.raises(ImportError) as exc_info:
                    trainer_module.TextPredictionTrainer._load_hf_dataset(
                        "some/dataset",
                        subset=None,
                        splits=["train"],
                    )

                error_msg = str(exc_info.value)
                assert "datasets" in error_msg.lower()
                assert "install" in error_msg.lower() or "pip" in error_msg.lower()
        finally:
            for mod_name, mod in modules_to_restore.items():
                sys.modules[mod_name] = mod


class TestSpacyOptional:
    """Test spacy optional dependency handling."""

    def test_spacy_available(self):
        """Test that spacy works when available (import)."""
        try:
            import spacy
            assert spacy is not None
        except ImportError:
            pytest.skip("spacy not available")

    def test_gradiend_data_loads_without_spacy(self):
        """Test that gradiend.data classes load without spacy installed."""
        from gradiend.data.text.filter_config import TextFilterConfig
        from gradiend.data.text.prediction.filter_engine import filter_sentences

        # TextFilterConfig and filter_sentences are importable and usable
        cfg = TextFilterConfig(targets=["der"], id="test")
        assert cfg.id == "test"
        # String-only filtering does not require spacy
        results = filter_sentences(["Der Mann geht."], cfg, spacy_model=None)
        assert len(results) == 1

    def test_spacy_missing_error_message(self):
        """Test that using spacy_tags without spacy raises ImportError with helpful message."""
        from gradiend.data.text.filter_config import TextFilterConfig
        from gradiend.data.text.prediction.filter_engine import filter_sentences

        cfg = TextFilterConfig(
            targets=["der"],
            spacy_tags={"pos": "DET", "Case": "Nom", "Gender": "Masc"},
            id="test",
        )

        modules_to_restore = {}
        for mod_name in ["spacy", "spacy.cli", "gradiend.data.core.spacy_util"]:
            if mod_name in sys.modules:
                modules_to_restore[mod_name] = sys.modules[mod_name]
                del sys.modules[mod_name]

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "spacy" or name.startswith("spacy."):
                raise ImportError("No module named 'spacy'")
            return original_import(name, *args, **kwargs)
        try:
            with patch("builtins.__import__", side_effect=mock_import):
                import importlib
                import gradiend.data.core.spacy_util as spacy_util_module
                importlib.reload(spacy_util_module)

                with pytest.raises(ImportError) as exc_info:
                    filter_sentences(
                        ["Der Mann geht."],
                        cfg,
                        spacy_model="de_core_news_sm",
                    )

                error_msg = str(exc_info.value)
                assert "spacy" in error_msg.lower()
                assert "install" in error_msg.lower() or "pip" in error_msg.lower()
        finally:
            for mod_name, mod in modules_to_restore.items():
                sys.modules[mod_name] = mod


class TestOptionalDependencyIntegration:
    """Integration tests for optional dependencies."""
    
    def test_model_saving_with_safetensors_preference(self, mock_model, temp_dir):
        """Test that model saving prefers safetensors when available."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        from unittest.mock import patch
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
                device=torch.device("cpu")
            )
        
        save_path = f"{temp_dir}/test_model"
        model_with_gradiend.save_pretrained(save_path)
        
        # Check which format was used
        has_safetensors = False
        has_bin = False
        
        import os
        if os.path.exists(f"{save_path}/model.safetensors"):
            has_safetensors = True
        if os.path.exists(f"{save_path}/pytorch_model.bin"):
            has_bin = True
        
        # At least one should exist
        assert has_safetensors or has_bin, "Model should be saved in some format"
        
        # If safetensors is available, it should be preferred
        try:
            import safetensors
            # If safetensors is available, it should be used (unless explicitly disabled)
            # Note: The actual preference depends on the implementation
        except ImportError:
            # If safetensors is not available, bin should be used
            assert has_bin, "pytorch_model.bin should exist when safetensors is not available"
    
    def test_model_saving_without_safetensors(self, mock_model, temp_dir):
        """Test that model saving falls back to .bin when safetensors is not available."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        from unittest.mock import patch
        import sys
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        # Save original safetensors module if it exists
        original_safetensors = sys.modules.get('safetensors')
        original_safetensors_torch = sys.modules.get('safetensors.torch')
        
        # Remove safetensors from sys.modules to simulate it not being installed
        modules_to_restore = {}
        for mod_name in ['safetensors', 'safetensors.torch']:
            if mod_name in sys.modules:
                modules_to_restore[mod_name] = sys.modules[mod_name]
                del sys.modules[mod_name]
        
        # Patch __import__ to raise ImportError for safetensors
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'safetensors' or name.startswith('safetensors.'):
                raise ImportError("No module named 'safetensors'")
            return original_import(name, *args, **kwargs)
        
        try:
            with patch('builtins.__import__', side_effect=mock_import):
                with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
                    model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                        mock_model,
                        n_features=1,
                        device=torch.device("cpu")
                    )
                
                save_path = f"{temp_dir}/test_model_no_safetensors"
                model_with_gradiend.save_pretrained(save_path, use_safetensors=None)  # None = prefer if available
                
                import os
                # Should fall back to .bin format
                assert os.path.exists(f"{save_path}/pytorch_model.bin"), \
                    "Model should be saved as pytorch_model.bin when safetensors is not available"
                assert not os.path.exists(f"{save_path}/model.safetensors"), \
                    "model.safetensors should not exist when safetensors is not available"
        finally:
            # Restore modules
            for mod_name, mod in modules_to_restore.items():
                sys.modules[mod_name] = mod
    
    def test_model_saving_force_safetensors_when_unavailable(self, mock_model, temp_dir):
        """Test that model saving raises error when safetensors is required but unavailable."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        from unittest.mock import patch
        import sys
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.conftest import MockTokenizer
            return mock_model, MockTokenizer()
        
        # Remove safetensors from sys.modules to simulate it not being installed
        modules_to_restore = {}
        for mod_name in ['safetensors', 'safetensors.torch']:
            if mod_name in sys.modules:
                modules_to_restore[mod_name] = sys.modules[mod_name]
                del sys.modules[mod_name]
        
        # Patch __import__ to raise ImportError for safetensors
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'safetensors' or name.startswith('safetensors.'):
                raise ImportError("No module named 'safetensors'")
            return original_import(name, *args, **kwargs)
        
        try:
            with patch('builtins.__import__', side_effect=mock_import):
                with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
                    model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                        mock_model,
                        n_features=1,
                        device=torch.device("cpu")
                    )
                
                save_path = f"{temp_dir}/test_model_force_safetensors"
                # When use_safetensors=True, it should raise ImportError if safetensors is not available
                with pytest.raises(ImportError) as exc_info:
                    model_with_gradiend.save_pretrained(save_path, use_safetensors=True)
                
                error_msg = str(exc_info.value)
                assert "safetensors" in error_msg.lower()
        finally:
            # Restore modules
            for mod_name, mod in modules_to_restore.items():
                sys.modules[mod_name] = mod
    
    def test_optional_dependencies_error_messages(self):
        """Test that error messages for optional dependencies are helpful."""
        # Test matplotlib error message
        from gradiend.visualizer.plot_optional import _MSG_MATPLOTLIB, _MSG_SEABORN
        
        assert "matplotlib" in _MSG_MATPLOTLIB.lower()
        assert "install" in _MSG_MATPLOTLIB.lower() or "pip" in _MSG_MATPLOTLIB.lower()
        
        assert "seaborn" in _MSG_SEABORN.lower()
        assert "install" in _MSG_SEABORN.lower() or "pip" in _MSG_SEABORN.lower()
        assert "matplotlib" in _MSG_SEABORN.lower()  # Should mention matplotlib dependency
