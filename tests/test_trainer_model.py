"""
Tests for Trainer model handling: get_model returns in-memory model during training,
base_model_path vs model_path, and require_gradiend_model flag.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch

from gradiend.trainer import Trainer
from gradiend.util.paths import resolve_decoder_stats_path
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel, GradiendModel
from tests.conftest import SimpleMockModel


def _make_param_map_spec():
    """Param map spec for SimpleMockModel encoder.0.0.weight has shape (64, 64)."""
    return {"encoder.0.0.weight": {"shape": (64, 64), "repr": "all"}}


class MockModelWithGradiendForTest(ModelWithGradiend):
    """Minimal ModelWithGradiend subclass for testing."""

    def _save_model(self, save_directory, **kwargs):
        pass

    def create_gradients(self, *args, **kwargs):
        return torch.randn(64)

    @classmethod
    def _load_model(cls, load_directory, base_model_id=None, gradiend_kwargs=None, **kwargs):
        base = SimpleMockModel(name_or_path=base_model_id or load_directory)
        return (base,)

    @classmethod
    def _create_gradiend(cls, base_model, load_directory, **kwargs):
        return ParamMappedGradiendModel(
            input_dim=64,
            latent_dim=1,
            param_map=_make_param_map_spec(),
        )


class MockTrainerForTest(Trainer):
    """Concrete Trainer for testing model handling."""

    @property
    def model_with_gradiend_cls(self):
        return MockModelWithGradiendForTest

    @property
    def default_model_with_gradiend_cls(self):
        return MockModelWithGradiendForTest

    def create_training_data(self, *args, **kwargs):
        from gradiend.trainer.core.dataset import GradientTrainingDataset

        base = SimpleMockModel()
        gradiend = GradiendModel(input_dim=64, latent_dim=1)
        m = MockModelWithGradiendForTest(base, gradiend)
        data = [{"factual": torch.randn(10), "alternative": torch.randn(10), "label": 1.0}] * 4

        def gc(x):
            return torch.randn(64)

        return GradientTrainingDataset(data, gc, source="factual", target="diff")

    def create_gradient_training_dataset(self, raw, model_with_gradiend, **kwargs):
        return self.create_training_data()

    def _get_decoder_eval_dataframe(self, tokenizer, **kwargs):
        import pandas as pd

        return pd.DataFrame({"text": ["a"], "label": ["x"], "factual_id": [1], "alternative_id": [2]}), pd.DataFrame()

    def _get_decoder_eval_targets(self):
        return {"x": ["x"]}

    def evaluate_base_model(self, model, tokenizer, **kwargs):
        return {"lms": {"lms": 0.5}, "x": 0.5}

    def _analyze_encoder(self, model_with_gradiend=None, **kwargs):
        import pandas as pd

        return pd.DataFrame({
            "encoded": [0.1, -0.2, 0.3],
            "label": [1.0, -1.0, 1.0],
            "source_id": ["a", "b", "a"],
            "target_id": ["b", "a", "b"],
            "type": ["training", "training", "training"],
        })


class TestGetModelDuringTraining:
    """Test that get_model returns in-memory model during training."""

    def test_get_model_returns_model_instance_during_training(self, temp_dir):
        """During training, evaluate_encoder should use the in-memory model, not load a new one."""
        args = TrainingArguments(
            experiment_dir=temp_dir,
            train_batch_size=2,
            num_train_epochs=1,
            max_steps=2,
            eval_steps=1,
            do_eval=True,
        )
        trainer = MockTrainerForTest(
            model="mock-base",
            args=args,
        )

        # Simulate training: _train sets _model_instance
        mock_model = MagicMock()
        mock_model.name_or_path = "mock-model"
        trainer._model_instance = mock_model
        trainer._model_arg = "mock-base"

        # get_model without load_directory should return in-memory model
        got = trainer.get_model(use_cache=False)
        assert got is mock_model, "get_model should return _model_instance when set (no load_directory)"

    def test_get_model_loads_from_load_directory_when_passed(self, temp_dir):
        """When load_directory is explicitly passed, get_model loads from that path (not _model_instance)."""
        args = TrainingArguments(experiment_dir=temp_dir)
        trainer = MockTrainerForTest(model="mock-base", args=args)
        trainer._model_instance = MagicMock()

        mock_model = MagicMock()
        # Patch where Trainer looks up create_model_with_gradiend (super() in get_model)
        with patch(
            "gradiend.trainer.trainer.FeatureLearningDefinition.create_model_with_gradiend",
            return_value=mock_model,
        ) as mock_create:
            load_path = os.path.join(temp_dir, "checkpoint")
            got = trainer.get_model(load_directory=load_path)
            assert got is mock_model
            assert mock_create.called
            # load_directory may be first or second positional (after self) depending on patch binding
            pos = mock_create.call_args[0]
            kw = mock_create.call_args[1]
            load_dir_arg = (pos[0] if len(pos) > 0 else None) or (pos[1] if len(pos) > 1 else None) or kw.get("load_directory")
            assert load_dir_arg is not None, f"load_directory should be passed: call_args={mock_create.call_args}"
            assert load_path in str(load_dir_arg) or os.path.normpath(load_path) in str(load_dir_arg)


class TestBaseModelPathVsModelPath:
    """Test base_model_path and model_path naming."""

    def test_base_model_path_unchanged_after_training(self, temp_dir):
        """base_model_path returns original model; model_path changes after train()."""
        args = TrainingArguments(experiment_dir=temp_dir)
        trainer = MockTrainerForTest(model="bert-base-cased", args=args)

        assert trainer.base_model_path == "bert-base-cased"
        assert trainer.model_path == "bert-base-cased"

        # Simulate post-training: _model_arg updated to output dir
        out_path = os.path.join(temp_dir, "runs", "model")
        trainer._model_arg = out_path

        assert trainer.base_model_path == "bert-base-cased", "base_model_path should never change"
        assert trainer.model_path == out_path, "model_path should reflect current path"


class TestRequireGradiendModel:
    """Test require_gradiend_model flag in from_pretrained."""

    def test_require_gradiend_model_raises_when_base_model_path(self, temp_dir):
        """When require_gradiend_model=True and path is a base model (e.g. bert-base-cased), raise FileNotFoundError."""
        args = TrainingArguments(experiment_dir=temp_dir)
        trainer = MockTrainerForTest(model="mock-base", args=args)

        # load_model expects a GRADIEND checkpoint; a base model path should fail
        with pytest.raises(FileNotFoundError) as exc_info:
            trainer.load_model("/nonexistent/not-a-gradiend-checkpoint")

        assert "GRADIEND checkpoint" in str(exc_info.value) or "require_gradiend_model" in str(exc_info.value)


class TestSelectAndSaveChangedModel:
    """Test rewrite_base_model behavior."""

    def _decoder_results_with_class_keys(self):
        """Decoder results in typical evaluate_decoder format (<class_id> keys)."""
        return {
            "summary": {
                "masc_nom": {"feature_factor": 1.0, "learning_rate": 1e-4, "value": 0.8},
                "fem_nom": {"feature_factor": -1.0, "learning_rate": 1e-4, "value": 0.7},
            },
            "grid": {},
        }

    def test_rewrite_base_model_accepts_target_class_as_class_id(self):
        """target_class as class id (e.g. 'masc_nom') should match summary key directly."""
        args = TrainingArguments(experiment_dir=None)
        trainer = MockTrainerForTest(model="mock-base", args=args)
        mock_model = MagicMock()
        mock_model.rewrite_base_model = MagicMock(return_value=MagicMock())
        trainer._model_instance = mock_model
        trainer._model_arg = "mock-base"

        decoder_results = self._decoder_results_with_class_keys()

        changed = trainer.rewrite_base_model(
            decoder_results=decoder_results,
            target_class="masc_nom",
        )

        assert changed is not None
        mock_model.rewrite_base_model.assert_called_once_with(
            learning_rate=1e-4,
            feature_factor=1.0,
        )

    def test_rewrite_base_model_accepts_target_class_as_class_id_for_fem_nom(self):
        """target_class 'fem_nom' should match summary key directly."""
        args = TrainingArguments(experiment_dir=None)
        trainer = MockTrainerForTest(model="mock-base", args=args)
        mock_model = MagicMock()
        mock_model.rewrite_base_model = MagicMock(return_value=MagicMock())
        trainer._model_instance = mock_model
        trainer._model_arg = "mock-base"

        decoder_results = self._decoder_results_with_class_keys()

        changed = trainer.rewrite_base_model(
            decoder_results=decoder_results,
            target_class="fem_nom",
        )

        assert changed is not None
        mock_model.rewrite_base_model.assert_called_once_with(
            learning_rate=1e-4,
            feature_factor=-1.0,
        )

    def test_rewrite_base_model_target_class_list_accepts_multiple_class_ids(self):
        """target_class as list can specify multiple class ids."""
        args = TrainingArguments(experiment_dir=None)
        trainer = MockTrainerForTest(model="mock-base", args=args)
        mock_model = MagicMock()
        mock_model.rewrite_base_model = MagicMock(side_effect=lambda **kw: MagicMock())
        trainer._model_instance = mock_model
        trainer._model_arg = "mock-base"

        decoder_results = self._decoder_results_with_class_keys()

        changed_list = trainer.rewrite_base_model(
            decoder_results=decoder_results,
            target_class=["masc_nom", "fem_nom"],
        )

        assert len(changed_list) == 2
        assert mock_model.rewrite_base_model.call_count == 2
        calls = mock_model.rewrite_base_model.call_args_list
        assert calls[0][1]["feature_factor"] == 1.0  # masc_nom
        assert calls[1][1]["feature_factor"] == -1.0  # fem_nom

    def test_rewrite_base_model_loads_from_cache_when_decoder_results_omitted(self, tmp_path):
        """When decoder_results=None and decoder stats cache exists, load from disk."""
        args = TrainingArguments(experiment_dir=str(tmp_path))
        trainer = MockTrainerForTest(model="mock-base", args=args)
        mock_model = MagicMock()
        mock_model.rewrite_base_model = MagicMock(return_value=MagicMock())
        trainer._model_instance = mock_model
        trainer._model_arg = "mock-base"

        # Write decoder_stats cache (as evaluate_decoder with use_cache=True would populate)
        stats_content = {
            "summary": {
                "masc_nom": {"feature_factor": 1.0, "learning_rate": 1e-4, "value": 0.8},
            },
            "grid": [],
        }
        stats_path = resolve_decoder_stats_path(
            str(tmp_path),
            metric_name="masc_nom",
        )
        assert stats_path is not None
        with open(stats_path, "w") as f:
            json.dump(stats_content, f, indent=2)

        changed = trainer.rewrite_base_model(
            decoder_results=None,
            target_class="masc_nom",
        )

        assert changed is not None
        mock_model.rewrite_base_model.assert_called_once_with(
            learning_rate=1e-4,
            feature_factor=1.0,
        )

    def test_rewrite_base_model_raises_when_decoder_results_omitted_and_no_cache(self, tmp_path):
        """When decoder_results=None and no decoder stats cache exists, raise."""
        args = TrainingArguments(experiment_dir=str(tmp_path))
        trainer = MockTrainerForTest(model="mock-base", args=args)
        trainer._model_instance = MagicMock()
        trainer._model_arg = "mock-base"

        with pytest.raises(ValueError) as exc_info:
            trainer.rewrite_base_model(
                decoder_results=None,
                target_class="masc_nom",
            )

        assert "No decoder results cache found" in str(exc_info.value)
        assert "evaluate_decoder" in str(exc_info.value)

    def test_rewrite_base_model_raises_without_save_path_when_output_dir_provided(self):
        """When experiment_dir is not set and output_dir is empty, rewrite_base_model raises when trying to save."""
        args = TrainingArguments(experiment_dir=None)
        trainer = MockTrainerForTest(model="mock-base", args=args)
        trainer._model_instance = MagicMock()

        decoder_results = {
            "summary": {
                "x": {"feature_factor": 0.5, "learning_rate": 1e-4},
            }
        }

        with pytest.raises(ValueError) as exc_info:
            trainer.rewrite_base_model(
                decoder_results=decoder_results,
                target_class="x",
                output_dir="",  # Empty string should trigger save path check
            )

        assert "Cannot save" in str(exc_info.value) or "no output path" in str(exc_info.value)
        assert "experiment_dir" in str(exc_info.value) or "output_dir" in str(exc_info.value)

    def test_rewrite_base_model_saves_when_output_dir_provided(self, tmp_path):
        """When output_dir is provided, rewrite_base_model saves models and returns paths."""
        args = TrainingArguments(experiment_dir=str(tmp_path))
        trainer = MockTrainerForTest(model="mock-base", args=args)
        
        # Create a mock model that can be saved
        mock_model = MagicMock()
        mock_rewritten = MagicMock()
        mock_model.rewrite_base_model = MagicMock(return_value=mock_rewritten)
        trainer._model_instance = mock_model
        trainer._model_arg = "mock-base"

        decoder_results = {
            "summary": {
                "masc_nom": {"feature_factor": 1.0, "learning_rate": 1e-4, "value": 0.8},
            },
            "grid": {},
        }

        output_dir = os.path.join(tmp_path, "saved_model")
        os.makedirs(output_dir, exist_ok=True)

        result = trainer.rewrite_base_model(
            decoder_results=decoder_results,
            target_class="masc_nom",
            output_dir=output_dir,
        )

        # Should return a path (string), not a model
        assert isinstance(result, str)
        assert os.path.exists(result)
        mock_model.rewrite_base_model.assert_called_once_with(
            learning_rate=1e-4,
            feature_factor=1.0,
        )
        # Verify save_pretrained was called
        assert mock_rewritten.save_pretrained.called

    def test_rewrite_base_model_returns_model_when_output_dir_not_provided(self):
        """When output_dir is not provided, rewrite_base_model returns models in memory."""
        args = TrainingArguments(experiment_dir=None)
        trainer = MockTrainerForTest(model="mock-base", args=args)
        mock_model = MagicMock()
        mock_rewritten = MagicMock()
        mock_model.rewrite_base_model = MagicMock(return_value=mock_rewritten)
        trainer._model_instance = mock_model
        trainer._model_arg = "mock-base"

        decoder_results = {
            "summary": {
                "masc_nom": {"feature_factor": 1.0, "learning_rate": 1e-4, "value": 0.8},
            },
            "grid": {},
        }

        result = trainer.rewrite_base_model(
            decoder_results=decoder_results,
            target_class="masc_nom",
            # No output_dir
        )

        # Should return a model, not a path
        assert result is mock_rewritten
        mock_model.rewrite_base_model.assert_called_once_with(
            learning_rate=1e-4,
            feature_factor=1.0,
        )
        # save_pretrained should not be called when output_dir is not provided
        if hasattr(mock_rewritten, 'save_pretrained'):
            assert not mock_rewritten.save_pretrained.called

    def test_rewrite_base_model_saves_multiple_models_with_experiment_dir(self, tmp_path):
        """When multiple target classes and experiment_dir, rewrite_base_model saves all models."""
        args = TrainingArguments(experiment_dir=str(tmp_path))
        trainer = MockTrainerForTest(model="mock-base", args=args)
        
        mock_model = MagicMock()
        mock_rewritten1 = MagicMock()
        mock_rewritten2 = MagicMock()
        mock_model.rewrite_base_model = MagicMock(side_effect=[mock_rewritten1, mock_rewritten2])
        trainer._model_instance = mock_model
        trainer._model_arg = "mock-base"

        decoder_results = self._decoder_results_with_class_keys()

        result = trainer.rewrite_base_model(
            decoder_results=decoder_results,
            target_class=["masc_nom", "fem_nom"],
            output_dir=str(tmp_path),  # With experiment_dir, this is used as parent
        )

        # Should return list of paths
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(p, str) for p in result)
        assert mock_model.rewrite_base_model.call_count == 2
        assert mock_rewritten1.save_pretrained.called
        assert mock_rewritten2.save_pretrained.called

    def test_rewrite_base_model_raises_when_multiple_keys_without_experiment_dir(self):
        """When multiple target classes without experiment_dir, rewrite_base_model raises."""
        args = TrainingArguments(experiment_dir=None)
        trainer = MockTrainerForTest(model="mock-base", args=args)
        trainer._model_instance = MagicMock()

        decoder_results = self._decoder_results_with_class_keys()

        with pytest.raises(ValueError) as exc_info:
            trainer.rewrite_base_model(
                decoder_results=decoder_results,
                target_class=["masc_nom", "fem_nom"],
                output_dir="./output",  # Single output_dir not enough for multiple keys
            )

        assert "multiple" in str(exc_info.value).lower() or "experiment_dir" in str(exc_info.value)

    def test_rewrite_base_model_returns_single_path_for_single_key_with_output_dir(self, tmp_path):
        """When single target_class with output_dir, rewrite_base_model returns single path string."""
        args = TrainingArguments(experiment_dir=str(tmp_path))
        trainer = MockTrainerForTest(model="mock-base", args=args)
        
        mock_model = MagicMock()
        mock_rewritten = MagicMock()
        mock_model.rewrite_base_model = MagicMock(return_value=mock_rewritten)
        trainer._model_instance = mock_model
        trainer._model_arg = "mock-base"

        decoder_results = {
            "summary": {
                "masc_nom": {"feature_factor": 1.0, "learning_rate": 1e-4, "value": 0.8},
            },
            "grid": {},
        }

        output_dir = os.path.join(tmp_path, "saved_model")
        os.makedirs(output_dir, exist_ok=True)

        result = trainer.rewrite_base_model(
            decoder_results=decoder_results,
            target_class="masc_nom",  # Single key
            output_dir=output_dir,
        )

        # Should return a single string path, not a list
        assert isinstance(result, str)
        assert not isinstance(result, list)
