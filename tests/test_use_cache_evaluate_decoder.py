"""Test that evaluate_decoder works after training is skipped due to use_cache=True."""

import json
import os
import tempfile

import pandas as pd
import pytest

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.util.paths import has_saved_model, resolve_output_path, ARTIFACT_MODEL


def _per_class_dict():
    """Minimal per-class data for trainer (train + test splits for decoder eval)."""
    return {
        "3SG": pd.DataFrame({
            "masked": ["[MASK] went home", "[MASK] went home"],
            "split": ["train", "test"],
            "3SG": ["he", "he"],
        }),
        "3PL": pd.DataFrame({
            "masked": ["[MASK] went home", "[MASK] went home"],
            "split": ["train", "test"],
            "3PL": ["they", "they"],
        }),
    }


def _create_stub_saved_model(output_dir: str) -> None:
    """Create minimal files so has_saved_model(output_dir) returns True."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({"model_type": "gradiend"}, f)
    # pytorch_model.bin: tiny 4-byte file suffices for has_saved_model check
    weights_path = os.path.join(output_dir, "pytorch_model.bin")
    with open(weights_path, "wb") as f:
        f.write(b"\x00\x00\x00\x00")


class TestEvaluateDecoderAfterUseCacheSkip:
    """evaluate_decoder should work when training was skipped due to use_cache=True."""

    def test_target_classes_loaded_after_use_cache_skip(self, tmp_path):
        """When train() skips due to use_cache, _ensure_data_for_training runs so target_classes are available."""
        experiment_dir = str(tmp_path)
        output_dir = resolve_output_path(experiment_dir, None, ARTIFACT_MODEL)
        assert output_dir is not None
        _create_stub_saved_model(output_dir)
        assert has_saved_model(output_dir)

        args = TrainingArguments(
            experiment_dir=experiment_dir,
            use_cache=True,
            train_batch_size=2,
            max_steps=2,
        )
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            args=args,
            use_class_names_as_columns=True,
        )

        # Train should skip (use_cache + saved model exists)
        result = trainer.train()
        assert trainer._last_train_used_cache is True

        # target_classes must be set (from _ensure_data_for_training)
        assert trainer.target_classes is not None
        assert set(trainer.target_classes) == {"3SG", "3PL"}

    def test_default_decoder_feature_factors_has_classes_after_use_cache_skip(self, tmp_path):
        """default_decoder_feature_factors gets classes from trainer (no ValueError) when train() was skipped by use_cache."""
        experiment_dir = str(tmp_path)
        output_dir = resolve_output_path(experiment_dir, None, ARTIFACT_MODEL)
        assert output_dir is not None
        _create_stub_saved_model(output_dir)
        assert has_saved_model(output_dir)

        args = TrainingArguments(
            experiment_dir=experiment_dir,
            use_cache=True,
            train_batch_size=2,
            max_steps=2,
        )
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            args=args,
            use_class_names_as_columns=True,
        )

        trainer.train()
        assert trainer._last_train_used_cache is True

        # Before fix: default_decoder_feature_factors would raise "classes must be provided"
        # because target_classes was None when training was skipped.
        from gradiend.evaluator.decoder import default_decoder_feature_factors
        from unittest.mock import MagicMock

        mock_model = MagicMock()
        mock_model.feature_class_encoding_direction = {"3SG": 1.0, "3PL": -1.0}
        factors = default_decoder_feature_factors(trainer, model_with_gradiend=mock_model)
        assert factors is not None
        assert len(factors) == 2  # one per target class
