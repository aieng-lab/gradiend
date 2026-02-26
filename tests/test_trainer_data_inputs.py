"""Tests for TextPredictionTrainer data inputs: Path, DataFrame, dict, eval_neutral_data path."""

from pathlib import Path

import pandas as pd
import pytest

from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from gradiend.trainer.text.prediction.unified_data import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_FACTUAL_CLASS,
    resolve_dataframe,
)


def _merged_factual_df(rows=None):
    """Minimal merged (factual-only) DataFrame: masked, split, label_class, label."""
    if rows is None:
        rows = [
            {"masked": "[MASK] here", "split": "train", "label_class": "3SG", "label": "he"},
            {"masked": "[MASK] there", "split": "train", "label_class": "3PL", "label": "they"},
        ]
    return pd.DataFrame(rows)


def _merged_with_alternative_df():
    """Merged DataFrame with explicit alternative columns (label_class, label, alternative, alternative_class)."""
    return pd.DataFrame([
        {"masked": "[MASK] here", "split": "train", "label_class": "3SG", "label": "he", "alternative_class": "3PL", "alternative": "they"},
        {"masked": "[MASK] there", "split": "train", "label_class": "3PL", "label": "they", "alternative_class": "3SG", "alternative": "he"},
    ])


def _per_class_dict():
    """Per-class dict compatible with trainer."""
    return {
        "3SG": pd.DataFrame({
            "masked": ["[MASK] here"],
            "split": ["train"],
            "3SG": ["he"],
        }),
        "3PL": pd.DataFrame({
            "masked": ["[MASK] there"],
            "split": ["train"],
            "3PL": ["they"],
        }),
    }


class TestTrainerDataAsPath:
    """data=Path(...) or data=str path loads CSV/Parquet and builds unified data."""

    def test_data_as_path_loads_csv(self, tmp_path):
        path = tmp_path / "training.csv"
        _merged_factual_df().to_csv(path, index=False)
        config = TextPredictionConfig(
            data=path,
            target_classes=["3SG", "3PL"],
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert UNIFIED_FACTUAL_CLASS in trainer._combined_data.columns
        assert UNIFIED_ALTERNATIVE_CLASS in trainer._combined_data.columns
        assert UNIFIED_FACTUAL in trainer._combined_data.columns
        assert UNIFIED_ALTERNATIVE in trainer._combined_data.columns
        assert len(trainer._combined_data) == 2

    def test_data_as_str_path_loads_csv(self, tmp_path):
        path = str(tmp_path / "training.csv")
        _merged_factual_df().to_csv(path, index=False)
        config = TextPredictionConfig(data=path, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert len(trainer._combined_data) >= 1


class TestTrainerDataAsDataFrame:
    """data=DataFrame (merged) builds unified data."""

    def test_data_as_dataframe_merged(self):
        df = _merged_factual_df()
        config = TextPredictionConfig(data=df, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert UNIFIED_FACTUAL in trainer._combined_data.columns
        assert UNIFIED_ALTERNATIVE in trainer._combined_data.columns


class TestTrainerDataAsDict:
    """data=dict (per-class) builds merged then unified."""

    def test_data_as_per_class_dict(self):
        class_dfs = _per_class_dict()
        config = TextPredictionConfig(
            data=class_dfs,
            target_classes=["3SG", "3PL"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert set(trainer._all_classes) == {"3SG", "3PL"}
        assert UNIFIED_FACTUAL_CLASS in trainer._combined_data.columns

    def test_target_classes_not_in_data_raises_early(self):
        """When target_classes include a class not in the data, raise a clear ValueError before building unified data."""
        class_dfs = _per_class_dict()
        config = TextPredictionConfig(
            data=class_dfs,
            target_classes=["3SG", "NonExistent"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(ValueError) as exc_info:
            trainer._ensure_data()
        assert "target_classes" in str(exc_info.value)
        assert "NonExistent" in str(exc_info.value)
        assert "not present" in str(exc_info.value).lower() or "not in" in str(exc_info.value).lower()
        assert "3SG" in str(exc_info.value) or "3PL" in str(exc_info.value)

    def test_per_class_dict_without_target_classes_infers_from_data_and_run_id_none(self):
        """With data=dict and no explicit target_classes or run_id, target_classes are inferred and decoder eval targets can be inferred."""
        class_dfs = _per_class_dict()
        config = TextPredictionConfig(
            data=class_dfs,
            run_id=None,
            use_class_names_as_columns=True,
        )
        assert config.target_classes is None
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        assert trainer.run_id is None
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert trainer.target_classes is not None
        assert set(trainer.target_classes) == {"3SG", "3PL"}
        targets = trainer._infer_decoder_eval_targets()
        assert targets
        assert "3SG" in targets and "3PL" in targets
        assert isinstance(targets["3SG"], list) and isinstance(targets["3PL"], list)
        assert len(targets["3SG"]) >= 1 and len(targets["3PL"]) >= 1


class TestEvalNeutralDataAsPath:
    """eval_neutral_data=Path or str path is resolved by resolve_dataframe."""

    def test_eval_neutral_data_path_resolved(self, tmp_path):
        neutral_path = tmp_path / "neutral.csv"
        pd.DataFrame({"text": ["Neutral sentence one.", "Another neutral."]}).to_csv(neutral_path, index=False)
        config = TextPredictionConfig(
            data=_merged_factual_df(),
            target_classes=["3SG", "3PL"],
            eval_neutral_data=neutral_path,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        resolved = resolve_dataframe(config.eval_neutral_data)
        assert resolved is not None and "text" in resolved.columns and len(resolved) == 2


class TestStandardPipelinePerDataFormat:
    """Standard pipeline (ensure_data → create_training_data) works for each data input format."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Load tokenizer once per class to avoid repeated from_pretrained (~10s each)."""
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def test_standard_pipeline_per_class_dict(self, tokenizer):
        """Per-class dict: trainer → ensure_data → create_training_data yields non-empty dataset."""
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1

    def test_standard_pipeline_merged_dataframe_pair_mode(self, tokenizer):
        """Merged DataFrame (factual-only, pair): pipeline works."""
        config = TextPredictionConfig(
            data=_merged_factual_df(),
            target_classes=["3SG", "3PL"],
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1

    def test_standard_pipeline_merged_dataframe_with_alternative_cols(self, tokenizer):
        """Merged DataFrame with alternative/alternative_class columns: pipeline works."""
        config = TextPredictionConfig(
            data=_merged_with_alternative_df(),
            target_classes=["3SG", "3PL"],
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1

    def test_standard_pipeline_csv_path(self, tokenizer, tmp_path):
        """Data as CSV path (merged format with pair): pipeline works."""
        path = tmp_path / "train.csv"
        _merged_factual_df().to_csv(path, index=False)
        config = TextPredictionConfig(
            data=path,
            target_classes=["3SG", "3PL"],
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1

    def test_standard_pipeline_parquet_path(self, tokenizer, tmp_path):
        """Data as Parquet path (merged format with pair): pipeline works."""
        pytest.importorskip("pyarrow")
        path = tmp_path / "train.parquet"
        _merged_factual_df().to_parquet(path, index=False)
        config = TextPredictionConfig(
            data=path,
            target_classes=["3SG", "3PL"],
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1

    def test_standard_pipeline_per_class_dict_no_explicit_target_classes(self, tokenizer):
        """Per-class dict without target_classes (inferred from data): pipeline works."""
        config = TextPredictionConfig(
            data=_per_class_dict(),
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer.target_classes is not None
        assert set(trainer.target_classes) == {"3SG", "3PL"}
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1
