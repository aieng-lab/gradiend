"""Tests for TextPredictionTrainer data inputs: Path, DataFrame, dict, eval_neutral_data path."""

from pathlib import Path

import pandas as pd
import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from gradiend.trainer.core.unified_data import (
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


def _per_class_dict_three_classes():
    """Per-class dict with three classes (3SG, 3PL, Other) for identity-for-other tests."""
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
        "Other": pd.DataFrame({
            "masked": ["[MASK] person"],
            "split": ["train"],
            "Other": ["they"],
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

    def test_data_as_directory_loads_training_csv(self, tmp_path):
        _merged_factual_df().to_csv(tmp_path / "training.csv", index=False)
        config = TextPredictionConfig(data=tmp_path, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert len(trainer._combined_data) == 2

    def test_missing_data_path_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "training.csv"
        config = TextPredictionConfig(data=missing, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            trainer._ensure_data()

    def test_missing_path_does_not_reach_merged_to_unified(self, tmp_path):
        """Regression: Path must not be passed to merged_to_unified (no AttributeError on .copy)."""
        missing = tmp_path / "training.csv"
        config = TextPredictionConfig(data=missing, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(FileNotFoundError) as exc_info:
            trainer._ensure_data()
        assert "copy" not in str(exc_info.value)
        assert "PosixPath" not in str(exc_info.value)
        assert "WindowsPath" not in str(exc_info.value)

    def test_empty_directory_raises_file_not_found(self, tmp_path):
        config = TextPredictionConfig(data=tmp_path, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(FileNotFoundError, match=r"training\.csv or training\.parquet"):
            trainer._ensure_data()

    def test_data_as_directory_parquet(self, tmp_path):
        pytest.importorskip("pyarrow")
        _merged_factual_df().to_parquet(tmp_path / "training.parquet", index=False)
        config = TextPredictionConfig(data=tmp_path, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert len(trainer._combined_data) == 2

    def test_invalid_data_type_raises_type_error(self):
        config = TextPredictionConfig(data=[1, 2, 3], target_classes=["3SG", "3PL"])  # type: ignore[arg-type]
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(TypeError, match="config.data"):
            trainer._ensure_data()


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
        targets, has_overlap = trainer._infer_decoder_eval_targets()
        assert targets
        assert "3SG" in targets and "3PL" in targets
        assert has_overlap is False
        assert isinstance(targets["3SG"], list) and isinstance(targets["3PL"], list)
        assert len(targets["3SG"]) >= 1 and len(targets["3PL"]) >= 1

    def test_infer_decoder_eval_targets_marks_overlapping_tokens_for_row_wise_fallback(self):
        """Auto-inferred decoder eval targets mark overlap so decoder eval can use row-wise targets."""
        # Both classes use the same factual token "x", which should trigger row-wise fallback.
        class_dfs = {
            "A": pd.DataFrame(
                {
                    "masked": ["[MASK] x"],
                    "split": ["train"],
                    "A": ["x"],
                }
            ),
            "B": pd.DataFrame(
                {
                    "masked": ["[MASK] x"],
                    "split": ["train"],
                    "B": ["x"],
                }
            ),
        }
        config = TextPredictionConfig(
            data=class_dfs,
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        targets, has_overlap = trainer._infer_decoder_eval_targets()
        assert targets == {"A": ["x"], "B": ["x"]}
        assert has_overlap is True


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
    @classmethod
    def tokenizer(cls):
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


class TestAddIdentityForOtherClasses:
    """Identity transitions are added only for non-target classes (all_classes \\ target_classes)."""

    @pytest.fixture(scope="class")
    @classmethod
    def tokenizer(cls):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def test_add_identity_not_added_when_all_classes_equal_target_classes(self, tokenizer):
        """When all_classes is set to target_classes, no identity rows are added."""
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            all_classes=["3SG", "3PL"],
            use_class_names_as_columns=True,
        )
        training_args = TrainingArguments(add_identity_for_other_classes=True)
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=training_args,
        )
        trainer._ensure_data()
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        df = training_data.data
        identity_rows = df[df["factual_id"] == df["alternative_id"]]
        assert len(identity_rows) == 0, (
            "Expected no identity rows when all_classes equals target_classes"
        )

    def test_add_identity_only_for_non_target_classes(self, tokenizer):
        """When all_classes has extra classes, identity rows are added only for non-target classes."""
        config = TextPredictionConfig(
            data=_per_class_dict_three_classes(),
            target_classes=["3SG", "3PL"],
            use_class_names_as_columns=True,
        )
        training_args = TrainingArguments(add_identity_for_other_classes=True)
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=training_args,
        )
        trainer._ensure_data()
        assert set(trainer.all_classes) == {"3SG", "3PL", "Other"}
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        df = training_data.data
        identity_rows = df[df["factual_id"] == df["alternative_id"]]
        assert len(identity_rows) >= 1, "Expected at least one identity row for non-target class Other"
        non_target = {"Other"}
        for _, row in identity_rows.iterrows():
            assert row["factual_id"] in non_target, (
                f"Identity row should be for non-target class only, got factual_id={row['factual_id']!r}"
            )
