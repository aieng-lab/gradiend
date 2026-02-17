"""Tests for data generation: TextPredictionDataCreator output formats, auto-split, preprocess (newline splitting)."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gradiend.data.text import TextPreprocessConfig, iter_sentences_from_texts, preprocess_texts
from gradiend.data.text.filter_config import TextFilterConfig
from gradiend.data.text.prediction.creator import TextPredictionDataCreator


class TestPreprocessNewlineSplitting:
    """Sentences must not contain \\n; newlines are split first."""

    def test_iter_sentences_from_texts_none_config_splits_on_newline(self):
        texts = ["line one\nline two", "single"]
        out = list(iter_sentences_from_texts(texts, config=None))
        assert out == ["line one", "line two", "single"]
        assert all("\n" not in s for s in out)

    def test_preprocess_texts_split_to_sentences_splits_newline_first(self):
        config = TextPreprocessConfig(split_to_sentences=True)
        texts = ["First. Second.\nThird. Fourth."]
        out = preprocess_texts(texts, config=config)
        assert all("\n" not in s for s in out)
        # Newlines are split first, then regex on .!? so we get four sentences
        assert "First." in out and "Second." in out and "Third." in out and "Fourth." in out
        assert len(out) == 4


class TestCreatorOutputFormats:
    """Creator output: merged has label_class, label, feature_class_id (string id)."""

    def test_generate_training_data_unified_has_merged_columns(self):
        creator = TextPredictionDataCreator(
            base_data=["He walks.", "She runs.", "They play."],
            feature_targets=[
                TextFilterConfig(id="3SG", targets=["he", "she"]),
                TextFilterConfig(id="3PL", targets=["they"]),
            ],
            seed=42,
        )
        result = creator.generate_training_data(max_size_per_class=10, format="unified")
        assert "label_class" in result.columns
        assert "label" in result.columns
        assert "masked" in result.columns
        assert "split" in result.columns
        assert "feature_class_id" in result.columns
        assert result["feature_class_id"].dtype == object or result["feature_class_id"].iloc[0] in ("3SG", "3PL")

    def test_generate_training_data_auto_split_proportions(self):
        creator = TextPredictionDataCreator(
            base_data=["He walks."] * 20 + ["They run."] * 20,
            feature_targets=[
                TextFilterConfig(id="3SG", targets=["he"]),
                TextFilterConfig(id="3PL", targets=["they"]),
            ],
            seed=42,
        )
        result = creator.generate_training_data(
            max_size_per_class=30,
            format="unified",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        splits = result["split"].value_counts()
        assert "train" in splits.index
        assert "validation" in splits.index
        assert "test" in splits.index
        total = len(result)
        assert splits.get("train", 0) <= int(total * 0.8) + 2
        assert splits.get("validation", 0) <= int(total * 0.1) + 2
        assert splits.get("test", 0) <= int(total * 0.1) + 2

    def test_generate_training_data_ratios_must_sum_to_one(self):
        creator = TextPredictionDataCreator(
            base_data=["He walks."],
            feature_targets=[TextFilterConfig(targets=["he"])],
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            creator.generate_training_data(
                train_ratio=0.5,
                val_ratio=0.2,
                test_ratio=0.2,
            )


class TestCreatorOutputDirAndSave:
    """Creator can write to output_dir with default basenames."""

    def test_generate_training_data_writes_to_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            creator = TextPredictionDataCreator(
                base_data=["He walks.", "They run."],
                feature_targets=[
                    TextFilterConfig(id="3SG", targets=["he"]),
                    TextFilterConfig(id="3PL", targets=["they"]),
                ],
                output_dir=str(base),
                output_format="csv",
                seed=42,
            )
            creator.generate_training_data(max_size_per_class=5, format="unified")
            training_csv = base / "training.csv"
            assert training_csv.is_file()
            df = pd.read_csv(training_csv)
            assert "label_class" in df.columns and "label" in df.columns

    def test_generate_neutral_data_stops_at_max_size(self):
        creator = TextPredictionDataCreator(
            base_data=[f"Sentence {i} here." for i in range(50)],
            feature_targets=[TextFilterConfig(targets=["he"])],
            seed=42,
        )
        neutral = creator.generate_neutral_data(max_size=5)
        assert len(neutral) <= 5
        assert "text" in neutral.columns
