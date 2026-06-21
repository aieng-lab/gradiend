"""Tests for encoder split resolution and cross-split generalization metrics."""

import pandas as pd
import pytest

from gradiend.evaluator.encoder_metrics import get_encoder_metrics_from_dataframe
from gradiend.util.encoder_splits import order_split_names, resolve_encoder_splits


class TestResolveEncoderSplits:
    def test_single_split(self):
        assert resolve_encoder_splits("test") == ["test"]
        assert resolve_encoder_splits("val") == ["validation"]

    def test_all_uses_available(self):
        assert resolve_encoder_splits("all", available=["train", "test"]) == ["train", "test"]

    def test_sequence(self):
        assert resolve_encoder_splits(["train", "test"]) == ["train", "test"]

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            resolve_encoder_splits([])

    def test_order_split_names_canonical(self):
        assert order_split_names(["test", "train", "validation"]) == ["train", "validation", "test"]
        assert order_split_names(["val", "test"]) == ["validation", "test"]

    def test_resolve_encoder_splits_sequence_is_ordered(self):
        assert resolve_encoder_splits(["test", "validation", "train"]) == ["train", "validation", "test"]


class TestSplitGeneralization:
    def test_agreement_per_feature_class(self):
        encoder_df = pd.DataFrame(
            {
                "encoded": [0.8, 0.82, -0.75, -0.73, 0.0],
                "label": [1.0, 1.0, -1.0, -1.0, 0.0],
                "source_id": ["positive", "positive", "negative", "negative", "neutral"],
                "target_id": ["negative"] * 5,
                "data_split": ["train", "test", "train", "test", "test"],
                "type": ["training"] * 5,
            }
        )
        result = get_encoder_metrics_from_dataframe(encoder_df)
        sg = result["split_generalization"]
        assert sg["splits_compared"] == ["train", "test"]
        assert sg["agreement_by_feature_class"]["positive"] > 0.95
        assert sg["agreement_by_feature_class"]["negative"] > 0.95
        assert "mean_by_feature_class_by_split" in sg

    def test_neutral_rows_do_not_pollute_split_generalization(self):
        encoder_df = pd.DataFrame(
            {
                "encoded": [0.8, 0.82, -0.75, -0.73, 0.0, 0.01],
                "label": [1.0, 1.0, -1.0, -1.0, 0.0, 0.0],
                "source_id": ["positive", "positive", "negative", "negative", "neutral", "neutral"],
                "target_id": ["negative"] * 6,
                "data_split": ["train", "test", "train", "test", "test", "test"],
                "type": [
                    "training",
                    "training",
                    "training",
                    "training",
                    "neutral_dataset",
                    "neutral_training_masked",
                ],
                "neutral_variant": [None, None, None, None, "neutral_dataset", "neutral_training_masked"],
            }
        )
        result = get_encoder_metrics_from_dataframe(encoder_df)
        sg = result["split_generalization"]
        assert "neutral" not in sg["agreement_by_feature_class"]
