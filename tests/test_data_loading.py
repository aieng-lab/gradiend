"""
Tests for data loading methods with mocked HuggingFace API.

Covers resolve_base_data (HF path), resolve_dataframe (HF id), and load_hf_per_class.
All HF calls are mocked so tests do not hit the network.
"""

import importlib.util
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

HAS_DATASETS = importlib.util.find_spec("datasets") is not None

from gradiend.data.core.base_loader import resolve_base_data
from gradiend.trainer.text.prediction.unified_data import (
    load_hf_per_class,
    resolve_dataframe,
)


def _mock_hf_dataset(df: pd.DataFrame):
    """Build a mock HF Dataset with to_pandas() returning the DataFrame.
    No .items() so load_hf_per_class uses the single-dataset branch."""
    class _SingleDataset:
        def to_pandas(self):
            return df.copy()
    return _SingleDataset()


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets not installed")
class TestResolveBaseDataHfMocked:
    """resolve_base_data with str = HuggingFace dataset ID (mocked)."""

    @patch("datasets.load_dataset")
    def test_hf_id_returns_text_column_and_shuffles(self, mock_load_dataset):
        mock_load_dataset.return_value = _mock_hf_dataset(
            pd.DataFrame({"text": ["a", "b", "c"]})
        )
        texts = resolve_base_data(
            "org/dataset-id",
            text_column="text",
            max_size=10,
            seed=42,
        )
        mock_load_dataset.assert_called_once_with(
            "org/dataset-id", split="train", trust_remote_code=False
        )
        assert len(texts) == 3
        assert set(texts) == {"a", "b", "c"}

    @patch("datasets.load_dataset")
    def test_hf_id_with_hf_config_calls_load_dataset_with_config(self, mock_load_dataset):
        mock_load_dataset.return_value = _mock_hf_dataset(
            pd.DataFrame({"text": ["one"]})
        )
        texts = resolve_base_data(
            "org/dataset",
            text_column="text",
            hf_config="en",
            seed=42,
        )
        mock_load_dataset.assert_called_once_with(
            "org/dataset", "en", split="train", trust_remote_code=False
        )
        assert texts == ["one"]

    @patch("datasets.load_dataset")
    def test_hf_id_missing_text_column_raises(self, mock_load_dataset):
        mock_load_dataset.return_value = _mock_hf_dataset(
            pd.DataFrame({"other": [1, 2]})
        )
        with pytest.raises(ValueError, match="missing column 'text'"):
            resolve_base_data("org/ds", text_column="text", seed=42)

    @patch("datasets.load_dataset")
    def test_hf_id_custom_text_column(self, mock_load_dataset):
        mock_load_dataset.return_value = _mock_hf_dataset(
            pd.DataFrame({"body": ["x", "y"]})
        )
        texts = resolve_base_data(
            "org/ds",
            text_column="body",
            seed=42,
        )
        assert len(texts) == 2 and set(texts) == {"x", "y"}


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets not installed")
class TestResolveDataframeHfMocked:
    """resolve_dataframe with str = HuggingFace dataset ID (mocked)."""

    @patch("datasets.load_dataset")
    def test_str_hf_id_returns_dataframe(self, mock_load_dataset):
        mock_load_dataset.return_value = _mock_hf_dataset(
            pd.DataFrame({"masked": ["[MASK] hi"], "split": ["train"], "label": ["x"]})
        )
        out = resolve_dataframe("org/dataset-id", split="train")
        mock_load_dataset.assert_called_once_with(
            "org/dataset-id", split="train", trust_remote_code=True
        )
        assert out is not None
        assert len(out) == 1
        assert "masked" in out.columns

    @patch("datasets.load_dataset")
    def test_str_hf_id_respects_max_rows(self, mock_load_dataset):
        mock_load_dataset.return_value = _mock_hf_dataset(
            pd.DataFrame({"a": list(range(20))})
        )
        out = resolve_dataframe("org/ds", split="train", max_rows=5)
        assert out is not None
        assert len(out) == 5


@pytest.mark.skipif(not HAS_DATASETS, reason="datasets not installed")
class TestLoadHfPerClassMocked:
    """load_hf_per_class with mocked get_dataset_config_names and load_dataset."""

    @patch("datasets.load_dataset")
    @patch("datasets.get_dataset_config_names")
    def test_loads_single_class_by_name(
        self, mock_get_configs, mock_load_dataset
    ):
        mock_get_configs.return_value = ["class_a", "class_b"]
        mock_load_dataset.return_value = _mock_hf_dataset(
            pd.DataFrame({
                "masked": ["[MASK] x"],
                "split": ["train"],
            })
        )
        result = load_hf_per_class(
            "org/dataset",
            classes=["class_a"],
            masked_col="masked",
            split_col="split",
        )
        mock_load_dataset.assert_called_once_with(
            "org/dataset", "class_a", trust_remote_code=True
        )
        assert "class_a" in result
        assert len(result["class_a"]) == 1
        assert "masked" in result["class_a"].columns

    @patch("datasets.load_dataset")
    @patch("datasets.get_dataset_config_names")
    def test_all_configs_loads_each_subset(
        self, mock_get_configs, mock_load_dataset
    ):
        mock_get_configs.return_value = ["A", "B"]
        def load_side_effect(name, config, **kwargs):
            return _mock_hf_dataset(
                pd.DataFrame({
                    "masked": [f"[MASK] {config}"],
                    "split": ["train"],
                })
            )
        mock_load_dataset.side_effect = load_side_effect

        result = load_hf_per_class(
            "org/ds",
            classes="all",
            masked_col="masked",
            split_col="split",
        )
        assert mock_load_dataset.call_count == 2
        assert set(result.keys()) == {"A", "B"}
        assert result["A"]["masked"].iloc[0] == "[MASK] A"
        assert result["B"]["masked"].iloc[0] == "[MASK] B"

    @patch("datasets.load_dataset")
    @patch("datasets.get_dataset_config_names")
    def test_missing_masked_col_raises(
        self, mock_get_configs, mock_load_dataset
    ):
        mock_get_configs.return_value = ["X"]
        mock_load_dataset.return_value = _mock_hf_dataset(
            pd.DataFrame({"other": [1], "split": ["train"]})
        )
        with pytest.raises(ValueError, match="missing column 'masked'"):
            load_hf_per_class(
                "org/ds",
                classes=["X"],
                masked_col="masked",
                split_col="split",
            )
