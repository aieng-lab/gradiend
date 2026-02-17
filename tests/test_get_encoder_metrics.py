"""
Tests for get_encoder_metrics and evaluate_encoder.

Verifies that both cached results and explicit encoder_df work correctly.
"""

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

from gradiend.evaluator.encoder_metrics import get_encoder_metrics_from_dataframe
from gradiend.trainer.core.feature_definition import FeatureLearningDefinition
from gradiend.util.paths import resolve_encoder_analysis_path


class _MinimalFeatureDef(FeatureLearningDefinition):
    """Minimal concrete subclass for testing get_encoder_metrics and evaluate_encoder."""

    def __init__(self, experiment_dir, use_cache: bool = False):
        super().__init__(target_classes=["A", "B"])
        self._experiment_dir_val = experiment_dir
        self._training_args = type("Args", (), {"use_cache": use_cache, "experiment_dir": experiment_dir})()

    def _experiment_dir(self) -> str:
        return self._experiment_dir_val

    def create_training_data(self, *args, **kwargs):
        raise NotImplementedError("Not used in these tests")

    def create_gradient_training_dataset(self, *args, **kwargs):
        raise NotImplementedError("Not used in these tests")

    def _get_decoder_eval_dataframe(self, *args, **kwargs):
        raise NotImplementedError("Not used in these tests")

    def _get_decoder_eval_targets(self) -> Dict[str, List[str]]:
        return {}

    def evaluate_base_model(self, *args, **kwargs):
        raise NotImplementedError("Not used in these tests")

    def _analyze_encoder(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError("Not used in these tests")

    def _default_from_training_args(self, value: Any, name: str, fallback: Any = None) -> Any:
        if value is not None:
            return value
        return getattr(self._training_args, name, fallback)


def _make_encoder_df(n=20):
    """Create a minimal encoder DataFrame for testing."""
    import numpy as np

    return pd.DataFrame({
        "encoded": np.random.randn(n).astype(float).tolist(),
        "label": [1.0] * (n // 2) + [-1.0] * (n - n // 2),
        "source_id": ["A"] * (n // 2) + ["B"] * (n - n // 2),
        "target_id": ["B"] * (n // 2) + ["A"] * (n - n // 2),
        "type": ["training"] * n,
    })


class TestGetEncoderMetricsFromDataframe:
    """Test get_encoder_metrics_from_dataframe (standalone function)."""

    def test_computes_metrics_from_dataframe(self):
        encoder_df = _make_encoder_df(20)
        result = get_encoder_metrics_from_dataframe(encoder_df)
        assert "n_samples" in result
        assert result["n_samples"] == 20
        assert "all_data" in result
        assert "correlation" in result["all_data"]
        assert "accuracy" in result["all_data"]
        assert "training_only" in result
        assert "target_classes_only" in result
        assert "boundaries" in result

    def test_empty_dataframe_raises(self):
        encoder_df = pd.DataFrame(columns=["encoded", "label", "type"])
        with pytest.raises(ValueError, match="empty"):
            get_encoder_metrics_from_dataframe(encoder_df)

    def test_mean_by_class_includes_identity_label_0(self):
        """mean_by_class must include identity classes (label 0) for convergence plot."""
        import numpy as np

        encoder_df = pd.DataFrame({
            "encoded": [0.5, -0.5, 0.02],
            "label": [1.0, -1.0, 0.0],
            "source_id": ["A", "B", "C"],
            "target_id": ["B", "A", "C"],
            "type": ["training", "training", "training"],
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)
        mean_by_class = result.get("mean_by_class", {})
        assert 0.0 in mean_by_class or 0 in mean_by_class, "mean_by_class must include label 0 (identity)"
        assert 1.0 in mean_by_class or 1 in mean_by_class
        assert -1.0 in mean_by_class or -1 in mean_by_class

    def test_mean_by_feature_class_includes_identity_classes(self):
        """mean_by_feature_class must include identity feature classes for convergence plot."""
        encoder_df = pd.DataFrame({
            "encoded": [0.5, -0.5, 0.02, -0.01],
            "label": [1.0, -1.0, 0.0, 0.0],
            "source_id": ["masc_nom", "fem_nom", "neut_nom", "neut_nom"],
            "target_id": ["fem_nom", "masc_nom", "neut_nom", "neut_nom"],
            "type": ["training", "training", "training", "training"],
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)
        mean_by_feature_class = result.get("mean_by_feature_class", {})
        assert "neut_nom" in mean_by_feature_class, "mean_by_feature_class must include identity class neut_nom"
        assert "masc_nom" in mean_by_feature_class
        assert "fem_nom" in mean_by_feature_class


class TestEncoderMetricsDetailed:
    """
    Detailed encoder metric tests with known correlations and expected values.

    These tests use constructed data (high correlation, adversarial/anti-correlated)
    to assert that the unified metrics contain the expected entries and values.
    """

    REQUIRED_KEYS = frozenset({
        "n_samples", "sample_counts", "all_data", "training_only",
        "target_classes_only", "boundaries", "correlation", "mean_by_class",
        "mean_by_type",
    })

    def test_high_correlation_data_entries_and_values(self):
        """
        Data with encoded ≈ label (high positive correlation) must produce:
        - correlation > 0.9
        - mean_by_class close to label values
        - all required unified metric keys present
        """
        # Perfect positive: encoded = label
        n = 20
        labels = [1.0] * (n // 2) + [-1.0] * (n - n // 2)
        encoded = [float(l) for l in labels]
        encoder_df = pd.DataFrame({
            "encoded": encoded,
            "label": labels,
            "source_id": ["A"] * (n // 2) + ["B"] * (n - n // 2),
            "target_id": ["B"] * (n // 2) + ["A"] * (n - n // 2),
            "type": ["training"] * n,
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        # Assert required keys
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Unified metrics must contain '{key}'"

        # Correlation should be 1.0 (perfect positive)
        assert result["correlation"] > 0.99, "Perfect positive data should yield correlation > 0.99"
        assert result["training_only"]["correlation"] > 0.99

        # mean_by_class: label 1 -> mean ~1, label -1 -> mean ~-1
        mbc = result["mean_by_class"]
        mbc_1 = mbc.get(1.0) or mbc.get(1)
        mbc_m1 = mbc.get(-1.0) or mbc.get(-1)
        assert mbc_1 is not None and abs(float(mbc_1) - 1.0) < 0.01
        assert mbc_m1 is not None and abs(float(mbc_m1) - (-1.0)) < 0.01

        # accuracy should be high
        assert result["training_only"]["accuracy"] > 0.95

        # boundaries present
        b = result["boundaries"]
        assert "neg_boundary" in b
        assert "pos_boundary" in b

    def test_adversarial_anticorrelated_data(self):
        """
        Adversarial data: encoded = -label (anti-correlated).
        - correlation < -0.9
        - mean_by_class[1] ~ -1, mean_by_class[-1] ~ 1
        """
        n = 20
        labels = [1.0] * (n // 2) + [-1.0] * (n - n // 2)
        encoded = [-float(l) for l in labels]
        encoder_df = pd.DataFrame({
            "encoded": encoded,
            "label": labels,
            "source_id": ["A"] * (n // 2) + ["B"] * (n - n // 2),
            "target_id": ["B"] * (n // 2) + ["A"] * (n - n // 2),
            "type": ["training"] * n,
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        assert result["correlation"] < -0.99, "Anti-correlated data should yield correlation < -0.99"

        mbc = result["mean_by_class"]
        mbc_1 = mbc.get(1.0) or mbc.get(1)
        mbc_m1 = mbc.get(-1.0) or mbc.get(-1)
        assert mbc_1 is not None and abs(float(mbc_1) - (-1.0)) < 0.01
        assert mbc_m1 is not None and abs(float(mbc_m1) - 1.0) < 0.01

    def test_identity_class_encoded_near_zero(self):
        """
        Identity rows (label 0) with encoded near 0: mean_by_class[0] ~ 0.
        """
        encoder_df = pd.DataFrame({
            "encoded": [0.8, -0.8, 0.0, 0.01, -0.01],
            "label": [1.0, -1.0, 0.0, 0.0, 0.0],
            "source_id": ["A", "B", "C", "C", "C"],
            "target_id": ["B", "A", "C", "C", "C"],
            "type": ["training"] * 5,
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        mbc = result["mean_by_class"]
        mbc_0 = mbc.get(0.0) or mbc.get(0)
        assert mbc_0 is not None, "mean_by_class must include label 0"
        assert abs(float(mbc_0)) < 0.05, "Identity encoded near 0 should yield mean_by_class[0] ~ 0"

    def test_mean_by_feature_class_values_when_encoded_matches_class(self):
        """
        When each feature class has distinct encoded values, mean_by_feature_class
        should reflect those values.
        """
        encoder_df = pd.DataFrame({
            "encoded": [1.0, 1.0, -1.0, -1.0, 0.0],
            "label": [1.0, 1.0, -1.0, -1.0, 0.0],
            "source_id": ["masc", "masc", "fem", "fem", "neut"],
            "target_id": ["fem", "fem", "masc", "masc", "neut"],
            "type": ["training"] * 5,
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        mbfc = result["mean_by_feature_class"]
        assert abs(mbfc["masc"] - 1.0) < 0.01
        assert abs(mbfc["fem"] - (-1.0)) < 0.01
        assert abs(mbfc["neut"]) < 0.01

    def test_sample_counts_and_mean_by_feature_class_include_all_classes(self):
        """
        sample_counts.by_type must reflect all rows; mean_by_feature_class must
        include identity class C (from unified metrics merge).
        """
        encoder_df = pd.DataFrame({
            "encoded": [0.5, -0.5, 0.0, 0.0],
            "label": [1.0, -1.0, 0.0, 0.0],
            "source_id": ["A", "B", "C", "C"],
            "target_id": ["B", "A", "C", "C"],
            "type": ["training", "training", "training", "training"],
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        sc = result["sample_counts"]
        assert "by_type" in sc
        assert sc["by_type"].get("training", 0) >= 4

        # mean_by_feature_class includes identity C (from mean_aggregate_mask, not training_mask)
        mbfc = result.get("mean_by_feature_class", {})
        assert "C" in mbfc
        assert "A" in mbfc
        assert "B" in mbfc


class TestGetEncoderMetricsWithEncoderDf:
    """Test get_encoder_metrics with encoder_df explicitly provided."""

    def test_get_encoder_metrics_with_encoder_df(self, tmp_path):
        """get_encoder_metrics should compute metrics from encoder_df without cache."""
        defn = _MinimalFeatureDef(experiment_dir=str(tmp_path))

        encoder_df = _make_encoder_df(20)
        metrics = defn.get_encoder_metrics(model_path=str(tmp_path), encoder_df=encoder_df)

        assert metrics is not None
        assert metrics["n_samples"] == 20
        assert "all_data" in metrics
        assert "training_only" in metrics
        assert "correlation" in metrics["all_data"]

    def test_get_encoder_metrics_with_empty_encoder_df_returns_none(self, tmp_path):
        defn = _MinimalFeatureDef(experiment_dir=str(tmp_path))

        encoder_df = pd.DataFrame(columns=["encoded", "label", "type"])
        metrics = defn.get_encoder_metrics(model_path=str(tmp_path), encoder_df=encoder_df)

        assert metrics is None


class TestGetEncoderMetricsWithCache:
    """Test get_encoder_metrics with use_cache=True."""

    def test_get_encoder_metrics_from_cache(self, tmp_path):
        """get_encoder_metrics should load from cache when use_cache=True and cache exists."""
        defn = _MinimalFeatureDef(experiment_dir=str(tmp_path), use_cache=True)

        # Create cache: CSV + JSON (get_model_metrics reads CSV and creates JSON)
        cache_path = resolve_encoder_analysis_path(tmp_path, None, split="test")
        assert cache_path is not None
        encoder_df = _make_encoder_df(10)
        encoder_df.to_csv(cache_path, index=False)

        from gradiend.evaluator.encoder_metrics import get_model_metrics

        get_model_metrics(cache_path, use_cache=False)

        metrics = defn.get_encoder_metrics(model_path=str(tmp_path), use_cache=True, split="test")

        assert metrics is not None
        assert metrics["n_samples"] == 10
        assert "all_data" in metrics

    def test_get_encoder_metrics_without_cache_or_df_raises(self, tmp_path):
        defn = _MinimalFeatureDef(experiment_dir=str(tmp_path), use_cache=False)

        with pytest.raises(ValueError, match="encoder_df or use_cache"):
            defn.get_encoder_metrics(model_path=str(tmp_path), use_cache=False)


class TestEncoderMetricsSubsetMasks:
    """
    Tests for correct behavior of all_data, training_only, target_classes_only
    across different scenarios: with/without neutral data, with/without identity (label 0).
    """

    def test_no_neutral_data_all_equals_training_only(self):
        """When no neutral types, all_data and training_only should be identical."""
        # Only type "training", labels -1, 1 (no identity)
        encoder_df = pd.DataFrame({
            "encoded": [1.0, 1.0, -1.0, -1.0],
            "label": [1.0, 1.0, -1.0, -1.0],
            "source_id": ["A", "A", "B", "B"],
            "target_id": ["B", "B", "A", "A"],
            "type": ["training"] * 4,
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        # all_data and training_only use same rows
        assert result["all_data"]["correlation"] == result["training_only"]["correlation"]
        assert result["all_data"]["accuracy"] == result["training_only"]["accuracy"]
        # target_classes_only same (no identity)
        assert result["target_classes_only"]["correlation"] == result["training_only"]["correlation"]

    def test_with_neutral_data_all_differs_from_training_only(self):
        """
        When neutral types present, all_data includes them; training_only excludes them.
        Neutral rows with encoded != label dilute all_data correlation.
        """
        # Training: encoded=label (perfect correlation). Neutral: label 0 but encoded off (0.8, -0.8)
        # so they dilute Pearson correlation.
        encoder_df = pd.DataFrame({
            "encoded": [1.0, 1.0, -1.0, -1.0, 0.8, -0.8, 0.5],  # 4 training + 3 neutral
            "label": [1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
            "source_id": ["A", "A", "B", "B", "neutral", "neutral", "neutral"],
            "target_id": ["B", "B", "A", "A", "neutral", "neutral", "neutral"],
            "type": ["training", "training", "training", "training",
                    "neutral_dataset", "neutral_dataset", "neutral_training_masked"],
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        # training_only: 4 rows (type non-neutral), perfect correlation
        assert result["training_only"]["correlation"] > 0.99
        assert result["training_only"]["accuracy"] > 0.95
        # all_data: 7 rows (includes neutral with encoded != label 0), correlation diluted
        assert result["all_data"]["correlation"] < result["training_only"]["correlation"]
        # n_samples and by_type
        assert result["n_samples"] == 7
        assert result["sample_counts"]["by_type"].get("training", 0) == 4
        assert result["sample_counts"]["by_type"].get("neutral_dataset", 0) == 2
        assert result["sample_counts"]["by_type"].get("neutral_training_masked", 0) == 1

    def test_training_only_includes_identity_label_0(self):
        """
        training_only includes identity transitions (label 0, type training).
        target_classes_only excludes label 0.
        """
        encoder_df = pd.DataFrame({
            "encoded": [1.0, -1.0, 0.0, 0.0],  # A, B, identity C
            "label": [1.0, -1.0, 0.0, 0.0],
            "source_id": ["A", "B", "C", "C"],
            "target_id": ["B", "A", "C", "C"],
            "type": ["training", "training", "training", "training"],
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        # training_only: all 4 rows (includes identity)
        assert result["training_only"]["correlation"] is not None
        assert 0.0 in result["mean_by_class"] or 0 in result["mean_by_class"]
        # target_classes_only: only A, B (excludes identity)
        assert result["target_classes_only"]["correlation"] is not None
        # With perfect separation: target_classes_only uses binary; training_only uses ternary
        assert result["target_classes_only"]["accuracy"] > 0.95
        assert result["training_only"]["accuracy"] > 0.9

    def test_target_classes_only_excludes_identity_affects_accuracy(self):
        """
        target_classes_only excludes label 0; uses binary classification.
        When identity (label 0) has encoded near boundary, target_classes_only
        accuracy can differ from training_only.
        """
        # Training: A=1, B=-1. Identity C: encoded=0.3 (just above neutral_boundary 0)
        # Ternary: 0.3 -> class 1. Binary: 0.3 >= 0 -> 1.
        encoder_df = pd.DataFrame({
            "encoded": [1.0, -1.0, 0.3],  # A, B, C (identity)
            "label": [1.0, -1.0, 0.0],
            "source_id": ["A", "B", "C"],
            "target_id": ["B", "A", "C"],
            "type": ["training", "training", "training"],
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        # training_only: 3 rows (ternary: -1, 0, 1)
        # target_classes_only: 2 rows (A, B only; binary)
        assert result["n_samples"] == 3
        assert result["training_only"]["correlation"] is not None
        assert result["target_classes_only"]["correlation"] is not None
        # target_classes_only uses only A,B - perfect binary separation
        assert result["target_classes_only"]["accuracy"] == 1.0
        assert result["target_classes_only"]["correlation"] > 0.99

    def test_neutral_dilutes_all_data_not_training_only(self):
        """
        Neutral rows (label 0, encoded != 0) dilute all_data correlation; training_only unaffected.
        """
        n_train = 20
        labels_train = [1.0] * (n_train // 2) + [-1.0] * (n_train - n_train // 2)
        encoded_train = [float(l) for l in labels_train]  # perfect for training
        n_neutral = 30
        # Neutral: label 0 but encoded scattered (0.5, -0.5, etc.) to dilute Pearson
        encoded_neutral = [0.5] * 10 + [-0.5] * 10 + [0.3] * 10
        encoder_df = pd.DataFrame({
            "encoded": encoded_train + encoded_neutral,
            "label": labels_train + [0.0] * n_neutral,
            "source_id": ["A"] * (n_train // 2) + ["B"] * (n_train - n_train // 2) + ["neut"] * n_neutral,
            "target_id": ["B"] * (n_train // 2) + ["A"] * (n_train - n_train // 2) + ["neut"] * n_neutral,
            "type": ["training"] * n_train + ["neutral_dataset"] * n_neutral,
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        assert result["training_only"]["correlation"] > 0.99
        assert result["all_data"]["correlation"] < result["training_only"]["correlation"]
        assert result["all_data"]["accuracy"] < result["training_only"]["accuracy"]

    def test_sample_counts_by_type_matches_data(self):
        """sample_counts.by_type must match actual type distribution."""
        encoder_df = pd.DataFrame({
            "encoded": [1.0, -1.0, 0.0, 0.0, 0.0],
            "label": [1.0, -1.0, 0.0, 0.0, 0.0],
            "source_id": ["A", "B", "n1", "n2", "n3"],
            "target_id": ["B", "A", "n1", "n2", "n3"],
            "type": ["training", "training", "neutral_dataset", "neutral_dataset", "neutral_training_masked"],
        })
        result = get_encoder_metrics_from_dataframe(encoder_df)

        sc = result["sample_counts"]["by_type"]
        assert sc.get("training", 0) == 2
        assert sc.get("neutral_dataset", 0) == 2
        assert sc.get("neutral_training_masked", 0) == 1
        assert result["n_samples"] == 5


class TestEvaluateEncoderWithEncoderDf:
    """Test evaluate_encoder with encoder_df (EncoderEvaluator and Evaluator)."""

    def test_encoder_evaluator_with_encoder_df_returns_metrics(self):
        """EncoderEvaluator.evaluate_encoder(encoder_df=...) should return unified metrics."""
        from gradiend.evaluator.encoder import EncoderEvaluator
        from tests.test_evaluator import MockTrainer

        evaluator = EncoderEvaluator()
        args = type("Args", (), {"use_cache": False, "experiment_dir": None, "encoder_eval_max_size": 100})()
        trainer = MockTrainer(training_args=args)

        encoder_df = _make_encoder_df(15)
        result = evaluator.evaluate_encoder(trainer, encoder_df=encoder_df)

        assert result is not None
        assert result["n_samples"] == 15
        assert "all_data" in result
        assert "correlation" in result
        assert "mean_by_class" in result

    def test_encoder_evaluator_with_empty_encoder_df_returns_empty(self):
        """EncoderEvaluator.evaluate_encoder(encoder_df=empty) should return empty dict."""
        from gradiend.evaluator.encoder import EncoderEvaluator
        from tests.test_evaluator import MockTrainer

        evaluator = EncoderEvaluator()
        args = type("Args", (), {"use_cache": False})()
        trainer = MockTrainer(training_args=args)

        encoder_df = pd.DataFrame(columns=["encoded", "label", "type"])
        result = evaluator.evaluate_encoder(trainer, encoder_df=encoder_df)

        assert result == {}


class TestUseCacheWithoutExperimentDir:
    """
    When use_cache=True and experiment_dir is not set, an error must be raised.

    Caching requires a location (experiment_dir or equivalent) to store/load from.
    """

    def test_get_encoder_metrics_use_cache_no_experiment_dir_raises(self):
        """get_encoder_metrics(use_cache=True) with experiment_dir=None raises."""
        defn = _MinimalFeatureDef(experiment_dir=None, use_cache=True)

        with pytest.raises(ValueError, match="Cannot resolve cache path|experiment_dir"):
            defn.get_encoder_metrics(model_path="/some/path", use_cache=True, split="test")

    def test_evaluate_encoder_use_cache_no_experiment_dir_raises(self):
        """evaluate_encoder(use_cache=True) with experiment_dir=None raises."""
        from gradiend.evaluator.encoder import EncoderEvaluator
        from gradiend.trainer.core.dataset import GradientTrainingDataset
        from tests.test_evaluator import MockTrainer, MockTrainingData

        evaluator = EncoderEvaluator()
        args = type("Args", (), {"use_cache": True, "experiment_dir": None, "encoder_eval_max_size": 100})()
        trainer = MockTrainer(training_args=args)
        trainer.experiment_dir = None
        trainer._model = None

        with pytest.raises(ValueError, match="use_cache=True.*experiment_dir"):
            evaluator.evaluate_encoder(trainer, use_cache=True)

    def test_evaluate_decoder_use_cache_no_experiment_dir_raises(self):
        """evaluate_decoder(use_cache=True) with experiment_dir=None raises."""
        from gradiend.evaluator.decoder import DecoderEvaluator
        from tests.test_evaluator import MockTrainer, MockModelWithGradiend

        evaluator = DecoderEvaluator()
        args = type("Args", (), {"use_cache": True, "experiment_dir": None})()
        trainer = MockTrainer(training_args=args)
        trainer.experiment_dir = None
        trainer._model = MockModelWithGradiend()

        with pytest.raises(ValueError, match="use_cache=True.*experiment_dir|output_path"):
            evaluator.evaluate_decoder(trainer, use_cache=True, feature_factors=[-1.0], lrs=[1e-2])
