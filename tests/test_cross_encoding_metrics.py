import pandas as pd
import pytest

import gradiend.comparison.cross_encoding as cross_encoding_module
from gradiend.comparison.cross_encoding import (
    _resolve_full_eval,
    can_normalize_cross_encoding_by_diagonal,
    compute_cross_encoding_matrix,
    normalize_cross_encoding_rows_by_diagonal,
)
from gradiend.trainer.suite import PositiveTrainerSuite, SuitePairDefinition


class _DummyTrainer:
    model_path = "/tmp/model"

    def __init__(self, target_classes=None, **kwargs):
        if target_classes is None:
            target_classes = kwargs["target_classes"]
        self.target_classes = list(target_classes)
        self.training_args = None
        self._training_args = None

    def evaluate_encoder(self, **kwargs):
        return {"encoder_df": _make_sentiment_encoder_df()}


def _make_encoder_df():
    return pd.DataFrame(
        [
            {"source_id": "commutative_plus", "target_id": "non_commutative_plus", "encoded": 0.9},
            {"source_id": "commutative_plus", "target_id": "non_commutative_plus", "encoded": 0.7},
            {"source_id": "non_commutative_plus", "target_id": "commutative_plus", "encoded": 0.3},
            {"source_id": "non_commutative_plus", "target_id": "commutative_plus", "encoded": 0.1},
        ]
    )


def _make_sentiment_encoder_df():
    return pd.DataFrame(
        [
            {"source_id": "good", "target_id": "bad", "encoded": 0.9},
            {"source_id": "good", "target_id": "bad", "encoded": 0.7},
            {"source_id": "bad", "target_id": "good", "encoded": 0.2},
            {"source_id": "happy", "target_id": "sad", "encoded": 0.8},
            {"source_id": "sad", "target_id": "happy", "encoded": 0.1},
        ]
    )


def test_resolve_full_eval_defaults_to_test_split_only():
    assert _resolve_full_eval(None, "test") is True
    assert _resolve_full_eval(None, "validation") is False
    assert _resolve_full_eval(False, "test") is False
    assert _resolve_full_eval(True, "validation") is True


def test_cross_encoding_metrics_cover_positive_negative_and_difference(monkeypatch):
    monkeypatch.setattr(
        cross_encoding_module,
        "_load_cached_encoder_df",
        lambda trainer, split, max_size: _make_encoder_df(),
    )

    trainers = {
        "plus": _DummyTrainer(["commutative_plus", "non_commutative_plus"]),
        "times": _DummyTrainer(["commutative_plus", "non_commutative_plus"]),
    }

    positive = compute_cross_encoding_matrix(trainers, metric="positive_mean", use_cache=True, run_evaluation=False)
    negative = compute_cross_encoding_matrix(trainers, metric="negative_mean", use_cache=True, run_evaluation=False)
    difference = compute_cross_encoding_matrix(trainers, metric="positive_minus_negative", use_cache=True, run_evaluation=False)

    assert positive["measure"] == "cross_encoding_positive_mean"
    assert negative["measure"] == "cross_encoding_negative_mean"
    assert difference["measure"] == "cross_encoding_positive_minus_negative"
    assert positive["full_eval"] is True
    assert positive["matrix"][0][0] == pytest.approx(0.8)
    assert negative["matrix"][0][0] == pytest.approx(0.2)
    assert difference["matrix"][0][0] == pytest.approx(0.6)


def test_positive_suite_cross_encoding_uses_positive_feature_definition(monkeypatch):
    monkeypatch.setattr(
        cross_encoding_module,
        "_load_cached_encoder_df",
        lambda trainer, split, max_size: _make_sentiment_encoder_df(),
    )

    suite = object.__new__(PositiveTrainerSuite)
    suite.trainers = {
        "good__bad": _DummyTrainer(["good", "bad"]),
        "happy__sad": _DummyTrainer(["happy", "sad"]),
    }
    suite.pair_definitions = {
        "good__bad": SuitePairDefinition(target_classes=("good", "bad"), positive_class="good"),
        "happy__sad": SuitePairDefinition(target_classes=("happy", "sad"), positive_class="happy"),
    }
    suite._resolve_suite_seed_selection = lambda seed_selection=None: "best"
    suite._resolve_suite_dispersion = lambda dispersion=None: "none"
    suite.evaluate_encoder = lambda **kwargs: None

    result = suite.compute_cross_encoding_matrix(use_cache=True, run_evaluation=False)

    assert result["positive_class_by_column"] == {
        "good__bad": "good",
        "happy__sad": "happy",
    }
    assert result["negative_class_by_column"] == {
        "good__bad": "bad",
        "happy__sad": "sad",
    }
    assert result["matrix"][0][0] == pytest.approx(0.8)
    assert result["matrix"][0][1] == pytest.approx(0.8)


def test_positive_suite_cross_encoding_does_not_recompute_bad_cache(monkeypatch):
    bad_cache = pd.DataFrame(
        [
            {"source_id": "good", "target_id": "bad", "encoded": 0.9},
            {"source_id": "bad", "target_id": "good", "encoded": 0.2},
        ]
    )

    monkeypatch.setattr(cross_encoding_module, "_load_cached_encoder_df", lambda trainer, split, max_size: bad_cache)
    monkeypatch.setattr(cross_encoding_module, "_load_eval_model_for_trainer", lambda trainer, load_directory=None: object())

    suite = object.__new__(PositiveTrainerSuite)
    suite.trainers = {
        "good__bad": _DummyTrainer(["good", "bad"]),
        "happy__sad": _DummyTrainer(["happy", "sad"]),
    }
    suite.pair_definitions = {
        "good__bad": SuitePairDefinition(target_classes=("good", "bad"), positive_class="good"),
        "happy__sad": SuitePairDefinition(target_classes=("happy", "sad"), positive_class="happy"),
    }
    suite._resolve_suite_seed_selection = lambda seed_selection=None: "best"
    suite._resolve_suite_dispersion = lambda dispersion=None: "none"

    def evaluate_encoder(**kwargs):
        if kwargs.get("use_cache") is False:
            pytest.fail("use_cache=True must not silently recompute bad caches")

    suite.evaluate_encoder = evaluate_encoder

    with pytest.raises(ValueError, match="Cross-encoding found no rows"):
        suite.compute_cross_encoding_matrix(use_cache=True, run_evaluation=True)


def test_compute_cross_encoding_matrix_passes_full_eval_to_encoder(monkeypatch):
    calls = []

    class _EvalTrainer(_DummyTrainer):
        def evaluate_encoder(self, **kwargs):
            calls.append(kwargs)
            return {"encoder_df": _make_encoder_df()}

        model_path = "/tmp/model"

    monkeypatch.setattr(
        cross_encoding_module,
        "_load_cached_encoder_df",
        lambda trainer, split, max_size: None,
    )
    monkeypatch.setattr(
        cross_encoding_module,
        "_load_eval_model_for_trainer",
        lambda trainer, load_directory=None: object(),
    )

    trainers = {
        "plus": _EvalTrainer(["commutative_plus", "non_commutative_plus"]),
        "times": _EvalTrainer(["commutative_plus", "non_commutative_plus"]),
    }
    compute_cross_encoding_matrix(
        trainers,
        metric="positive_mean",
        use_cache=False,
        run_evaluation=True,
        full_eval=False,
        split="test",
    )
    assert calls
    assert calls[0]["include_other_classes"] is False


def test_can_normalize_cross_encoding_by_diagonal_requires_matching_ids():
    assert can_normalize_cross_encoding_by_diagonal(
        {
            "model_ids": ["A", "B"],
            "column_ids": ["A", "B"],
            "matrix": [[1.0, 0.5], [0.2, 1.0]],
        }
    )
    assert not can_normalize_cross_encoding_by_diagonal(
        {
            "model_ids": ["A", "B"],
            "column_ids": ["B", "A"],
            "matrix": [[1.0, 0.5], [0.2, 1.0]],
        }
    )


def test_normalize_cross_encoding_rows_by_diagonal_scales_rows_and_stats():
    comparison_data = {
        "measure": "cross_encoding_positive_mean",
        "model_ids": ["plus", "times"],
        "matrix": [[2.0, 1.0], [3.0, 6.0]],
        "cell_stats": [
            [{"aggregate": 2.0, "n": 2, "std": 0.5, "min": 1.5, "max": 2.5, "scores": [1.5, 2.5]}, {"aggregate": 1.0, "n": 2, "range_half_width": 0.25}],
            [{"aggregate": 3.0, "n": 2, "minmax": [2.0, 4.0]}, {"aggregate": 6.0, "n": 2, "std": 1.5}],
        ],
    }

    normalized = normalize_cross_encoding_rows_by_diagonal(comparison_data)

    assert normalized["matrix"][0] == pytest.approx([1.0, 0.5])
    assert normalized["matrix"][1] == pytest.approx([0.5, 1.0])
    assert normalized["measure"] == "cross_encoding_positive_mean_row_normalized"
    assert normalized["cell_stats"][0][0]["aggregate"] == pytest.approx(1.0)
    assert normalized["cell_stats"][0][0]["std"] == pytest.approx(0.25)
    assert normalized["cell_stats"][0][1]["range_half_width"] == pytest.approx(0.125)
    assert normalized["cell_stats"][1][0]["minmax"] == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    assert normalized["row_normalized_by_diagonal"] is True


def test_resolve_trainer_load_directory_prefers_experiment_checkpoint():
    import json
    import os
    import tempfile

    from gradiend.comparison.cross_encoding import _resolve_trainer_load_directory

    with tempfile.TemporaryDirectory(prefix="resolve_ckpt_") as temp:
        exp_dir = os.path.join(temp, "gender_de_der__dem")
        model_dir = os.path.join(exp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "gradiend_context.json"), "w", encoding="utf-8") as handle:
            json.dump({"source": "a", "target": "b"}, handle)
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
            json.dump({"architecture": {"input_dim": 4}}, handle)
        with open(os.path.join(model_dir, "model.safetensors"), "wb") as handle:
            handle.write(b"")

        class _Trainer:
            run_id = "gender_de_der__dem"
            model_path = "google-bert/bert-base-multilingual-cased"

            @property
            def experiment_dir(self):
                return exp_dir

        assert _resolve_trainer_load_directory(_Trainer()) == os.path.normpath(model_dir)
