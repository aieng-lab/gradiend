import pandas as pd
import pytest

import gradiend.comparison.cross_encoding as cross_encoding_module
from gradiend.comparison.cross_encoding import (
    _resolve_full_eval,
    compute_cross_encoding_matrix,
    normalize_cross_encoding_rows_by_diagonal,
)


class _DummyTrainer:
    def __init__(self, target_classes):
        self.target_classes = list(target_classes)
        self.training_args = None
        self._training_args = None


def _make_encoder_df():
    return pd.DataFrame(
        [
            {"source_id": "commutative_plus", "target_id": "non_commutative_plus", "encoded": 0.9},
            {"source_id": "commutative_plus", "target_id": "non_commutative_plus", "encoded": 0.7},
            {"source_id": "non_commutative_plus", "target_id": "commutative_plus", "encoded": 0.3},
            {"source_id": "non_commutative_plus", "target_id": "commutative_plus", "encoded": 0.1},
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


def test_normalize_cross_encoding_rows_by_diagonal_scales_rows_and_stats():
    comparison_data = {
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
    assert normalized["cell_stats"][0][0]["aggregate"] == pytest.approx(1.0)
    assert normalized["cell_stats"][0][0]["std"] == pytest.approx(0.25)
    assert normalized["cell_stats"][0][1]["range_half_width"] == pytest.approx(0.125)
    assert normalized["cell_stats"][1][0]["minmax"] == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    assert normalized["row_normalized_by_diagonal"] is True
