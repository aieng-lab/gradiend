"""Regression: default feature-factor sign depends on encoder source."""

from __future__ import annotations

import pytest

from gradiend.evaluator.decoder import (
    compute_metric_summaries,
    derive_default_feature_factor,
    LMSThresholdPolicy,
)
from gradiend.model._source_target import feature_factor_from_encoding_direction
from tests.conftest import MockTokenizer


class _MockModel:
    def __init__(self, source: str):
        self.source = source
        self.feature_class_encoding_direction = {"3SG": 1.0, "3PL": -1.0}
        self.tokenizer = MockTokenizer()


class _Trainer:
    def __init__(self, model: _MockModel):
        self._model = model

    def get_model(self):
        return self._model


@pytest.mark.parametrize(
    "source,direction,expected",
    [
        ("factual", 1.0, -1.0),
        ("diff", 1.0, -1.0),
        ("alternative", 1.0, 1.0),
        ("factual", -1.0, 1.0),
        ("alternative", -1.0, -1.0),
    ],
)
def test_feature_factor_from_encoding_direction(source, direction, expected):
    assert feature_factor_from_encoding_direction(direction, source) == expected


@pytest.mark.parametrize(
    "source,expected_3sg,expected_3pl",
    [
        ("factual", -1.0, 1.0),
        ("diff", -1.0, 1.0),
        ("alternative", 1.0, -1.0),
    ],
)
def test_derive_default_feature_factor_by_source(source, expected_3sg, expected_3pl):
    model = _MockModel(source)
    trainer = _Trainer(model)
    assert derive_default_feature_factor(trainer, model, class_name="3SG") == expected_3sg
    assert derive_default_feature_factor(trainer, model, class_name="3PL") == expected_3pl


def test_strengthen_summary_filters_to_class_feature_factor():
    """With both ff in grid, strengthen 3SG must not pick the other class's ff."""
    results = {
        "base": {"id": "base", "lms": {"lms": 1.0}, "probs": {}},
        (-1.0, 0.01): {
            "id": {"feature_factor": -1.0, "learning_rate": 0.01},
            "lms": {"lms": 1.0},
            "probs": {"3SG": 0.9, "3PL": 0.1},
        },
        (1.0, 0.01): {
            "id": {"feature_factor": 1.0, "learning_rate": 0.01},
            "lms": {"lms": 1.0},
            "probs": {"3SG": 0.2, "3PL": 0.8},
        },
    }
    class_to_ff = {"3SG": 1.0, "3PL": -1.0}
    summary = compute_metric_summaries(
        results,
        metrics=["3SG"],
        selector=LMSThresholdPolicy(ratio=0.0),
        class_to_ff=class_to_ff,
        feature_factor_from_id=lambda cid: cid["feature_factor"] if isinstance(cid, dict) else cid[0],
        lr_from_id=lambda cid: cid["learning_rate"] if isinstance(cid, dict) else cid[1],
    )
    assert summary["3SG"]["feature_factor"] == 1.0
    assert summary["3SG"]["value"] == 0.2
