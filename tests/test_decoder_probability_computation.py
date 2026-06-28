"""
Tests for decoder probability computation in evaluate_base_model.

All data is built in-memory. The tests intentionally avoid external datasets or
files so they remain stable across local and CI environments.
"""

import pandas as pd
import pytest

from gradiend.trainer.text.prediction.prediction_objective import PredictionObjective
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from tests.conftest import MockTokenizer, SimpleMockModel


def _with_required_splits(rows):
    """Replicate tiny rows over train/validation/test, as current trainer validation requires all three."""
    out = []
    for split in ("train", "validation", "test"):
        for row in rows:
            copied = dict(row)
            copied["split"] = split
            out.append(copied)
    return pd.DataFrame(out)


def _two_class_data(left="3SG", right="3PL", left_token="he", right_token="they"):
    return _with_required_splits(
        [
            {
                "masked": "[MASK] here",
                "label_class": left,
                "label": left_token,
                "alternative_class": right,
                "alternative": right_token,
            },
            {
                "masked": "[MASK] there",
                "label_class": right,
                "label": right_token,
                "alternative_class": left,
                "alternative": left_token,
            },
        ]
    )


def _trainer(config):
    trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
    trainer._ensure_data()
    return trainer


class TestEvaluateBaseModelProbabilityComputation:
    def test_evaluate_base_model_returns_probs_dict(self, monkeypatch):
        config = TextPredictionConfig(
            data=_two_class_data(),
            target_classes=["3SG", "3PL"],
            decoder_eval_targets={"3SG": ["he", "He"], "3PL": ["they", "They"]},
            masked_col="masked",
        )
        trainer = _trainer(config)

        calls = {}

        def fake_score(self, *args, **kwargs):
            calls["score_kwargs"] = kwargs
            return {
                "3PL": {"3SG": 0.8, "3PL": 0.2},
                "3SG": {"3SG": 0.6, "3PL": 0.4},
            }

        monkeypatch.setattr(PredictionObjective, "score_probability_shift", fake_score)
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [
                    {"masked": "[MASK] here", "label_class": "3SG", "alternative_id": "3PL"},
                    {"masked": "[MASK] there", "label_class": "3PL", "alternative_id": "3SG"},
                ]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral text"}]),
            use_cache=False,
        )

        assert result["probs"] == {"3SG": 0.8, "3PL": 0.4}
        assert result["lms"] == {"lms": 0.5}
        assert calls["score_kwargs"]["dataset_class_col"] == "label_class"

    def test_evaluate_base_model_counterfactual_evaluation(self, monkeypatch):
        config = TextPredictionConfig(
            data=_two_class_data(),
            target_classes=["3SG", "3PL"],
            decoder_eval_targets={"3SG": ["he"], "3PL": ["they"]},
            decoder_eval_prob_on_other_class=True,
            masked_col="masked",
        )
        trainer = _trainer(config)

        monkeypatch.setattr(
            PredictionObjective,
            "score_probability_shift",
            lambda *args, **kwargs: {"3PL": {"3SG": 0.8, "3PL": 0.2}, "3SG": {"3SG": 0.6, "3PL": 0.4}},
        )
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [
                    {"masked": "[MASK] here", "label_class": "3SG", "alternative_id": "3PL"},
                    {"masked": "[MASK] there", "label_class": "3PL", "alternative_id": "3SG"},
                ]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral_data"}]),
            use_cache=False,
        )

        assert result["probs"]["3SG"] == 0.8
        assert result["probs"]["3PL"] == 0.4

    def test_evaluate_base_model_multiple_tokens_per_class(self, monkeypatch):
        config = TextPredictionConfig(
            data=_two_class_data("masc_nom", "fem_nom", "der", "die"),
            target_classes=["masc_nom", "fem_nom"],
            decoder_eval_targets={"masc_nom": ["der", "Der"], "fem_nom": ["die", "Die"]},
            masked_col="masked",
        )
        trainer = _trainer(config)
        calls = {}

        def fake_score(self, *args, **kwargs):
            calls["targets"] = kwargs["targets"]
            return {
                "fem_nom": {"masc_nom": 0.8, "fem_nom": 0.2},
                "masc_nom": {"masc_nom": 0.2, "fem_nom": 0.8},
            }

        monkeypatch.setattr(PredictionObjective, "score_probability_shift", fake_score)
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [
                    {"masked": "[MASK] here", "label_class": "masc_nom", "alternative_id": "fem_nom"},
                    {"masked": "[MASK] there", "label_class": "fem_nom", "alternative_id": "masc_nom"},
                ]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral_data"}]),
            use_cache=False,
        )

        assert calls["targets"] == {"masc_nom": ["der", "Der"], "fem_nom": ["die", "Die"]}
        assert result["probs"] == {"masc_nom": 0.8, "fem_nom": 0.8}

    def test_explicit_decoder_targets_with_overlap_warns_not_errors(self):
        config = TextPredictionConfig(
            data=_two_class_data("C1", "C2", "x", "y"),
            target_classes=["C1", "C2"],
            decoder_eval_targets={"C1": ["x", "+"], "C2": ["+", "y"]},
            masked_col="masked",
        )
        trainer = _trainer(config)

        normalized = trainer._validate_explicit_decoder_eval_targets(config.decoder_eval_targets)

        assert normalized["C1"] == ["x", "+"]
        assert normalized["C2"] == ["+", "y"]

    def test_explicit_decoder_targets_invalid_class_key_raises(self):
        config = TextPredictionConfig(
            data=_two_class_data("REAL", "OTHER", "x", "z"),
            target_classes=["REAL", "OTHER"],
            decoder_eval_targets={"REAL": ["x"], "TYPO": ["y"]},
            masked_col="masked",
        )
        trainer = _trainer(config)

        with pytest.raises(ValueError) as exc_info:
            trainer._validate_explicit_decoder_eval_targets(config.decoder_eval_targets)

        assert "TYPO" in str(exc_info.value)
        assert "REAL" in str(exc_info.value) or "known classes" in str(exc_info.value)

    def test_resolve_decoder_eval_targets_returns_tuple(self):
        config = TextPredictionConfig(
            data=_two_class_data("C1", "C2", "x", "y"),
            target_classes=["C1", "C2"],
            decoder_eval_targets={"C1": ["t1"], "C2": ["t2"]},
            masked_col="masked",
        )
        trainer = _trainer(config)

        targets, use_row_wise = trainer._resolve_decoder_eval_targets(training_like_df=None)

        assert use_row_wise is False
        assert targets == {"C1": ["t1"], "C2": ["t2"]}

    def test_decoder_eval_targets_label_uses_row_wise_mode(self):
        config = TextPredictionConfig(
            data=_two_class_data("commutative", "non-commutative", "=", "!="),
            target_classes=["commutative", "non-commutative"],
            decoder_eval_targets="label",
            masked_col="masked",
        )
        trainer = _trainer(config)

        targets, use_row_wise = trainer._resolve_decoder_eval_targets(training_like_df=None)

        assert targets is None
        assert use_row_wise is True
