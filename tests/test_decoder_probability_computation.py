"""
Tests for decoder probability computation in evaluate_base_model.

All data is built in-memory. The tests intentionally avoid external datasets or
files so they remain stable across local and CI environments.
"""

import pandas as pd
import pytest

from gradiend.trainer.text.prediction.prediction_objective import PredictionObjective
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from tests.testing_mocks import MockTokenizer, SimpleMockModel


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

    def test_evaluate_base_model_row_wise_exposes_counterfactual_probs(self, monkeypatch):
        """Row-wise scoring must still expose probs[T]=P(T) on the other class's dataset."""
        config = TextPredictionConfig(
            data=_two_class_data("M", "F", "he", "she"),
            target_classes=["M", "F"],
            decoder_eval_targets="label",
            decoder_eval_prob_on_other_class=True,
            masked_col="masked",
        )
        trainer = _trainer(config)

        def fake_score(self, *args, **kwargs):
            assert kwargs.get("use_row_wise") is True
            return {
                "F": {"M": 0.81, "F": 0.19},
                "M": {"M": 0.55, "F": 0.45},
            }

        monkeypatch.setattr(PredictionObjective, "score_probability_shift", fake_score)
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [
                    {"masked": "[MASK] here", "label_class": "M", "label": "he", "alternative": "she", "alternative_id": "F"},
                    {"masked": "[MASK] there", "label_class": "F", "label": "she", "alternative": "he", "alternative_id": "M"},
                ]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral_data"}]),
            use_cache=False,
        )

        assert result["probs"]["M"] == 0.81
        assert result["probs"]["F"] == 0.45
        assert result["probs_factual"]["M"] == 0.55
        assert result["probs_factual"]["F"] == 0.19

    def test_evaluate_base_model_row_wise_with_only_the_other_classes_rows(self, monkeypatch):
        """Strengthening evaluates only the other class's rows; probs still holds P(T) on them."""
        config = TextPredictionConfig(
            data=_two_class_data("M", "F", "he", "she"),
            target_classes=["M", "F"],
            decoder_eval_targets="label",
            decoder_eval_prob_on_other_class=True,
            masked_col="masked",
        )
        trainer = _trainer(config)
        monkeypatch.setattr(PredictionObjective, "score_probability_shift", lambda self, *a, **k: {"F": {"M": 0.81, "F": 0.19}})
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [{"masked": "[MASK] there", "label_class": "F", "label": "she", "alternative": "he", "alternative_id": "M"}]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral_data"}]),
            use_cache=False,
        )

        assert result["probs"] == {"M": 0.81}
        assert result["probs_factual"] == {"F": 0.19}

    def test_evaluate_base_model_row_wise_with_three_classes(self, monkeypatch):
        """With three classes, probs[T] is P(T) measured on the rows of another class."""
        words = {"A": "a", "B": "b", "C": "c"}
        rows = [
            {"masked": "[MASK] z", "label_class": x, "label": words[x], "alternative_class": y, "alternative": words[y]}
            for x in words
            for y in words
            if x != y
        ]
        config = TextPredictionConfig(
            data=_with_required_splits(rows),
            target_classes=["A", "B", "C"],
            decoder_eval_targets="label",
            decoder_eval_prob_on_other_class=True,
            masked_col="masked",
        )
        trainer = _trainer(config)
        by_dataset = {
            "A": {"A": 0.50, "B": 0.11, "C": 0.12},
            "B": {"B": 0.60, "A": 0.21, "C": 0.22},
            "C": {"C": 0.70, "A": 0.31, "B": 0.32},
        }
        monkeypatch.setattr(PredictionObjective, "score_probability_shift", lambda self, *a, **k: by_dataset)
        monkeypatch.setattr(PredictionObjective, "compute_lms", lambda *args, **kwargs: {"lms": 0.5})

        result = trainer.evaluate_base_model(
            model=SimpleMockModel(),
            tokenizer=MockTokenizer(),
            training_like_df=pd.DataFrame(
                [
                    {"masked": "[MASK] z", "label_class": c, "label": words[c], "alternative": "x", "alternative_id": "A"}
                    for c in words
                ]
            ),
            neutral_df=pd.DataFrame([{"text": "neutral_data"}]),
            use_cache=False,
        )

        for target in words:
            measured_elsewhere = {by_dataset[other][target] for other in words if other != target}
            assert result["probs"][target] in measured_elsewhere
            assert result["probs_factual"][target] == by_dataset[target][target]
