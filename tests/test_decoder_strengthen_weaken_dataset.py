"""
Tests that decoder evaluation uses the correct probability on the correct dataset
for strengthening and weakening.

- Strengthen class X: maximize P(X) on the *other* class's dataset → eval on other's data,
  result key = dataset class (e.g. "3PL"), value = P(3SG) on 3PL data; summary aliased to "3SG".
- Weaken class X: maximize (1 - P(X) on X's dataset) → eval on X's data,
  result key = "X_weaken", value = 1 - P(X) on X data.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock

from gradiend.evaluator.decoder import DecoderEvaluator
from tests.conftest import MockTokenizer


class MockModelWithGradiend3SG3PL:
    """Mock ModelWithGradiend with 3SG/3PL classes for decoder tests."""

    def __init__(self):
        self.name_or_path = "mock-model"
        self.base_model = MagicMock()
        self.tokenizer = MockTokenizer()
        self.feature_class_encoding_direction = {"3SG": 1.0, "3PL": -1.0}

    def rewrite_base_model(self, **kwargs):
        return self


class TrainerForStrengthenWeakenTest:
    """
    Mock trainer that records which training_like_df was passed to evaluate_base_model
    and returns controlled probs/probs_factual so we can assert correct dataset and metric.
    """

    def __init__(self):
        self._training_args = MagicMock()
        self._training_args.use_cache = False
        self._training_args.decoder_eval_max_size_training_like = 50
        self._training_args.decoder_eval_max_size_neutral = 50
        self.experiment_dir = None
        self.run_id = None
        self.target_classes = ["3SG", "3PL"]
        self._model = MockModelWithGradiend3SG3PL()
        self._evaluate_base_model_calls = []

    def _default_from_training_args(self, value, name, fallback=None):
        if value is not None:
            return value
        return getattr(self._training_args, name, fallback)

    def get_model(self):
        return self._model

    def get_target_feature_classes(self):
        return self.target_classes

    def _model_for_decoder_eval(self, model_with_gradiend):
        return model_with_gradiend

    def _get_decoder_eval_dataframe(self, tokenizer, **kwargs):
        """Full dataframe with both 3SG and 3PL so decoder can restrict by direction."""
        training_like_df = pd.DataFrame([
            {"masked": "[MASK] a", "label_class": "3SG", "label": "he"},
            {"masked": "[MASK] b", "label_class": "3PL", "label": "they"},
        ])
        neutral_df = pd.DataFrame([{"text": "neutral"}])
        return training_like_df, neutral_df

    def evaluate_base_model(self, base_model, tokenizer, *, training_like_df=None, **kwargs):
        """Record which dataset classes were in training_like_df; return probs for that dataset."""
        if training_like_df is not None and hasattr(training_like_df, "columns"):
            col = "label_class" if "label_class" in training_like_df.columns else "factual_id"
            if col in training_like_df.columns:
                dataset_classes = sorted(training_like_df[col].dropna().astype(str).unique().tolist())
            else:
                dataset_classes = []
        else:
            dataset_classes = []
        self._evaluate_base_model_calls.append({"dataset_classes": dataset_classes})

        # Return probs keyed by dataset class (convention: probs[dataset_class] = P(other) on that dataset).
        # For strengthen we need only the other class's data → probs has key "3PL" when eval 3SG.
        # For weaken we need class's data → probs_factual has key "3SG" when weaken 3SG.
        probs = {}
        probs_factual = {}
        for c in dataset_classes:
            # Simulate: P(other) on this dataset. For single class "3PL", other = "3SG" → value is P(3SG) on 3PL
            others = [x for x in self.target_classes if x != c]
            if others:
                probs[c] = 0.9 if c == "3PL" else 0.2  # P(3SG) on 3PL high for strengthen selection
            probs_factual[c] = 0.1 if c == "3SG" else 0.8  # P(3SG) on 3SG low → 3SG_weaken = 0.9 high

        result = {
            "lms": {"lms": 0.99},
            "probs": probs,
            "probs_factual": probs_factual,
            "id": "base",
        }
        # Grid entries get id from decoder; base is overwritten to "base"
        return result


class TrainerForStrengthenWeakenTestGrid:
    """
    Like above but returns different values for base vs grid so the selector picks a grid candidate.
    Also records dataset_classes per call so we can assert strengthen uses only 3PL and weaken only 3SG.
    """

    def __init__(self):
        self._training_args = MagicMock()
        self._training_args.use_cache = False
        self._training_args.decoder_eval_max_size_training_like = 50
        self._training_args.decoder_eval_max_size_neutral = 50
        self.experiment_dir = None
        self.run_id = None
        self.target_classes = ["3SG", "3PL"]
        self._model = MockModelWithGradiend3SG3PL()
        self._evaluate_base_model_calls = []
        self._call_count = 0

    def _default_from_training_args(self, value, name, fallback=None):
        if value is not None:
            return value
        return getattr(self._training_args, name, fallback)

    def get_model(self):
        return self._model

    def get_target_feature_classes(self):
        return self.target_classes

    def _model_for_decoder_eval(self, model_with_gradiend):
        return model_with_gradiend

    def _get_decoder_eval_dataframe(self, tokenizer, **kwargs):
        training_like_df = pd.DataFrame([
            {"masked": "[MASK] a", "label_class": "3SG", "label": "he"},
            {"masked": "[MASK] b", "label_class": "3PL", "label": "they"},
        ])
        neutral_df = pd.DataFrame([{"text": "neutral"}])
        return training_like_df, neutral_df

    def evaluate_base_model(self, base_model, tokenizer, *, training_like_df=None, **kwargs):
        if training_like_df is not None and hasattr(training_like_df, "columns"):
            col = "label_class" if "label_class" in training_like_df.columns else "factual_id"
            if col in training_like_df.columns:
                dataset_classes = sorted(training_like_df[col].dropna().astype(str).unique().tolist())
            else:
                dataset_classes = []
        else:
            dataset_classes = []
        self._evaluate_base_model_calls.append({"dataset_classes": dataset_classes})
        self._call_count += 1
        is_base = self._call_count == 1

        probs = {}
        probs_factual = {}
        for c in dataset_classes:
            others = [x for x in self.target_classes if x != c]
            if others:
                # Strengthen 3SG: we want P(3SG) on 3PL. Base low, modified high so selector picks modified.
                probs[c] = 0.15 if is_base else 0.85
            probs_factual[c] = 0.2 if c == "3SG" else 0.7
            if c == "3SG":
                probs_factual[c] = 0.1 if is_base else 0.05  # weaken = 1 - factual → high for modified

        result = {
            "lms": {"lms": 0.99},
            "probs": probs,
            "probs_factual": probs_factual,
        }
        return result


class TestDecoderStrengthenWeakenDataset:
    """Ensure correct probability on correct dataset for strengthen and weaken."""

    def test_strengthen_uses_other_class_dataset(self):
        """Strengthen class X must evaluate on the *other* class's dataset (P(X) on other's data)."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTest()
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            increase_target_probabilities=True,
            feature_factors=[-1.0],
            lrs=[1e-2],
            plot=False,
        )
        # Decoder should restrict to other class only: 3PL
        assert len(trainer._evaluate_base_model_calls) >= 1
        for call in trainer._evaluate_base_model_calls:
            assert call["dataset_classes"] == ["3PL"], (
                "Strengthen 3SG must use only the other class's dataset (3PL), not 3SG. "
                "We maximize P(3SG) on 3PL data."
            )
        # Summary for 3SG should be present (aliased from 3PL metric)
        assert "3SG" in result
        assert "value" in result["3SG"]
        assert result["3SG"]["value"] == 0.9  # our mock returns probs["3PL"] = 0.9

    def test_strengthen_summary_value_is_p_target_on_other_dataset(self):
        """Selected summary value for strengthen 3SG is the max P(3SG) on 3PL across candidates."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTestGrid()
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            increase_target_probabilities=True,
            feature_factors=[-1.0],
            lrs=[1e-2],
            plot=False,
        )
        for call in trainer._evaluate_base_model_calls:
            assert call["dataset_classes"] == ["3PL"], (
                "Strengthen must evaluate on other class's dataset (3PL)."
            )
        # Selector should pick the candidate with higher P(3SG) on 3PL (our grid returns 0.85 vs base 0.15)
        assert "3SG" in result
        assert result["3SG"]["value"] == 0.85

    def test_weaken_uses_own_class_dataset(self):
        """Weaken class X must evaluate on X's own dataset (P(X) on X data; we maximize 1 - P(X))."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTest()
        trainer._evaluate_base_model_calls = []
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            increase_target_probabilities=False,
            feature_factors=[1.0],
            lrs=[1e-2],
            plot=False,
        )
        for call in trainer._evaluate_base_model_calls:
            assert call["dataset_classes"] == ["3SG"], (
                "Weaken 3SG must use the class's own dataset (3SG). "
                "We maximize (1 - P(3SG) on 3SG data)."
            )
        assert "3SG_weaken" in result
        assert "value" in result["3SG_weaken"]
        # Mock returns probs_factual["3SG"] = 0.1 → 3SG_weaken = 0.9
        assert result["3SG_weaken"]["value"] == pytest.approx(0.9, abs=1e-5)

    def test_weaken_summary_value_is_one_minus_p_on_own_dataset(self):
        """Selected summary for weaken 3SG is 1 - P(3SG) on 3SG data; selector picks max."""
        evaluator = DecoderEvaluator()
        trainer = TrainerForStrengthenWeakenTestGrid()
        trainer._call_count = 0
        result = evaluator.evaluate_decoder(
            trainer,
            target_class="3SG",
            increase_target_probabilities=False,
            feature_factors=[1.0],
            lrs=[1e-2],
            plot=False,
        )
        for call in trainer._evaluate_base_model_calls:
            assert call["dataset_classes"] == ["3SG"], (
                "Weaken must evaluate on class's own dataset (3SG)."
            )
        # We return factual 3SG: base 0.1 (weaken 0.9), modified 0.05 (weaken 0.95) → selector picks 0.95
        assert "3SG_weaken" in result
        assert result["3SG_weaken"]["value"] == pytest.approx(0.95, abs=1e-5)
