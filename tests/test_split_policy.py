"""Tests for split policy and target-pair coverage validation."""

from __future__ import annotations

import pandas as pd
import pytest

from gradiend.evaluator.encoder_metrics import get_encoder_metrics_from_dataframe
from gradiend.util.split_policy import (
    SplitPolicy,
    min_vocabulary_keys_for_split_ratios,
    validate_data_split_policy,
    validate_target_class_vocabulary_coverage,
    validate_target_pair_encoder_split_coverage,
    vocabulary_held_out_viable_for_target_pair,
)
from gradiend.trainer.core.unified_schema import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_MASKED,
    UNIFIED_SPLIT,
    UNIFIED_TRANSITION,
    transition_id,
)


class TestSplitPolicy:
    def test_generalization_pair_train_test(self):
        policy = SplitPolicy.from_available(["test", "train", "validation"])
        assert policy.generalization_pair() == ("train", "test")

    def test_eval_falls_back_to_test_without_validation(self):
        policy = SplitPolicy.from_available(["train", "test"])
        assert policy.split_for_role("eval") == "test"
        assert policy.split_for_role("decoder") == "test"

    def test_train_only_requires_do_eval_off(self):
        policy = SplitPolicy.from_available(["train"])
        validate_data_split_policy(policy, do_eval=False)
        with pytest.raises(ValueError, match="do_eval=True"):
            validate_data_split_policy(policy, do_eval=True)

    def test_vocabulary_held_out_requires_test(self):
        policy = SplitPolicy.from_available(["train", "validation"])
        with pytest.raises(ValueError, match="vocabulary-held-out"):
            validate_data_split_policy(policy, vocabulary_held_out=True, do_eval=False)


class TestMinVocabularyKeys:
    def test_three_way_split_needs_three_keys(self):
        assert min_vocabulary_keys_for_split_ratios(0.8, 0.1, 0.1) == 3

    def test_train_test_only_needs_two_keys(self):
        assert min_vocabulary_keys_for_split_ratios(0.9, 0.0, 0.1) == 2


class TestVocabularyHeldOutViability:
    def test_single_token_class_not_viable(self):
        rows = []
        for cls, tok in (("3SG", "he"), ("3PL", "they")):
            other = "3PL" if cls == "3SG" else "3SG"
            for _ in range(12):
                rows.append(
                    {
                        UNIFIED_MASKED: "The runner [MASK] fast.",
                        UNIFIED_SPLIT: "train",
                        UNIFIED_FACTUAL_CLASS: cls,
                        UNIFIED_ALTERNATIVE_CLASS: other,
                        UNIFIED_FACTUAL: tok,
                        UNIFIED_ALTERNATIVE: other,
                        UNIFIED_TRANSITION: transition_id(cls, other),
                    }
                )
        df = pd.DataFrame(rows)
        assert vocabulary_held_out_viable_for_target_pair(
            df,
            ["3SG", "3PL"],
            group_col=UNIFIED_FACTUAL,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        ) is False

    def test_many_tokens_per_class_is_viable(self):
        rows = []
        for cls, toks in (("positive", ["love", "great", "nice"]), ("negative", ["hate", "awful", "bad"])):
            other = "negative" if cls == "positive" else "positive"
            for tok in toks:
                for _ in range(12):
                    rows.append(
                        {
                            UNIFIED_MASKED: "I feel [MASK].",
                            UNIFIED_SPLIT: "train",
                            UNIFIED_FACTUAL_CLASS: cls,
                            UNIFIED_ALTERNATIVE_CLASS: other,
                            UNIFIED_FACTUAL: tok,
                            UNIFIED_ALTERNATIVE: other,
                            UNIFIED_TRANSITION: transition_id(cls, other),
                        }
                    )
        df = pd.DataFrame(rows)
        assert vocabulary_held_out_viable_for_target_pair(
            df,
            ["positive", "negative"],
            group_col=UNIFIED_FACTUAL,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        ) is True


class TestTargetClassVocabularyCoverage:
    def _unified_two_tokens(self) -> pd.DataFrame:
        rows = []
        for cls, toks in (("positive", ["love", "great"]), ("negative", ["hate", "awful"])):
            other = "negative" if cls == "positive" else "positive"
            for tok in toks:
                rows.append(
                    {
                        UNIFIED_MASKED: "I feel [MASK].",
                        UNIFIED_SPLIT: "train",
                        UNIFIED_FACTUAL_CLASS: cls,
                        UNIFIED_ALTERNATIVE_CLASS: other,
                        UNIFIED_FACTUAL: tok,
                        UNIFIED_ALTERNATIVE: other,
                        UNIFIED_TRANSITION: transition_id(cls, other),
                    }
                )
        return pd.DataFrame(rows)

    def test_two_tokens_fails_for_three_way_split(self):
        with pytest.raises(ValueError, match="requires at least 3 distinct"):
            validate_target_class_vocabulary_coverage(
                self._unified_two_tokens(),
                ["positive", "negative"],
                train_ratio=0.34,
                val_ratio=0.33,
                test_ratio=0.33,
                vocabulary_held_out=True,
                factual_class_col=UNIFIED_FACTUAL_CLASS,
                factual_col=UNIFIED_FACTUAL,
                split_col=UNIFIED_SPLIT,
                group_key=[str.casefold],
            )


class TestTargetPairCoverage:
    def test_missing_test_rows_raises(self):
        encoder_df = pd.DataFrame(
            {
                "encoded": [0.9, -0.9, 0.8, -0.8, -0.7],
                "label": [1.0, -1.0, 1.0, -1.0, -1.0],
                "source_id": ["christian", "muslim", "christian", "muslim", "jewish"],
                "target_id": ["muslim", "christian", "muslim", "christian", "muslim"],
                "data_split": ["train", "train", "train", "test", "test"],
                "type": ["training"] * 5,
            }
        )
        with pytest.raises(ValueError, match="christian.*test"):
            validate_target_pair_encoder_split_coverage(
                encoder_df,
                ["christian", "muslim"],
                ("train", "test"),
            )

    def test_split_generalization_uses_target_pair_only(self):
        encoder_df = pd.DataFrame(
            {
                "encoded": [0.9, -0.9, 0.8, -0.8],
                "label": [1.0, -1.0, 1.0, -1.0],
                "source_id": ["christian", "muslim", "christian", "muslim"],
                "target_id": ["muslim", "christian", "muslim", "christian"],
                "data_split": ["train", "train", "test", "test"],
                "type": ["training"] * 4,
            }
        )
        metrics = get_encoder_metrics_from_dataframe(
            encoder_df,
            target_classes=["christian", "muslim"],
            generalization_splits=("train", "test"),
        )
        sg = metrics["split_generalization"]
        assert set(sg["agreement_by_feature_class"]) == {"christian", "muslim"}
        assert "jewish" not in sg["agreement_by_feature_class"]

    def test_incomplete_coverage_raises_in_metrics(self):
        encoder_df = pd.DataFrame(
            {
                "encoded": [0.9, -0.9, 0.8, -0.8, 0.0],
                "label": [1.0, -1.0, 1.0, -1.0, 0.0],
                "source_id": ["christian", "muslim", "christian", "muslim", "jewish"],
                "target_id": ["muslim", "christian", "muslim", "christian", "muslim"],
                "data_split": ["train", "train", "train", "test", "test"],
                "type": ["training"] * 5,
            }
        )
        with pytest.raises(ValueError, match="christian"):
            get_encoder_metrics_from_dataframe(
                encoder_df,
                target_classes=["christian", "muslim"],
                generalization_splits=("train", "test"),
            )
