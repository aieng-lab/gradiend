"""Tests for class_merge_map: merging base classes into higher-level classes."""

import pandas as pd
import pytest

from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from gradiend.trainer.text.prediction.unified_data import (
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_TRANSITION,
)


def _four_class_data():
    """Merged DataFrame with 4 base classes (1SG, 1PL, 3SG, 3PL) and explicit alternative columns."""
    return pd.DataFrame([
        {"masked": "[MASK] here", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "1PL", "alternative": "we"},
        {"masked": "[MASK] there", "split": "train", "label_class": "1PL", "label": "we", "alternative_class": "1SG", "alternative": "I"},
        {"masked": "[MASK] went", "split": "train", "label_class": "3SG", "label": "he", "alternative_class": "3PL", "alternative": "they"},
        {"masked": "[MASK] left", "split": "train", "label_class": "3PL", "label": "they", "alternative_class": "3SG", "alternative": "he"},
    ])


def _four_class_per_class_dict():
    """Per-class dict with 1SG, 1PL, 3SG, 3PL."""
    return {
        "1SG": pd.DataFrame({"masked": ["[MASK] here"], "split": ["train"], "1SG": ["I"]}),
        "1PL": pd.DataFrame({"masked": ["[MASK] there"], "split": ["train"], "1PL": ["we"]}),
        "3SG": pd.DataFrame({"masked": ["[MASK] went"], "split": ["train"], "3SG": ["he"]}),
        "3PL": pd.DataFrame({"masked": ["[MASK] left"], "split": ["train"], "3PL": ["they"]}),
    }


class TestClassMergeMapDataLayer:
    """class_merge_map applies at data layer; combined_data returns merged view."""

    def test_effective_data_has_merged_class_names(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        cd = trainer.combined_data
        assert cd is not None
        merged_factual = set(cd[UNIFIED_FACTUAL_CLASS].unique())
        merged_alt = set(cd[UNIFIED_ALTERNATIVE_CLASS].unique())
        assert merged_factual <= {"singular", "plural"}
        assert merged_alt <= {"singular", "plural"}
        assert "1SG" not in merged_factual and "3SG" not in merged_factual

    def test_target_classes_inferred_when_merge_has_two_keys(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        assert config.target_classes is None
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        assert trainer.target_classes is not None
        assert set(trainer.target_classes) == {"singular", "plural"}
        assert trainer.pair == ("singular", "plural")

    def test_all_classes_from_effective_data(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert set(trainer.all_classes) == {"singular", "plural"}

    def test_transitions_use_merged_names(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        cd = trainer.combined_data
        transitions = set(cd[UNIFIED_TRANSITION].unique())
        assert "singular→plural" in transitions or "plural→singular" in transitions
        assert "1SG→1PL" not in transitions


class TestClassMergeMapValidation:
    """Validation: duplicate base class, base in multiple merges."""

    def test_duplicate_base_class_raises(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3SG"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(ValueError, match="3SG.*multiple merged"):
            trainer._ensure_data()

    def test_explicit_target_classes_with_merge(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            target_classes=["singular", "plural"],
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer.target_classes == ["singular", "plural"]


class TestClassMergeMapNeutral:
    """Base classes not in any merge value remain as neutral (unmapped)."""

    def test_unmapped_base_class_stays_in_data(self):
        """Class '2' (you) not in merge map stays as-is in effective data."""
        df = pd.DataFrame([
            {"masked": "[MASK] you", "split": "train", "label_class": "2", "label": "you", "alternative_class": "1SG", "alternative": "I"},
            {"masked": "[MASK] here", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "2", "alternative": "you"},
        ])
        config = TextPredictionConfig(
            data=df,
            target_classes=["singular", "plural"],
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        cd = trainer.combined_data
        assert "2" in set(cd[UNIFIED_FACTUAL_CLASS].unique()) or "2" in set(cd[UNIFIED_ALTERNATIVE_CLASS].unique())
        assert "singular" in set(cd[UNIFIED_FACTUAL_CLASS].unique())


class TestClassMergeMapCreateTrainingData:
    """create_training_data with class_merge_map yields correct labels."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("bert-base-uncased")

    def test_create_training_data_merged_pipeline(self, tokenizer):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1


class TestClassMergeMapDecoderTargets:
    """Decoder eval targets work via effective data (merged class names)."""

    def test_infer_decoder_targets_merged_classes(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        targets = trainer._infer_decoder_eval_targets()
        assert "singular" in targets
        assert "plural" in targets
        assert "I" in targets["singular"] or "he" in targets["singular"]
        assert "we" in targets["plural"] or "they" in targets["plural"]
