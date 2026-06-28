"""Tests for vocabulary resplit, encoder split options, and split_generalization."""

from __future__ import annotations

import pandas as pd
import pytest

from gradiend.data.core import (
    apply_split_group_key,
    normalize_split_ratios,
    resplit_unified_dataframe,
    split_dataframe_by_group_key,
)
from gradiend.evaluator.encoder_metrics import get_encoder_metrics_from_dataframe
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
from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer
from gradiend.util.encoder_splits import resolve_encoder_splits


def _race_like_unified() -> pd.DataFrame:
    rows = []
    for cls, tokens in (
        ("white", ["White", "white", "w1", "w2"]),
        ("black", ["Black", "black", "b1", "b2"]),
    ):
        other = "black" if cls == "white" else "white"
        for tok in tokens:
            rows.append(
                {
                    UNIFIED_MASKED: f"The person is [MASK].",
                    UNIFIED_SPLIT: "train",
                    UNIFIED_FACTUAL_CLASS: cls,
                    UNIFIED_ALTERNATIVE_CLASS: other,
                    UNIFIED_FACTUAL: tok,
                    UNIFIED_ALTERNATIVE: other,
                    UNIFIED_TRANSITION: transition_id(cls, other),
                }
            )
    return pd.DataFrame(rows)


class TestNormalizeSplitRatios:
    def test_tuple(self):
        assert normalize_split_ratios((0.8, 0.1, 0.1)) == (0.8, 0.1, 0.1)

    def test_dict(self):
        assert normalize_split_ratios({"train": 0.7, "validation": 0.2, "test": 0.1}) == (0.7, 0.2, 0.1)

    def test_dict_val_alias(self):
        assert normalize_split_ratios({"train": 0.7, "val": 0.2, "test": 0.1}) == (0.7, 0.2, 0.1)


class TestSplitDataframeByGroupKey:
    def test_two_keys_three_splits_raises(self):
        df = pd.DataFrame({"label": ["love", "great", "love", "great"], "value": [1, 2, 3, 4]})
        with pytest.raises(ValueError, match="requires at least 3 distinct"):
            split_dataframe_by_group_key(
                df,
                "label",
                0.34,
                0.33,
                0.33,
                seed=42,
                group_key=[str.casefold],
                class_id="positive",
            )

    def test_five_keys_get_nonempty_test_bucket(self):
        df = pd.DataFrame({"label": [f"w{i}" for i in range(5)] * 2, "value": range(10)})
        out = split_dataframe_by_group_key(
            df,
            "label",
            0.34,
            0.33,
            0.33,
            seed=42,
            group_key=[str.casefold],
        )
        assert "test" in set(out["split"].unique())

    def test_balanced_cycle_rotates_each_word_through_60_20_20_over_five_seed_slots(self):
        df = pd.DataFrame({"label": [f"w{i}" for i in range(10)]})
        assignments = []
        for cycle_index in range(5):
            out = split_dataframe_by_group_key(
                df,
                "label",
                0.6,
                0.2,
                0.2,
                seed=42,
                group_key=[str.casefold],
                balanced_cycle_index=cycle_index,
                balanced_cycle_length=5,
            )
            assignments.append(out.set_index("label")["split"].to_dict())

        # In a five-slot 60/20/20 cycle, every canonical word should appear in
        # train three times, validation once, and test once. This is the split
        # policy used by multi-seed held-out-target analysis; convergence can
        # later filter which trained seed checkpoints are included in plots.
        for word in df["label"]:
            word_splits = [assignment[word] for assignment in assignments]
            assert word_splits.count("train") == 3
            assert word_splits.count("validation") == 1
            assert word_splits.count("test") == 1

    def test_balanced_cycle_prioritizes_previous_heldout_words_as_next_train(self):
        df = pd.DataFrame({"label": [f"w{i}" for i in range(10)]})
        first = split_dataframe_by_group_key(
            df,
            "label",
            0.6,
            0.2,
            0.2,
            seed=42,
            group_key=[str.casefold],
            balanced_cycle_index=0,
            balanced_cycle_length=2,
        ).set_index("label")["split"].to_dict()
        second = split_dataframe_by_group_key(
            df,
            "label",
            0.6,
            0.2,
            0.2,
            seed=42,
            group_key=[str.casefold],
            balanced_cycle_index=1,
            balanced_cycle_length=2,
        ).set_index("label")["split"].to_dict()

        # With only two seed slots, perfect 60/20/20 coverage per word is
        # impossible. The balanced policy therefore maximizes useful reuse:
        # every word held out in the first run is pulled into training in the
        # second run, and the second run's held-out set comes from the first
        # run's training words.
        first_heldout = {word for word, split in first.items() if split != "train"}
        second_heldout = {word for word, split in second.items() if split != "train"}
        assert first_heldout
        assert all(second[word] == "train" for word in first_heldout)
        assert second_heldout <= {word for word, split in first.items() if split == "train"}


class TestResplitUnified:
    def test_casefold_groups_white_variants(self):
        df = _race_like_unified()
        out = resplit_unified_dataframe(
            df,
            group_col=UNIFIED_FACTUAL,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=42,
            group_key=[str.strip, str.casefold],
        )
        white_splits = out.loc[out[UNIFIED_FACTUAL].str.casefold() == "white", UNIFIED_SPLIT].unique()
        assert len(white_splits) == 1

    def test_different_seeds_change_assignment(self):
        rows = []
        other = "black"
        for i, tok in enumerate(["w1", "w2", "w3", "w4", "w5", "w6"]):
            rows.append(
                {
                    UNIFIED_MASKED: f"Person {i} [MASK].",
                    UNIFIED_SPLIT: "train",
                    UNIFIED_FACTUAL_CLASS: "white",
                    UNIFIED_ALTERNATIVE_CLASS: other,
                    UNIFIED_FACTUAL: tok,
                    UNIFIED_ALTERNATIVE: other,
                    UNIFIED_TRANSITION: transition_id("white", other),
                }
            )
        df = pd.DataFrame(rows)
        out_a = resplit_unified_dataframe(
            df,
            group_col=UNIFIED_FACTUAL,
            train_ratio=0.34,
            val_ratio=0.33,
            test_ratio=0.33,
            seed=1,
            group_key=[str.casefold],
        )
        out_b = resplit_unified_dataframe(
            df,
            group_col=UNIFIED_FACTUAL,
            train_ratio=0.34,
            val_ratio=0.33,
            test_ratio=0.33,
            seed=99,
            group_key=[str.casefold],
        )

        def _token_split_map(frame: pd.DataFrame) -> dict[str, str]:
            result: dict[str, str] = {}
            for key in frame[UNIFIED_FACTUAL].map(lambda x: apply_split_group_key(x, [str.casefold])).unique():
                splits = frame.loc[
                    frame[UNIFIED_FACTUAL].map(lambda x: apply_split_group_key(x, [str.casefold])) == key,
                    UNIFIED_SPLIT,
                ].unique()
                result[str(key)] = str(splits[0])
            return result

        assert _token_split_map(out_a) != _token_split_map(out_b)

    def test_aligned_alternatives_do_not_cross_vocab_splits(self):
        rows = []
        tokens_by_class = {
            "positive": ["love", "happy", "great", "best", "glad"],
            "negative": ["hate", "sad", "bad", "worst", "angry"],
        }
        for cls, tokens in tokens_by_class.items():
            other = "negative" if cls == "positive" else "positive"
            leaked_alt = tokens_by_class[other][0]
            for tok in tokens:
                rows.append(
                    {
                        UNIFIED_MASKED: "I feel [MASK].",
                        UNIFIED_SPLIT: "train",
                        UNIFIED_FACTUAL_CLASS: cls,
                        UNIFIED_ALTERNATIVE_CLASS: other,
                        UNIFIED_FACTUAL: tok,
                        UNIFIED_ALTERNATIVE: leaked_alt,
                        UNIFIED_TRANSITION: transition_id(cls, other),
                    }
                )
        out = resplit_unified_dataframe(
            pd.DataFrame(rows),
            group_col=UNIFIED_FACTUAL,
            train_ratio=0.34,
            val_ratio=0.33,
            test_ratio=0.33,
            seed=42,
            group_key=[str.casefold],
            align_alternatives_with_split_vocab=True,
        )

        def key(value: str) -> str:
            return apply_split_group_key(value, [str.casefold])

        vocab_by_split_class = {
            (split, cls): set(group[UNIFIED_FACTUAL].map(key))
            for (split, cls), group in out.groupby([UNIFIED_SPLIT, UNIFIED_FACTUAL_CLASS])
        }
        for _, row in out.iterrows():
            allowed = vocab_by_split_class[(row[UNIFIED_SPLIT], row[UNIFIED_ALTERNATIVE_CLASS])]
            assert key(row[UNIFIED_ALTERNATIVE]) in allowed


class TestTrainerEarlyVocabularyValidation:
    def test_two_tokens_per_class_fails_at_resplit(self):
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
        trainer = TextPredictionTrainer(
            model="distilbert-base-cased",
            target_classes=["positive", "negative"],
            split_col=None,
            split_group_key=[str.casefold],
            split_ratios=(0.34, 0.33, 0.33),
            args=__import__("gradiend").TrainingArguments(seed=1, experiment_dir=None),
        )
        trainer._combined_data_template = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="requires at least 3 distinct"):
            trainer._apply_vocabulary_splits(1)


class TestTrainerDefaultSplitCol:
    def test_omitted_split_col_keeps_random_splits_for_single_token_classes(self):
        rows = []
        for cls, tok in (("3SG", "he"), ("3PL", "they")):
            other = "3PL" if cls == "3SG" else "3SG"
            for i, split in enumerate(["train", "validation", "test"] * 4):
                rows.append(
                    {
                        "masked": f"Person {i} [MASK] quickly.",
                        "split": split,
                        "label_class": cls,
                        "alternative_class": other,
                        "label": tok,
                        "alternative": "they" if other == "3PL" else "he",
                    }
                )
        rows.append(
            {
                "masked": "I [MASK] here.",
                "split": "train",
                "label_class": "1SG",
                "alternative_class": "1PL",
                "label": "I",
                "alternative": "we",
            }
        )
        trainer = TextPredictionTrainer(
            model="t5-small",
            data=pd.DataFrame(rows),
            target_classes=["3SG", "3PL"],
            args=__import__("gradiend").TrainingArguments(seed=1, experiment_dir=None),
        )
        trainer._ensure_data()
        assert trainer.config.split_col == "split"
        assert trainer._combined_data_template is None

    def test_explicit_split_col_none_still_raises_when_not_viable(self):
        rows = []
        for cls, tok in (("3SG", "he"), ("3PL", "they")):
            other = "3PL" if cls == "3SG" else "3SG"
            for _ in range(12):
                rows.append(
                    {
                        "masked": "The runner [MASK] fast.",
                        "split": "train",
                        "label_class": cls,
                        "alternative_class": other,
                        "label": tok,
                        "alternative": "they" if other == "3PL" else "he",
                    }
                )
        with pytest.raises(ValueError, match="split_col=None requires"):
            trainer = TextPredictionTrainer(
                model="t5-small",
                data=pd.DataFrame(rows),
                target_classes=["3SG", "3PL"],
                split_col=None,
                args=__import__("gradiend").TrainingArguments(seed=1, experiment_dir=None),
            )
            trainer._ensure_data()


class TestTrainerVocabularyResplit:
    def test_split_col_none_applies_resplit(self):
        rows = []
        for cls, toks in (
            ("white", ["White", "white", "w1", "w2", "w3"]),
            ("black", ["Black", "black", "b1", "b2", "b3"]),
        ):
            other = "black" if cls == "white" else "white"
            for tok in toks:
                for _ in range(3):
                    rows.append(
                        {
                            "masked": "The [MASK] person.",
                            "split": "train",
                            "label_class": cls,
                            "label": tok,
                            "alternative_class": other,
                            "alternative": other,
                        }
                    )
        merged = pd.DataFrame(rows)
        trainer = TextPredictionTrainer(
            model="distilbert-base-cased",
            data=merged,
            target_classes=["white", "black"],
            split_col=None,
            split_group_key=[str.strip, str.casefold],
            split_ratios=(0.34, 0.33, 0.33),
            args=__import__("gradiend").TrainingArguments(seed=7, experiment_dir=None),
        )
        trainer._ensure_data()
        combined = trainer.combined_data
        assert combined is not None
        for key in combined[UNIFIED_FACTUAL].map(lambda x: apply_split_group_key(x, [str.casefold])).unique():
            splits = combined.loc[
                combined[UNIFIED_FACTUAL].map(lambda x: apply_split_group_key(x, [str.casefold])) == key,
                UNIFIED_SPLIT,
            ].unique()
            assert len(splits) == 1

        vocab_by_split_class = {
            (split, cls): set(group[UNIFIED_FACTUAL].map(lambda x: apply_split_group_key(x, [str.casefold])))
            for (split, cls), group in combined.groupby([UNIFIED_SPLIT, UNIFIED_FACTUAL_CLASS])
        }
        for _, row in combined.iterrows():
            allowed = vocab_by_split_class[(row[UNIFIED_SPLIT], row[UNIFIED_ALTERNATIVE_CLASS])]
            alt_key = apply_split_group_key(row[UNIFIED_ALTERNATIVE], [str.casefold])
            assert alt_key in allowed

    def test_refresh_stable_vs_per_seed(self):
        from gradiend import TrainingArguments

        trainer = TextPredictionTrainer(
            model="distilbert-base-cased",
            target_classes=["white", "black"],
            split_col=None,
            split_group_key=[str.casefold],
            split_ratios=(0.34, 0.33, 0.33),
            args=TrainingArguments(seed=5, split_resplit_per_seed=False, experiment_dir=None),
        )
        rows = []
        other = "black"
        for i, tok in enumerate(["w1", "w2", "w3", "w4", "w5", "w6"]):
            rows.append(
                {
                    UNIFIED_MASKED: f"Person {i} [MASK].",
                    UNIFIED_SPLIT: "train",
                    UNIFIED_FACTUAL_CLASS: "white",
                    UNIFIED_ALTERNATIVE_CLASS: other,
                    UNIFIED_FACTUAL: tok,
                    UNIFIED_ALTERNATIVE: other,
                    UNIFIED_TRANSITION: transition_id("white", other),
                }
            )
        trainer._combined_data_template = pd.DataFrame(rows)
        trainer._apply_vocabulary_splits(5)
        trainer._data_loaded = True
        stable_map = trainer._combined_data.set_index(UNIFIED_FACTUAL)[UNIFIED_SPLIT].to_dict()
        trainer._refresh_data_splits_for_seed(99, trainer.training_args)
        assert trainer._combined_data.set_index(UNIFIED_FACTUAL)[UNIFIED_SPLIT].to_dict() == stable_map

        trainer.training_args.split_resplit_per_seed = True
        trainer._refresh_data_splits_for_seed(99, trainer.training_args)
        per_seed_map = trainer._combined_data.set_index(UNIFIED_FACTUAL)[UNIFIED_SPLIT].to_dict()
        trainer._refresh_data_splits_for_seed(5, trainer.training_args)
        other_map = trainer._combined_data.set_index(UNIFIED_FACTUAL)[UNIFIED_SPLIT].to_dict()
        assert per_seed_map != other_map or stable_map != other_map

    def test_balanced_cycle_refresh_uses_explicit_cycle_slot_not_model_seed(self):
        from gradiend import TrainingArguments

        trainer = TextPredictionTrainer(
            model="distilbert-base-cased",
            target_classes=["white", "black"],
            split_col=None,
            split_group_key=[str.casefold],
            split_ratios=(0.6, 0.2, 0.2),
            args=TrainingArguments(
                seed=100,
                max_seeds=5,
                split_resplit_per_seed=True,
                split_resplit_strategy="balanced_cycle",
                min_convergent_seeds=5,
                experiment_dir=None,
            ),
        )
        rows = []
        for i, tok in enumerate([f"w{i}" for i in range(8)]):
            rows.append(
                {
                    UNIFIED_MASKED: f"Person {i} [MASK].",
                    UNIFIED_SPLIT: "train",
                    UNIFIED_FACTUAL_CLASS: "white",
                    UNIFIED_ALTERNATIVE_CLASS: "black",
                    UNIFIED_FACTUAL: tok,
                    UNIFIED_ALTERNATIVE: "black",
                    UNIFIED_TRANSITION: transition_id("white", "black"),
                }
            )
        trainer._combined_data_template = pd.DataFrame(rows)
        trainer._data_loaded = True

        trainer._refresh_data_splits_for_seed(100, trainer.training_args, split_cycle_index=2, split_cycle_length=5)
        first = trainer._combined_data.set_index(UNIFIED_FACTUAL)[UNIFIED_SPLIT].to_dict()
        trainer._refresh_data_splits_for_seed(999, trainer.training_args, split_cycle_index=2, split_cycle_length=5)
        second = trainer._combined_data.set_index(UNIFIED_FACTUAL)[UNIFIED_SPLIT].to_dict()

        # During managed multi-seed training, model seed and data split slot are
        # deliberately decoupled. A failed model seed can move its split slot to
        # the back of the queue; later evaluation must reconstruct the same
        # slot from seed-report metadata rather than from seed_value - base_seed.
        assert second == first


class TestEncoderSplitOptions:
    @pytest.mark.parametrize(
        "split,expected",
        [
            ("test", ["test"]),
            ("all", ["train", "validation", "test"]),
            (["train", "test"], ["train", "test"]),
        ],
    )
    def test_resolve_encoder_splits(self, split, expected):
        assert resolve_encoder_splits(split) == expected

    def test_split_generalization_present_for_multi_split_rows(self):
        encoder_df = pd.DataFrame(
            {
                "encoded": [0.9, 0.88, -0.8, -0.82],
                "label": [1.0, 1.0, -1.0, -1.0],
                "source_id": ["white", "white", "black", "black"],
                "target_id": ["black"] * 4,
                "data_split": ["train", "test", "train", "test"],
                "type": ["training"] * 4,
            }
        )
        metrics = get_encoder_metrics_from_dataframe(encoder_df)
        sg = metrics["split_generalization"]
        assert "agreement" in sg
        assert "white" in sg["agreement_by_feature_class"]
        assert "black" in sg["agreement_by_feature_class"]

    def test_create_training_data_preserves_data_split_for_split_all(self):
        from tests.conftest import MockTokenizer

        rows = []
        for cls, tok, split in (
            ("white", "w1", "train"),
            ("white", "w2", "test"),
            ("white", "w3", "validation"),
            ("black", "b1", "train"),
            ("black", "b2", "test"),
            ("black", "b3", "validation"),
        ):
            other = "black" if cls == "white" else "white"
            rows.append(
                {
                    UNIFIED_MASKED: "The person is [MASK].",
                    UNIFIED_SPLIT: split,
                    UNIFIED_FACTUAL_CLASS: cls,
                    UNIFIED_ALTERNATIVE_CLASS: other,
                    UNIFIED_FACTUAL: tok,
                    UNIFIED_ALTERNATIVE: other,
                    UNIFIED_TRANSITION: transition_id(cls, other),
                }
            )
        trainer = TextPredictionTrainer(
            model="distilbert-base-cased",
            target_classes=["white", "black"],
            args=__import__("gradiend").TrainingArguments(seed=1, experiment_dir=None),
        )
        trainer._combined_data = pd.DataFrame(rows)
        trainer._data_loaded = True
        tokenizer = MockTokenizer()
        ds = trainer.create_training_data(tokenizer, split="all", batch_size=1)
        splits_seen: set[str] = set()
        for i in range(len(ds)):
            item = ds[i]
            assert "data_split" in item, f"missing data_split at index {i}"
            splits_seen.add(str(item["data_split"]))
        assert splits_seen == {"train", "test", "validation"}
