"""Tests for data generation: TextPredictionDataCreator output formats, auto-split, preprocess (newline splitting)."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from gradiend.data.core import apply_split_group_key, split_dataframe_by_group_key
from gradiend.data.text import TextPreprocessConfig, iter_sentences_from_texts, preprocess_texts
from gradiend.data.text.classification.creator import TextClassificationDataCreator
from gradiend.data.text.filter_config import TextFilterConfig
from gradiend.data.text.prediction import filter_engine
from gradiend.data.text.prediction.creator import TextPredictionDataCreator


class TestPreprocessNewlineSplitting:
    """Sentences must not contain \\n; newlines are split first."""

    def test_iter_sentences_from_texts_none_config_splits_on_newline(self):
        texts = ["line one\nline two", "single"]
        out = list(iter_sentences_from_texts(texts, config=None))
        assert out == ["line one", "line two", "single"]
        assert all("\n" not in s for s in out)

    def test_preprocess_texts_split_to_sentences_splits_newline_first(self):
        config = TextPreprocessConfig(split_to_sentences=True)
        texts = ["First. Second.\nThird. Fourth."]
        out = preprocess_texts(texts, config=config)
        assert all("\n" not in s for s in out)
        # Newlines are split first, then regex on .!? so we get four sentences
        assert "First." in out and "Second." in out and "Third." in out and "Fourth." in out
        assert len(out) == 4

    def test_preprocess_texts_split_to_sentence_windows_is_non_overlapping(self):
        config = TextPreprocessConfig(split_to_sentences=2)
        texts = ["First. Second. Third. Fourth. Fifth."]
        out = preprocess_texts(texts, config=config)
        assert out == ["First. Second.", "Third. Fourth.", "Fifth."]

    def test_iter_sentences_from_texts_split_to_sentence_windows(self):
        config = TextPreprocessConfig(split_to_sentences=2)
        out = list(iter_sentences_from_texts(["First. Second. Third. Fourth."], config=config))
        assert out == ["First. Second.", "Third. Fourth."]


class TestCreatorOutputFormats:
    """Creator output: merged has label_class, label, feature_class_id (string id)."""

    def test_generate_training_data_unified_has_merged_columns(self):
        creator = TextPredictionDataCreator(
            base_data=["He walks.", "She runs.", "They play."],
            feature_targets=[
                TextFilterConfig(id="3SG", targets=["he", "she"]),
                TextFilterConfig(id="3PL", targets=["they"]),
            ],
            min_left_context_words=0,
            seed=42,
        )
        result = creator.generate_training_data(
            max_size_per_class=10, format="unified", min_rows_per_class_for_split=0
        )
        assert "label_class" in result.columns
        assert "label" in result.columns
        assert "masked" in result.columns
        assert "split" in result.columns
        assert "feature_class_id" in result.columns
        assert result["feature_class_id"].dtype == object or result["feature_class_id"].iloc[0] in ("3SG", "3PL")

    def test_generate_training_data_strict_balances_per_target_token(self):
        creator = TextPredictionDataCreator(
            base_data=["He walks."] * 30 + ["She runs."] * 5 + ["They play."] * 20,
            feature_targets=[
                TextFilterConfig(id="3SG", targets=["he", "she"]),
                TextFilterConfig(id="3PL", targets=["they"]),
            ],
            min_left_context_words=0,
            seed=42,
        )
        result = creator.generate_training_data(
            max_size_per_class=50,
            format="unified",
            min_rows_per_class_for_split=0,
            balance="strict",
            min_rows_per_target_for_balance=10,
        )
        sg = result[result["label_class"] == "3SG"]
        counts = sg["label"].value_counts()
        assert len(counts) >= 2
        assert counts["He"] == counts["She"] == 10

    def test_generate_training_data_strict_cross_class_cap_preserves_per_target_balance(self):
        creator = TextPredictionDataCreator(
            base_data=["He walks."] * 30 + ["She runs."] * 5 + ["They play."] * 20,
            feature_targets=[
                TextFilterConfig(id="3SG", targets=["he", "she"]),
                TextFilterConfig(id="3PL", targets=["they"]),
            ],
            min_left_context_words=0,
            seed=42,
        )
        result = creator.generate_training_data(
            max_size_per_class=50,
            format="unified",
            min_rows_per_class_for_split=0,
            balance="strict",
            min_rows_per_target_for_balance=10,
        )
        sg = result[result["label_class"] == "3SG"]
        pl = result[result["label_class"] == "3PL"]
        sg_counts = sg["label"].value_counts()
        assert sg_counts["He"] == sg_counts["She"]
        assert len(sg) == len(pl)
        assert sg_counts["He"] == len(pl) // len(sg_counts)

    def test_generate_training_data_try_preserves_target_counts(self):
        creator = TextPredictionDataCreator(
            base_data=["He walks."] * 30 + ["She runs."] * 5 + ["They play."] * 20,
            feature_targets=[
                TextFilterConfig(id="3SG", targets=["he", "she"]),
                TextFilterConfig(id="3PL", targets=["they"]),
            ],
            min_left_context_words=0,
            seed=42,
        )
        result = creator.generate_training_data(
            max_size_per_class=50,
            format="unified",
            min_rows_per_class_for_split=0,
            balance="try",
            min_rows_per_target_for_balance=10,
        )
        sg = result[result["label_class"] == "3SG"]
        counts = sg["label"].value_counts()
        assert counts["He"] > counts["She"]

    def test_generate_training_data_min_left_context_words(self):
        creator = TextPredictionDataCreator(
            base_data=[
                "They arrived.",
                "Yesterday They arrived.",
                "Yesterday afternoon They arrived.",
            ],
            feature_targets=[
                TextFilterConfig(id="3PL", targets=["they"]),
            ],
            min_left_context_words=2,
            seed=42,
        )
        result = creator.generate_training_data(
            max_size_per_class=10,
            format="unified",
            min_rows_per_class_for_split=0,
            balance=False,
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
        )
        assert len(result) == 1
        assert result.iloc[0]["masked"] == "Yesterday afternoon [MASK] arrived."

    def test_generate_training_data_min_left_context_words_per_config_override(self):
        creator = TextPredictionDataCreator(
            base_data=[
                "They arrived.",
                "Yesterday They arrived.",
                "Yesterday afternoon They arrived.",
            ],
            feature_targets=[
                TextFilterConfig(id="3PL", targets=["they"], min_left_context_words=2),
            ],
            min_left_context_words=10,
            seed=42,
        )
        result = creator.generate_training_data(
            max_size_per_class=10,
            format="unified",
            min_rows_per_class_for_split=0,
            balance=False,
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
        )
        assert len(result) == 1
        assert result.iloc[0]["masked"] == "Yesterday afternoon [MASK] arrived."

    def test_generate_training_data_sentence_window_allows_second_sentence_context(self):
        creator = TextPredictionDataCreator(
            base_data=["Alice waved. They arrived. They left."],
            preprocess=TextPreprocessConfig(split_to_sentences=2),
            feature_targets=[
                TextFilterConfig(id="3PL", targets=["they"]),
            ],
            min_left_context_words=2,
            seed=42,
        )
        result = creator.generate_training_data(
            max_size_per_class=10,
            format="unified",
            min_rows_per_class_for_split=0,
            balance=False,
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
        )
        assert len(result) == 1
        assert result.iloc[0]["masked"] == "Alice waved. [MASK] arrived."

    def test_generate_training_data_auto_split_proportions(self):
        creator = TextPredictionDataCreator(
            base_data=["He walks."] * 20 + ["They run."] * 20,
            feature_targets=[
                TextFilterConfig(id="3SG", targets=["he"]),
                TextFilterConfig(id="3PL", targets=["they"]),
            ],
            min_left_context_words=0,
            seed=42,
        )
        result = creator.generate_training_data(
            max_size_per_class=30,
            format="unified",
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
        splits = result["split"].value_counts()
        assert "train" in splits.index
        assert "validation" in splits.index
        assert "test" in splits.index
        total = len(result)
        assert splits.get("train", 0) <= int(total * 0.8) + 2
        assert splits.get("validation", 0) <= int(total * 0.1) + 2
        assert splits.get("test", 0) <= int(total * 0.1) + 2

    def test_generate_training_data_ratios_must_sum_to_one(self):
        creator = TextPredictionDataCreator(
            base_data=["He walks."],
            feature_targets=[TextFilterConfig(targets=["he"])],
        )
        with pytest.raises(ValueError, match="sum to 1.0"):
            creator.generate_training_data(
                train_ratio=0.5,
                val_ratio=0.2,
                test_ratio=0.2,
                min_rows_per_class_for_split=0,
            )

    def test_generate_training_data_excludes_and_saves_incomplete_classes_by_default(self):
        """Classes below the split minimum are saved separately and excluded from main data."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            creator = TextPredictionDataCreator(
                base_data=["He walks."] * 12 + ["They play."],
                feature_targets=[
                    TextFilterConfig(id="3SG", targets=["he"]),
                    TextFilterConfig(id="3PL", targets=["they"]),
                ],
                min_left_context_words=0,
                output_dir=str(base),
                output_format="csv",
                seed=42,
            )
            result = creator.generate_training_data(max_size_per_class=20, format="unified")
            assert set(result["label_class"]) == {"3SG"}

            training_csv = base / "training.csv"
            incomplete_csv = base / "training_incomplete_classes.csv"
            assert training_csv.is_file()
            assert incomplete_csv.is_file()
            saved_main = pd.read_csv(training_csv)
            saved_incomplete = pd.read_csv(incomplete_csv)
            assert set(saved_main["label_class"]) == {"3SG"}
            assert set(saved_incomplete["label_class"]) == {"3PL"}

    def test_generate_training_data_raise_on_incomplete_classes_after_saving(self):
        """Opt-in strict mode raises after saving main and incomplete-class files."""
        with tempfile.TemporaryDirectory() as tmp:
            creator = TextPredictionDataCreator(
                base_data=["He walks."] * 12 + ["They play."],
                feature_targets=[
                    TextFilterConfig(id="3SG", targets=["he"]),
                    TextFilterConfig(id="3PL", targets=["they"]),
                ],
                min_left_context_words=0,
                output_dir=tmp,
                output_format="csv",
                seed=42,
            )
            with pytest.raises(ValueError, match="at least 10 rows per class"):
                creator.generate_training_data(
                    max_size_per_class=20,
                    format="unified",
                    raise_on_incomplete_classes=True,
                )
            assert (Path(tmp) / "training.csv").is_file()
            assert (Path(tmp) / "training_incomplete_classes.csv").is_file()
            # With min_rows_per_class_for_split=0, should succeed
            result = creator.generate_training_data(
                max_size_per_class=20, format="unified", min_rows_per_class_for_split=0
            )
            assert len(result) > 0


class TestCreatorOutputDirAndSave:
    """Creator can write to output_dir with default basenames."""

    def test_generate_training_data_writes_to_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            creator = TextPredictionDataCreator(
                base_data=["He walks.", "They run."],
                feature_targets=[
                    TextFilterConfig(id="3SG", targets=["he"]),
                    TextFilterConfig(id="3PL", targets=["they"]),
                ],
                min_left_context_words=0,
                output_dir=str(base),
                output_format="csv",
                seed=42,
            )
            creator.generate_training_data(
                max_size_per_class=5, format="unified", min_rows_per_class_for_split=0
            )
            training_csv = base / "training.csv"
            assert training_csv.is_file()
            df = pd.read_csv(training_csv)
            assert "label_class" in df.columns and "label" in df.columns

    def test_generate_neutral_data_stops_at_max_size(self):
        creator = TextPredictionDataCreator(
            base_data=[f"Sentence {i} here." for i in range(50)],
            feature_targets=[TextFilterConfig(targets=["he"])],
            seed=42,
        )
        neutral = creator.generate_neutral_data(max_size=5)
        assert len(neutral) <= 5
        assert "text" in neutral.columns

    def test_use_cache_returns_cached_training_when_available(self):
        """When use_cache=True and output_dir is set, generate_training_data returns cached data if file exists."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            creator = TextPredictionDataCreator(
                base_data=["He walks.", "She runs.", "They play."],
                feature_targets=[
                    TextFilterConfig(id="3SG", targets=["he", "she"]),
                    TextFilterConfig(id="3PL", targets=["they"]),
                ],
                output_dir=str(base),
                output_format="csv",
                seed=42,
            )
            first = creator.generate_training_data(
                max_size_per_class=10, format="unified", min_rows_per_class_for_split=0
            )
            assert (base / "training.csv").is_file()
            creator_cached = TextPredictionDataCreator(
                base_data=["He walks.", "She runs.", "They play."],
                feature_targets=[
                    TextFilterConfig(id="3SG", targets=["he", "she"]),
                    TextFilterConfig(id="3PL", targets=["they"]),
                ],
                output_dir=str(base),
                output_format="csv",
                seed=42,
                use_cache=True,
            )
            cached = creator_cached.generate_training_data(
                max_size_per_class=10, format="unified", min_rows_per_class_for_split=0
            )
            pd.testing.assert_frame_equal(first, cached)

    def test_use_cache_returns_cached_neutral_when_available(self):
        """When use_cache=True and output_dir is set, generate_neutral_data returns cached data if file exists."""
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            creator = TextPredictionDataCreator(
                base_data=["Hello world.", "Another sentence."],
                feature_targets=[TextFilterConfig(targets=["he"])],
                output_dir=str(base),
                output_format="csv",
                seed=42,
            )
            first = creator.generate_neutral_data(max_size=10)
            assert (base / "neutral.csv").is_file()
            creator_cached = TextPredictionDataCreator(
                base_data=["Hello world.", "Another sentence."],
                feature_targets=[TextFilterConfig(targets=["he"])],
                output_dir=str(base),
                output_format="csv",
                seed=42,
                use_cache=True,
            )
            cached = creator_cached.generate_neutral_data(max_size=10)
            pd.testing.assert_frame_equal(first, cached)

    def test_generate_training_data_saves_partial_on_keyboard_interrupt(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            original_match = filter_engine._match_sentence_compiled
            calls = {"n": 0}

            def interrupt_after_first_sentence(sentence, compiled, doc=None, nlp=None):
                calls["n"] += 1
                if calls["n"] > 2:
                    raise KeyboardInterrupt
                return original_match(sentence, compiled, doc=doc, nlp=nlp)

            monkeypatch.setattr(filter_engine, "_match_sentence_compiled", interrupt_after_first_sentence)
            creator = TextPredictionDataCreator(
                base_data=["He and they run.", "He walks.", "They play."],
                feature_targets=[
                    TextFilterConfig(id="3SG", targets=["he"]),
                    TextFilterConfig(id="3PL", targets=["they"]),
                ],
                min_left_context_words=0,
                output_dir=str(base),
                output_format="csv",
                seed=42,
            )

            result = creator.generate_training_data(format="unified")

            training_csv = base / "training.csv"
            assert training_csv.is_file()
            saved = pd.read_csv(training_csv)
            assert len(result) == len(saved) > 0
            assert set(saved["label_class"]).issubset({"3SG", "3PL"})

    def test_classification_training_saves_partial_on_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            calls = {"n": 0}

            def label_fn(text):
                calls["n"] += 1
                if calls["n"] > 2:
                    raise KeyboardInterrupt
                return "contains_a" if "a" in text else "other"

            creator = TextClassificationDataCreator(
                base_data=["alpha", "beta", "gamma"],
                label_fn=label_fn,
                output_dir=str(base),
                output_format="csv",
                seed=42,
            )

            result = creator.generate_training_data()

            training_csv = base / "training.csv"
            assert training_csv.is_file()
            saved = pd.read_csv(training_csv)
            assert len(result) == len(saved) == 2
            assert set(saved["text"]) == {"alpha", "beta"}
            assert "split" in saved.columns

    def test_many_classes_use_summary_logging(self, caplog):
        feature_targets = [
            TextFilterConfig(id=f"class_{i}", targets=[f"token{i}"])
            for i in range(12)
        ]
        base_data = [f"token{i} appears." for i in range(12)]
        creator = TextPredictionDataCreator(
            base_data=base_data,
            feature_targets=feature_targets,
            min_left_context_words=0,
            seed=42,
        )

        with caplog.at_level("INFO"):
            creator.generate_training_data(
                max_size_per_class=1,
                format="unified",
                min_rows_per_class_for_split=0,
            )

        log_text = "\n".join(record.getMessage() for record in caplog.records)
        assert "Training filter stats: classes=12" in log_text
        assert "Count examples:" in log_text
        assert "Training filter stats (instances per group):" not in log_text

    def test_capped_classes_stop_being_checked(self, monkeypatch):
        feature_targets = [
            TextFilterConfig(id=f"class_{i}", targets=[f"token{i}"])
            for i in range(3)
        ]
        base_data = [
            "token0 token1 token2",
            "token0 token1 token2",
            "token0 token1 token2",
        ]
        calls = {"class_0": 0, "class_1": 0, "class_2": 0}
        original_match = filter_engine._match_sentence_compiled

        def count_calls(sentence, compiled, doc=None, nlp=None):
            calls[compiled.class_id] += 1
            return original_match(sentence, compiled, doc=doc, nlp=nlp)

        monkeypatch.setattr(filter_engine, "_match_sentence_compiled", count_calls)
        creator = TextPredictionDataCreator(
            base_data=base_data,
            feature_targets=feature_targets,
            min_left_context_words=0,
            seed=42,
        )

        creator.generate_training_data(
            max_size_per_class=1,
            format="unified",
            min_rows_per_class_for_split=0,
        )

        assert calls == {"class_0": 1, "class_1": 1, "class_2": 1}


class TestSplitByTarget:
    """Vocabulary-held-out splits: each target token confined to one split."""

    def test_split_dataframe_by_group_key_no_label_spans_splits(self):
        df = pd.DataFrame(
            {
                "label": ["love", "love", "hate", "hate", "amazing", "amazing"],
                "text": ["a", "b", "c", "d", "e", "f"],
            }
        )
        out = split_dataframe_by_group_key(df, "label", 0.6, 0.2, 0.2, seed=42)
        for label in out["label"].unique():
            splits = out.loc[out["label"] == label, "split"].unique()
            assert len(splits) == 1, f"label {label!r} appears in splits {splits.tolist()}"

    def test_generate_training_data_split_group_col(self):
        creator = TextPredictionDataCreator(
            base_data=[
                "I love this.",
                "I love that.",
                "It is great.",
                "A great result.",
                "They hate it.",
                "They hate them.",
                "That was awful.",
                "An awful result.",
                "This is bad.",
                "A bad result.",
                "It is amazing.",
                "So amazing.",
            ],
            feature_targets=[
                TextFilterConfig(id="positive", targets=["love", "amazing", "great"]),
                TextFilterConfig(id="negative", targets=["hate", "awful", "bad"]),
            ],
            min_left_context_words=0,
            seed=42,
            split_group_col="label",
        )
        result = creator.generate_training_data(
            max_size_per_class=20,
            format="unified",
            min_rows_per_class_for_split=0,
        )
        for label in result["label"].dropna().unique():
            splits = result.loc[result["label"] == label, "split"].unique()
            assert len(splits) == 1

    def test_split_group_key_casefold_collapses_variants(self):
        df = pd.DataFrame(
            {
                "label": ["amazing", "AMAZING", "Amazing", "hate", "HATE", "great"],
                "text": list("abcdef"),
            }
        )
        out = split_dataframe_by_group_key(
            df,
            "label",
            0.6,
            0.2,
            0.2,
            seed=42,
            group_key=[str.strip, str.casefold],
        )
        amazing_splits = out.loc[out["label"].str.casefold() == "amazing", "split"].unique()
        assert len(amazing_splits) == 1
        hate_splits = out.loc[out["label"].str.casefold() == "hate", "split"].unique()
        assert len(hate_splits) == 1
        assert apply_split_group_key(" AMAZING ", [str.strip, str.casefold]) == "amazing"
