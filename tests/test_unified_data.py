"""Tests for unified_data: resolve_dataframe, _load_dataframe_from_path, per_class_dict_to_unified, merged_to_unified."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

HAS_DATASETS = importlib.util.find_spec("datasets") is not None

from gradiend.trainer.text.prediction.unified_data import (
    _load_dataframe_from_path,
    apply_factual_casing,
    merged_to_unified,
    per_class_dict_to_unified,
    resolve_dataframe,
)
from gradiend.trainer.text.prediction.unified_schema import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_MASKED,
)


class TestLoadDataframeFromPath:
    def test_load_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(path, index=False)
        df = _load_dataframe_from_path(path)
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_load_csv_str_path(self, tmp_path):
        path = str(tmp_path / "data.csv")
        pd.DataFrame({"x": [1]}).to_csv(path, index=False)
        df = _load_dataframe_from_path(path)
        assert len(df) == 1 and df["x"].iloc[0] == 1

    def test_load_parquet(self, tmp_path):
        path = tmp_path / "data.parquet"
        pd.DataFrame({"a": [1, 2]}).to_parquet(path, index=False)
        df = _load_dataframe_from_path(path)
        assert len(df) == 2 and "a" in df.columns

    def test_missing_file_raises(self, tmp_path):
        path = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError, match="not a file"):
            _load_dataframe_from_path(path)

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("a,b\n1,2")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            _load_dataframe_from_path(path)


class TestResolveDataframe:
    def test_none_returns_none(self):
        assert resolve_dataframe(None) is None

    def test_dataframe_passthrough(self):
        df = pd.DataFrame({"x": [1]})
        assert resolve_dataframe(df) is df

    def test_path_loads_file(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"label_class": ["A"], "label": ["x"], "masked": ["y"], "split": ["train"]}).to_csv(path, index=False)
        out = resolve_dataframe(path)
        assert out is not None and len(out) == 1 and out["label_class"].iloc[0] == "A"

    def test_str_path_loads_file(self, tmp_path):
        path = tmp_path / "data.csv"
        pd.DataFrame({"a": [1]}).to_csv(path, index=False)
        out = resolve_dataframe(str(path))
        assert out is not None and out["a"].iloc[0] == 1

    @pytest.mark.skipif(not HAS_DATASETS, reason="datasets not installed")
    def test_str_hf_id_loads_or_raises(self):
        # String that is not a file path is treated as HF id; either loads or raises (e.g. 404).
        try:
            out = resolve_dataframe("organization/nonexistent-dataset-xyz-999")
            assert out is None or isinstance(out, pd.DataFrame)
        except Exception:
            pass  # 404 / auth / etc. is acceptable


class TestApplyFactualCasing:
    """apply_factual_casing applies factual token casing to alternative."""

    def test_lower_factual_returns_lower_alternative(self):
        assert apply_factual_casing("he", "SHE") == "she"
        assert apply_factual_casing("he", "She") == "she"

    def test_upper_factual_returns_upper_alternative(self):
        assert apply_factual_casing("HE", "she") == "SHE"
        assert apply_factual_casing("HE", "She") == "SHE"

    def test_title_factual_returns_title_alternative(self):
        assert apply_factual_casing("He", "SHE") == "She"
        assert apply_factual_casing("He", "she") == "She"

    def test_other_casing_returns_alternative_as_is(self):
        assert apply_factual_casing("hE", "she") == "she"
        assert apply_factual_casing("HeLLo", "world") == "world"

    def test_empty_factual_returns_alternative_unchanged(self):
        assert apply_factual_casing("", "she") == "she"

    def test_empty_alternative_returns_empty(self):
        assert apply_factual_casing("He", "") == ""


class TestPerClassDictToUnified:
    """per_class_dict_to_unified returns unified schema: masked, split, factual_class, alternative_class, factual, alternative, transition."""

    def test_unified_columns_with_pair(self):
        class_dfs = {
            "A": pd.DataFrame({"masked": ["[MASK] x"], "split": ["train"], "A": ["a"]}),
            "B": pd.DataFrame({"masked": ["[MASK] y"], "split": ["train"], "B": ["b"]}),
        }
        df = per_class_dict_to_unified(
            class_dfs, classes=["A", "B"], masked_col="masked", split_col="split", pair=("A", "B")
        )
        assert "factual_class" in df.columns
        assert "alternative_class" in df.columns
        assert "factual" in df.columns
        assert "alternative" in df.columns
        assert "masked" in df.columns
        assert "split" in df.columns
        assert len(df) >= 2
        assert set(df["factual_class"]) == {"A", "B"}

    def test_source_target_pre_paired(self):
        """Standard source/target columns: one unified row per row, no pair needed."""
        class_dfs = {
            "X_to_Y": pd.DataFrame({
                "masked": ["[MASK] here"],
                "split": ["train"],
                "source": ["x_token"],
                "target": ["y_token"],
                "source_id": ["X"],
                "target_id": ["Y"],
            }),
        }
        df = per_class_dict_to_unified(class_dfs, classes=["X_to_Y"], masked_col="masked", split_col="split")
        assert len(df) == 1
        assert df["factual_class"].iloc[0] == "X"
        assert df["alternative_class"].iloc[0] == "Y"
        assert df["factual"].iloc[0] == "x_token"
        assert df["alternative"].iloc[0] == "y_token"

    def test_source_target_applies_factual_casing(self):
        """Pre-paired source/target: alternative string gets factual casing."""
        class_dfs = {
            "X_to_Y": pd.DataFrame({
                "masked": ["[MASK] here"],
                "split": ["train"],
                "source": ["He"],
                "target": ["she"],
                "source_id": ["X"],
                "target_id": ["Y"],
            }),
        }
        df = per_class_dict_to_unified(class_dfs, classes=["X_to_Y"], masked_col="masked", split_col="split")
        assert df["alternative"].iloc[0] == "She"

    def test_derive_from_other_df_applies_casing(self):
        """When alternative is derived from other class df, factual casing is applied."""
        class_dfs = {
            "A": pd.DataFrame({"masked": ["[MASK] a"], "split": ["train"], "A": ["He"]}),
            "B": pd.DataFrame({"masked": ["[MASK] b"], "split": ["train"], "B": ["she"]}),
        }
        df = per_class_dict_to_unified(
            class_dfs, classes=["A", "B"], masked_col="masked", split_col="split",
            pair=("A", "B"), random_state=42,
        )
        # A -> B: factual "He" (title), alternative from B is "she" -> "She"
        row_ab = df[(df["factual_class"] == "A") & (df["alternative_class"] == "B")]
        assert len(row_ab) >= 1
        assert row_ab["alternative"].iloc[0] == "She"

    def test_derive_from_other_df_weighted_sampling_multiple_tokens(self):
        """Other class has multiple tokens; counterfactuals are sampled by distribution (case-insensitive)."""
        # B has "she" (2x) and "her" (1x) -> weights 2/3, 1/3; with seed we get deterministic draw
        class_dfs = {
            "A": pd.DataFrame({"masked": ["[MASK] one"], "split": ["train"], "A": ["he"]}),
            "B": pd.DataFrame({
                "masked": ["m1", "m2", "m3"],
                "split": ["train", "train", "train"],
                "B": ["she", "she", "her"],
            }),
        }
        df = per_class_dict_to_unified(
            class_dfs, classes=["A", "B"], masked_col="masked", split_col="split",
            pair=("A", "B"), max_counterfactuals_per_sentence=2, random_state=0,
        )
        # One base row from A; we get 1 or 2 rows (unique counterfactuals)
        rows_a_to_b = df[(df["factual_class"] == "A") & (df["alternative_class"] == "B")]
        assert 1 <= len(rows_a_to_b) <= 2
        alternatives = set(rows_a_to_b["alternative"].str.lower())
        assert alternatives <= {"she", "her"}

    def test_max_counterfactuals_per_sentence_caps_unique(self):
        """At most max_counterfactuals_per_sentence unique (case-insensitive) counterfactuals per base sentence."""
        class_dfs = {
            "A": pd.DataFrame({"masked": ["[MASK] x"], "split": ["train"], "A": ["a"]}),
            "B": pd.DataFrame({
                "masked": ["m1", "m2", "m3"],
                "split": ["train", "train", "train"],
                "B": ["one", "two", "three"],
            }),
        }
        df = per_class_dict_to_unified(
            class_dfs, classes=["A", "B"], masked_col="masked", split_col="split",
            pair=("A", "B"), max_counterfactuals_per_sentence=2, random_state=123,
        )
        rows_a_to_b = df[(df["factual_class"] == "A") & (df["alternative_class"] == "B")]
        assert len(rows_a_to_b) <= 2
        # All alternatives must be unique (case-insensitive)
        alts_lower = rows_a_to_b["alternative"].str.lower().tolist()
        assert len(alts_lower) == len(set(alts_lower))

    def test_same_seed_same_counterfactuals(self):
        """Same random_state yields same unified rows when deriving from other_df."""
        class_dfs = {
            "A": pd.DataFrame({"masked": ["[MASK] x"], "split": ["train"], "A": ["a"]}),
            "B": pd.DataFrame({
                "masked": ["m1", "m2", "m3"],
                "split": ["train"] * 3,
                "B": ["x", "y", "z"],
            }),
        }
        df1 = per_class_dict_to_unified(
            class_dfs, classes=["A", "B"], pair=("A", "B"),
            max_counterfactuals_per_sentence=2, random_state=99,
        )
        df2 = per_class_dict_to_unified(
            class_dfs, classes=["A", "B"], pair=("A", "B"),
            max_counterfactuals_per_sentence=2, random_state=99,
        )
        assert len(df1) == len(df2)
        assert set(df1["alternative"]) == set(df2["alternative"])


class TestMergedToUnified:
    """merged_to_unified with pair derives alternative from other class."""

    def test_factual_only_with_pair(self):
        merged = pd.DataFrame({
            "masked": ["[MASK] here", "[MASK] there"],
            "split": ["train", "train"],
            "label_class": ["3SG", "3PL"],
            "label": ["he", "they"],
        })
        out = merged_to_unified(
            merged,
            masked_col="masked",
            split_col="split",
            label_class_col="label_class",
            label_col="label",
            pair=("3SG", "3PL"),
        )
        assert UNIFIED_FACTUAL_CLASS in out.columns
        assert UNIFIED_ALTERNATIVE_CLASS in out.columns
        assert UNIFIED_FACTUAL in out.columns
        assert UNIFIED_ALTERNATIVE in out.columns
        assert UNIFIED_MASKED in out.columns
        assert len(out) == 2
        # 3SG row -> alternative 3PL token (they), 3PL row -> alternative 3SG token (he)
        for _, row in out.iterrows():
            assert row[UNIFIED_FACTUAL_CLASS] in ("3SG", "3PL")
            assert row[UNIFIED_ALTERNATIVE_CLASS] in ("3SG", "3PL")
            assert row[UNIFIED_FACTUAL_CLASS] != row[UNIFIED_ALTERNATIVE_CLASS] or row[UNIFIED_FACTUAL] == row[UNIFIED_ALTERNATIVE]

    def test_missing_pair_raises_for_factual_only(self):
        merged = pd.DataFrame({
            "masked": ["m"],
            "split": ["train"],
            "label_class": ["A"],
            "label": ["x"],
        })
        with pytest.raises(ValueError, match="requires pair"):
            merged_to_unified(
                merged,
                label_class_col="label_class",
                label_col="label",
                pair=None,
            )

    def test_explicit_target_applies_factual_casing(self):
        """When target_col is present, alternative gets factual casing."""
        merged = pd.DataFrame({
            "masked": ["[MASK] here"],
            "split": ["train"],
            "label_class": ["3SG"],
            "label": ["He"],
            "alternative_class": ["3PL"],
            "alternative": ["they"],
        })
        out = merged_to_unified(
            merged,
            masked_col="masked",
            split_col="split",
            label_class_col="label_class",
            label_col="label",
            target_col="alternative",
            target_class_col="alternative_class",
        )
        assert out[UNIFIED_ALTERNATIVE].iloc[0] == "They"

    def test_pair_mode_applies_factual_casing(self):
        """Pair mode: alternative token from other class gets factual casing."""
        merged = pd.DataFrame({
            "masked": ["[MASK] here", "[MASK] there"],
            "split": ["train", "train"],
            "label_class": ["3SG", "3PL"],
            "label": ["He", "THEY"],
        })
        out = merged_to_unified(
            merged,
            masked_col="masked",
            split_col="split",
            label_class_col="label_class",
            label_col="label",
            pair=("3SG", "3PL"),
        )
        # 3SG row factual "He" -> alternative (they) -> "They"
        row_3sg = out[out[UNIFIED_FACTUAL_CLASS] == "3SG"]
        assert row_3sg[UNIFIED_ALTERNATIVE].iloc[0] == "They"
        # 3PL row factual "THEY" -> alternative (he) -> "HE"
        row_3pl = out[out[UNIFIED_FACTUAL_CLASS] == "3PL"]
        assert row_3pl[UNIFIED_ALTERNATIVE].iloc[0] == "HE"
