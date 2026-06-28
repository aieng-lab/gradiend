"""Tests for TrainerSuite feature-class inference from unresolved trainer data inputs."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from gradiend.trainer.suite import (
    SuitePairDefinition,
    SymmetricTrainerSuite,
    _default_child_id,
    _default_child_label,
    _feature_classes_from_tabular,
    _feature_classes_from_unified,
    _infer_target_classes_from_pair_inputs,
    _infer_target_classes_from_inputs,
    _normalize_class_merge_transition_groups,
    _normalize_pair_definitions,
    _resolve_suite_training_view,
)
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from gradiend.trainer.core.unified_schema import (
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL_CLASS,
)
def _merged_factual_df(rows=None):
    if rows is None:
        rows = [
            {"masked": "[MASK] here", "split": "train", "label_class": "3SG", "label": "he"},
            {"masked": "[MASK] there", "split": "train", "label_class": "3PL", "label": "they"},
        ]
    return pd.DataFrame(rows)


def _merged_with_alternative_df(*, splits=None):
    if splits is None:
        splits = ["train"]
    rows = []
    for split in splits:
        rows.extend(
            [
                {
                    "masked": "[MASK] here",
                    "split": split,
                    "label_class": "3SG",
                    "label": "he",
                    "alternative_class": "3PL",
                    "alternative": "they",
                },
                {
                    "masked": "[MASK] there",
                    "split": split,
                    "label_class": "3PL",
                    "label": "they",
                    "alternative_class": "3SG",
                    "alternative": "he",
                },
            ]
        )
    return pd.DataFrame(rows)


@pytest.fixture
def data_dir():
    with tempfile.TemporaryDirectory(prefix="suite_data_") as tmp:
        yield Path(tmp)


def _per_class_dict():
    return {
        "3SG": pd.DataFrame(
            {
                "masked": ["[MASK] here"],
                "split": ["train"],
                "3SG": ["he"],
            }
        ),
        "3PL": pd.DataFrame(
            {
                "masked": ["[MASK] there"],
                "split": ["train"],
                "3PL": ["they"],
            }
        ),
    }


def _per_class_dict_three_classes():
    return {
        "3SG": pd.DataFrame({"masked": ["[MASK] here"], "split": ["train"], "3SG": ["he"]}),
        "3PL": pd.DataFrame({"masked": ["[MASK] there"], "split": ["train"], "3PL": ["they"]}),
        "Other": pd.DataFrame({"masked": ["[MASK] person"], "split": ["train"], "Other": ["they"]}),
    }


class TestSuiteFeatureClassHelpers:
    def test_feature_classes_from_unified(self):
        unified = pd.DataFrame(
            {
                UNIFIED_FACTUAL_CLASS: ["3SG", "3PL", "3SG"],
                UNIFIED_ALTERNATIVE_CLASS: ["3PL", "3SG", "3PL"],
            }
        )
        assert _feature_classes_from_unified(unified) == ["3SG", "3PL"]

    def test_feature_classes_from_tabular_uses_label_class_col_kwarg(self):
        df = _merged_factual_df()
        classes = _feature_classes_from_tabular(
            df,
            config=None,
            trainer_kwargs={"label_class_col": "label_class"},
        )
        assert classes == ["3SG", "3PL"]

    def test_feature_classes_from_tabular_reads_config_columns(self):
        df = pd.DataFrame(
            {
                "src_cls": ["A", "B"],
                "tgt_cls": ["B", "A"],
            }
        )
        config = type("Cfg", (), {"label_class_col": "src_cls", "alternative_class_col": "tgt_cls"})()
        classes = _feature_classes_from_tabular(df, config=config)
        assert classes == ["A", "B"]


class TestResolveSuiteTrainingView:
    def test_resolve_in_memory_dataframe(self):
        df = _merged_factual_df()
        resolved = _resolve_suite_training_view(
            trainer_cls=TextPredictionTrainer,
            trainer_args=("bert-base-uncased",),
            trainer_kwargs={"data": df},
        )
        assert resolved is df

    def test_resolve_per_class_dict(self):
        class_dfs = _per_class_dict()
        resolved = _resolve_suite_training_view(
            trainer_cls=TextPredictionTrainer,
            trainer_args=("bert-base-uncased",),
            trainer_kwargs={"data": class_dfs},
        )
        assert resolved is class_dfs
        assert set(resolved.keys()) == {"3SG", "3PL"}

    def test_resolve_csv_path_falls_back_to_raw_table(self, data_dir):
        path = data_dir / "training.csv"
        _merged_factual_df().to_csv(path, index=False)
        resolved = _resolve_suite_training_view(
            trainer_cls=TextPredictionTrainer,
            trainer_args=("bert-base-uncased",),
            trainer_kwargs={"data": str(path), "masked_col": "masked", "split_col": "split"},
        )
        assert isinstance(resolved, pd.DataFrame)
        assert "label_class" in resolved.columns
        assert set(resolved["label_class"]) == {"3SG", "3PL"}

    def test_resolve_csv_path_peeks_unified_when_alternative_columns_present(self, data_dir):
        path = data_dir / "training.csv"
        _merged_with_alternative_df(splits=["train", "validation", "test"]).to_csv(path, index=False)
        resolved = _resolve_suite_training_view(
            trainer_cls=TextPredictionTrainer,
            trainer_args=("bert-base-uncased",),
            trainer_kwargs={
                "data": str(path),
                "alternative_col": "alternative",
                "alternative_class_col": "alternative_class",
            },
        )
        assert isinstance(resolved, pd.DataFrame)
        assert UNIFIED_FACTUAL_CLASS in resolved.columns
        assert UNIFIED_ALTERNATIVE_CLASS in resolved.columns

    @patch.object(TextPredictionTrainer, "peek_unified_training_data", return_value=None)
    @patch("gradiend.trainer.suite.load_hf_per_class")
    def test_resolve_hf_per_class_id(self, mock_load_hf, _mock_peek):
        class_dfs = _per_class_dict()
        mock_load_hf.return_value = class_dfs
        resolved = _resolve_suite_training_view(
            trainer_cls=TextPredictionTrainer,
            trainer_args=("bert-base-uncased",),
            trainer_kwargs={"data": "org/example-dataset", "all_classes": ["3SG", "3PL"]},
        )
        mock_load_hf.assert_called_once()
        assert resolved is class_dfs


class TestNormalizeClassMergeTransitionGroups:
    def test_accepts_cluster_list(self):
        normalized = _normalize_class_merge_transition_groups([["1SG", "3SG"], ["1PL", "3PL"]])
        assert normalized == [["1SG", "3SG"], ["1PL", "3PL"]]

    def test_rejects_dict_format(self):
        with pytest.raises(TypeError, match="list of class clusters"):
            _normalize_class_merge_transition_groups({"1SG": ["3SG"]})

    def test_normalize_pair_definitions_preserves_transition_groups(self):
        definitions = _normalize_pair_definitions(
            [
                SuitePairDefinition(
                    target_classes=("1st", "3rd"),
                    child_id="merged_person_1vs3",
                    class_merge_map={"1st": ["1SG", "1PL"], "3rd": ["3SG", "3PL"]},
                    class_merge_transition_groups=[["1SG", "3SG"], ["1PL", "3PL"]],
                ),
            ],
            pair_id_fn=_default_child_id,
            pair_label_fn=_default_child_label,
        )
        assert definitions is not None
        assert definitions[0].class_merge_transition_groups == [["1SG", "3SG"], ["1PL", "3PL"]]


class TestInferTargetClassesFromPairInputs:
    def test_unions_classes_from_pair_definitions(self):
        classes = _infer_target_classes_from_pair_inputs(
            pair_definitions=[
                SuitePairDefinition(target_classes=("white", "black")),
                SuitePairDefinition(target_classes=("white", "asian")),
                SuitePairDefinition(target_classes=("black", "asian")),
            ],
        )
        assert classes == ["white", "black", "asian"]

    def test_suite_builds_with_hf_data_and_pair_definitions_without_network(self):
        pair_definitions = [
            SuitePairDefinition(target_classes=("white", "black"), child_id="race_white_black"),
            SuitePairDefinition(target_classes=("white", "asian"), child_id="race_white_asian"),
            SuitePairDefinition(target_classes=("black", "asian"), child_id="race_black_asian"),
        ]
        with patch("gradiend.trainer.suite.load_hf_per_class") as mock_load_hf:
            mock_load_hf.side_effect = AssertionError("load_hf_per_class should not run")
            suite = SymmetricTrainerSuite(
                TextPredictionTrainer,
                model="bert-base-uncased",
                data="aieng-lab/gradiend_race_data",
                masked_col="masked",
                pair_definitions=pair_definitions,
            )
        assert suite.target_classes == ["white", "black", "asian"]
        assert len(suite.pairs) == 3
        mock_load_hf.assert_not_called()

    def test_infers_from_pair_definitions_without_loading_hf_data(self):
        with patch("gradiend.trainer.suite.load_hf_per_class") as mock_load_hf:
            mock_load_hf.side_effect = AssertionError("load_hf_per_class should not run")
            classes = _infer_target_classes_from_inputs(
                target_classes=None,
                trainer_kwargs={"data": "org/example-hf-dataset"},
                trainer_cls=TextPredictionTrainer,
                trainer_args=("bert-base-uncased",),
                pair_definitions=[
                    SuitePairDefinition(target_classes=("christian", "muslim")),
                    SuitePairDefinition(target_classes=("christian", "jewish")),
                    SuitePairDefinition(target_classes=("muslim", "jewish")),
                ],
            )
        assert classes == ["christian", "muslim", "jewish"]
        mock_load_hf.assert_not_called()


class TestInferTargetClassesFromInputs:
    def test_explicit_target_classes_win(self):
        classes = _infer_target_classes_from_inputs(
            target_classes=["A", "B"],
            trainer_kwargs={"data": _merged_factual_df(), "all_classes": ["X", "Y"]},
            trainer_cls=TextPredictionTrainer,
        )
        assert classes == ["A", "B"]

    def test_all_classes_used_before_data_resolution(self):
        classes = _infer_target_classes_from_inputs(
            target_classes=None,
            trainer_kwargs={"all_classes": ["3SG", "3PL", "Other"]},
            trainer_cls=TextPredictionTrainer,
        )
        assert classes == ["3SG", "3PL", "Other"]

    def test_infers_from_in_memory_dict(self):
        classes = _infer_target_classes_from_inputs(
            target_classes=None,
            trainer_kwargs={"data": _per_class_dict_three_classes(), "use_class_names_as_columns": True},
            trainer_cls=TextPredictionTrainer,
        )
        assert classes == ["3SG", "3PL", "Other"]

    def test_infers_from_in_memory_dataframe(self):
        classes = _infer_target_classes_from_inputs(
            target_classes=None,
            trainer_kwargs={"data": _merged_factual_df()},
            trainer_cls=TextPredictionTrainer,
        )
        assert classes == ["3SG", "3PL"]

    def test_infers_from_csv_path_without_target_classes(self, data_dir):
        path = data_dir / "training.csv"
        _merged_factual_df().to_csv(path, index=False)
        classes = _infer_target_classes_from_inputs(
            target_classes=None,
            trainer_kwargs={"data": str(path), "masked_col": "masked", "split_col": "split"},
            trainer_cls=TextPredictionTrainer,
            trainer_args=("bert-base-uncased",),
        )
        assert classes == ["3SG", "3PL"]

    def test_raises_when_classes_cannot_be_inferred(self):
        with pytest.raises(ValueError, match="could not determine feature classes"):
            _infer_target_classes_from_inputs(
                target_classes=None,
                trainer_kwargs={},
                trainer_cls=TextPredictionTrainer,
                trainer_args=("bert-base-uncased",),
            )


class TestPeekUnifiedTrainingData:
    def test_peek_merged_dataframe_with_alternative_columns(self):
        unified = TextPredictionTrainer.peek_unified_training_data(
            data=_merged_with_alternative_df(splits=["train", "validation", "test"]),
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        assert unified is not None
        assert UNIFIED_FACTUAL_CLASS in unified.columns
        assert UNIFIED_ALTERNATIVE_CLASS in unified.columns
        assert set(unified[UNIFIED_FACTUAL_CLASS]) == {"3SG", "3PL"}

    def test_peek_csv_path_with_explicit_alternative_columns(self, data_dir):
        path = data_dir / "training.csv"
        _merged_with_alternative_df(splits=["train", "validation", "test"]).to_csv(path, index=False)
        unified = TextPredictionTrainer.peek_unified_training_data(
            data=str(path),
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        assert unified is not None
        assert len(unified) == 6
        assert UNIFIED_FACTUAL_CLASS in unified.columns


class TestSymmetricTrainerSuiteDataInputs:
    def test_infers_from_csv_path_without_explicit_target_classes(self, data_dir):
        path = data_dir / "training.csv"
        _merged_factual_df().to_csv(path, index=False)
        suite = SymmetricTrainerSuite(
            TextPredictionTrainer,
            model="bert-base-uncased",
            data=str(path),
            masked_col="masked",
            split_col="split",
        )
        assert suite.target_classes == ["3SG", "3PL"]
        assert suite.pairs == [("3PL", "3SG")]

    def test_uses_all_classes_for_unresolved_path(self, data_dir):
        path = data_dir / "training.csv"
        _merged_factual_df().to_csv(path, index=False)
        suite = SymmetricTrainerSuite(
            TextPredictionTrainer,
            model="bert-base-uncased",
            data=str(path),
            all_classes=["3SG", "3PL"],
            masked_col="masked",
            split_col="split",
        )
        assert suite.target_classes == ["3SG", "3PL"]

    def test_infers_from_path_object(self, data_dir):
        path = data_dir / "training.csv"
        _merged_factual_df().to_csv(path, index=False)
        suite = SymmetricTrainerSuite(
            TextPredictionTrainer,
            model="bert-base-uncased",
            data=path,
            masked_col="masked",
            split_col="split",
        )
        assert suite.target_classes == ["3SG", "3PL"]

    def test_infers_from_in_memory_dataframe(self):
        suite = SymmetricTrainerSuite(
            TextPredictionTrainer,
            model="bert-base-uncased",
            data=_merged_factual_df(),
        )
        assert suite.target_classes == ["3SG", "3PL"]
        assert len(suite.pairs) == 1

    def test_infers_three_classes_from_per_class_dict(self):
        suite = SymmetricTrainerSuite(
            TextPredictionTrainer,
            model="bert-base-uncased",
            data=_per_class_dict_three_classes(),
            use_class_names_as_columns=True,
        )
        assert suite.target_classes == ["3SG", "3PL", "Other"]
        assert len(suite.pairs) == 3

    def test_default_child_ids_sort_class_names_alphabetically(self):
        suite = SymmetricTrainerSuite(
            TextPredictionTrainer,
            model="bert-base-uncased",
            data=_per_class_dict_three_classes(),
            use_class_names_as_columns=True,
            target_pairs=[("3SG", "3PL"), ("Other", "3SG")],
        )
        assert set(suite.trainers) == {"3PL__3SG", "3SG__Other"}
        assert suite.pair_by_id["3PL__3SG"] == ("3PL", "3SG")
        assert ("3SG", "3PL") not in suite.pairs

    def test_infers_from_config_data_path(self, data_dir):
        path = data_dir / "training.csv"
        _merged_factual_df().to_csv(path, index=False)
        config = TextPredictionConfig(data=path, masked_col="masked", split_col="split")
        suite = SymmetricTrainerSuite(
            TextPredictionTrainer,
            model="bert-base-uncased",
            config=config,
        )
        assert suite.target_classes == ["3SG", "3PL"]

    def test_pair_definitions_validated_against_resolved_csv(self, data_dir):
        path = data_dir / "training.csv"
        _merged_with_alternative_df(splits=["train", "validation", "test"]).to_csv(path, index=False)
        SymmetricTrainerSuite(
            TextPredictionTrainer,
            model="bert-base-uncased",
            data=str(path),
            alternative_col="alternative",
            alternative_class_col="alternative_class",
            pair_definitions=[
                SuitePairDefinition(target_classes=("3SG", "3PL"), child_id="pair_3sg_3pl"),
            ],
        )

    def test_pair_definitions_rejected_when_transition_missing(self, data_dir):
        path = data_dir / "training.csv"
        _merged_with_alternative_df(splits=["train", "validation", "test"]).to_csv(path, index=False)
        with pytest.raises(ValueError, match="not present in the training data"):
            SymmetricTrainerSuite(
                TextPredictionTrainer,
                model="bert-base-uncased",
                data=str(path),
                alternative_col="alternative",
                alternative_class_col="alternative_class",
                pair_definitions=[
                    SuitePairDefinition(target_classes=("3SG", "Other"), child_id="missing_pair"),
                ],
            )

    def test_skips_generated_incomplete_classes_from_sidecar(self, data_dir):
        path = data_dir / "training.csv"
        pd.DataFrame(
            [
                {"masked": "[MASK] a", "split": "train", "label_class": "A", "label": "a"},
                {"masked": "[MASK] c", "split": "train", "label_class": "C", "label": "c"},
            ]
        ).to_csv(path, index=False)
        pd.DataFrame(
            [{"masked": "[MASK] b", "split": "train", "label_class": "B", "label": "b"}]
        ).to_csv(data_dir / "training_incomplete_classes.csv", index=False)

        suite = SymmetricTrainerSuite(
            TextPredictionTrainer,
            model="bert-base-uncased",
            data=path,
            target_classes=["A", "B", "C"],
        )
        assert suite.target_classes == ["A", "C"]
        assert suite.pairs == [("A", "C")]

    def test_raises_when_feature_classes_cannot_be_resolved(self):
        with pytest.raises(ValueError, match="could not determine feature classes"):
            SymmetricTrainerSuite(
                TextPredictionTrainer,
                model="bert-base-uncased",
            )
