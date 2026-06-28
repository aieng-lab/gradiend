import pandas as pd
import pytest

from gradiend.comparison.anchor_aligned import (
    aggregate_anchor_aligned_encoding_rows,
    build_anchor_aligned_encoding_rows,
    compute_anchor_aligned_encoding_matrix,
    compute_dense_anchor_aligned_encoding_matrix,
)
from gradiend.trainer.core.unified_schema import transition_id


def test_anchor_aligned_encoding_orients_second_class_and_aggregates():
    encoder_summary = {
        "ab": {
            "correlation": 0.9,
            "encoder_df": pd.DataFrame(
                [
                    {"source_id": "A", "encoded": 0.8, "type": "training"},
                    {"source_id": "B", "encoded": -0.6, "type": "training"},
                ]
            ),
        },
        "ac": {
            "correlation": 0.7,
            "encoder_df": pd.DataFrame(
                [
                    {"source_id": "A", "encoded": 0.4, "type": "training"},
                    {"source_id": "C", "encoded": -0.2, "type": "training"},
                ]
            ),
        },
    }
    pair_by_id = {"ab": ("A", "B"), "ac": ("A", "C")}
    result = compute_anchor_aligned_encoding_matrix(
        pair_by_id=pair_by_id,
        encoder_summary=encoder_summary,
        feature_classes=["A", "B", "C"],
        aggregate="mean",
    )

    matrix = pd.DataFrame(result["matrix"], index=result["rows"], columns=result["columns"])
    assert matrix.loc["A", "A"] == pytest.approx(0.6)
    assert matrix.loc["A", "B"] == pytest.approx(-0.6)
    assert matrix.loc["B", "B"] == pytest.approx(0.6)
    assert matrix.loc["A", "C"] == pytest.approx(-0.2)
    assert result["measure"] == "anchor_aligned_encoding_factual_mean"


def test_anchor_aligned_encoding_averages_across_seed_encoder_dfs():
    encoder_summary = {
        "ab": {
            "encoder_dfs": [
                pd.DataFrame(
                    [
                        {"source_id": "A", "target_id": "B", "encoded": 1.0, "type": "training"},
                        {"source_id": "B", "target_id": "A", "encoded": -1.0, "type": "training"},
                    ]
                ),
                pd.DataFrame(
                    [
                        {"source_id": "A", "target_id": "B", "encoded": 0.2, "type": "training"},
                        {"source_id": "B", "target_id": "A", "encoded": -0.2, "type": "training"},
                    ]
                ),
            ],
        }
    }
    pair_by_id = {"ab": ("A", "B")}
    result = compute_anchor_aligned_encoding_matrix(
        pair_by_id=pair_by_id,
        encoder_summary=encoder_summary,
        feature_classes=["A", "B"],
        aggregate="mean",
    )
    matrix = pd.DataFrame(result["matrix"], index=result["rows"], columns=result["columns"])
    assert matrix.loc["A", "A"] == pytest.approx(0.6)
    assert matrix.loc["A", "B"] == pytest.approx(-0.6)
    assert matrix.loc["B", "B"] == pytest.approx(0.6)


def test_build_anchor_aligned_encoding_rows_skips_neutral_rows():
    aligned = build_anchor_aligned_encoding_rows(
        pair_by_id={"ab": ("A", "B")},
        encoder_summary={
            "ab": {
                "encoder_df": pd.DataFrame(
                    [
                        {"source_id": "A", "encoded": 1.0, "type": "training"},
                        {"source_id": "A", "encoded": 9.0, "type": "neutral_dataset"},
                    ]
                )
            }
        },
        feature_classes=["A", "B"],
    )
    assert len(aligned) == 2
    by_anchor = aligned.set_index("anchor_class")["aligned_mean"].to_dict()
    assert by_anchor["A"] == pytest.approx(1.0)
    assert by_anchor["B"] == pytest.approx(-1.0)


def test_aggregate_anchor_aligned_encoding_rows_reindexes_classes():
    aligned = pd.DataFrame(
        [
            {"anchor_class": "A", "eval_class": "B", "aligned_mean": 0.5},
        ]
    )
    matrix = aggregate_anchor_aligned_encoding_rows(aligned, ["A", "B", "C"], aggregate="mean")
    assert list(matrix.index) == ["A", "B", "C"]
    assert list(matrix.columns) == ["A", "B", "C"]
    assert matrix.loc["A", "B"] == pytest.approx(0.5)
    assert pd.isna(matrix.loc["B", "A"])


def test_anchor_aligned_counts_distinct_gradiend_transition_contributions():
    encoder_summary = {
        "ab": {
            "encoder_df": pd.DataFrame(
                [
                    {
                        "factual_id": "A",
                        "counterfactual_id": "B",
                        "transition_id": "A->B",
                        "encoded": 1.0,
                        "type": "training",
                    },
                    {
                        "factual_id": "A",
                        "counterfactual_id": "B",
                        "transition_id": "A->B",
                        "encoded": 3.0,
                        "type": "training",
                    },
                ]
            )
        },
        "ac": {
            "encoder_df": pd.DataFrame(
                [
                    {
                        "factual_id": "A",
                        "counterfactual_id": "C",
                        "transition_id": "A->C",
                        "encoded": 5.0,
                        "type": "training",
                    },
                ]
            )
        },
    }
    result = compute_anchor_aligned_encoding_matrix(
        pair_by_id={"ab": ("A", "B"), "ac": ("A", "C")},
        encoder_summary=encoder_summary,
        feature_classes=["A", "B", "C"],
        alignment="factual",
    )
    matrix = pd.DataFrame(result["matrix"], index=result["rows"], columns=result["columns"])
    counts = pd.DataFrame(result["n_matrix"], index=result["rows"], columns=result["columns"])
    raw_counts = pd.DataFrame(result["raw_n_matrix"], index=result["rows"], columns=result["columns"])

    assert matrix.loc["A", "A"] == pytest.approx(3.5)
    assert counts.loc["A", "A"] == 2
    assert raw_counts.loc["A", "A"] == 3
    assert counts.loc["B", "A"] == 1
    assert raw_counts.loc["B", "A"] == 2


def test_anchor_aligned_factual_view_flips_when_source_is_alternative():
    encoder_summary = {
        "ab": {
            "encoder_df": pd.DataFrame(
                [
                    {
                        "factual_id": "A",
                        "counterfactual_id": "B",
                        "transition_id": "A->B",
                        "label": 1.0,
                        "encoded": 1.0,
                        "type": "training",
                    },
                ]
            )
        }
    }
    pair_by_id = {"ab": ("A", "B")}
    factual = compute_anchor_aligned_encoding_matrix(
        pair_by_id=pair_by_id,
        encoder_summary=encoder_summary,
        feature_classes=["A", "B"],
        alignment="factual",
        source_by_id={"ab": "factual"},
    )
    alternative = compute_anchor_aligned_encoding_matrix(
        pair_by_id=pair_by_id,
        encoder_summary=encoder_summary,
        feature_classes=["A", "B"],
        alignment="factual",
        source_by_id={"ab": "alternative"},
    )
    m_f = pd.DataFrame(factual["matrix"], index=factual["rows"], columns=factual["columns"])
    m_a = pd.DataFrame(alternative["matrix"], index=alternative["rows"], columns=alternative["columns"])
    assert m_f.loc["A", "A"] == pytest.approx(1.0)
    assert m_a.loc["A", "A"] == pytest.approx(-1.0)


def test_anchor_aligned_counterfactual_and_transition_alignment():
    encoder_summary = {
        "ab": {
            "encoder_df": pd.DataFrame(
                [
                    {
                        "factual_id": "A",
                        "counterfactual_id": "B",
                        "transition_id": "A->B",
                        "encoded": 2.0,
                        "type": "training",
                    },
                ]
            )
        }
    }
    pair_by_id = {"ab": ("A", "B")}

    by_counterfactual = compute_anchor_aligned_encoding_matrix(
        pair_by_id=pair_by_id,
        encoder_summary=encoder_summary,
        feature_classes=["A", "B"],
        alignment="counterfactual",
    )
    cf_counts = pd.DataFrame(
        by_counterfactual["n_matrix"],
        index=by_counterfactual["rows"],
        columns=by_counterfactual["columns"],
    )
    assert cf_counts.loc["A", "B"] == 1
    assert cf_counts.loc["A", "A"] == 0

    by_transition = compute_anchor_aligned_encoding_matrix(
        pair_by_id=pair_by_id,
        encoder_summary=encoder_summary,
        feature_classes=["A", "B"],
        alignment="transition",
    )
    assert by_transition["columns"] == [transition_id("A", "B")]
    transition_matrix = pd.DataFrame(
        by_transition["matrix"],
        index=by_transition["rows"],
        columns=by_transition["columns"],
    )
    ab = transition_id("A", "B")
    assert transition_matrix.loc["A", ab] == pytest.approx(2.0)
    assert transition_matrix.loc["B", ab] == pytest.approx(-2.0)


def test_factual_and_counterfactual_alignment_differ_when_source_is_alternative():
    encoder_summary = {
        "ab": {
            "encoder_df": pd.DataFrame(
                [
                    {
                        "factual_id": "A",
                        "counterfactual_id": "B",
                        "source_id": "B",
                        "target_id": "B",
                        "input_type": "alternative",
                        "encoded": 2.0,
                        "type": "training",
                    },
                ]
            )
        }
    }
    pair_by_id = {"ab": ("A", "B")}

    by_factual = compute_anchor_aligned_encoding_matrix(
        pair_by_id=pair_by_id,
        encoder_summary=encoder_summary,
        feature_classes=["A", "B"],
        alignment="factual",
        source_by_id={"ab": "alternative"},
    )
    by_counterfactual = compute_anchor_aligned_encoding_matrix(
        pair_by_id=pair_by_id,
        encoder_summary=encoder_summary,
        feature_classes=["A", "B"],
        alignment="counterfactual",
        source_by_id={"ab": "alternative"},
    )
    factual_matrix = pd.DataFrame(
        by_factual["matrix"],
        index=by_factual["rows"],
        columns=by_factual["columns"],
    )
    counterfactual_matrix = pd.DataFrame(
        by_counterfactual["matrix"],
        index=by_counterfactual["rows"],
        columns=by_counterfactual["columns"],
    )

    assert factual_matrix.loc["A", "A"] == pytest.approx(-2.0)
    assert factual_matrix.loc["A", "B"] == 0 or pd.isna(factual_matrix.loc["A", "B"])
    assert counterfactual_matrix.loc["A", "B"] == pytest.approx(2.0)
    assert counterfactual_matrix.loc["A", "A"] == 0 or pd.isna(counterfactual_matrix.loc["A", "A"])


def test_dense_anchor_aligned_cross_task_fills_cross_domain_columns(monkeypatch):
    trainers = {
        "gender": object(),
        "sentiment": object(),
    }

    def _fake_cross_task_summary(trainers_arg, feature_classes, **kwargs):
        assert trainers_arg is trainers
        return {
            "gender": {
                "encoder_df": pd.DataFrame(
                    [
                        {
                            "factual_id": "F",
                            "counterfactual_id": "M",
                            "transition_id": "F->M",
                            "encoded": 0.8,
                            "type": "training",
                        },
                        {
                            "factual_id": "positive",
                            "counterfactual_id": "negative",
                            "transition_id": "positive->negative",
                            "encoded": 0.1,
                            "type": "training",
                        },
                    ]
                )
            },
            "sentiment": {
                "encoder_df": pd.DataFrame(
                    [
                        {
                            "factual_id": "F",
                            "counterfactual_id": "M",
                            "transition_id": "F->M",
                            "encoded": 0.2,
                            "type": "training",
                        },
                    ]
                )
            },
        }

    monkeypatch.setattr(
        "gradiend.comparison.feature_cross_encoding.build_cross_task_encoder_summary",
        _fake_cross_task_summary,
    )
    monkeypatch.setattr(
        "gradiend.comparison.anchor_aligned.pair_by_id_from_trainers",
        lambda _: {"gender": ("F", "M"), "sentiment": ("positive", "negative")},
    )

    sparse = compute_anchor_aligned_encoding_matrix(
        pair_by_id={"gender": ("F", "M")},
        encoder_summary={
            "gender": {
                "encoder_df": pd.DataFrame(
                    [
                        {
                            "factual_id": "F",
                            "counterfactual_id": "M",
                            "encoded": 0.8,
                            "type": "training",
                        },
                        {
                            "factual_id": "M",
                            "counterfactual_id": "F",
                            "encoded": -0.6,
                            "type": "training",
                        },
                    ]
                )
            }
        },
        feature_classes=["F", "M", "positive"],
    )
    sparse_matrix = pd.DataFrame(sparse["matrix"], index=sparse["rows"], columns=sparse["columns"])
    assert pd.isna(sparse_matrix.loc["F", "positive"])

    dense = compute_dense_anchor_aligned_encoding_matrix(
        trainers,
        ["F", "M", "positive"],
        alignment="factual",
    )
    dense_matrix = pd.DataFrame(dense["matrix"], index=dense["rows"], columns=dense["columns"])
    assert dense_matrix.loc["F", "positive"] == pytest.approx(0.1)
    assert dense_matrix.loc["F", "F"] == pytest.approx(0.8)


def test_encoder_rows_label_source_id_by_actual_gradient_source():
    from gradiend.util.encoding_rows import encode_dataset_to_rows

    class _Dataset:
        source = "alternative"

        def __len__(self):
            return 1

        def __iter__(self):
            yield {
                "source": "gradient",
                "label": 1.0,
                "factual_id": "he",
                "alternative_id": "she",
                "feature_class_id": "he->she",
            }

    class _Model:
        def encode(self, grad, return_float=True):
            assert grad == "gradient"
            return 0.25

    rows = encode_dataset_to_rows(_Model(), _Dataset())

    assert rows[0]["source_id"] == "she"
    assert rows[0]["factual_id"] == "he"
    assert rows[0]["counterfactual_id"] == "she"
    assert rows[0]["transition_id"] == "he->she"
    assert rows[0]["input_type"] == "alternative"


def test_orient_encoder_df_by_label_correlation_flips_inverted_encodings():
    from gradiend.comparison.common import orient_encoder_df_by_label_correlation

    inverted = pd.DataFrame(
        [
            {"label": 1.0, "encoded": -0.8, "type": "training"},
            {"label": -1.0, "encoded": 0.6, "type": "training"},
        ]
    )
    oriented = orient_encoder_df_by_label_correlation(inverted)
    assert oriented["encoded"].tolist() == pytest.approx([0.8, -0.6])


def test_anchor_aligned_matrix_positive_after_orientation_correction():
    from gradiend.comparison.common import orient_encoder_df_by_label_correlation

    inverted = pd.DataFrame(
        [
            {
                "factual_id": "A",
                "counterfactual_id": "B",
                "transition_id": "A->B",
                "label": 1.0,
                "encoded": -0.9,
                "input_type": "alternative",
                "type": "training",
            },
            {
                "factual_id": "B",
                "counterfactual_id": "A",
                "transition_id": "B->A",
                "label": -1.0,
                "encoded": 0.9,
                "input_type": "alternative",
                "type": "training",
            },
        ]
    )
    oriented = orient_encoder_df_by_label_correlation(inverted)
    result = compute_anchor_aligned_encoding_matrix(
        pair_by_id={"ab": ("A", "B")},
        encoder_summary={"ab": {"encoder_df": oriented}},
        feature_classes=["A", "B"],
        alignment="factual",
    )
    matrix = pd.DataFrame(result["matrix"], index=result["rows"], columns=result["columns"])
    assert matrix.loc["A", "A"] == pytest.approx(0.9)
    assert matrix.loc["B", "B"] == pytest.approx(0.9)
    assert matrix.loc["A", "B"] == pytest.approx(-0.9)


def test_anchor_aligned_includes_cross_domain_label_zero_rows():
    """Cross-domain probes (label=0) must contribute to off-diagonal cells."""
    aligned = build_anchor_aligned_encoding_rows(
        pair_by_id={"ab": ("A", "B")},
        encoder_summary={
            "ab": {
                "encoder_df": pd.DataFrame(
                    [
                        {
                            "factual_id": "A",
                            "counterfactual_id": "B",
                            "transition_id": "A->B",
                            "label": 1.0,
                            "encoded": 1.0,
                            "type": "training",
                        },
                        {
                            "factual_id": "C",
                            "counterfactual_id": "D",
                            "transition_id": "C->D",
                            "label": 0.0,
                            "encoded": 5.0,
                            "type": "training",
                        },
                    ]
                )
            }
        },
        feature_classes=["A", "B", "C", "D"],
    )
    matrix = aggregate_anchor_aligned_encoding_rows(
        aligned, ["A", "B", "C", "D"], aggregate="mean"
    )
    assert matrix.loc["A", "A"] == pytest.approx(1.0)
    assert matrix.loc["A", "C"] == pytest.approx(5.0)


def test_sample_cross_task_eval_rows_keeps_each_factual_class():
    from gradiend.comparison.feature_cross_encoding import _sample_cross_task_eval_rows

    df = pd.DataFrame(
        {
            "factual_class": ["1SG"] * 100 + ["1PL"] * 3 + ["2SGPL"] * 3,
            "alternative_class": ["3SG"] * 100 + ["3PL"] * 3 + ["3SG"] * 3,
            "masked": ["m"] * 106,
            "split": ["test"] * 106,
            "factual": ["a"] * 106,
            "alternative": ["b"] * 106,
            "transition": ["t"] * 106,
        }
    )
    sampled = _sample_cross_task_eval_rows(df, max_size=50)
    assert set(sampled["factual_class"].astype(str)) == {"1SG", "1PL", "2SGPL"}
