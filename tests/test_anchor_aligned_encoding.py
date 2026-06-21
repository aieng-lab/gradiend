import pandas as pd
import pytest

from gradiend.comparison.anchor_aligned import (
    aggregate_anchor_aligned_encoding_rows,
    build_anchor_aligned_encoding_rows,
    compute_anchor_aligned_encoding_matrix,
)


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
    assert by_transition["columns"] == ["A->B"]
    transition_matrix = pd.DataFrame(
        by_transition["matrix"],
        index=by_transition["rows"],
        columns=by_transition["columns"],
    )
    assert transition_matrix.loc["A", "A->B"] == pytest.approx(2.0)
    assert transition_matrix.loc["B", "A->B"] == pytest.approx(-2.0)


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
