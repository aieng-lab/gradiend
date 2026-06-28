import pandas as pd
import pytest

from gradiend.comparison.encoder_aggregation import (
    aggregate_encoder_dataframes,
    aggregate_encoder_dataframes_registry,
    encoder_probe_key_from_row,
    encoder_probe_keys,
)
from gradiend.trainer.core.multi_seed import aggregate_eval_results


def _probe_row(*, masked="[MASK] x", factual="A", alternative="B", encoded=1.0):
    return {
        "masked": masked,
        "source_id": factual,
        "target_id": alternative,
        "encoded": encoded,
        "type": "training",
    }


def test_encoder_probe_key_from_row_uses_source_id_when_factual_id_missing():
    row = pd.Series(_probe_row())
    assert encoder_probe_key_from_row(row) == ("[MASK] x", "A", "B")


def test_encoder_probe_key_from_row_prefers_factual_id_columns():
    row = pd.Series(
        {
            "masked": "[MASK] y",
            "factual_id": "F",
            "target_id": "T",
            "source_id": "ignored",
            "encoded": 0.1,
        }
    )
    assert encoder_probe_key_from_row(row) == ("[MASK] y", "F", "T")


def test_aggregate_encoder_dataframes_means_matching_probes():
    frames = [
        pd.DataFrame([_probe_row(encoded=1.0)]),
        pd.DataFrame([_probe_row(encoded=0.2)]),
    ]
    result = aggregate_encoder_dataframes(frames)
    assert len(result) == 1
    assert float(result.iloc[0]["encoded"]) == pytest.approx(0.6)


def test_aggregate_encoder_dataframes_source_id_rows():
    """Regression: encoder rows from encode_dataset_to_rows use source_id/target_id."""
    frames = [
        pd.DataFrame([_probe_row(encoded=1.0)]),
        pd.DataFrame([_probe_row(encoded=0.0)]),
    ]
    result = aggregate_encoder_dataframes(frames)
    assert len(result) == 1
    assert float(result.iloc[0]["encoded"]) == pytest.approx(0.5)


def test_aggregate_encoder_dataframes_keeps_distinct_probes():
    frames = [
        pd.DataFrame(
            [
                _probe_row(masked="[MASK] a", factual="A", alternative="B", encoded=1.0),
                _probe_row(masked="[MASK] c", factual="C", alternative="D", encoded=0.4),
            ]
        ),
        pd.DataFrame(
            [
                _probe_row(masked="[MASK] a", factual="A", alternative="B", encoded=0.0),
                _probe_row(masked="[MASK] c", factual="C", alternative="D", encoded=0.6),
            ]
        ),
    ]
    result = aggregate_encoder_dataframes(frames)
    assert len(result) == 2
    by_pair = {
        (row["source_id"], row["target_id"]): float(row["encoded"])
        for _, row in result.iterrows()
    }
    assert by_pair[("A", "B")] == pytest.approx(0.5)
    assert by_pair[("C", "D")] == pytest.approx(0.5)


def test_aggregate_encoder_dataframes_single_frame_passthrough():
    frame = pd.DataFrame([_probe_row(encoded=0.42)])
    result = aggregate_encoder_dataframes([frame])
    assert len(result) == 1
    assert float(result.iloc[0]["encoded"]) == pytest.approx(0.42)


def test_aggregate_encoder_dataframes_empty_input():
    assert aggregate_encoder_dataframes([]).empty
    assert aggregate_encoder_dataframes([pd.DataFrame()]).empty


def test_aggregate_encoder_dataframes_registry_matches_helper():
    frames = [pd.DataFrame([_probe_row(encoded=1.0)]), pd.DataFrame([_probe_row(encoded=0.0)])]
    direct = aggregate_encoder_dataframes(frames)
    via_registry = aggregate_encoder_dataframes_registry(frames)
    assert float(direct.iloc[0]["encoded"]) == pytest.approx(float(via_registry.iloc[0]["encoded"]))


def test_aggregate_eval_results_promotes_encoder_df():
    frames = [
        pd.DataFrame([{"masked": "m", "factual_id": "A", "target_id": "B", "encoded": 1.0}]),
        pd.DataFrame([{"masked": "m", "factual_id": "A", "target_id": "B", "encoded": 0.0}]),
    ]
    merged = aggregate_eval_results(
        [{"encoder_df": frames[0], "correlation": 0.5}, {"encoder_df": frames[1], "correlation": 0.7}],
        [10, 11],
        aggregate="mean",
        dispersion="none",
    )
    assert "encoder_df" in merged
    assert float(merged["encoder_df"].iloc[0]["encoded"]) == pytest.approx(0.5)
    assert merged["correlation"] == pytest.approx(0.6)
    assert merged["seeds"]["n"] == 2


def test_aggregate_eval_results_encoder_df_per_seed_when_requested():
    frames = [
        pd.DataFrame([_probe_row(encoded=1.0)]),
        pd.DataFrame([_probe_row(encoded=0.0)]),
    ]
    merged = aggregate_eval_results(
        [{"encoder_df": frames[0]}, {"encoder_df": frames[1]}],
        [1, 2],
        aggregate="mean",
        dispersion="std",
        return_per_seed=True,
    )
    assert merged["seeds"]["per_seed"][1]["encoder_df"].iloc[0]["encoded"] == pytest.approx(1.0)
    assert merged["seeds"]["per_seed"][2]["encoder_df"].iloc[0]["encoded"] == pytest.approx(0.0)
    assert float(merged["encoder_df"].iloc[0]["encoded"]) == pytest.approx(0.5)


def test_encoder_probe_keys_collects_unique_probes():
    df = pd.DataFrame(
        [
            _probe_row(masked="[MASK] a", factual="A", alternative="B"),
            _probe_row(masked="[MASK] a", factual="A", alternative="B"),
            _probe_row(masked="[MASK] c", factual="C", alternative="D"),
        ]
    )
    keys = encoder_probe_keys(df)
    assert keys == {
        ("[MASK] a", "A", "B"),
        ("[MASK] c", "C", "D"),
    }
