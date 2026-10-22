"""Tests for loading cross-encoding inputs from saved run folders (no trainers, no data)."""

import json

import pandas as pd
import pytest

from gradiend.comparison import compute_anchor_aligned_encoding_matrix, load_saved_cross_encoding_inputs


def _make_run(root, run_id, direction, rows, *, source="alternative", cache_name="encoded_values_max_size_50_split_test.csv"):
    run = root / run_id
    model = run / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text(json.dumps({"architecture": {"input_dim": 1}}))
    (model / "model.safetensors").write_bytes(b"")
    (model / "gradiend_context.json").write_text(
        json.dumps({"source": source, "target": "diff", "feature_class_encoding_direction": direction})
    )
    if cache_name:
        pd.DataFrame(rows).to_csv(run / cache_name, index=False)
    return run


def _rows(a, b):
    return [
        {"source_id": a, "encoded": 0.8, "type": "training"},
        {"source_id": b, "encoded": -0.6, "type": "training"},
    ]


def test_loads_pairs_sources_and_cached_encodings(tmp_path):
    _make_run(tmp_path, "ab", {"A": 1, "B": -1}, _rows("A", "B"))
    _make_run(tmp_path, "ac", {"A": 1, "C": -1}, _rows("A", "C"), source="factual")
    (tmp_path / "README.md").write_text("not a run")

    pair_by_id, source_by_id, summary = load_saved_cross_encoding_inputs(str(tmp_path))

    assert pair_by_id == {"ab": ("A", "B"), "ac": ("A", "C")}
    assert source_by_id == {"ab": "alternative", "ac": "factual"}
    assert list(summary["ab"]["encoder_df"]["encoded"]) == [0.8, -0.6]

    result = compute_anchor_aligned_encoding_matrix(pair_by_id=pair_by_id, encoder_summary=summary, feature_classes=["A", "B", "C"])
    matrix = pd.DataFrame(result["matrix"], index=result["rows"], columns=result["columns"])
    assert matrix.loc["A", "B"] == pytest.approx(-0.6)


def test_run_ids_select_runs(tmp_path):
    _make_run(tmp_path, "ab", {"A": 1, "B": -1}, _rows("A", "B"))
    _make_run(tmp_path, "ac", {"A": 1, "C": -1}, _rows("A", "C"))
    pair_by_id, _, summary = load_saved_cross_encoding_inputs(str(tmp_path), ["ac"])
    assert list(pair_by_id) == ["ac"] and list(summary) == ["ac"]


def test_missing_cache_raises(tmp_path):
    _make_run(tmp_path, "ab", {"A": 1, "B": -1}, _rows("A", "B"), cache_name=None)
    with pytest.raises(FileNotFoundError, match="no cached encoder evaluation"):
        load_saved_cross_encoding_inputs(str(tmp_path))


def test_several_caches_raise(tmp_path):
    run = _make_run(tmp_path, "ab", {"A": 1, "B": -1}, _rows("A", "B"))
    pd.DataFrame(_rows("A", "B")).to_csv(run / "encoded_values_max_size_100_split_test.csv", index=False)
    with pytest.raises(ValueError, match="several cached encoder evaluations"):
        load_saved_cross_encoding_inputs(str(tmp_path))


def test_non_binary_run_raises(tmp_path):
    _make_run(tmp_path, "abc", {"A": 1, "B": -1, "C": 0}, _rows("A", "B"))
    with pytest.raises(ValueError, match="exactly two classes"):
        load_saved_cross_encoding_inputs(str(tmp_path))
