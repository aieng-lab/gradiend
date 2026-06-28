"""Decoder grid utility helpers."""

from gradiend.evaluator.decoder_eval_utils import parse_grid_candidate_id


def test_parse_grid_candidate_id_tuple():
    assert parse_grid_candidate_id((1.0, 0.01)) == (1.0, 0.01)


def test_parse_grid_candidate_id_entry_id_dict():
    entry = {"id": {"feature_factor": -1.0, "learning_rate": 0.1}}
    assert parse_grid_candidate_id("ignored", entry) == (-1.0, 0.1)
