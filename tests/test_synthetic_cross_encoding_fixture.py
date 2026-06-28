"""Tests for synthetic cross-encoding documentation fixture."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gradiend.comparison.anchor_aligned import compute_anchor_aligned_encoding_matrix
from gradiend.comparison.feature_cross_encoding import compute_gradiend_transition_cross_encoding_matrix
from synthetic_cross_encoding_fixture import (
    FEATURE_ORDER,
    PAIR_BY_ID,
    SOURCE_BY_ID,
    TRAINER_ORDER,
    WORKED_DIAGONAL,
    WORKED_OFF_DIAGONAL,
    WORKED_NEGATIVE_DIAGONAL,
    build_synthetic_encoder_summary,
    cell_contribution_rows,
    dummy_trainers,
    preanchor_highlight_sets,
    synthetic_transition_order,
)


def _oriented_matrix():
    encoder_summary = build_synthetic_encoder_summary(full_pool=False)
    payload = compute_anchor_aligned_encoding_matrix(
        pair_by_id=PAIR_BY_ID,
        encoder_summary=encoder_summary,
        feature_classes=list(FEATURE_ORDER),
        alignment="counterfactual",
        source_by_id=SOURCE_BY_ID,
    )
    return pd.DataFrame(payload["matrix"], index=payload["rows"], columns=payload["columns"]), payload


def test_synthetic_fixture_oriented_diagonal_matches_hand_computed_mean():
    matrix, _ = _oriented_matrix()
    anchor = WORKED_DIAGONAL["anchor"]
    column = WORKED_DIAGONAL["column"]
    assert matrix.loc[anchor, column] == pytest.approx(0.55)


def test_synthetic_fixture_off_diagonal_differs_from_diagonal():
    matrix, _ = _oriented_matrix()
    diag = matrix.loc[WORKED_DIAGONAL["anchor"], WORKED_DIAGONAL["column"]]
    off = matrix.loc[WORKED_OFF_DIAGONAL["anchor"], WORKED_OFF_DIAGONAL["column"]]
    assert off != pytest.approx(diag)
    assert off == pytest.approx(-0.2375)
    assert diag > 0
    assert off < 0


def test_synthetic_race_diagonals_have_mixed_signs():
    matrix, _ = _oriented_matrix()
    assert matrix.loc["asian", "asian"] > 0
    assert matrix.loc["white", "white"] < 0
    assert matrix.loc["black", "black"] < 0


def test_synthetic_gender_block_reflects_inverted_second_class():
    matrix, _ = _oriented_matrix()
    assert matrix.loc["he", "he"] == pytest.approx(0.82)
    assert matrix.loc["she", "she"] == pytest.approx(-0.88)
    assert matrix.loc["he", "she"] == pytest.approx(0.88)
    assert matrix.loc["she", "he"] == pytest.approx(-0.82)


def test_synthetic_feature_order_is_domain_grouped():
    assert FEATURE_ORDER == ("white", "black", "asian", "he", "she")


def test_synthetic_preanchor_highlights_cover_all_contributions():
    _, oriented = _oriented_matrix()
    contrib = cell_contribution_rows(
        oriented,
        anchor=WORKED_DIAGONAL["anchor"],
        column=WORKED_DIAGONAL["column"],
    )
    trainers, transitions = preanchor_highlight_sets(contrib, same_family_only=True)
    assert trainers == ["race_black_asian", "race_white_asian"]
    assert transitions == ["black→asian", "white→asian"]
    assert len(contrib) == 4


def test_synthetic_unrelated_gradiend_does_not_contribute_to_race_row():
    _, oriented = _oriented_matrix()
    contrib = cell_contribution_rows(
        oriented,
        anchor=WORKED_DIAGONAL["anchor"],
        column=WORKED_DIAGONAL["column"],
    )
    assert "gender_he_she" not in contrib["trainer_id"].astype(str).tolist()


def test_synthetic_negative_diagonal_white_white():
    matrix, _ = _oriented_matrix()
    anchor = WORKED_NEGATIVE_DIAGONAL["anchor"]
    column = WORKED_NEGATIVE_DIAGONAL["column"]
    assert matrix.loc[anchor, column] == pytest.approx(-0.495)


def test_synthetic_preanchor_matrix_is_complete():
    encoder_summary = build_synthetic_encoder_summary(full_pool=True)
    preanchor = compute_gradiend_transition_cross_encoding_matrix(
        dummy_trainers(),
        trainer_order=list(TRAINER_ORDER),
        transition_order=synthetic_transition_order(),
        encoder_summary=encoder_summary,
    )
    assert len(preanchor["columns"]) == 20
    assert len(preanchor["rows"]) == 4
    assert "gender_he_she" in preanchor["rows"]
    for row in preanchor["matrix"]:
        for value in row:
            assert not (isinstance(value, float) and math.isnan(value))
