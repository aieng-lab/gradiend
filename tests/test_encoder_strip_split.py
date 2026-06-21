"""Tests for encoder strip-by-split plot helpers."""

from __future__ import annotations

import pandas as pd

from gradiend.visualizer.encoder_strip_split import (
    _label_indices_from_mode,
    aggregate_strip_targets_to_mean,
    select_encoded_outlier_indices,
    select_trend_sample_indices,
)


def test_select_encoded_outlier_indices_per_group():
    df = pd.DataFrame(
        {
            "x_group": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "encoded": [0.0, 0.1, 0.2, 5.0, 0.0, 0.1, -4.0, 0.2],
        }
    )
    outliers = select_encoded_outlier_indices(df)
    assert set(outliers) == {3, 6}


def test_label_indices_outliers_mode():
    df = pd.DataFrame(
        {
            "x_group": ["pos"] * 4,
            "encoded": [0.0, 0.1, 0.2, 3.0],
            "text": ["a", "b", "c", "outlier"],
        }
    )
    indices = _label_indices_from_mode(
        df,
        label_points="outliers",
        label_indices=None,
        outlier_method="iqr",
        outlier_k=1.5,
    )
    assert indices == [3]


def test_label_indices_outliers_plus_sample():
    df = pd.DataFrame(
        {
            "x_group": ["pos"] * 8,
            "data_split": ["train"] * 8,
            "encoded": [1.0, 1.0, 1.0, 1.0, 0.2, 0.3, 0.4, -0.5],
            "factual_token": ["good"] * 8,
        }
    )
    indices = _label_indices_from_mode(
        df,
        label_points="outliers+sample",
        label_indices=None,
        outlier_method="iqr",
        outlier_k=1.5,
        label_sample_per_group=2,
        sample_group_cols=("x_group", "data_split"),
    )
    assert len(indices) >= 2


def test_select_trend_sample_indices_even_spacing():
    df = pd.DataFrame(
        {
            "target_token": ["good"] * 6,
            "data_split": ["train"] * 6,
            "encoded": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        }
    )
    indices = select_trend_sample_indices(
        df,
        group_cols=("target_token", "data_split"),
        n_per_group=3,
    )
    assert len(indices) == 3


def test_label_indices_explicit_union():
    df = pd.DataFrame({"x_group": ["a", "b"], "encoded": [0.0, 1.0]})
    indices = _label_indices_from_mode(
        df,
        label_points=False,
        label_indices=[1],
        outlier_method="iqr",
        outlier_k=1.5,
    )
    assert indices == [1]


def test_aggregate_strip_targets_to_mean_per_group_and_split():
    df = pd.DataFrame(
        {
            "x_group": ["positive", "positive", "positive", "positive"],
            "data_split": ["train", "train", "test", "test"],
            "type": ["training"] * 4,
            "factual_token": ["good", "good", "good", "nice"],
            "encoded": [0.7, 0.9, 0.2, 0.6],
            "text": ["a", "b", "c", "d"],
        }
    )

    out = aggregate_strip_targets_to_mean(df)

    assert len(out) == 3
    train_good = out[
        (out["x_group"] == "positive")
        & (out["data_split"] == "train")
        & (out["factual_token"] == "good")
    ]
    assert float(train_good.iloc[0]["encoded"]) == 0.8


def test_default_strip_by_split_figsize_scales_with_groups():
    from gradiend.visualizer.encoder_strip_split import _default_strip_by_split_figsize

    assert _default_strip_by_split_figsize(2) == (6.0, 4.5)
    assert _default_strip_by_split_figsize(5) == (9.0, 4.5)
