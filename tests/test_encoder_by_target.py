"""Tests for encoder by-target plot framing."""

from __future__ import annotations

import pandas as pd
import pytest

from gradiend.visualizer.encoder_by_target import (
    build_encoder_target_plot_frame,
    plot_encoder_by_target,
    plot_encoder_by_target_seed_grid,
)


def test_build_encoder_target_plot_frame_orders_targets_within_class():
    encoder_df = pd.DataFrame(
        {
            "encoded": [0.9, 0.8, -0.9, -0.7, 0.85, -0.8],
            "source_id": ["positive", "positive", "negative", "negative", "positive", "negative"],
            "factual_token": ["good", "bad", "terrible", "awful", "nice", "sad"],
            "type": ["training"] * 6,
            "data_split": ["train", "test", "train", "validation", "validation", "test"],
        }
    )
    plot_df, target_order, split_order, target_to_class, target_to_split = (
        build_encoder_target_plot_frame(
            encoder_df,
            class_order=["positive", "negative"],
        )
    )
    assert split_order == ["train", "validation", "test"]
    assert target_order == ["good", "nice", "bad", "terrible", "awful", "sad"]
    assert target_to_split["good"] == "train"
    assert target_to_split["nice"] == "validation"
    assert target_to_split["bad"] == "test"
    assert target_to_class["good"] == "positive"
    assert target_to_class["awful"] == "negative"
    assert len(plot_df) == 6


def test_plot_encoder_by_target_seed_grid_writes_shared_axis_plot(tmp_path):
    encoder_df = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "encoded": [0.9, -0.8, 0.85, -0.75],
            "source_id": ["positive", "negative", "positive", "negative"],
            "source_token": ["good", "bad", "good", "bad"],
            "type": ["training"] * 4,
            "data_split": ["train", "validation", "test", "train"],
            "input_type": ["factual"] * 4,
        }
    )
    output = tmp_path / "seed_grid.pdf"

    path = plot_encoder_by_target_seed_grid(
        encoder_df,
        class_order=["positive", "negative"],
        output=str(output),
        show=False,
    )

    # The multi-seed API uses this plot for convergent held-out-target
    # analysis. A single rendered file means all seed rows shared one target
    # order/x-axis instead of producing one unrelated figure per seed.
    assert path == str(output)
    assert output.is_file()


def test_plot_encoder_by_target_seed_grid_writes_interactive_html(tmp_path):
    pytest.importorskip("plotly")
    encoder_df = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "encoded": [0.9, -0.8, 0.85, -0.75],
            "source_id": ["positive", "negative", "positive", "negative"],
            "source_token": ["good", "bad", "good", "bad"],
            "masked": ["This is [MASK].", "That is [MASK].", "This is [MASK].", "That is [MASK]."],
            "type": ["training"] * 4,
            "data_split": ["train", "validation", "test", "train"],
            "input_type": ["factual"] * 4,
        }
    )
    output = tmp_path / "seed_grid.html"

    path = plot_encoder_by_target_seed_grid(
        encoder_df,
        class_order=["positive", "negative"],
        output=str(output),
        show=False,
        interactive=True,
    )

    assert path == str(output)
    assert output.is_file()


def test_plot_encoder_by_target_seed_grid_writes_errorbar_summary(tmp_path):
    encoder_df = pd.DataFrame(
        {
            "seed": [0, 0, 0, 1, 1, 1],
            "encoded": [0.9, 0.7, -0.8, 0.8, 0.6, -0.7],
            "source_id": ["positive", "positive", "negative", "positive", "positive", "negative"],
            "source_token": ["good", "good", "bad", "good", "good", "bad"],
            "type": ["training"] * 6,
            "data_split": ["train", "train", "validation", "test", "test", "train"],
            "input_type": ["factual"] * 6,
        }
    )
    output = tmp_path / "seed_errorbar.pdf"

    path = plot_encoder_by_target_seed_grid(
        encoder_df,
        class_order=["positive", "negative"],
        output=str(output),
        show=False,
        plot_style="errorbar",
    )

    assert path == str(output)
    assert output.is_file()


def test_plot_encoder_by_target_seed_grid_writes_combined_seed_strip(tmp_path):
    encoder_df = pd.DataFrame(
        {
            "seed": [0, 0, 1, 1],
            "encoded": [0.9, -0.8, 0.85, -0.75],
            "source_id": ["positive", "negative", "positive", "negative"],
            "source_token": ["good", "bad", "good", "bad"],
            "type": ["training"] * 4,
            "data_split": ["train", "validation", "test", "train"],
            "input_type": ["factual"] * 4,
        }
    )
    output = tmp_path / "seed_combined_strip.pdf"

    path = plot_encoder_by_target_seed_grid(
        encoder_df,
        class_order=["positive", "negative"],
        output=str(output),
        show=False,
        combine_seed_rows=True,
    )

    assert path == str(output)
    assert output.is_file()


def test_build_encoder_target_plot_frame_swaps_display_class_for_counterfactual_source():
    """Counterfactual-source plots display targets under the counterfactual class.

    This reproduces source="alternative" sentiment rows. The raw row is a
    positive factual example ("good") with a negative counterfactual target
    ("bad"). For the by-target plot the x-axis target is the counterfactual
    token, so the displayed feature-class bracket must also be the
    counterfactual class. Otherwise negative targets such as "bad" appear under
    the positive bracket, which was the bug reported from train_sentiment.py.
    """
    encoder_df = pd.DataFrame(
        {
            "encoded": [-0.9, 0.8],
            # source_id intentionally follows the factual/raw row class here.
            # The plotter must not trust it blindly for counterfactual input.
            "source_id": ["positive", "negative"],
            "target_id": ["negative", "positive"],
            "input_type": ["alternative", "alternative"],
            "factual_token": ["good", "bad"],
            "alternative_token": ["bad", "good"],
            "type": ["training"] * 2,
            "data_split": ["train"] * 2,
        }
    )

    plot_df, target_order, _, target_to_class, _ = build_encoder_target_plot_frame(
        encoder_df,
        class_order=["positive", "negative"],
    )

    # The counterfactual target "good" belongs under the positive bracket; the
    # counterfactual target "bad" belongs under the negative bracket.
    assert target_order == ["good", "bad"]
    assert target_to_class["good"] == "positive"
    assert target_to_class["bad"] == "negative"
    assert plot_df["target_token"].tolist() == ["bad", "good"]
    assert plot_df["feature_class"].tolist() == ["negative", "positive"]


def test_build_encoder_target_plot_frame_keeps_factual_source_class_unchanged():
    """Factual-source plots keep the ordinary factual token/class pairing."""
    encoder_df = pd.DataFrame(
        {
            "encoded": [0.9, -0.8],
            "source_id": ["positive", "negative"],
            "target_id": ["negative", "positive"],
            "input_type": ["factual", "factual"],
            "factual_token": ["good", "bad"],
            "alternative_token": ["bad", "good"],
            "type": ["training"] * 2,
            "data_split": ["train"] * 2,
        }
    )

    plot_df, target_order, _, target_to_class, _ = build_encoder_target_plot_frame(
        encoder_df,
        class_order=["positive", "negative"],
    )

    assert target_order == ["good", "bad"]
    assert target_to_class["good"] == "positive"
    assert target_to_class["bad"] == "negative"
    assert plot_df["target_token"].tolist() == ["good", "bad"]
    assert plot_df["feature_class"].tolist() == ["positive", "negative"]


def test_plot_encoder_by_target_rejects_unsupported_style_generically():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    encoder_df = pd.DataFrame(
        {
            "encoded": [0.9],
            "source_id": ["positive"],
            "factual_token": ["good"],
            "type": ["training"],
            "data_split": ["train"],
        }
    )
    with pytest.raises(ValueError, match="plot_style must be one of"):
        plot_encoder_by_target(
            encoder_df=encoder_df,
            plot_style="ridgeline",
            show=False,
            output="unused.png",
        )


def test_plot_encoder_by_target_rejects_removed_point_label_kwargs():
    encoder_df = pd.DataFrame(
        {
            "encoded": [0.9],
            "source_id": ["positive"],
            "factual_token": ["good"],
            "type": ["training"],
            "data_split": ["train"],
        }
    )
    with pytest.raises(TypeError, match="does not accept point-label arguments"):
        plot_encoder_by_target(
            encoder_df=encoder_df,
            label_points="outliers+sample",
            show=False,
            output="unused.png",
        )


def test_box_plot_centers_targets_when_each_target_has_one_split():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from unittest.mock import patch

    import matplotlib.pyplot as plt
    import seaborn as sns

    encoder_df = pd.DataFrame(
        {
            "encoded": [0.9, 0.7, -0.9, -0.8],
            "source_id": ["positive", "positive", "negative", "negative"],
            "factual_token": ["good", "nice", "awful", "sad"],
            "type": ["training"] * 4,
            "data_split": ["train", "validation", "test", "test"],
        }
    )
    captured_dodge: list[bool] = []
    original_boxplot = sns.boxplot

    def _record_boxplot(*args, **kwargs):
        captured_dodge.append(bool(kwargs.get("dodge")))
        return original_boxplot(*args, **kwargs)

    with patch("matplotlib.pyplot.show"), patch("matplotlib.figure.Figure.savefig"), patch(
        "seaborn.boxplot", side_effect=_record_boxplot
    ):
        plot_encoder_by_target(
            encoder_df=encoder_df,
            plot_style="box",
            show=False,
            output="box_test.png",
        )
    plt.close("all")

    assert captured_dodge == [False]


def test_strip_plot_centers_targets_when_each_target_has_one_split():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from unittest.mock import patch

    import matplotlib.pyplot as plt
    import seaborn as sns

    encoder_df = pd.DataFrame(
        {
            "encoded": [0.9, 0.7, -0.9, -0.8],
            "source_id": ["positive", "positive", "negative", "negative"],
            "factual_token": ["good", "nice", "awful", "sad"],
            "type": ["training"] * 4,
            "data_split": ["train", "validation", "test", "test"],
        }
    )
    captured_dodge: list[bool] = []
    original_stripplot = sns.stripplot

    def _record_stripplot(*args, **kwargs):
        captured_dodge.append(bool(kwargs.get("dodge")))
        return original_stripplot(*args, **kwargs)

    with patch("matplotlib.pyplot.show"), patch("matplotlib.figure.Figure.savefig"), patch(
        "seaborn.stripplot", side_effect=_record_stripplot
    ):
        plot_encoder_by_target(
            encoder_df=encoder_df,
            plot_style="strip",
            show=False,
            output="strip_test.png",
        )
    plt.close("all")

    assert captured_dodge == [False]


def test_violin_plot_centers_targets_when_each_target_has_one_split():
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")
    from unittest.mock import patch

    import matplotlib.pyplot as plt
    import seaborn as sns

    encoder_df = pd.DataFrame(
        {
            "encoded": [0.9, 0.7, -0.9, -0.8],
            "source_id": ["positive", "positive", "negative", "negative"],
            "factual_token": ["good", "nice", "awful", "sad"],
            "type": ["training"] * 4,
            "data_split": ["train", "validation", "test", "test"],
        }
    )
    captured_dodge: list[bool] = []
    original_violinplot = sns.violinplot

    def _record_violinplot(*args, **kwargs):
        captured_dodge.append(bool(kwargs.get("dodge")))
        return original_violinplot(*args, **kwargs)

    with patch("matplotlib.pyplot.show"), patch("matplotlib.figure.Figure.savefig"), patch(
        "seaborn.violinplot", side_effect=_record_violinplot
    ):
        plot_encoder_by_target(
            encoder_df=encoder_df,
            plot_style="violin",
            show=False,
            output="violin_test.png",
        )
    plt.close("all")

    assert captured_dodge == [False]


def test_draw_split_spine_marks_one_line_per_split_run():
    pytest.importorskip("matplotlib")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    from gradiend.visualizer.encoder_by_target import _draw_split_spine_marks, _split_color_map

    target_order = ["good", "nice", "bad", "awful", "sad"]
    target_to_split = {
        "good": "train",
        "nice": "validation",
        "bad": "test",
        "awful": "validation",
        "sad": "test",
    }
    target_to_class = {
        "good": "positive",
        "nice": "positive",
        "bad": "positive",
        "awful": "negative",
        "sad": "negative",
    }
    split_colors = _split_color_map(["train", "validation", "test"])

    fig, ax = plt.subplots()
    n_lines_before = len(ax.lines)
    ax.set_xticks(range(len(target_order)))
    ax.set_xticklabels(target_order)
    _draw_split_spine_marks(
        ax,
        target_order=target_order,
        target_to_split=target_to_split,
        target_to_class=target_to_class,
        class_order=["positive", "negative"],
        split_color_map=split_colors,
    )
    labels = ax.get_xticklabels()
    assert all(label.get_fontweight() in ("normal", 400, "regular") for label in labels)
    assert mcolors.to_hex(labels[0].get_color()) == mcolors.to_hex("0.15") or labels[0].get_color() in (
        "black",
        "#000000",
        (0.0, 0.0, 0.0, 1.0),
    )
    new_lines = ax.lines[n_lines_before:]
    assert len(new_lines) == 5
    line_colors = {mcolors.to_hex(line.get_color()) for line in new_lines}
    assert mcolors.to_hex(split_colors["train"]) in line_colors
    assert mcolors.to_hex(split_colors["validation"]) in line_colors
    assert mcolors.to_hex(split_colors["test"]) in line_colors
    plt.close(fig)


def test_tighten_target_axis_xlim_matches_group_padding():
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    from gradiend.visualizer.encoder_by_target import _TARGET_GROUP_PAD, _tighten_target_axis_xlim

    fig, ax = plt.subplots()
    _tighten_target_axis_xlim(ax, 50)
    assert ax.get_xlim() == (-_TARGET_GROUP_PAD, 49 + _TARGET_GROUP_PAD)
    plt.close(fig)


def test_feature_class_group_xlim_widens_gap_between_adjacent_brackets():
    from gradiend.visualizer.encoder_by_target import (
        _TARGET_GROUP_INNER_PAD,
        _TARGET_GROUP_PAD,
        _feature_class_group_xlim,
    )

    classes_present = ["positive", "negative"]
    pos_x0, pos_x1 = _feature_class_group_xlim(list(range(10)), "positive", classes_present)
    neg_x0, neg_x1 = _feature_class_group_xlim(list(range(10, 20)), "negative", classes_present)

    assert pos_x0 == -_TARGET_GROUP_PAD
    assert pos_x1 == 9 + _TARGET_GROUP_INNER_PAD
    assert neg_x0 == 10 - _TARGET_GROUP_INNER_PAD
    assert neg_x1 == 19 + _TARGET_GROUP_PAD
    assert neg_x0 - pos_x1 == pytest.approx(1.0 - 2 * _TARGET_GROUP_INNER_PAD)
