"""Tests for encoder by-target plot framing."""

from __future__ import annotations

import pandas as pd
import pytest

from gradiend.visualizer.encoder_by_target import build_encoder_target_plot_frame, plot_encoder_by_target


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
