"""Tests for user-facing Plotly hover labels in encoder visualizations."""

from __future__ import annotations

import pandas as pd
import pytest


def _hovertemplate(fig) -> str:
    templates = [
        getattr(trace, "hovertemplate", "") or ""
        for trace in getattr(fig, "data", ())
    ]
    return "\n".join(templates)


def test_encoder_scatter_prefers_text_column_with_mask():
    from gradiend.visualizer.encoder_scatter import _first_masked_or_non_empty_column

    df = pd.DataFrame(
        {
            "display_text": ["Alice arrived."],
            "masked": ["[MASK] arrived."],
        }
    )

    assert _first_masked_or_non_empty_column(df, ("display_text", "masked")) == "masked"


def test_encoder_by_target_prefers_text_column_with_mask():
    from gradiend.visualizer.encoder_by_target import _text_column

    df = pd.DataFrame(
        {
            "display_text": ["Alice arrived."],
            "masked": ["[MASK] arrived."],
        }
    )

    assert _text_column(df) == "masked"


def test_encoder_scatter_truncates_hover_text_around_mask_cleanly():
    from gradiend.visualizer.encoder_scatter import _truncate_text_around_mask

    text = (
        "Before the important opening phrase with several words "
        "[MASK] "
        "after the important closing phrase with several more words"
    )

    out = _truncate_text_around_mask(text, max_chars=70, line_chars=0)

    assert "[MASK]" in out
    assert out.startswith("...")
    assert out.endswith("...")
    assert "phrase with several words [MASK] after the important" in out


def test_encoder_scatter_hover_text_limit_can_be_disabled():
    from gradiend.visualizer.encoder_scatter import _truncate_text_around_mask

    text = " ".join(["prefix"] * 40) + " [MASK] " + " ".join(["suffix"] * 40)

    assert _truncate_text_around_mask(text, max_chars=0, line_chars=0) == text


def test_encoder_scatter_hover_uses_user_facing_labels():
    pytest.importorskip("plotly")

    from gradiend.visualizer.encoder_scatter import plot_encoder_scatter

    fig = plot_encoder_scatter(
        encoder_df=pd.DataFrame(
            {
                "encoded": [[0.1], [0.2]],
                "source_id": ["positive", "negative"],
                "text_:hover": ["A friendly sentence.", "Another sentence."],
                "data_split": ["test", "test"],
                "type": ["training", "training"],
            }
        ),
        color_by="missing_label",
        x_col="source_id",
        show=False,
    )

    hovertemplate = _hovertemplate(fig)
    assert "Text=%{customdata" in hovertemplate
    assert "Split=%{customdata" in hovertemplate
    assert "text_:hover" not in hovertemplate
    assert "data_split" not in hovertemplate
    assert "Type=" not in hovertemplate


def test_encoder_by_target_interactive_hover_uses_user_facing_labels():
    pytest.importorskip("plotly")

    from gradiend.visualizer.encoder_by_target import plot_encoder_by_target

    fig = plot_encoder_by_target(
        encoder_df=pd.DataFrame(
            {
                "type": ["training", "training"],
                "encoded": [[0.1], [0.2]],
                "source_id": ["positive", "negative"],
                "source_token": ["good", "bad"],
                "data_split": ["train", "test"],
                "text": ["This is good.", "This is bad."],
            }
        ),
        interactive=True,
        show=False,
    )

    hovertemplate = _hovertemplate(fig)
    assert "Text=%{customdata" in hovertemplate
    assert "Split=%{customdata" in hovertemplate
    assert "text_hover" not in hovertemplate
    assert "plot_hue" not in hovertemplate
    assert "Type=" not in hovertemplate
