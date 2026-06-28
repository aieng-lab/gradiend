"""Tests for heatmap cell highlight overlays."""

from __future__ import annotations

import pytest


def test_highlight_heatmap_cells_cartesian_product():
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    from gradiend.visualizer.heatmaps.highlight import highlight_heatmap_cells

    _, ax = plt.subplots()
    highlight_heatmap_cells(
        ax,
        row_labels=["r1", "r2"],
        col_labels=["c1", "c2", "c3"],
        row_subset=["r1"],
        col_subset=["c2", "c3"],
    )
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rects) == 2
    coords = sorted((rect.get_x(), rect.get_y()) for rect in rects)
    assert coords == [(1.0, 0.0), (2.0, 0.0)]
    plt.close("all")


def test_highlight_heatmap_cells_explicit_pairs():
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    from gradiend.visualizer.heatmaps.highlight import highlight_heatmap_cells

    _, ax = plt.subplots()
    highlight_heatmap_cells(
        ax,
        row_labels=["A", "B"],
        col_labels=["X", "Y"],
        cells=[("B", "Y")],
    )
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rects) == 1
    assert rects[0].get_x() == pytest.approx(1.0)
    assert rects[0].get_y() == pytest.approx(1.0)
    plt.close("all")


def test_highlight_heatmap_cells_requires_selector():
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt

    from gradiend.visualizer.heatmaps.highlight import highlight_heatmap_cells

    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="Provide cells or both"):
        highlight_heatmap_cells(
            ax,
            row_labels=["A"],
            col_labels=["X"],
        )
    plt.close("all")
