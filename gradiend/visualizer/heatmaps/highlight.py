"""Overlay helpers for comparison heatmaps."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Set, Tuple


def highlight_heatmap_cells(
    ax,
    *,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    cells: Optional[Iterable[Tuple[str, str]]] = None,
    row_subset: Optional[Iterable[str]] = None,
    col_subset: Optional[Iterable[str]] = None,
    edgecolor: str = "#E67E22",
    linewidth: float = 2.5,
    linestyle: str = "-",
    zorder: int = 10,
) -> None:
    """Draw rectangle outlines on heatmap cells (seaborn heatmap coordinates).

    Highlight either explicit ``(row_id, col_id)`` pairs via ``cells``, or the
    Cartesian product of ``row_subset`` × ``col_subset`` when both are set.

    Args:
        ax: Matplotlib axis returned by ``plot_comparison_heatmap`` / seaborn heatmap.
        row_labels: Row ids in plot order (y axis, top to bottom).
        col_labels: Column ids in plot order (x axis, left to right).
        cells: Optional explicit ``(row_id, col_id)`` pairs to outline.
        row_subset: Optional row ids; use with ``col_subset`` for a block highlight.
        col_subset: Optional column ids; use with ``row_subset`` for a block highlight.
        edgecolor: Rectangle edge color.
        linewidth: Rectangle edge width.
        linestyle: Rectangle edge linestyle.
        zorder: Drawing order (above heatmap cells).
    """
    from matplotlib.patches import Rectangle

    row_index = {str(label): index for index, label in enumerate(row_labels)}
    col_index = {str(label): index for index, label in enumerate(col_labels)}

    targets: Set[Tuple[int, int]] = set()
    if cells is not None:
        for row_id, col_id in cells:
            row_key, col_key = str(row_id), str(col_id)
            if row_key in row_index and col_key in col_index:
                targets.add((row_index[row_key], col_index[col_key]))
    elif row_subset is not None and col_subset is not None:
        for row_id in row_subset:
            for col_id in col_subset:
                row_key, col_key = str(row_id), str(col_id)
                if row_key in row_index and col_key in col_index:
                    targets.add((row_index[row_key], col_index[col_key]))
    else:
        raise ValueError("Provide cells or both row_subset and col_subset")

    for row_idx, col_idx in sorted(targets):
        ax.add_patch(
            Rectangle(
                (col_idx, row_idx),
                1,
                1,
                fill=False,
                edgecolor=edgecolor,
                linewidth=linewidth,
                linestyle=linestyle,
                zorder=zorder,
            )
        )
