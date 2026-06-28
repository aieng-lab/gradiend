"""Base comparison heatmap plotting utilities."""

from __future__ import annotations

from numbers import Real
from typing import Any, Dict, List, Optional, Tuple, Union

import math

from gradiend.util.deprecation import warn_deprecated_annot_fmt
from gradiend.visualizer.heatmaps.ordering import _reorder_comparison_data
from gradiend.visualizer.plot_optional import _require_matplotlib, _require_seaborn
from gradiend.visualizer.plot_style import disable_usetex_for_axis_text
from gradiend.visualizer.labels import (
    converged_for_trainer,
    format_label_with_convergence,
    resolve_axis_convergence_for_comparison_heatmap,
)

# Matrix-computation options that must not be forwarded to plot_comparison_heatmap.
_MATRIX_COMPUTE_KWARGS = frozenset({"seed_aggregate", "dispersion", "seed_selection"})


def filter_comparison_heatmap_plot_kwargs(kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop matrix-computation kwargs that high-level plot wrappers may receive."""
    if not kwargs:
        return {}
    return {k: v for k, v in kwargs.items() if k not in _MATRIX_COMPUTE_KWARGS}


def _encoding_measure_is_signed(measure: Optional[str]) -> bool:
    """Whether encoded-value heatmaps use a diverging (signed) color scale."""
    if not measure:
        return False
    name = str(measure)
    if name in {
        "cross_encoding_positive_minus_negative",
        "gradiend_feature_cross_encoding_mean",
        "gradiend_transition_cross_encoding_mean",
    }:
        return True
    if name.startswith("anchor_aligned_encoding_"):
        return not (name.endswith("_count") or name.endswith("_raw_count"))
    return False


def _default_colormap_for_measure(measure: Optional[str], cmap: str) -> str:
    if cmap != "viridis":
        return cmap
    if _encoding_measure_is_signed(measure):
        return "coolwarm"
    return cmap


def _symmetric_value_limits(mat_arr: Any) -> Tuple[float, float]:
    import numpy as np

    max_abs = max(1.0, float(np.nanmax(np.abs(mat_arr))))
    return -max_abs, max_abs


def _validate_numeric_optional(name: str, value: Optional[Union[int, float]]) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number or None")


def _validate_fontsize_optional(name: str, value: Optional[Union[int, float]]) -> None:
    _validate_numeric_optional(name, value)
    if value is not None and float(value) <= 0:
        raise ValueError(f"{name} must be > 0")


def plot_comparison_heatmap(
    comparison_data: Dict[str, Any],
    *,
    order: Union[str, List[str]] = "input",
    cluster: bool = False,
    annot: Union[bool, str] = "auto",
    fmt: Optional[str] = None,
    annot_fmt: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: Optional[Union[str, bool]] = False,
    output_path: Optional[str] = None,
    show: bool = True,
    return_data: bool = True,
    return_fig_ax: bool = False,
    ax: Optional[Any] = None,
    pretty_groups: Optional[Dict[str, List[str]]] = None,
    scale: str = "linear",
    scale_gamma: Optional[float] = None,
    annot_fontsize: Optional[Union[int, float]] = None,
    tick_label_fontsize: Optional[Union[int, float]] = None,
    axis_label_fontsize: Optional[Union[int, float]] = None,
    group_label_fontsize: Optional[Union[int, float]] = None,
    group_label_rotation_top: Union[int, float] = 0,
    group_label_rotation_right: Union[int, float] = 0,
    cbar_pad: Optional[float] = None,
    cbar_fontsize: Optional[Union[int, float]] = None,
    cbar_shrink: Optional[float] = None,
    cbar_label: Optional[str] = None,
    percentages: bool = False,
    row_metric: Optional[Dict[str, float]] = None,
    row_metric_label: Optional[str] = None,
    row_metric_cmap: str = "magma",
    row_metric_vmin: Optional[float] = None,
    row_metric_vmax: Optional[float] = None,
    row_label_mapping: Optional[Dict[str, str]] = None,
    column_label_mapping: Optional[Dict[str, str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    dispersion_display: str = "none",
    seed_annotation: Union[bool, Dict[str, Any]] = False,
    models: Optional[Dict[str, object]] = None,
    highlight_non_convergence: bool = True,
) -> Any:
    """Plot a precomputed comparison matrix as a heatmap.

    Args:
        comparison_data: Matrix payload with at least ``matrix`` and ``model_ids``.
        order: ``"input"``, ``"cluster"``, or an explicit row/column order.
        cluster: Whether to cluster rows/columns.
        annot: Whether/how to annotate cells.
        fmt: Deprecated alias for ``annot_fmt``.
        annot_fmt: Cell annotation format.
        figsize: Optional figure size.
        cmap: Heatmap colormap.
        vmin: Optional lower value bound.
        vmax: Optional upper value bound.
        title: Optional title, or False to omit.
        output_path: Optional output path.
        show: Whether to display the plot.
        return_data: Whether to include reordered comparison data.
        return_fig_ax: Whether to include matplotlib figure/axis.
        ax: Optional existing matplotlib axis.
        pretty_groups: Optional id groups shown as brackets.
        scale: Color scale type.
        scale_gamma: Optional gamma for power scaling.
        annot_fontsize: Optional annotation font size.
        tick_label_fontsize: Optional tick-label font size.
        axis_label_fontsize: Optional x/y axis title font size. When omitted but
            ``tick_label_fontsize`` is set, defaults to tick size + 4 and is never
            smaller than tick size + 1.
        group_label_fontsize: Optional group-label font size.
        group_label_rotation_top: Rotation for top group labels.
        group_label_rotation_right: Rotation for right group labels.
        cbar_pad: Optional colorbar padding.
        cbar_fontsize: Optional colorbar tick and label font size.
        cbar_shrink: Optional colorbar shrink factor (width relative to heatmap).
        cbar_label: Optional colorbar axis label.
        percentages: Whether to show values as percentages.
        row_metric: Optional side metric by row id.
        row_metric_label: Label for the side metric.
        row_metric_cmap: Colormap for the side metric.
        row_metric_vmin: Optional side-metric lower bound.
        row_metric_vmax: Optional side-metric upper bound.
        row_label_mapping: Optional mapping for row labels.
        column_label_mapping: Optional mapping for column labels.
        xlabel: Optional x-axis label (heatmap columns).
        ylabel: Optional y-axis label (heatmap rows).
        dispersion_display: How to show dispersion values.
        seed_annotation: Whether/how to annotate seed counts.
        models: Optional model mapping for non-convergence label lookup.
        highlight_non_convergence: Whether labels mark non-converged runs.
    """
    warn_deprecated_annot_fmt(fmt=fmt, annot_fmt=annot_fmt, stacklevel=1)
    if not isinstance(comparison_data, dict):
        raise TypeError("comparison_data must be a dict")
    if "matrix" not in comparison_data or "model_ids" not in comparison_data:
        raise ValueError("comparison_data must contain 'matrix' and 'model_ids'")
    if not isinstance(cmap, str) or not cmap:
        raise TypeError("cmap must be a non-empty string")
    if not (isinstance(annot, bool) or annot == "auto"):
        raise ValueError("annot must be True, False, or 'auto'")
    if dispersion_display not in {"none", "stacked", "corner_glyph"}:
        raise ValueError("dispersion_display must be 'none', 'stacked', or 'corner_glyph'")
    if not (isinstance(seed_annotation, bool) or isinstance(seed_annotation, dict)):
        raise TypeError("seed_annotation must be bool or dict")
    _validate_numeric_optional("vmin", vmin)
    _validate_numeric_optional("vmax", vmax)
    _validate_numeric_optional("scale_gamma", scale_gamma)
    _validate_fontsize_optional("annot_fontsize", annot_fontsize)
    _validate_fontsize_optional("tick_label_fontsize", tick_label_fontsize)
    _validate_fontsize_optional("axis_label_fontsize", axis_label_fontsize)
    _validate_fontsize_optional("group_label_fontsize", group_label_fontsize)
    _validate_numeric_optional("group_label_rotation_top", group_label_rotation_top)
    _validate_numeric_optional("group_label_rotation_right", group_label_rotation_right)
    _validate_fontsize_optional("cbar_fontsize", cbar_fontsize)
    _validate_numeric_optional("cbar_pad", cbar_pad)
    _validate_numeric_optional("cbar_shrink", cbar_shrink)
    _validate_numeric_optional("row_metric_vmin", row_metric_vmin)
    _validate_numeric_optional("row_metric_vmax", row_metric_vmax)
    if scale not in {"linear", "log", "sqrt", "power"}:
        raise ValueError("scale must be 'linear', 'log', 'sqrt', or 'power'")
    if scale == "power" and (scale_gamma is None or float(scale_gamma) <= 0):
        raise ValueError("scale_gamma must be > 0 when scale='power'")

    plt = _require_matplotlib()
    sns = _require_seaborn()

    if row_label_mapping:
        comparison_data = dict(comparison_data)
        comparison_data["row_labels"] = {str(k): str(v) for k, v in row_label_mapping.items()}
    if column_label_mapping:
        comparison_data = dict(comparison_data)
        comparison_data["column_labels"] = {str(k): str(v) for k, v in column_label_mapping.items()}
    comparison_data = _reorder_comparison_data(
        comparison_data,
        order=order,
        cluster=cluster,
        pretty_groups=pretty_groups,
    )
    row_ids = comparison_data["model_ids"]
    col_ids = comparison_data.get("column_ids", row_ids)
    rectangular = "column_ids" in comparison_data
    mat = comparison_data["matrix"]
    row_labels_map = comparison_data.get("row_labels") or {}
    column_labels_map = comparison_data.get("column_labels") or {}
    row_ticklabels = [row_labels_map.get(mid, mid) for mid in row_ids]
    column_ticklabels = [column_labels_map.get(mid, mid) for mid in col_ids]
    if highlight_non_convergence:
        row_convergence, col_convergence = resolve_axis_convergence_for_comparison_heatmap(
            comparison_data,
            models=models,
            row_ids=row_ids,
            column_ids=col_ids,
        )

        def _maybe_mark(
            label: str,
            mid: str,
            *,
            axis_convergence: Dict[str, Optional[bool]],
        ) -> str:
            converged = None
            key = str(mid)
            if models is not None and mid in models:
                converged = converged_for_trainer(models[mid])
            elif key in axis_convergence:
                converged = axis_convergence[key]
            return format_label_with_convergence(
                str(label),
                converged=converged,
                highlight_non_convergence=highlight_non_convergence,
            )

        row_ticklabels = [
            _maybe_mark(lbl, mid, axis_convergence=row_convergence)
            for lbl, mid in zip(row_ticklabels, row_ids)
        ]
        column_ticklabels = [
            _maybe_mark(lbl, mid, axis_convergence=col_convergence)
            for lbl, mid in zip(column_ticklabels, col_ids)
        ]
    n_rows = len(row_ids)
    n_cols = len(col_ids)
    custom_vmin = vmin is not None
    custom_vmax = vmax is not None

    if percentages:
        if (
            comparison_data.get("measure") == "topk_overlap"
            and comparison_data.get("value") == "intersection"
            and "resolved_topk" in comparison_data
        ):
            resolved_topk = comparison_data["resolved_topk"]
            display_mat: List[List[float]] = []
            for mi, row in zip(row_ids, mat):
                display_row: List[float] = []
                for mj, value in zip(col_ids, row):
                    denom = min(int(resolved_topk[mi]), int(resolved_topk[mj]))
                    display_row.append((float(value) / denom * 100.0) if denom else 0.0)
                display_mat.append(display_row)
            mat = display_mat
            if custom_vmin or custom_vmax:
                unique_denoms = {
                    min(int(resolved_topk[mi]), int(resolved_topk[mj]))
                    for mi in row_ids
                    for mj in col_ids
                }
                if len(unique_denoms) != 1:
                    raise ValueError(
                        "Custom vmin/vmax for percentage top-k overlap counts require uniform resolved top-k sizes"
                    )
                denom = next(iter(unique_denoms))
                scale_factor = (100.0 / float(denom)) if denom else 0.0
                if vmin is not None:
                    vmin = float(vmin) * scale_factor
                if vmax is not None:
                    vmax = float(vmax) * scale_factor
        else:
            mat = [[float(v) * 100.0 for v in row] for row in mat]
            if vmin is not None:
                vmin = float(vmin) * 100.0
            if vmax is not None:
                vmax = float(vmax) * 100.0

    if annot_fmt is not None:
        fmt = annot_fmt
    elif fmt is None:
        fmt = ".0f" if percentages else ".2f"

    if figsize is None:
        if rectangular:
            figsize = (max(14.0, n_cols * 0.45), max(8.0, n_rows * 0.35))
        else:
            s = max(14.0, n_rows * 0.4)
            figsize = (s, s)

    _annot = annot
    if _annot == "auto":
        _annot = (n_rows * n_cols) <= 1600

    if title in {None, True}:
        measure = comparison_data.get("measure", "comparison")
        part = comparison_data.get("part")
        title_map = {
            "cosine": "Cosine similarity",
            "cosine_signed": "Signed cosine similarity",
            "spearman": "Spearman similarity",
            "spearman_signed": "Signed Spearman similarity",
            "mass_overlap": "Mass overlap",
            "cross_encoding_positive_mean": "Cross-encoding true-class mean",
            "cross_encoding_negative_mean": "Cross-encoding false-class mean",
            "cross_encoding_positive_minus_negative": "Cross-encoding true-minus-false mean",
            "gradiend_feature_cross_encoding_mean": "GRADIEND × feature-class cross-encoding (mean)",
            "gradiend_feature_cross_encoding_count": "GRADIEND × feature-class cross-encoding (eval count)",
            "gradiend_transition_cross_encoding_mean": "GRADIEND × transition cross-encoding (mean)",
            "gradiend_transition_cross_encoding_count": "GRADIEND × transition cross-encoding (eval count)",
        }
        if str(measure).startswith("anchor_aligned_encoding_"):
            parts = str(measure).split("_")
            alignment = parts[3] if len(parts) > 3 else "factual"
            aggregate = parts[4] if len(parts) > 4 else "mean"
            measure_title = f"Anchor-aligned encoded value ({alignment}, {aggregate})"
        else:
            measure_title = title_map.get(measure, str(measure).replace("_", " "))
        if part:
            title = f"{measure_title} ({part})"
        else:
            title = measure_title

    import numpy as np
    from matplotlib.colors import LogNorm, PowerNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mat_arr = np.array(mat, dtype=float)
    measure = comparison_data.get("measure")
    value_name = comparison_data.get("value")
    row_normalized_by_diagonal = bool(comparison_data.get("row_normalized_by_diagonal"))
    normalized_cross_encoding = row_normalized_by_diagonal and str(measure).startswith("cross_encoding_")
    cross_encoding_difference = measure == "cross_encoding_positive_minus_negative"
    signed_encoding = _encoding_measure_is_signed(measure)
    signed_measures = {"cosine_signed", "spearman_signed", "cross_encoding_positive_minus_negative"}
    bounded_unit_measures = {"cosine", "cosine_signed", "spearman", "spearman_signed", "mass_overlap", "cross_encoding_positive_mean", "cross_encoding_negative_mean", "cross_encoding_positive_minus_negative"}
    cmap = _default_colormap_for_measure(measure, cmap)
    if vmin is None:
        if cross_encoding_difference or signed_encoding:
            vmin, _ = _symmetric_value_limits(mat_arr)
        elif measure in signed_measures:
            vmin = -100.0 if percentages else -1.0
        else:
            vmin = 0.0
    if vmax is None:
        if cross_encoding_difference or signed_encoding:
            _, vmax = _symmetric_value_limits(mat_arr)
        elif normalized_cross_encoding:
            vmax = max(1.0, float(np.nanmax(mat_arr)))
        elif measure in signed_measures:
            vmax = 100.0 if percentages else 1.0
        elif percentages:
            vmax = 100.0
        elif measure == "topk_overlap" and value_name == "intersection":
            vmax = float(np.nanmax(mat_arr))
        elif measure in bounded_unit_measures or value_name == "intersection_frac":
            vmax = 1.0
        else:
            vmax = float(np.nanmax(mat_arr))

    norm = None
    eps = max(1e-10, np.finfo(float).tiny)
    if scale == "log":
        norm = LogNorm(vmin=max(eps, float(vmin)), vmax=float(vmax))
    elif scale == "sqrt":
        norm = PowerNorm(gamma=0.5, vmin=float(vmin), vmax=float(vmax))
    elif scale == "power":
        norm = PowerNorm(gamma=float(scale_gamma), vmin=float(vmin), vmax=float(vmax))

    cell_stats = comparison_data.get("cell_stats") or []
    custom_cell_annotation = dispersion_display == "stacked" and bool(cell_stats)

    annot_kws = {}
    if annot_fontsize is not None:
        annot_kws["fontsize"] = annot_fontsize

    cbar_kws = {"shrink": 0.75 if cbar_shrink is None else float(cbar_shrink)}
    if cbar_pad is not None:
        cbar_kws["pad"] = cbar_pad
    if cbar_label:
        cbar_kws["label"] = cbar_label

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(
        mat_arr,
        ax=ax,
        xticklabels=column_ticklabels,
        yticklabels=row_ticklabels,
        cmap=cmap,
        norm=norm,
        vmin=None if norm else vmin,
        vmax=None if norm else vmax,
        annot=False if custom_cell_annotation else _annot,
        fmt=fmt,
        annot_kws=annot_kws,
        square=not rectangular,
        cbar=True,
        cbar_kws=cbar_kws,
        linewidths=0.5,
        linecolor="white",
    )
    fig = ax.get_figure()
    cbar_ax = fig.axes[1] if len(fig.axes) >= 2 else None

    if row_metric:
        metric_vals: List[float] = []
        for mid in row_ids:
            v = row_metric.get(mid)
            metric_vals.append(float(v) if isinstance(v, (int, float)) else np.nan)
        arr = np.asarray(metric_vals, dtype=float).reshape(-1, 1)
        if not np.all(np.isnan(arr)):
            m_vmin = row_metric_vmin if row_metric_vmin is not None else float(np.nanmin(arr))
            m_vmax = row_metric_vmax if row_metric_vmax is not None else float(np.nanmax(arr))
            divider_metric = make_axes_locatable(ax)
            ax_metric = divider_metric.append_axes("left", size="5%", pad=0.2)
            sns.heatmap(
                arr,
                ax=ax_metric,
                cmap=row_metric_cmap,
                vmin=m_vmin,
                vmax=m_vmax,
                cbar=False,
                xticklabels=[row_metric_label] if row_metric_label else [],
                yticklabels=[],
                square=True,
                linewidths=0.5,
                linecolor="white",
            )
            ax_metric.yaxis.set_ticks_position("left")
            ax_metric.tick_params(axis="x", rotation=90)

    if tick_label_fontsize is not None:
        for label in ax.get_xticklabels():
            label.set_fontsize(tick_label_fontsize)
        for label in ax.get_yticklabels():
            label.set_fontsize(tick_label_fontsize)

    disable_usetex_for_axis_text(ax)

    if isinstance(title, str):
        ax.set_title(title, usetex=False)

    resolved_axis_label_fontsize = axis_label_fontsize
    if resolved_axis_label_fontsize is None and tick_label_fontsize is not None:
        resolved_axis_label_fontsize = float(tick_label_fontsize) + 4
    elif (
        resolved_axis_label_fontsize is not None
        and tick_label_fontsize is not None
        and float(resolved_axis_label_fontsize) <= float(tick_label_fontsize)
    ):
        resolved_axis_label_fontsize = float(tick_label_fontsize) + 2

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=resolved_axis_label_fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=resolved_axis_label_fontsize)

    active_groups = comparison_data.get("pretty_groups")
    if active_groups is not None:
        divider = make_axes_locatable(ax)
        id_to_group = {mid: gname for gname, ids in active_groups.items() for mid in ids}
        group_col_spans = {}
        group_row_spans = {}
        axis_ids = row_ids if rectangular else row_ids
        for gname, ids in active_groups.items():
            indices = [axis_ids.index(mid) for mid in ids if mid in axis_ids]
            if indices:
                if not rectangular:
                    group_col_spans[gname] = (min(indices), max(indices))
                group_row_spans[gname] = (min(indices), max(indices))

        if not rectangular:
            for j in range(1, n_cols):
                if id_to_group.get(axis_ids[j]) != id_to_group.get(axis_ids[j - 1]):
                    ax.axvline(x=j, color="white", linewidth=2.5, zorder=5)
        for i in range(1, n_rows):
            if id_to_group.get(axis_ids[i]) != id_to_group.get(axis_ids[i - 1]):
                ax.axhline(y=i, color="white", linewidth=2.5, zorder=5)

        line_margin = 0.1
        group_fontsize = (
            group_label_fontsize
            if group_label_fontsize is not None
            else (tick_label_fontsize + 2 if tick_label_fontsize is not None else max(11, min(14, 280 / max(n_rows, 1))))
        )

        if not rectangular:
            ax_top = divider.append_axes("top", size="8%", pad=0.02)
            ax_top.set_xlim(0, n_cols)
            ax_top.set_ylim(0, 1)
            ax_top.set_aspect("auto")
            ax_top.axis("off")
            if title:
                ax_top.set_title(title, fontsize=plt.rcParams["axes.titlesize"], usetex=False)
            ax.set_title("")

            for gname, (start, end) in group_col_spans.items():
                x1, x2 = start, end + 1
                x_center = (x1 + x2 - 1) / 2 + 0.5
                ax_top.hlines(0.12, x1 + line_margin, x2 - line_margin, colors="gray", linewidth=3)
                ax_top.text(
                    x_center, 0.28, gname,
                    rotation=90 + float(group_label_rotation_top), ha="center", va="bottom",
                    fontsize=group_fontsize, transform=ax_top.transData,
                )
            disable_usetex_for_axis_text(ax_top)
        else:
            ax_top = None

        ax_right = divider.append_axes("right", size="8%", pad=0.02)
        ax_right.set_xlim(0, 1)
        ax_right.set_ylim(n_rows, 0)
        ax_right.set_aspect("auto")
        ax_right.axis("off")
        for gname, (start, end) in group_row_spans.items():
            y1, y2 = start, end + 1
            y_center = (y1 + y2 - 1) / 2 + 0.5
            ax_right.vlines(0.12, y1 + line_margin, y2 - line_margin, colors="gray", linewidth=3)
            ax_right.text(
                0.28, y_center, gname,
                rotation=0 + float(group_label_rotation_right), ha="left", va="center",
                fontsize=group_fontsize, transform=ax_right.transData,
            )
        disable_usetex_for_axis_text(ax_right)

    if cbar_ax is not None:
        if cbar_fontsize is not None:
            cbar_ax.tick_params(labelsize=cbar_fontsize)
            if cbar_label:
                cbar_ax.set_ylabel(cbar_label, fontsize=cbar_fontsize)

    if custom_cell_annotation:
        def _format_secondary(stat: Dict[str, Any]) -> str:
            dispersion = str(comparison_data.get("dispersion", "none"))
            if dispersion == "std" and isinstance(stat.get("std"), (int, float)):
                return f"+/-{float(stat['std']):.2f}"
            if dispersion == "range" and isinstance(stat.get("range_half_width"), (int, float)):
                return f"+/-{float(stat['range_half_width']):.2f}"
            if dispersion == "minmax" and isinstance(stat.get("min"), (int, float)) and isinstance(stat.get("max"), (int, float)):
                return f"[{float(stat['min']):.2f},{float(stat['max']):.2f}]"
            return ""
        base_font = annot_fontsize if annot_fontsize is not None else max(7, min(12, int(220 / max(n_rows, 1))))
        secondary_font = max(6, int(round(base_font * 0.8)))
        for i in range(n_rows):
            for j in range(n_cols):
                stat = cell_stats[i][j] if i < len(cell_stats) and j < len(cell_stats[i]) else None
                value = mat_arr[i, j]
                if stat is None or not isinstance(value, (int, float)) or math.isnan(float(value)):
                    continue
                primary = ax.text(j + 0.5, i + 0.42, f"{float(value):.2f}", ha="center", va="center", fontsize=base_font)
                primary.set_usetex(False)
                secondary = _format_secondary(stat)
                if secondary:
                    secondary_text = ax.text(j + 0.5, i + 0.72, secondary, ha="center", va="center", fontsize=secondary_font)
                    secondary_text.set_usetex(False)

    if dispersion_display == "corner_glyph" and cell_stats:
        from matplotlib.patches import Circle
        dispersion_key = str(comparison_data.get("dispersion", "none"))
        magnitudes: List[float] = []
        for row in cell_stats:
            for stat in row:
                if not isinstance(stat, dict):
                    continue
                if dispersion_key == "std" and isinstance(stat.get("std"), (int, float)):
                    magnitudes.append(float(stat["std"]))
                elif dispersion_key == "range" and isinstance(stat.get("range_half_width"), (int, float)):
                    magnitudes.append(float(stat["range_half_width"]))
        max_mag = max(magnitudes) if magnitudes else 0.0
        if max_mag > 0:
            for i in range(n_rows):
                for j in range(n_cols):
                    stat = cell_stats[i][j] if i < len(cell_stats) and j < len(cell_stats[i]) else None
                    if not isinstance(stat, dict):
                        continue
                    mag = None
                    if dispersion_key == "std" and isinstance(stat.get("std"), (int, float)):
                        mag = float(stat["std"])
                    elif dispersion_key == "range" and isinstance(stat.get("range_half_width"), (int, float)):
                        mag = float(stat["range_half_width"])
                    if mag is None or mag <= 0:
                        continue
                    radius = 0.05 + 0.12 * (mag / max_mag)
                    ax.add_patch(Circle((j + 0.82, i + 0.18), radius=radius, fill=False, linewidth=1.0, edgecolor="black"))

    if seed_annotation and ("global_n" in comparison_data or "global_n_range" in comparison_data):
        if "global_n" in comparison_data:
            seed_label = f"n={int(comparison_data['global_n'])}"
        else:
            lo, hi = comparison_data["global_n_range"]
            seed_label = f"n={int(lo)}-{int(hi)}"
        cfg = {"x": 0.995, "y": 0.005, "ha": "right", "va": "bottom", "fontsize": tick_label_fontsize or 10}
        if isinstance(seed_annotation, dict):
            cfg.update(seed_annotation)
        ax.text(float(cfg.pop("x")), float(cfg.pop("y")), seed_label, transform=ax.transAxes, **cfg)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()

    if return_data:
        returned = dict(comparison_data)
        returned["matrix"] = mat
        if return_fig_ax:
            return returned, fig, ax
        return returned
    if return_fig_ax:
        return fig, ax
    return {}
