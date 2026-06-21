"""Top-k overlap heatmap wrappers built on the generalized comparison heatmap."""

from __future__ import annotations

from typing import Dict, Union, List, Optional, Tuple

from gradiend.visualizer.heatmaps import plot_similarity_heatmap
from gradiend.visualizer.heatmaps.similarity import _extract_best_correlation_for_models


def plot_topk_overlap_heatmap(
    models: Dict[str, object],
    topk: Union[int, float] = 1000,
    part: str = "decoder-weight",
    value: str = "intersection",
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
    ax: Optional[object] = None,
    pretty_groups: Optional[Dict[str, List[str]]] = None,
    scale: str = "linear",
    scale_gamma: Optional[float] = None,
    annot_fontsize: Optional[Union[int, float]] = None,
    tick_label_fontsize: Optional[Union[int, float]] = None,
    group_label_fontsize: Optional[Union[int, float]] = None,
    group_label_rotation_top: Union[int, float] = 0,
    group_label_rotation_right: Union[int, float] = 0,
    cbar_pad: Optional[float] = None,
    cbar_fontsize: Optional[Union[int, float]] = None,
    percentages: bool = True,
    row_metric: Optional[Dict[str, float]] = None,
    row_metric_label: Optional[str] = "corr",
    row_metric_cmap: str = "magma",
    row_metric_vmin: Optional[float] = None,
    row_metric_vmax: Optional[float] = None,
    row_label_mapping: Optional[Dict[str, str]] = None,
    column_label_mapping: Optional[Dict[str, str]] = None,
    highlight_non_convergence: bool = True,
):
    """
    Plot pairwise top-k overlap between GRADIEND models.

    Args:
        models: Mapping of display/model ids to GRADIEND models.
        topk: Number or fraction of top weights to compare.
        part: Model part used to select weights.
        value: Overlap value to display.
        order: ``"input"``, ``"cluster"``, or explicit model order.
        cluster: Whether to cluster rows/columns.
        annot: Whether/how to annotate cells.
        fmt: Deprecated alias for ``annot_fmt``.
        annot_fmt: Cell annotation format.
        figsize: Optional figure size.
        cmap: Heatmap colormap.
        vmin: Optional lower value bound.
        vmax: Optional upper value bound.
        title: Optional plot title, or False to omit.
        output_path: Optional output path.
        show: Whether to display the plot.
        return_data: Whether to include computed data in the return payload.
        return_fig_ax: Whether to include matplotlib figure/axis.
        ax: Optional existing matplotlib axis.
        pretty_groups: Optional model-id groups displayed as brackets.
        scale: Color scale type.
        scale_gamma: Optional gamma for power scaling.
        annot_fontsize: Optional annotation font size.
        tick_label_fontsize: Optional tick-label font size.
        group_label_fontsize: Optional group-label font size.
        group_label_rotation_top: Rotation for top group labels.
        group_label_rotation_right: Rotation for right group labels.
        cbar_pad: Optional colorbar padding.
        cbar_fontsize: Optional colorbar font size.
        percentages: Whether overlap values are shown as percentages.
        row_metric: Optional side metric by row id.
        row_metric_label: Label for the side metric.
        row_metric_cmap: Colormap for the side metric.
        row_metric_vmin: Optional side-metric lower bound.
        row_metric_vmax: Optional side-metric upper bound.
        row_label_mapping: Optional mapping for row display labels.
        column_label_mapping: Optional mapping for column display labels.
        highlight_non_convergence: Whether labels mark non-converged runs.
    """
    return plot_similarity_heatmap(
        models,
        measure="topk_overlap",
        part=part,
        topk=topk,
        value=value,
        order=order,
        cluster=cluster,
        annot=annot,
        fmt=fmt,
        annot_fmt=annot_fmt,
        figsize=figsize,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title,
        output_path=output_path,
        show=show,
        return_data=return_data,
        return_fig_ax=return_fig_ax,
        ax=ax,
        pretty_groups=pretty_groups,
        scale=scale,
        scale_gamma=scale_gamma,
        annot_fontsize=annot_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        group_label_fontsize=group_label_fontsize,
        group_label_rotation_top=group_label_rotation_top,
        group_label_rotation_right=group_label_rotation_right,
        cbar_pad=cbar_pad,
        cbar_fontsize=cbar_fontsize,
        percentages=percentages,
        row_metric=row_metric,
        row_metric_label=row_metric_label,
        row_metric_cmap=row_metric_cmap,
        row_metric_vmin=row_metric_vmin,
        row_metric_vmax=row_metric_vmax,
        row_label_mapping=row_label_mapping,
        column_label_mapping=column_label_mapping,
        highlight_non_convergence=highlight_non_convergence,
    )


def plot_topk_overlap_heatmap_with_correlation(
    models: Dict[str, object],
    topk: Union[int, float] = 1000,
    part: str = "decoder-weight",
    value: str = "intersection_frac",
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
    ax: Optional[object] = None,
    pretty_groups: Optional[Dict[str, List[str]]] = None,
    scale: str = "linear",
    scale_gamma: Optional[float] = None,
    annot_fontsize: Optional[Union[int, float]] = None,
    tick_label_fontsize: Optional[Union[int, float]] = None,
    group_label_fontsize: Optional[Union[int, float]] = None,
    group_label_rotation_top: Union[int, float] = 0,
    group_label_rotation_right: Union[int, float] = 0,
    cbar_pad: Optional[float] = None,
    cbar_fontsize: Optional[Union[int, float]] = None,
    percentages: bool = False,
    row_label_mapping: Optional[Dict[str, str]] = None,
    column_label_mapping: Optional[Dict[str, str]] = None,
    highlight_non_convergence: bool = True,
):
    """Plot pairwise top-k overlap with best-correlation side annotations.

    Args:
        models: Mapping of display/model ids to GRADIEND models.
        topk: Number or fraction of top weights to compare.
        part: Model part used to select weights.
        value: Overlap value to display.
        order: ``"input"``, ``"cluster"``, or explicit model order.
        cluster: Whether to cluster rows/columns.
        annot: Whether/how to annotate cells.
        fmt: Deprecated alias for ``annot_fmt``.
        annot_fmt: Cell annotation format.
        figsize: Optional figure size.
        cmap: Heatmap colormap.
        vmin: Optional lower value bound.
        vmax: Optional upper value bound.
        title: Optional plot title, or False to omit.
        output_path: Optional output path.
        show: Whether to display the plot.
        return_data: Whether to include computed data in the return payload.
        return_fig_ax: Whether to include matplotlib figure/axis.
        ax: Optional existing matplotlib axis.
        pretty_groups: Optional model-id groups displayed as brackets.
        scale: Color scale type.
        scale_gamma: Optional gamma for power scaling.
        annot_fontsize: Optional annotation font size.
        tick_label_fontsize: Optional tick-label font size.
        group_label_fontsize: Optional group-label font size.
        group_label_rotation_top: Rotation for top group labels.
        group_label_rotation_right: Rotation for right group labels.
        cbar_pad: Optional colorbar padding.
        cbar_fontsize: Optional colorbar font size.
        percentages: Whether overlap values are shown as percentages.
        row_label_mapping: Optional mapping for row display labels.
        column_label_mapping: Optional mapping for column display labels.
        highlight_non_convergence: Whether labels mark non-converged runs.
    """
    row_metric = _extract_best_correlation_for_models(models)
    return plot_topk_overlap_heatmap(
        models,
        topk=topk,
        part=part,
        value=value,
        order=order,
        cluster=cluster,
        annot=annot,
        fmt=fmt,
        annot_fmt=annot_fmt,
        figsize=figsize,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        title=title,
        output_path=output_path,
        show=show,
        return_data=return_data,
        return_fig_ax=return_fig_ax,
        ax=ax,
        pretty_groups=pretty_groups,
        scale=scale,
        scale_gamma=scale_gamma,
        annot_fontsize=annot_fontsize,
        tick_label_fontsize=tick_label_fontsize,
        group_label_fontsize=group_label_fontsize,
        group_label_rotation_top=group_label_rotation_top,
        group_label_rotation_right=group_label_rotation_right,
        cbar_pad=cbar_pad,
        cbar_fontsize=cbar_fontsize,
        percentages=percentages,
        row_metric=row_metric or None,
        row_metric_label="corr",
        row_label_mapping=row_label_mapping,
        column_label_mapping=column_label_mapping,
        highlight_non_convergence=highlight_non_convergence,
    )
