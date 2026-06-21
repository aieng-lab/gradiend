"""Similarity heatmap wrappers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from gradiend.comparison import compute_similarity_matrix
from gradiend.trainer.core.stats import load_training_stats
from gradiend.util.logging import get_logger
from gradiend.visualizer.heatmaps.base import plot_comparison_heatmap

logger = get_logger(__name__)


def _extract_best_correlation_for_models(models: Dict[str, object]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for mid, model in models.items():
        model_path = getattr(model, "name_or_path", None)
        if not model_path:
            continue
        try:
            run_info = load_training_stats(model_path)
        except Exception as e:
            logger.debug("Could not load training stats for %s from %s: %s", mid, model_path, e)
            continue
        if not run_info:
            continue
        bsc = run_info.get("best_score_checkpoint") or {}
        corr = bsc.get("correlation")
        if isinstance(corr, (int, float)):
            metrics[mid] = float(corr)
    return metrics


def plot_similarity_heatmap(
    models: Dict[str, object],
    *,
    measure: str = "cosine",
    part: Optional[str] = None,
    topk: Optional[Union[int, float]] = None,
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
    ax: Optional[Any] = None,
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
    row_metric: Optional[Dict[str, float]] = None,
    row_metric_label: Optional[str] = None,
    row_metric_cmap: str = "magma",
    row_metric_vmin: Optional[float] = None,
    row_metric_vmax: Optional[float] = None,
    row_label_mapping: Optional[Dict[str, str]] = None,
    column_label_mapping: Optional[Dict[str, str]] = None,
    seed_aggregate: str = "mean",
    dispersion: str = "none",
    dispersion_display: str = "none",
    seed_annotation: Union[bool, Dict[str, Any]] = False,
    highlight_non_convergence: bool = True,
) -> Any:
    """Compute model similarity and plot it as a heatmap.

    Args:
        models: Mapping of model ids to GRADIEND models.
        measure: Similarity measure.
        part: Model part used by weight-based measures.
        topk: Number or fraction of top weights for top-k measures.
        value: Value variant for overlap-style measures.
        order: ``"input"``, ``"cluster"``, or explicit model order.
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
        return_data: Whether to include computed data.
        return_fig_ax: Whether to include matplotlib figure/axis.
        ax: Optional existing matplotlib axis.
        pretty_groups: Optional model-id groups shown as brackets.
        scale: Color scale type.
        scale_gamma: Optional gamma for power scaling.
        annot_fontsize: Optional annotation font size.
        tick_label_fontsize: Optional tick-label font size.
        group_label_fontsize: Optional group-label font size.
        group_label_rotation_top: Rotation for top group labels.
        group_label_rotation_right: Rotation for right group labels.
        cbar_pad: Optional colorbar padding.
        cbar_fontsize: Optional colorbar font size.
        percentages: Whether to show values as percentages.
        row_metric: Optional side metric by row id.
        row_metric_label: Label for the side metric.
        row_metric_cmap: Colormap for the side metric.
        row_metric_vmin: Optional side-metric lower bound.
        row_metric_vmax: Optional side-metric upper bound.
        row_label_mapping: Optional mapping for row labels.
        column_label_mapping: Optional mapping for column labels.
        seed_aggregate: Seed aggregation mode.
        dispersion: Dispersion mode.
        dispersion_display: How to show dispersion values.
        seed_annotation: Whether/how to annotate seed counts.
        highlight_non_convergence: Whether labels mark non-converged runs.
    """
    comparison_data = compute_similarity_matrix(
        models,
        measure=measure,
        part=part,
        topk=topk,
        value=value,
        seed_aggregate=seed_aggregate,
        dispersion=dispersion,
    )
    return plot_comparison_heatmap(
        comparison_data,
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
        dispersion_display=dispersion_display,
        seed_annotation=seed_annotation,
        models=models,
        highlight_non_convergence=highlight_non_convergence,
    )


def plot_similarity_heatmap_with_correlation(
    models: Dict[str, object],
    *,
    measure: str = "cosine",
    part: Optional[str] = None,
    topk: Optional[Union[int, float]] = None,
    value: str = "intersection_frac",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Plot similarity heatmap with best-correlation side annotations.

    Args:
        models: Mapping of model ids to GRADIEND models.
        measure: Similarity measure.
        part: Model part used by weight-based measures.
        topk: Number or fraction of top weights for top-k measures.
        value: Value variant for overlap-style measures.
        **kwargs: Additional options forwarded to ``plot_similarity_heatmap``.
    """
    row_metric = _extract_best_correlation_for_models(models)
    if not row_metric:
        raise ValueError("Could not extract any correlation metrics for the given models; cannot plot with correlation")
    return plot_similarity_heatmap(
        models,
        measure=measure,
        part=part,
        topk=topk,
        value=value,
        row_metric=row_metric,
        row_metric_label="corr",
        **kwargs,
    )
