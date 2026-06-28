"""Encoder-based comparison heatmap wrappers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from gradiend.comparison import (
    compute_anchor_aligned_encoding_matrix,
    compute_cross_encoding_matrix,
    compute_dense_anchor_aligned_encoding_matrix,
    compute_gradiend_feature_cross_encoding_matrix,
    compute_gradiend_transition_cross_encoding_matrix,
    normalize_cross_encoding_rows_by_diagonal,
    pair_by_id_from_trainers,
    source_by_id_from_trainers,
)
from gradiend.comparison.feature_cross_encoding import build_cross_task_encoder_summary
from gradiend.trainer.core.multi_seed import (
    resolve_dispersion_for_trainers,
    resolve_seed_selection_for_trainers,
)
from gradiend.visualizer.heatmaps.base import filter_comparison_heatmap_plot_kwargs, plot_comparison_heatmap

ORIENTED_CROSS_ENCODING_YLABEL = "Orienting feature"
ORIENTED_CROSS_ENCODING_XLABEL_FACTUAL = "Probe feature"
ORIENTED_CROSS_ENCODING_XLABEL_COUNTERFACTUAL = "Probe feature (counterfactual)"
ORIENTED_CROSS_ENCODING_XLABEL_TRANSITION = "Probe transition"
CROSS_ENCODING_CBAR_LABEL = "Encoding"
CROSS_ENCODING_ROW_NORMALIZED_CBAR_LABEL = "Relative encoding"


def _oriented_cross_encoding_axis_labels(alignment: str) -> tuple[str, str]:
    key = str(alignment).strip().lower()
    if key in {"counterfactual", "cf", "alternative", "alternatives"}:
        xlabel = ORIENTED_CROSS_ENCODING_XLABEL_COUNTERFACTUAL
    elif key in {"transition", "transitions"}:
        xlabel = ORIENTED_CROSS_ENCODING_XLABEL_TRANSITION
    else:
        xlabel = ORIENTED_CROSS_ENCODING_XLABEL_FACTUAL
    return ORIENTED_CROSS_ENCODING_YLABEL, xlabel


def _evaluate_encoder_one_trainer_on_gpu(trainer: object, **kwargs: Any) -> Any:
    """Run encoder eval on one trainer: move to CUDA, evaluate, move back to CPU."""
    moved_to_gpu = False
    if torch.cuda.is_available() and hasattr(trainer, "cuda"):
        trainer.cuda()
        moved_to_gpu = True
    try:
        return trainer.evaluate_encoder(**kwargs)
    finally:
        if moved_to_gpu and hasattr(trainer, "cpu"):
            trainer.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def plot_cross_encoding_heatmap(
    trainers: Dict[str, object],
    feature_classes: Optional[Sequence[str]] = None,
    *,
    alignment: str = "factual",
    column_ids: Optional[Sequence[str]] = None,
    encoder_summary: Optional[Dict[str, Any]] = None,
    split: str = "test",
    max_size: Optional[int] = None,
    use_cache: bool = True,
    full_eval: Optional[bool] = None,
    cross_task_eval: bool = False,
    aggregate: str = "mean",
    metric: str = "positive_mean",
    run_evaluation: bool = True,
    allow_incomplete: bool = False,
    seed_selection: Optional[str] = None,
    seed_aggregate: str = "mean",
    dispersion: Optional[str] = None,
    normalize: bool = False,
    order: Any = "input",
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
    row_label_mapping: Optional[Dict[str, str]] = None,
    column_label_mapping: Optional[Dict[str, str]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    dispersion_display: str = "none",
    seed_annotation: Union[bool, Dict[str, Any]] = False,
    highlight_non_convergence: bool = True,
    **plot_kwargs: Any,
) -> Dict[str, Any]:
    """
    Plot a cross-encoding heatmap for pairwise GRADIEND trainers.

    Pass ``feature_classes`` for oriented symmetric cross-encoding (anchor sign
    alignment and aggregation; see ``compute_anchor_aligned_encoding_matrix``).
    Omit ``feature_classes`` for directed positive-pair cross-encoding via
    ``compute_cross_encoding_matrix`` (default metric ``positive_mean``).

    Args:
        trainers: Mapping of ids to trainers.
        feature_classes: Feature classes used as oriented row anchors. When
            omitted, the directed positive-pair matrix is plotted instead.
        alignment: Column alignment for oriented mode (``factual``,
            ``counterfactual``, or ``transition``).
        column_ids: Optional explicit oriented-matrix columns.
        encoder_summary: Optional precomputed encoder summary for oriented mode.
        split: Encoder split used when evaluation is needed.
        max_size: Optional evaluation row cap.
        use_cache: Whether to use cached encoder analysis.
        full_eval: Whether encoder evaluation includes all transitions.
        cross_task_eval: Oriented mode only; use shared per-class test pool.
        aggregate: Oriented mode aggregate across trainers per anchor.
        metric: Directed mode cross-encoding metric.
        run_evaluation: Directed mode; run encoder eval before plotting.
        allow_incomplete: Directed mode; tolerate missing cells.
        seed_selection: Directed mode seed selection.
        seed_aggregate: Directed mode seed aggregate.
        dispersion: Directed mode dispersion statistic.
        normalize: If True, divide each row by its diagonal so self-encoding is
            1.0. Requires a square matrix with matching row/column ids (oriented
            factual/counterfactual mode, or directed trainer×trainer mode).
        order: Heatmap row/column order.
        cluster: Whether to cluster rows/columns.
        highlight_non_convergence: Whether labels mark non-converged runs.
        xlabel: Optional x-axis label. Oriented mode defaults to a probe-feature label.
        ylabel: Optional y-axis label. Oriented mode defaults to ``Orienting feature``.
        **plot_kwargs: Additional options forwarded to ``plot_comparison_heatmap``.
    """
    if cbar_label is None:
        cbar_label = (
            CROSS_ENCODING_ROW_NORMALIZED_CBAR_LABEL
            if normalize
            else CROSS_ENCODING_CBAR_LABEL
        )
    seed_selection = resolve_seed_selection_for_trainers(trainers, seed_selection)
    if dispersion is None:
        dispersion = resolve_dispersion_for_trainers(trainers, None)
    if feature_classes is not None:
        oriented_full_eval = True if full_eval is None else bool(full_eval)
        if encoder_summary is None and cross_task_eval:
            comparison_data = compute_dense_anchor_aligned_encoding_matrix(
                trainers,
                feature_classes,
                alignment=alignment,
                column_ids=column_ids,
                split=split,
                max_size=max_size,
                aggregate=aggregate,
                seed_selection=seed_selection,
                seed_aggregate=seed_aggregate,
                dispersion=dispersion,
            )
        else:
            if encoder_summary is None:
                if cross_task_eval:
                    encoder_summary = build_cross_task_encoder_summary(
                        trainers,
                        feature_classes,
                        split=split,
                        max_size=max_size,
                        use_cache=use_cache,
                        seed_selection=seed_selection,
                        seed_aggregate=seed_aggregate,
                        dispersion=dispersion,
                    )
                else:
                    encoder_summary = {}
                    include_other_classes = oriented_full_eval
                    for trainer_id, trainer in trainers.items():
                        if not hasattr(trainer, "evaluate_encoder"):
                            raise TypeError(f"Trainer {trainer_id!r} does not support evaluate_encoder")
                        encoder_summary[trainer_id] = _evaluate_encoder_one_trainer_on_gpu(
                            trainer,
                            split=split,
                            max_size=max_size,
                            use_cache=use_cache,
                            return_df=True,
                            plot=False,
                            include_other_classes=include_other_classes,
                        )
            comparison_data = compute_anchor_aligned_encoding_matrix(
                pair_by_id=pair_by_id_from_trainers(trainers),
                encoder_summary=encoder_summary,
                feature_classes=feature_classes,
                aggregate=aggregate,
                alignment=alignment,
                column_ids=column_ids,
                source_by_id=source_by_id_from_trainers(trainers),
            )
        if normalize:
            comparison_data = normalize_cross_encoding_rows_by_diagonal(comparison_data)
        resolved_order = list(feature_classes) if order == "input" else order
        default_ylabel, default_xlabel = _oriented_cross_encoding_axis_labels(alignment)
        return plot_comparison_heatmap(
            comparison_data,
            order=resolved_order,
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
            axis_label_fontsize=axis_label_fontsize,
            group_label_fontsize=group_label_fontsize,
            group_label_rotation_top=group_label_rotation_top,
            group_label_rotation_right=group_label_rotation_right,
            cbar_pad=cbar_pad,
            cbar_fontsize=cbar_fontsize,
            cbar_shrink=cbar_shrink,
            cbar_label=cbar_label,
            percentages=percentages,
            row_label_mapping=row_label_mapping,
            column_label_mapping=column_label_mapping,
            xlabel=default_xlabel if xlabel is None else xlabel,
            ylabel=default_ylabel if ylabel is None else ylabel,
            dispersion_display=dispersion_display,
            seed_annotation=seed_annotation,
            models=trainers,
            highlight_non_convergence=highlight_non_convergence,
            **filter_comparison_heatmap_plot_kwargs(plot_kwargs),
        )

    comparison_data = compute_cross_encoding_matrix(
        trainers,
        split=split,
        max_size=max_size,
        use_cache=use_cache,
        metric=metric,
        full_eval=full_eval,
        run_evaluation=run_evaluation,
        allow_incomplete=allow_incomplete,
        seed_selection=seed_selection,
        seed_aggregate=seed_aggregate,
        dispersion=dispersion,
    )
    if normalize:
        comparison_data = normalize_cross_encoding_rows_by_diagonal(comparison_data)
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
        axis_label_fontsize=axis_label_fontsize,
        group_label_fontsize=group_label_fontsize,
        group_label_rotation_top=group_label_rotation_top,
        group_label_rotation_right=group_label_rotation_right,
        cbar_pad=cbar_pad,
        cbar_fontsize=cbar_fontsize,
        cbar_shrink=cbar_shrink,
        cbar_label=cbar_label,
        percentages=percentages,
        row_label_mapping=row_label_mapping,
        column_label_mapping=column_label_mapping,
        xlabel=xlabel,
        ylabel=ylabel,
        dispersion_display=dispersion_display,
        seed_annotation=seed_annotation,
        models=trainers,
        highlight_non_convergence=highlight_non_convergence,
        **filter_comparison_heatmap_plot_kwargs(plot_kwargs),
    )


def plot_gradiend_feature_cross_encoding_heatmap(
    trainers: Dict[str, object],
    feature_classes: Sequence[str],
    *,
    trainer_order: Optional[Sequence[str]] = None,
    split: str = "test",
    max_size: Optional[int] = None,
    aggregate: str = "mean",
    order: Union[str, List[str]] = "input",
    cluster: bool = False,
    pretty_groups: Optional[Dict[str, List[str]]] = None,
    highlight_non_convergence: bool = True,
    **plot_kwargs: Any,
) -> Dict[str, Any]:
    """
    Plot a dense GRADIEND × feature-class cross-encoding matrix.

    Each row is one trained GRADIEND; each column is one feature class. Cell
    *(i, j)* is the mean encoded value when GRADIEND *i* encodes test-split
    snippets for class *j* (shared eval pool merged across trainers).

    Args:
        trainers: Mapping of ids to trainers.
        feature_classes: Feature classes used as columns.
        trainer_order: Optional explicit trainer row order.
        split: Encoder split to evaluate.
        max_size: Optional evaluation row cap.
        aggregate: ``"mean"`` for encoded values or ``"count"`` for counts.
        order: Heatmap row order.
        cluster: Whether to cluster rows/columns.
        pretty_groups: Optional groups shown as brackets.
        highlight_non_convergence: Whether labels mark non-converged runs.
        **plot_kwargs: Additional options forwarded to ``plot_comparison_heatmap``.
    """
    if aggregate not in {"mean", "count"}:
        raise ValueError("aggregate must be 'mean' or 'count'")
    comparison_data = compute_gradiend_feature_cross_encoding_matrix(
        trainers,
        feature_classes,
        trainer_order=trainer_order,
        split=split,
        max_size=max_size,
    )
    if aggregate == "count":
        comparison_data = dict(comparison_data)
        comparison_data["matrix"] = comparison_data["n_matrix"]
        comparison_data["measure"] = "gradiend_feature_cross_encoding_count"
    resolved_order = (
        list(order)
        if isinstance(order, list)
        else list(comparison_data["model_ids"])
        if order == "input"
        else order
    )
    plot_kwargs = dict(plot_kwargs)
    plot_kwargs.setdefault("cbar_label", CROSS_ENCODING_CBAR_LABEL)
    return plot_comparison_heatmap(
        comparison_data,
        order=resolved_order,
        cluster=cluster,
        pretty_groups=pretty_groups,
        models=trainers,
        highlight_non_convergence=highlight_non_convergence,
        **filter_comparison_heatmap_plot_kwargs(plot_kwargs),
    )


def plot_gradiend_transition_cross_encoding_heatmap(
    trainers: Dict[str, object],
    *,
    trainer_order: Optional[Sequence[str]] = None,
    transition_order: Optional[Sequence[str]] = None,
    encoder_summary: Optional[Dict[str, Any]] = None,
    split: str = "test",
    max_size: Optional[int] = None,
    aggregate: str = "mean",
    seed_selection: Optional[str] = None,
    seed_aggregate: str = "mean",
    dispersion: Optional[str] = None,
    order: Union[str, List[str]] = "input",
    cluster: bool = False,
    pretty_groups: Optional[Dict[str, List[str]]] = None,
    highlight_non_convergence: bool = True,
    **plot_kwargs: Any,
) -> Dict[str, Any]:
    """
    Plot GRADIEND × directed-transition cross-encoding (pre-anchor aggregation).

    Each row is one trained GRADIEND; each column is an input transition
    ``source->target`` from the shared cross-task test pool. This is the standard
    matrix before anchor sign alignment and anchor-class aggregation.
    """
    if aggregate not in {"mean", "count"}:
        raise ValueError("aggregate must be 'mean' or 'count'")
    seed_selection = resolve_seed_selection_for_trainers(trainers, seed_selection)
    if dispersion is None:
        dispersion = resolve_dispersion_for_trainers(trainers, None)
    comparison_data = compute_gradiend_transition_cross_encoding_matrix(
        trainers,
        trainer_order=trainer_order,
        transition_order=transition_order,
        encoder_summary=encoder_summary,
        split=split,
        max_size=max_size,
        seed_selection=seed_selection,
        seed_aggregate=seed_aggregate,
        dispersion=dispersion,
    )
    if aggregate == "count":
        comparison_data = dict(comparison_data)
        comparison_data["matrix"] = comparison_data["n_matrix"]
        comparison_data["measure"] = "gradiend_transition_cross_encoding_count"
    resolved_order = (
        list(order)
        if isinstance(order, list)
        else list(comparison_data["model_ids"])
        if order == "input"
        else order
    )
    plot_kwargs = dict(plot_kwargs)
    plot_kwargs.setdefault("cbar_label", CROSS_ENCODING_CBAR_LABEL)
    return plot_comparison_heatmap(
        comparison_data,
        order=resolved_order,
        cluster=cluster,
        pretty_groups=pretty_groups,
        models=trainers,
        highlight_non_convergence=highlight_non_convergence,
        **filter_comparison_heatmap_plot_kwargs(plot_kwargs),
    )
