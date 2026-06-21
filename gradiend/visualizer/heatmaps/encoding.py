"""Encoder-based comparison heatmap wrappers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from gradiend.comparison import (
    compute_anchor_aligned_encoding_matrix,
    compute_cross_encoding_matrix,
    compute_gradiend_feature_cross_encoding_matrix,
    normalize_cross_encoding_rows_by_diagonal,
    pair_by_id_from_trainers,
)
from gradiend.visualizer.heatmaps.base import plot_comparison_heatmap


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


def plot_anchor_aligned_encoding_heatmap(
    trainers: Dict[str, object],
    feature_classes: Sequence[str],
    *,
    alignment: str = "factual",
    column_ids: Optional[Sequence[str]] = None,
    encoder_summary: Optional[Dict[str, Any]] = None,
    split: str = "test",
    max_size: Optional[int] = None,
    use_cache: bool = True,
    full_eval: bool = True,
    aggregate: str = "mean",
    order: Any = "input",
    cluster: bool = False,
    pretty_groups: Optional[Dict[str, List[str]]] = None,
    highlight_non_convergence: bool = True,
    **plot_kwargs: Any,
) -> Dict[str, Any]:
    """
    Plot oriented cross-encoding for symmetric pairwise GRADIEND trainers.

    See ``compute_anchor_aligned_encoding_matrix`` for semantics. Rows are oriented
    feature anchors. Columns are factual features, counterfactual features, or
    directed transitions depending on ``alignment``.

    Args:
        trainers: Mapping of ids to pairwise trainers.
        feature_classes: Feature classes to use as oriented anchors.
        alignment: Column alignment mode.
        column_ids: Optional explicit output columns.
        encoder_summary: Optional precomputed encoder summary.
        split: Encoder split used when evaluation is needed.
        max_size: Optional evaluation row cap.
        use_cache: Whether to use cached encoder analysis.
        full_eval: Whether to evaluate all transitions.
        aggregate: Aggregate used when multiple trainers cover one anchor.
        order: Heatmap row/column order.
        cluster: Whether to cluster rows/columns.
        pretty_groups: Optional groups shown as brackets.
        highlight_non_convergence: Whether labels mark non-converged runs.
        **plot_kwargs: Additional options forwarded to ``plot_comparison_heatmap``.
    """
    if encoder_summary is None:
        encoder_summary = {}
        include_other_classes = bool(full_eval)
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
    )
    resolved_order = list(feature_classes) if order == "input" else order
    return plot_comparison_heatmap(
        comparison_data,
        order=resolved_order,
        cluster=cluster,
        pretty_groups=pretty_groups,
        models=trainers,
        highlight_non_convergence=highlight_non_convergence,
        **plot_kwargs,
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
    return plot_comparison_heatmap(
        comparison_data,
        order=resolved_order,
        cluster=cluster,
        pretty_groups=pretty_groups,
        models=trainers,
        highlight_non_convergence=highlight_non_convergence,
        **plot_kwargs,
    )


def plot_cross_encoding_heatmap(
    trainers: Dict[str, object],
    *,
    split: str = "test",
    max_size: Optional[int] = None,
    use_cache: bool = True,
    metric: str = "positive_mean",
    full_eval: Optional[bool] = None,
    run_evaluation: bool = True,
    allow_incomplete: bool = False,
    seed_selection: str = "best",
    seed_aggregate: str = "mean",
    dispersion: str = "none",
    normalize: bool = False,
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
    row_label_mapping: Optional[Dict[str, str]] = None,
    column_label_mapping: Optional[Dict[str, str]] = None,
    dispersion_display: str = "none",
    seed_annotation: Union[bool, Dict[str, Any]] = False,
    highlight_non_convergence: bool = True,
) -> Dict[str, Any]:
    """
    Compute a cross-encoding matrix for trainers and plot it as a heatmap.

    ``full_eval`` controls whether encoder evaluation includes all class transitions
    in the split (``True``, default for ``split='test'``) or only the trainer's
    target pair (``False``). This mirrors ``TrainerSuite.evaluate_encoder(full_eval=...)``.

    Args:
        trainers: Mapping of ids to trainers.
        split: Encoder split to evaluate.
        max_size: Optional evaluation row cap.
        use_cache: Whether to use cached encoder analysis.
        metric: Cross-encoding metric.
        full_eval: Whether to evaluate all transitions.
        run_evaluation: Whether to run encoder evaluation before plotting.
        allow_incomplete: Whether incomplete matrix entries are allowed.
        seed_selection: Seed-run selection for multi-seed trainers.
        seed_aggregate: Seed aggregation mode.
        dispersion: Dispersion mode.
        normalize: Whether to normalize rows by diagonal values.
        order: ``"input"``, ``"cluster"``, or explicit order.
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
        pretty_groups: Optional groups shown as brackets.
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
        row_label_mapping: Optional mapping for row labels.
        column_label_mapping: Optional mapping for column labels.
        dispersion_display: How to show dispersion values.
        seed_annotation: Whether/how to annotate seed counts.
        highlight_non_convergence: Whether labels mark non-converged runs.
    """
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
        group_label_fontsize=group_label_fontsize,
        group_label_rotation_top=group_label_rotation_top,
        group_label_rotation_right=group_label_rotation_right,
        cbar_pad=cbar_pad,
        cbar_fontsize=cbar_fontsize,
        percentages=percentages,
        row_label_mapping=row_label_mapping,
        column_label_mapping=column_label_mapping,
        seed_aggregate=seed_aggregate,
        dispersion=dispersion,
        dispersion_display=dispersion_display,
        seed_annotation=seed_annotation,
        models=trainers,
        highlight_non_convergence=highlight_non_convergence,
    )
