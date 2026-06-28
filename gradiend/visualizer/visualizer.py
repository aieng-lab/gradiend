"""
Visualizer(trainer): evaluation-related plots (encoder distributions, scatter, convergence).

Holds a reference to the trainer; exposes single-model visualization helpers.
Option A: Evaluator holds a Visualizer for evaluation-related plots.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd

from gradiend.visualizer.encoder_distributions import plot_encoder_distributions as _plot_encoder_distributions
from gradiend.visualizer.convergence import plot_training_convergence as _plot_training_convergence
from gradiend.visualizer.encoder_scatter import plot_encoder_scatter as _plot_encoder_scatter
from gradiend.visualizer.encoder_strip_split import plot_encoder_strip_by_split as _plot_encoder_strip_by_split
from gradiend.visualizer.encoder_by_target import plot_encoder_by_target as _plot_encoder_by_target
from gradiend.visualizer.probability_shifts import plot_probability_shifts as _plot_probability_shifts
from gradiend.visualizer.plot_delegation import see_implementation
from gradiend.visualizer.topk.venn_ import (
    compute_topk_sets,
    plot_topk_overlap_venn,
)


class Visualizer:
    """
    Visualizer bound to a trainer. Exposes single-model plots.
    User can subclass to customize plotting behavior.
    """

    def __init__(self, trainer: Any):
        self._trainer = trainer

    @property
    def trainer(self):
        """Trainer instance this visualizer delegates to."""
        return self._trainer

    def plot_encoder_distributions(
        self,
        encoder_df: Optional[pd.DataFrame] = None,
        *,
        output: Optional[str] = None,
        output_dir: Optional[str] = None,
        show: bool = True,
        title: Union[str, bool] = True,
        target_and_neutral_only: bool = True,
        split_plot_mode: str = "facet",
        include_neutral: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        img_format: str = "png",
        dpi: Optional[int] = None,
        highlight_non_convergence: Optional[bool] = None,
        return_fig_ax: bool = False,
        **kwargs: Any,
    ) -> str:
        """Plot encoder distributions for this trainer.

        Args:
            encoder_df: Optional precomputed encoder analysis DataFrame.
            output: Explicit output path.
            output_dir: Directory used when resolving the default output filename.
            show: Whether to display the plot.
            title: True for the default title, False for no title, or a custom title string.
            target_and_neutral_only: Restrict to target transitions and neutral rows.
            split_plot_mode: Multi-split layout mode.
            include_neutral: Include neutral encoder rows.
            figsize: Figure size in inches.
            img_format: File format used when saving.
            dpi: Optional savefig DPI.
            highlight_non_convergence: Append a non-convergence marker when requested.
            return_fig_ax: Return ``(fig, axes)`` instead of the output path.
            **kwargs: Forwarded to ``gradiend.visualizer.encoder_distributions.plot_encoder_distributions``.
        """
        return _plot_encoder_distributions(
            self._trainer,
            encoder_df=encoder_df,
            output=output,
            output_dir=output_dir,
            show=show,
            title=title,
            target_and_neutral_only=target_and_neutral_only,
            split_plot_mode=split_plot_mode,
            include_neutral=include_neutral,
            figsize=figsize,
            img_format=img_format,
            dpi=dpi,
            highlight_non_convergence=highlight_non_convergence,
            return_fig_ax=return_fig_ax,
            **kwargs,
        )
    plot_encoder_distributions.__doc__ = (
        "Plot encoder distributions (grouped split violins). Pass encoder_df for self-managed data."
        + see_implementation("gradiend.visualizer.encoder_distributions.plot_encoder_distributions")
    )

    def plot_topk_neuron_intersection(
        self,
        models: Optional[Dict[str, Any]] = None,
        topk: int = 100,
        part: str = "decoder-weight",
        **kwargs: Any,
    ) -> Any:
        """Plot a top-k neuron intersection Venn diagram.

        Args:
            models: Optional mapping of model label to model. When ``None``, uses this
                trainer's current model.
            topk: Number of top weights to include per model.
            part: Model part passed to ``get_topk_weights``.
            **kwargs: Forwarded to ``plot_topk_overlap_venn``.
        """
        if models is None:
            model = self._trainer.get_model()
            models = {getattr(model, "name_or_path", "model"): model} if model is not None else {}
        return plot_topk_overlap_venn(models, topk=topk, part=part, **kwargs)

    def plot_training_convergence(
        self,
        *,
        plot_mean_by_class: bool = True,
        plot_mean_by_feature_class: Optional[bool] = None,
        plot_correlation: bool = True,
        class_spread: Optional[Literal["minmax", "iqr", "ci95"]] = None,
        output: Optional[str] = None,
        show: bool = True,
        title: Union[str, bool] = True,
        figsize: Optional[Tuple[float, float]] = None,
        img_format: str = "png",
        dpi: Optional[int] = None,
        highlight_non_convergence: Optional[bool] = None,
        return_fig_ax: bool = False,
        **kwargs: Any,
    ) -> str:
        """Plot convergence statistics for this trainer.

        Args:
            plot_mean_by_class: Include mean encoded value by class.
            plot_mean_by_feature_class: Include mean encoded value by feature class. ``None``
                auto-disables redundant feature-class plots.
            plot_correlation: Include correlation over training steps.
            class_spread: Optional spread band behind class means.
                ``"minmax"`` shades min-max encoded values, ``"iqr"`` shades Q1-Q3,
                ``"ci95"`` shades mean +/- 1.96 standard errors,
                and ``None`` disables spread shading.
            output: Explicit output path.
            show: Whether to display the plot.
            title: True for the default title, False for no title, or a custom title string.
            figsize: Figure size in inches.
            img_format: File format used when saving.
            dpi: Optional savefig DPI.
            highlight_non_convergence: Append a non-convergence marker when requested.
            return_fig_ax: Return ``(fig, axes)`` instead of the output path.
            **kwargs: Forwarded to ``gradiend.visualizer.convergence.plot_training_convergence``.
        """
        return _plot_training_convergence(
            trainer=self._trainer,
            plot_mean_by_class=plot_mean_by_class,
            plot_mean_by_feature_class=plot_mean_by_feature_class,
            plot_correlation=plot_correlation,
            class_spread=class_spread,
            output=output,
            show=show,
            title=title,
            figsize=figsize,
            img_format=img_format,
            dpi=dpi,
            highlight_non_convergence=highlight_non_convergence,
            return_fig_ax=return_fig_ax,
            **kwargs,
        )
    plot_training_convergence.__doc__ = (
        "Plot training convergence (means by class/feature class and correlation). Uses trainer for stats."
        + see_implementation("gradiend.visualizer.convergence.plot_training_convergence")
    )

    def plot_encoder_scatter(
        self,
        encoder_df: Optional[pd.DataFrame] = None,
        *,
        color_by: str = "label",
        x_col: Optional[str] = None,
        label_name_mapping: Optional[dict] = None,
        max_points: Optional[int] = None,
        show: bool = True,
        title: Optional[str] = None,
        height: int = 500,
        split: str = "test",
        highlight_non_convergence: Optional[bool] = None,
        **kwargs: Any,
    ) -> Any:
        """Create an interactive encoder scatter plot for this trainer.

        Args:
            encoder_df: Optional precomputed encoder analysis DataFrame.
            color_by: Column used for point colors.
            x_col: Optional column used for the x-axis. Defaults to target token when available.
            label_name_mapping: Optional display-name mapping for color labels.
            max_points: Optional stratified point limit.
            show: Whether to display the Plotly figure.
            title: Optional plot title.
            height: Plotly figure height.
            split: Encoder split to compute when ``encoder_df`` is not supplied.
            highlight_non_convergence: Append a non-convergence marker when requested.
            **kwargs: Forwarded to ``gradiend.visualizer.encoder_scatter.plot_encoder_scatter``.
        """
        return _plot_encoder_scatter(
            trainer=self._trainer,
            encoder_df=encoder_df,
            color_by=color_by,
            x_col=x_col,
            label_name_mapping=label_name_mapping,
            max_points=max_points,
            show=show,
            title=title,
            height=height,
            split=split,
            highlight_non_convergence=highlight_non_convergence,
            **kwargs,
        )
    plot_encoder_scatter.__doc__ = (
        "Interactive 1D encoder scatter (categorical target x, encoded y), colored by label, with hover. For Jupyter."
        + see_implementation("gradiend.visualizer.encoder_scatter.plot_encoder_scatter")
    )

    def plot_encoder_strip_by_split(
        self,
        encoder_df: Optional[pd.DataFrame] = None,
        *,
        include_neutral: bool = False,
        title: Optional[str] = None,
        output: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        jitter: float = 0.08,
        dodge: bool = True,
        point_size: float = 5.0,
        label_points: Union[bool, Literal["outliers", "outliers+sample", "sample"], str] = False,
        highlight_non_convergence: Optional[bool] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Plot encoded values by feature group and data split for this trainer.

        Args:
            encoder_df: Optional precomputed encoder analysis DataFrame.
            include_neutral: Include neutral encoder rows.
            title: Optional plot title.
            output: Explicit output path.
            show: Whether to display the plot.
            figsize: Figure size in inches.
            jitter: Horizontal jitter width.
            dodge: Whether to dodge points by hue.
            point_size: Marker size.
            label_points: Point-label mode.
            highlight_non_convergence: Append a non-convergence marker when requested.
            **kwargs: Forwarded to ``gradiend.visualizer.encoder_strip_split.plot_encoder_strip_by_split``.
        """
        return _plot_encoder_strip_by_split(
            trainer=self._trainer,
            encoder_df=encoder_df,
            include_neutral=include_neutral,
            title=title,
            output=output,
            show=show,
            figsize=figsize,
            jitter=jitter,
            dodge=dodge,
            point_size=point_size,
            label_points=label_points,
            highlight_non_convergence=highlight_non_convergence,
            **kwargs,
        )
    plot_encoder_strip_by_split.__doc__ = (
        "Matplotlib strip scatter by feature group and data split, with optional point labels."
        + see_implementation("gradiend.visualizer.encoder_strip_split.plot_encoder_strip_by_split")
    )

    def plot_encoder_by_target(
        self,
        encoder_df: Optional[pd.DataFrame] = None,
        *,
        plot_style: Literal["strip", "box", "violin"] = "strip",
        title: Optional[str] = None,
        output: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        jitter: float = 0.25,
        dodge: bool = True,
        point_size: float = 1.5,
        interactive: bool = False,
        height: int = 520,
        legend_loc: str = "upper right",
        highlight_non_convergence: Optional[bool] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Plot encoded values by masked target token for this trainer.

        Args:
            encoder_df: Optional precomputed encoder analysis DataFrame.
            plot_style: Static plot style: ``strip``, ``box``, or ``violin``.
            title: Optional plot title.
            output: Explicit output path.
            show: Whether to display the plot.
            figsize: Static figure size in inches.
            jitter: Jitter width for static strip plots.
            dodge: Whether to dodge points by hue.
            point_size: Marker size for static strip plots.
            interactive: Return/write a Plotly interactive strip plot instead of a static plot.
            height: Plotly figure height for interactive plots.
            legend_loc: Matplotlib legend location for static plots.
            highlight_non_convergence: Append a non-convergence marker when requested.
            **kwargs: Forwarded to ``gradiend.visualizer.encoder_by_target.plot_encoder_by_target``.
        """
        return _plot_encoder_by_target(
            trainer=self._trainer,
            encoder_df=encoder_df,
            plot_style=plot_style,
            title=title,
            output=output,
            show=show,
            figsize=figsize,
            jitter=jitter,
            dodge=dodge,
            point_size=point_size,
            interactive=interactive,
            height=height,
            legend_loc=legend_loc,
            highlight_non_convergence=highlight_non_convergence,
            **kwargs,
        )
    plot_encoder_by_target.__doc__ = (
        "Encoder plot with target tokens on the x-axis, grouped by feature class, hue = split."
        + see_implementation("gradiend.visualizer.encoder_by_target.plot_encoder_by_target")
    )

    def plot_probability_shifts(
        self,
        decoder_results: Optional[Dict[str, Any]] = None,
        class_ids: Optional[List[str]] = None,
        target_class: Optional[str] = None,
        increase_target_probabilities: bool = True,
        use_cache: Optional[bool] = None,
        *,
        output: Optional[str] = None,
        show: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        highlight_non_convergence: Optional[bool] = None,
        return_fig_ax: bool = False,
        **kwargs: Any,
    ) -> str:
        """Plot decoder probability shifts for this trainer.

        Args:
            decoder_results: Optional precomputed decoder evaluation results.
            class_ids: Classes to include. Defaults to trainer classes.
            target_class: Target class whose probability shift is highlighted.
            increase_target_probabilities: Select strengthening or weakening summary.
            use_cache: Cache flag forwarded to trainer decoder evaluation/analysis.
            output: Explicit output path.
            show: Whether to display the plot.
            figsize: Figure size in inches.
            highlight_non_convergence: Append a non-convergence marker when requested.
            return_fig_ax: Return ``(fig, axes)`` instead of the output path.
            **kwargs: Forwarded to ``gradiend.visualizer.probability_shifts.plot_probability_shifts``.
        """
        if decoder_results is None:
            decoder_results = self.trainer.evaluate_decoder(use_cache=use_cache)

        plotting_data = self.trainer.analyze_decoder_for_plotting(
            decoder_results=decoder_results,
            class_ids=class_ids,
            use_cache=use_cache,
        )

        return _plot_probability_shifts(
            trainer=self.trainer,
            decoder_results=decoder_results,
            plotting_data=plotting_data,
            class_ids=class_ids,
            target_class=target_class,
            increase_target_probabilities=increase_target_probabilities,
            output=output,
            show=show,
            figsize=figsize,
            highlight_non_convergence=highlight_non_convergence,
            return_fig_ax=return_fig_ax,
            **kwargs,
        )
    plot_probability_shifts.__doc__ = (
        "Plot decoder probability shifts vs learning rate."
        + see_implementation("gradiend.visualizer.probability_shifts.plot_probability_shifts")
    )

    @staticmethod
    def compute_topk_sets(models: Dict[str, Any], topk: int = 100, part: str = "decoder-weight"):
        """Compute top-k weight sets for multiple models.

        Args:
            models: Mapping from model label to model with ``get_topk_weights``.
            topk: Number of top weights to select per model.
            part: Model part passed to ``get_topk_weights``.
        """
        return compute_topk_sets(models, topk=topk, part=part)
