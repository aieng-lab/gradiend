"""
Evaluator bound to a trainer; orchestrates encoder/decoder evaluation and
optionally delegates plotting to a Visualizer.

This module provides the high-level entry points to:
1) run encoder evaluation (gradient encodings + correlation metrics),
2) run decoder evaluation (grid search over feature_factor/lr + summaries),
3) merge results for convenience, and
4) produce evaluation-related plots if a Visualizer is configured.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import pandas as pd

from gradiend.evaluator.encoder import EncoderEvaluator
from gradiend.evaluator.decoder import DecoderEvaluator
from gradiend.util.logging import get_logger
from gradiend.visualizer.plot_delegation import see_implementation

logger = get_logger(__name__)


from gradiend.visualizer.visualizer import Visualizer
def _default_visualizer_class():
    return Visualizer


class Evaluator:
    """
    High-level evaluation coordinator bound to a trainer.

    This class owns an EncoderEvaluator and DecoderEvaluator and exposes
    convenience methods that pass the trainer through. It can also hold a
    Visualizer instance for evaluation-related plots.

    Subclasses can override evaluation or plotting methods to customize
    caching, metrics, or visualization behavior.
    """

    def __init__(
        self,
        trainer: Any,
        encoder_evaluator: Optional[EncoderEvaluator] = None,
        decoder_evaluator: Optional[DecoderEvaluator] = None,
        visualizer_class: Optional[Type] = None,
    ):
        self._trainer = trainer
        self._encoder_evaluator = encoder_evaluator or EncoderEvaluator()
        self._decoder_evaluator = decoder_evaluator or DecoderEvaluator()
        self._visualizer = None
        self._visualizer_class = visualizer_class if visualizer_class is not None else _default_visualizer_class()

    @property
    def trainer(self):
        return self._trainer

    def _get_visualizer(self):
        if self._visualizer is not None:
            return self._visualizer
        if self._visualizer_class is None:
            return None
        self._visualizer = self._visualizer_class(self._trainer)
        return self._visualizer

    def _delegate_to_visualizer(self, method_name: str, **kwargs: Any) -> Any:
        viz = self._get_visualizer()
        if viz is not None and hasattr(viz, method_name):
            return getattr(viz, method_name)(**kwargs)
        raise NotImplementedError(
            f"{method_name} requires a Visualizer; set visualizer_class on Evaluator or override this method."
        )

    def evaluate_encoder(
        self,
        encoder_df: Optional[Union[Any, Dict[str, Any]]] = None,
        eval_data: Any = None,
        use_cache: Optional[bool] = None,
        split: Optional[str] = None,
        max_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Run encoder evaluation and return encoding/correlation metrics.

        Args:
            encoder_df: Optional DataFrame or dict with "encoder_df" key. If provided,
                skips encoding and computes metrics from this data. Use
                evaluate_encoder(return_df=True) to get such a dict.
            eval_data: Optional pre-computed GradientTrainingDataset. If None and
                encoder_df is None, the trainer creates eval data via create_eval_data.
            use_cache: If True, reuse cached JSON result under experiment_dir when
                available. If None, defaults come from trainer training args.
            split: Dataset split for eval data creation. Default: "test".
            max_size: Maximum samples per variant for eval data creation.
            **kwargs: Forwarded to create_eval_data when encoder_df and eval_data are None.

        Returns:
            Dict with keys: correlation, mean_by_class, mean_by_type, n_samples,
            all_data, training_only, target_classes_only, boundaries; optionally
            neutral_mean_by_type, mean_by_feature_class, label_value_to_class_name.
        """
        resolved_df = encoder_df
        if isinstance(encoder_df, dict) and "encoder_df" in encoder_df:
            resolved_df = encoder_df["encoder_df"]
        return self._encoder_evaluator.evaluate_encoder(
            self._trainer,
            encoder_df=resolved_df,
            eval_data=eval_data,
            use_cache=use_cache,
            split=split,
            max_size=max_size,
            **kwargs,
        )

    def evaluate_decoder(
        self,
        model_with_gradiend: Any = None,
        feature_factors: Optional[list] = None,
        lrs: Optional[list] = None,
        use_cache: Optional[bool] = None,
        max_size_training_like: Optional[int] = None,
        max_size_neutral: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        training_like_df: Optional[Any] = None,
        neutral_df: Optional[Any] = None,
        selector: Optional[Any] = None,
        summary_extractor: Optional[Any] = None,
        summary_metrics: Optional[Any] = None,
        target_class: Optional[Any] = None,
        increase_target_probabilities: bool = True,
        plot: bool = False,
        show: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Run decoder grid evaluation and return summary + grid for one direction (strengthen or weaken).

        Only the dataset and feature-factor combinations required for the requested direction are computed.
        Use increase_target_probabilities=True (default) for strengthen, False for weaken.

        Args:
            model_with_gradiend: Optional ModelWithGradiend (or path) to evaluate.
                If None, the trainer's model is used.
            feature_factors: Optional list of feature factors to test. If None,
                derived from direction and target classes.
            lrs: Optional list of learning rates to test. If None, defaults are used.
            use_cache: If True, cached decoder grid results are reused when
                available under the trainer's experiment_dir. If None, defaults
                come from trainer training args.
            max_size_training_like: Maximum size for generated training-like eval data.
            max_size_neutral: Maximum size for generated neutral eval data (and LMS text cap).
            eval_batch_size: Common eval batch size used for LMS.
            training_like_df: Optional explicit training-like DataFrame.
            neutral_df: Optional explicit neutral DataFrame.
            selector: Optional SelectionPolicy for choosing best candidate per metric (e.g. LMSThresholdPolicy).
            summary_extractor: Optional callable(results) -> (candidates, ctx). Use to add derived metrics
                (e.g. bpi, fpi, mpi) to candidates; then pass summary_metrics.
            summary_metrics: Optional list of metric names to summarize (e.g. ["bpi", "fpi", "mpi"]).
            target_class: If set (str or list of str), evaluate only for this target class (or classes).
                Restricts feature factors and datasets for efficiency. When None, evaluates for all target classes.
            increase_target_probabilities: If True (default), compute strengthen summaries only (keys e.g. "3SG").
                If False, compute weaken summaries only (keys e.g. "3SG_weaken"). Only required combinations are evaluated.
            plot: If True, after selection run any missing dataset evaluations for plotting, update cache, then plot.
            show: If True, display the plot; if False, only save. When None and plot=True, defaults to True.

        Returns:
            Flat dict: for strengthen, keys like result['3SG']; for weaken, keys like result['3SG_weaken'].
            Each entry has value, feature_factor, learning_rate, id, strengthen, lms, base_lms. Plus 'grid'.
            When plot=True, also 'plot_paths' and 'plot_path'.
        """
        kwargs = dict(
            trainer=self._trainer,
            model_with_gradiend=model_with_gradiend,
            feature_factors=feature_factors,
            lrs=lrs,
            use_cache=use_cache,
            max_size_training_like=max_size_training_like,
            max_size_neutral=max_size_neutral,
            eval_batch_size=eval_batch_size,
            training_like_df=training_like_df,
            neutral_df=neutral_df,
            summary_metrics=summary_metrics,
            target_class=target_class,
            increase_target_probabilities=increase_target_probabilities,
            plot=plot,
            show=show if show is not None else plot,
        )
        if selector is not None:
            kwargs["selector"] = selector
        if summary_extractor is not None:
            kwargs["summary_extractor"] = summary_extractor
        return self._decoder_evaluator.evaluate_decoder(**kwargs)

    def evaluate(self, *, kwargs_encoder: dict = None, kwargs_decoder: dict = None, **kwargs: Any) -> Dict[str, Any]:
        """
        Run encoder and decoder evaluation and return a combined result.

        Args:
            kwargs_encoder: Optional dict of keyword arguments forwarded to
                evaluate_encoder.
            kwargs_decoder: Optional dict of keyword arguments forwarded to
                evaluate_decoder.
            **kwargs: Extra kwargs applied to both encoder and decoder
                evaluations (e.g., shared eval data settings).

        Returns:
            Dict with:
            - encoder: Result dict from evaluate_encoder.
            - decoder: Result dict from evaluate_decoder.
        """
        kwargs_encoder = kwargs_encoder or {}
        kwargs_decoder = kwargs_decoder or {}

        # Pass through any additional kwargs to both evaluators (e.g. for create_eval_data).
        kwargs_encoder.update(kwargs)
        kwargs_decoder.update(kwargs)

        enc = self.evaluate_encoder(**kwargs_encoder)
        dec = self.evaluate_decoder(**kwargs_decoder)

        return {"encoder": enc, "decoder": dec}

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
    ) -> Any:
        """
        Plot encoder value distributions for target and optional neutral rows.

        Args:
            encoder_df: Optional encoder-evaluation DataFrame. When omitted, the
                visualizer loads or computes the trainer's encoder analysis data.
            output: Optional explicit output file path.
            output_dir: Optional output directory used when ``output`` is omitted.
            show: Whether to display the figure interactively.
            title: Plot title. ``True`` uses the default title, ``False`` omits it.
            target_and_neutral_only: If True, omit identity/auxiliary training rows.
            split_plot_mode: How split-aware data is shown, e.g. ``"facet"``.
            include_neutral: If True, include neutral evaluation rows when present.
            figsize: Optional Matplotlib figure size.
            img_format: File format used for generated output paths.
            dpi: Optional figure DPI.
            highlight_non_convergence: Override whether non-convergent runs are
                marked in titles/labels. ``None`` uses trainer/config defaults.
            return_fig_ax: If True, return Matplotlib ``(fig, ax)`` instead of
                only the output path.
            **kwargs: Additional keyword arguments forwarded to the visualizer.

        Returns:
            Visualizer-specific result, usually an output path or ``(fig, ax)``.
        """
        return self._delegate_to_visualizer(
            "plot_encoder_distributions",
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
        plot_encoder_distributions.__doc__
        + see_implementation("gradiend.visualizer.encoder_distributions.plot_encoder_distributions")
    )

    def plot_training_convergence(
        self,
        *,
        plot_mean_by_class: bool = True,
        plot_mean_by_feature_class: Optional[bool] = None,
        plot_correlation: bool = True,
        output: Optional[str] = None,
        show: bool = True,
        title: Union[str, bool] = True,
        figsize: Optional[Tuple[float, float]] = None,
        img_format: str = "png",
        dpi: Optional[int] = None,
        highlight_non_convergence: Optional[bool] = None,
        return_fig_ax: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Plot convergence statistics collected during GRADIEND training.

        Args:
            plot_mean_by_class: Plot mean encoder values by label class.
            plot_mean_by_feature_class: Plot means grouped by feature class.
                ``None`` lets the visualizer decide from available statistics.
            plot_correlation: Plot correlation over training steps.
            output: Optional explicit output file path.
            show: Whether to display the figure interactively.
            title: Plot title. ``True`` uses the default title, ``False`` omits it.
            figsize: Optional Matplotlib figure size.
            img_format: File format used for generated output paths.
            dpi: Optional figure DPI.
            highlight_non_convergence: Override whether non-convergent runs are
                marked in titles/labels. ``None`` uses trainer/config defaults.
            return_fig_ax: If True, return Matplotlib ``(fig, ax)``.
            **kwargs: Additional keyword arguments forwarded to the visualizer.

        Returns:
            Visualizer-specific result, usually an output path or ``(fig, ax)``.
        """
        return self._delegate_to_visualizer(
            "plot_training_convergence",
            plot_mean_by_class=plot_mean_by_class,
            plot_mean_by_feature_class=plot_mean_by_feature_class,
            plot_correlation=plot_correlation,
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
        plot_training_convergence.__doc__
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
        """
        Create an interactive Plotly scatter plot for encoder outlier inspection.

        Args:
            encoder_df: Optional encoder-evaluation DataFrame. When omitted, the
                visualizer loads or computes encoder analysis data.
            color_by: Column used for point color, commonly ``"label"``.
            x_col: Column used for the categorical x-axis. ``None`` lets the
                visualizer choose a meaningful target/token column.
            label_name_mapping: Optional mapping from raw labels to display names.
            max_points: Optional cap on plotted rows.
            show: Whether to display the Plotly figure.
            title: Optional plot title.
            height: Plot height in pixels.
            split: Split to evaluate/load when ``encoder_df`` is omitted.
            highlight_non_convergence: Override non-convergence markers in title.
            **kwargs: Additional keyword arguments forwarded to the visualizer.

        Returns:
            Visualizer-specific result, typically an HTML path or Plotly figure.
        """
        return self._delegate_to_visualizer(
            "plot_encoder_scatter",
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
        plot_encoder_scatter.__doc__
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
        """
        Plot encoder values as a strip plot grouped by data split.

        Args:
            encoder_df: Optional encoder-evaluation DataFrame.
            include_neutral: If True, include neutral rows when available.
            title: Optional plot title.
            output: Optional explicit output file path.
            show: Whether to display the Matplotlib figure.
            figsize: Optional Matplotlib figure size.
            jitter: Horizontal jitter for points.
            dodge: If True, separate split hues within each group.
            point_size: Marker size.
            label_points: Whether and how to label points. Supported values are
                visualizer-defined, including ``False``, ``"outliers"``,
                ``"outliers+sample"``, and ``"sample"``.
            highlight_non_convergence: Override non-convergence markers in title.
            **kwargs: Additional keyword arguments forwarded to the visualizer.

        Returns:
            Output path when saved, otherwise visualizer-specific result.
        """
        return self._delegate_to_visualizer(
            "plot_encoder_strip_by_split",
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
        plot_encoder_strip_by_split.__doc__
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
        """
        Plot encoder values grouped by target token within feature class.

        Args:
            encoder_df: Optional encoder-evaluation DataFrame.
            plot_style: Plot style: ``"strip"``, ``"box"``, or ``"violin"``.
            title: Optional plot title.
            output: Optional explicit output file path.
            show: Whether to display the plot.
            figsize: Optional Matplotlib figure size for static plots.
            jitter: Horizontal jitter for strip points.
            dodge: If True, separate split hues within each target.
            point_size: Marker size for strip plots.
            interactive: If True, create the interactive Plotly strip variant.
            height: Plotly height in pixels for interactive plots.
            legend_loc: Static Matplotlib legend location.
            highlight_non_convergence: Override non-convergence markers in title.
            **kwargs: Additional keyword arguments forwarded to the visualizer.

        Returns:
            Output path when saved, otherwise visualizer-specific result.
        """
        return self._delegate_to_visualizer(
            "plot_encoder_by_target",
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
        plot_encoder_by_target.__doc__
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
    ) -> Any:
        """
        Plot decoder probability shifts over learning-rate/grid candidates.

        Args:
            decoder_results: Optional result from ``evaluate_decoder``. When
                omitted, cached decoder results may be loaded if available.
            class_ids: Optional class ids to include in the plot.
            target_class: Optional single target class to plot.
            increase_target_probabilities: True for strengthen plots, False for
                weaken plots.
            use_cache: Whether the visualizer may use cached decoder results.
            output: Optional explicit output file path.
            show: Whether to display the Matplotlib figure.
            figsize: Optional Matplotlib figure size.
            highlight_non_convergence: Override non-convergence markers in title.
            return_fig_ax: If True, return Matplotlib ``(fig, ax)``.
            **kwargs: Additional keyword arguments forwarded to the visualizer.

        Returns:
            Visualizer-specific result, usually an output path or ``(fig, ax)``.
        """
        return self._delegate_to_visualizer(
            "plot_probability_shifts",
            decoder_results=decoder_results,
            class_ids=class_ids,
            target_class=target_class,
            increase_target_probabilities=increase_target_probabilities,
            use_cache=use_cache,
            output=output,
            show=show,
            figsize=figsize,
            highlight_non_convergence=highlight_non_convergence,
            return_fig_ax=return_fig_ax,
            **kwargs,
        )
    plot_probability_shifts.__doc__ = (
        plot_probability_shifts.__doc__
        + see_implementation("gradiend.visualizer.probability_shifts.plot_probability_shifts")
    )
