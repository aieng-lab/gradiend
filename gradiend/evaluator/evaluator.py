"""
Evaluator bound to a trainer; orchestrates encoder/decoder evaluation and
optionally delegates plotting to a Visualizer.

This module provides the high-level entry points to:
1) run encoder evaluation (gradient encodings + correlation metrics),
2) run decoder evaluation (grid search over feature_factor/lr + summaries),
3) merge results for convenience, and
4) produce evaluation-related plots if a Visualizer is configured.
"""

from typing import Any, Dict, Optional, Type, Union

from gradiend.evaluator.encoder import EncoderEvaluator
from gradiend.evaluator.decoder import DecoderEvaluator
from gradiend.util.logging import get_logger

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
    ) -> Dict[str, Any]:
        """
        Run decoder grid evaluation and return summary + grid results.

        Args:
            model_with_gradiend: Optional ModelWithGradiend (or path) to evaluate.
                If None, the trainer's model is used.
            feature_factors: Optional list of feature factors to test. If None,
                DecoderEvaluator derives defaults from trainer target classes.
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

        Returns:
            A dict with:
            - summary: Per-metric selection summary (best candidate id, value,
              feature_factor, and learning_rate).
            - grid: Mapping of candidate id -> evaluation results, including a
              "base" entry for the unmodified model.
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

    def plot_encoder_distributions(self, **kwargs) -> Any:
        """
        Plot encoder distributions (typically a violin plot).

        Args:
            **kwargs: Forwarded to the Visualizer implementation.

        Returns:
            Whatever the Visualizer returns (often a matplotlib/seaborn figure).

        Raises:
            NotImplementedError: If no Visualizer is configured and this method
                is not overridden in a subclass.
        """
        return self._delegate_to_visualizer("plot_encoder_distributions", **kwargs)

    def plot_training_convergence(self, **kwargs) -> Any:
        """
        Plot training convergence (means by class/feature_class and correlation).

        Args:
            **kwargs: Forwarded to the Visualizer implementation.

        Returns:
            Whatever the Visualizer returns (often a matplotlib/seaborn figure).

        Raises:
            NotImplementedError: If no Visualizer is configured and this method
                is not overridden in a subclass.
        """
        return self._delegate_to_visualizer("plot_training_convergence", **kwargs)

    def plot_encoder_scatter(self, **kwargs) -> Any:
        """
        Plot interactive encoder scatter (Plotly: jitter x, encoded y, colored by label).

        Args:
            **kwargs: Forwarded to the Visualizer implementation (encoder_df, show, etc.).

        Returns:
            Plotly Figure or None if Plotly is not installed.

        Raises:
            NotImplementedError: If no Visualizer is configured.
        """
        return self._delegate_to_visualizer("plot_encoder_scatter", **kwargs)
