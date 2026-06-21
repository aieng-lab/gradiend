"""
Classification-specific evaluator and encoder evaluator.

Encoder eval branches on the actual eval data: if it has only one label class
(e.g. different texts / equal labels, or capped subset), we return correlation=0
and mean_by_class; otherwise we use the default encoder metrics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from gradiend.evaluator import Evaluator
from gradiend.evaluator.encoder import EncoderEvaluator
from gradiend.trainer.core.dataset import GradientTrainingDataset
from gradiend.util.encoding_rows import encode_dataset_to_rows
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def _rows_to_df(rows: list) -> pd.DataFrame:
    """Build encoder DataFrame from encoded rows (encoded, label, ...)."""
    if not rows:
        return pd.DataFrame(columns=["encoded", "label"])
    return pd.DataFrame([{"encoded": r.get("encoded"), "label": r.get("label", 0.0)} for r in rows])


def _single_class_result(df: pd.DataFrame) -> Dict[str, Any]:
    """Build result dict when eval data has only one (or zero) label class."""
    if df is None or len(df) == 0:
        return {"correlation": 0.0, "mean_by_class": {}, "n_samples": 0}
    if "encoded" not in df.columns or "label" not in df.columns:
        return {"correlation": 0.0, "mean_by_class": {}, "n_samples": len(df)}
    mean_by_class = df.groupby("label", dropna=False)["encoded"].agg("mean").astype(float)
    mean_by_class = {str(k): float(v) for k, v in mean_by_class.items()}
    return {"correlation": 0.0, "mean_by_class": mean_by_class, "n_samples": len(df)}


class ClassificationEncoderEvaluator(EncoderEvaluator):
    """
    Encoder evaluator for sequence classification. Encodes eval data first; if the
    result has only one label class, returns correlation=0 and mean_by_class
    (no call to get_encoder_metrics_from_dataframe). Otherwise delegates to the
    default encoder metrics (passes encoder_df to avoid re-encoding).
    """

    def evaluate_encoder(
        self,
        trainer: Any,
        encoder_df: Optional[pd.DataFrame] = None,
        eval_data: Any = None,
        use_cache: Optional[bool] = None,
        split: Optional[str] = None,
        max_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Evaluate a classification encoder with a single-class fallback.

        Args:
            trainer: Classification trainer providing ``get_model`` and
                ``create_eval_data``.
            encoder_df: Optional precomputed encoder DataFrame. If it contains
                one or zero label classes, returns a zero-correlation summary.
            eval_data: Optional precomputed ``GradientTrainingDataset``.
            use_cache: Forwarded to the base encoder evaluator for multi-class
                data.
            split: Dataset split used when creating eval data.
            max_size: Optional cap used when creating eval data.
            **kwargs: Additional arguments forwarded to ``create_eval_data`` or
                the base encoder evaluator.

        Returns:
            Encoder metric dict. Single-class data returns ``correlation=0.0``
            with class means instead of calling Pearson correlation.
        """
        model = trainer.get_model()
        if model is None:
            return {"correlation": 0.0, "mean_by_class": {}, "n_samples": 0}

        if encoder_df is not None:
            if encoder_df.empty:
                return _single_class_result(encoder_df)
            if encoder_df["label"].nunique() <= 1:
                return _single_class_result(encoder_df)
            return super().evaluate_encoder(
                trainer,
                encoder_df=encoder_df,
                eval_data=None,
                use_cache=use_cache,
                split=split,
                max_size=max_size,
                **kwargs,
            )

        # Encode eval data and branch on actual label count
        create_kwargs = {k: v for k, v in kwargs.items() if k not in ("eval_batch_size", "use_cache", "encoder_df", "return_df", "plot", "plot_kwargs")}
        if split is not None:
            create_kwargs["split"] = split
        if max_size is not None:
            create_kwargs["max_size"] = max_size
        if eval_data is None:
            eval_data = trainer.create_eval_data(model, **create_kwargs)
        if not isinstance(eval_data, GradientTrainingDataset):
            raise TypeError("Encoder evaluation expected a GradientTrainingDataset.")
        rows = encode_dataset_to_rows(model, eval_data)
        if not rows:
            return _single_class_result(pd.DataFrame())
        df = _rows_to_df(rows)
        if df["label"].nunique() <= 1:
            return _single_class_result(df)
        return super().evaluate_encoder(
            trainer,
            encoder_df=df,
            eval_data=None,
            use_cache=use_cache,
            split=split,
            max_size=max_size,
            **kwargs,
        )


class ClassificationEvaluator(Evaluator):
    """
    Evaluator for sequence classification. Uses ClassificationEncoderEvaluator so
    encoder eval branches on actual eval data (single-class vs multi-class).
    """

    def __init__(
        self,
        trainer: Any,
        encoder_evaluator: Optional[EncoderEvaluator] = None,
        decoder_evaluator: Any = None,
        visualizer_class: Optional[type] = None,
    ):
        if encoder_evaluator is None:
            encoder_evaluator = ClassificationEncoderEvaluator()
        super().__init__(
            trainer,
            encoder_evaluator=encoder_evaluator,
            decoder_evaluator=decoder_evaluator,
            visualizer_class=visualizer_class,
        )
