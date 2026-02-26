"""
Encoder evaluation: encode gradients and compute unified encoder metrics.

EncoderEvaluator runs encoding on evaluation data and delegates to
get_encoder_metrics_from_dataframe for all metrics (correlation, accuracy,
mean_by_class, mean_by_type, etc.). Single source of truth for encoder metrics.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional, Callable, Union

from tqdm import tqdm

import pandas as pd

from gradiend.evaluator.encoder_metrics import get_encoder_metrics_from_dataframe
from gradiend.trainer.core.dataset import GradientTrainingDataset
from gradiend.util.paths import resolve_encoder_eval_result_path, resolve_encoder_eval_result_path_legacy
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


def encode_dataset_to_rows(
    model_with_gradiend: Any,
    dataset: Any,
    row_extractor: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Encode a GradientTrainingDataset and return per-row dicts for building DataFrames.

    Each row has: encoded, label, source_id, target_id, plus optional fields
    provided by row_extractor (modality-specific, e.g. text).
    Used by EncoderEvaluator and by callers (e.g. _analyze_encoder) when training_rows
    are not available from cache.
    """
    rows: List[Dict[str, Any]] = []
    try:
        total = len(dataset)
    except (TypeError, AttributeError):
        total = None
    _tqdm_kw = dict(
        desc="Encoding",
        total=total,
        leave=False,
        ncols=80,
        dynamic_ncols=False,
        ascii=True,
        mininterval=0.5,
        position=0,
        disable=not sys.stderr.isatty(),
    )
    for entry in tqdm(dataset, **_tqdm_kw):
        grad = entry["source"]
        label = entry["label"]
        encoded_val = model_with_gradiend.encode(grad, return_float=True)
        row: Dict[str, Any] = {
            "encoded": encoded_val,
            "label": float(label),
            "source_id": entry.get("factual_id"),
            "target_id": entry.get("alternative_id"),
        }
        if row_extractor is not None:
            try:
                extra = row_extractor(entry)
                if isinstance(extra, dict) and extra:
                    row.update(extra)
            except Exception as e:
                logger.warning("Row extractor failed: %s", e)
        rows.append(row)
    return rows





def _cache_key_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Kwargs that affect create_eval_data / result, for cache key."""
    skip = {"eval_batch_size", "use_cache", "encoder_df", "return_df", "plot", "plot_kwargs"}
    return {k: v for k, v in kwargs.items() if k not in skip and v is not None}


def _rows_to_encoder_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert encoded rows to DataFrame for get_encoder_metrics_from_dataframe."""
    if not rows:
        return pd.DataFrame(columns=["encoded", "label", "type", "source_id", "target_id"])
    data: Dict[str, List[Any]] = {
        "encoded": [],
        "label": [],
        "type": [],
        "source_id": [],
        "target_id": [],
    }
    for r in rows:
        data["encoded"].append(r.get("encoded"))
        data["label"].append(r.get("label", 0.0))
        data["type"].append(r.get("type", "training"))
        data["source_id"].append(r.get("source_id"))
        data["target_id"].append(r.get("target_id"))
    return pd.DataFrame(data)


class EncoderEvaluator:
    """
    Encoder evaluation: encode gradients on eval data and compute label correlation.
    Uses trainer for model and create_eval_data; subclasses can override to
    customize behavior (caching, metrics).
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
        """
        Evaluate encoder on eval data: encode gradients and compute unified encoder metrics.

        Uses get_encoder_metrics_from_dataframe as single source of truth for all metrics.
        When encoder_df is provided, skips encoding and computes metrics directly from it.
        When experiment_dir is set, encoder metrics are written to the same path as the encoder
        analysis CSV but with .json extension (e.g. encoded_values_max_size_500_split_test.json).
        When use_cache=True and experiment_dir is set, loads from that path when the file exists.

        Args:
            trainer: Trainer (or protocol) with get_model() and create_eval_data().
            encoder_df: Optional DataFrame with encoded values. If provided, skips encoding
                and computes metrics from this DataFrame. Use when you already have
                encoder outputs (e.g. from evaluate_encoder(return_df=True)).
            eval_data: Optional pre-computed GradientTrainingDataset. If None and encoder_df
                is None, created via trainer.create_eval_data.
            use_cache: If True, use cached encoder evaluation result when available
                (requires experiment_dir).
            split: Dataset split for create_eval_data. Default: "test".
            max_size: Maximum samples per variant for create_eval_data.
            **kwargs: Passed to trainer.create_eval_data when eval_data and encoder_df are None.

        Returns:
            Dict with keys from get_encoder_metrics_from_dataframe: n_samples, sample_counts,
            all_data, training_only, target_classes_only, boundaries, correlation,
            mean_by_class, mean_by_type; optionally neutral_mean_by_type, mean_by_feature_class,
            label_value_to_class_name.
        """
        use_cache = trainer._default_from_training_args(use_cache, "use_cache", fallback=False)
        skip = {"eval_batch_size", "use_cache", "encoder_df", "return_df", "plot", "plot_kwargs"}
        create_kwargs = dict(kwargs)
        if split is not None:
            create_kwargs["split"] = split
        if max_size is not None:
            create_kwargs["max_size"] = max_size
        create_kwargs = {k: v for k, v in create_kwargs.items() if k not in skip}
        experiment_dir = getattr(trainer, "experiment_dir", None)
        if callable(experiment_dir):
            experiment_dir = experiment_dir()
        # Same key as encoder CSV (split, max_size) so JSON path = CSV path with .json
        metrics_path: Optional[str] = None
        if experiment_dir and str(experiment_dir).strip():
            key_kwargs = {k: create_kwargs.get(k) for k in ("split", "max_size") if create_kwargs.get(k) is not None}
            metrics_path = resolve_encoder_eval_result_path(experiment_dir, None, **key_kwargs)
        if use_cache and not metrics_path:
            raise ValueError(
                "evaluate_encoder(use_cache=True) requires experiment_dir to be set on the trainer. "
                "Set experiment_dir on TrainingArguments or pass encoder_df to compute from data."
            )
        def _load_metrics(path: str) -> Optional[Dict[str, Any]]:
            try:
                with open(path, "r") as f:
                    raw = json.load(f)
                for k in ("mean_by_class", "label_value_to_class_name"):
                    if k in raw and isinstance(raw[k], dict) and raw[k]:
                        try:
                            raw[k] = {float(x): v for x, v in raw[k].items()}
                        except (ValueError, TypeError):
                            pass
                return raw
            except Exception as e:
                logger.warning("Failed to load encoder eval cache %s: %s", path, e)
                return None

        if use_cache and metrics_path and os.path.isfile(metrics_path):
            raw = _load_metrics(metrics_path)
            if raw is not None:
                logger.info("Loaded cached encoder evaluation from %s", metrics_path)
                return raw
        if use_cache and experiment_dir and str(experiment_dir).strip():
            legacy_path = resolve_encoder_eval_result_path_legacy(experiment_dir, **key_kwargs)
            if legacy_path and legacy_path != metrics_path and os.path.isfile(legacy_path):
                raw = _load_metrics(legacy_path)
                if raw is not None:
                    logger.info("Loaded cached encoder evaluation from legacy path %s", legacy_path)
                    return raw
        if use_cache and metrics_path:
            logger.info("No encoder eval cache found at %s; computing fresh results.", metrics_path)

        model_with_gradiend = trainer.get_model()

        if encoder_df is not None:
            if encoder_df.empty:
                return {"n_samples": 0, "correlation": None}
            result = get_encoder_metrics_from_dataframe(encoder_df)
        else:
            if eval_data is None:
                eval_data = trainer.create_eval_data(
                    model_with_gradiend,
                    **create_kwargs,
                )
            if not isinstance(eval_data, GradientTrainingDataset):
                raise TypeError("EncoderEvaluator.evaluate_encoder expected a GradientTrainingDataset.")
            max_size = create_kwargs.get("max_size")
            try:
                n_eval = len(eval_data)
                if max_size is None and n_eval > 5000:
                    logger.warning(
                        "encoder eval: max_size is not set and eval data has %d samples across all classes. "
                        "Note: encoder_eval_max_size is applied per feature class; total samples may exceed this. "
                        "Computation may be slow. Consider setting encoder_eval_max_size, encoder_eval_train_max_size, "
                        "or max_size to cap.",
                        n_eval,
                    )
            except TypeError:
                pass
            training_rows = encode_dataset_to_rows(model_with_gradiend, eval_data)
            if not training_rows:
                return {"n_samples": 0, "correlation": None}
            df = _rows_to_encoder_df(training_rows)
            result = get_encoder_metrics_from_dataframe(df)

        if metrics_path:
            try:
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                with open(metrics_path, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info("Saved encoder metrics to %s", metrics_path)
            except Exception as e:
                logger.warning("Failed to save encoder metrics to %s: %s", metrics_path, e)

        return result
