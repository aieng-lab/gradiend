"""
Encoder evaluation: encode gradients and compute unified encoder metrics.

EncoderEvaluator runs encoding on evaluation data and delegates to
get_encoder_metrics_from_dataframe for all metrics (correlation, accuracy,
mean_by_class, mean_by_type, etc.). Single source of truth for encoder metrics.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from gradiend.evaluator.encoder_metrics import get_encoder_metrics_from_dataframe
from gradiend.trainer.core.dataset import GradientTrainingDataset
from gradiend.util.encoding_rows import encode_dataset_to_rows
from gradiend.util.split_policy import SplitPolicy
from gradiend.util.paths import resolve_encoder_eval_result_path, resolve_encoder_eval_result_path_legacy
from gradiend.util.logging import get_logger
from gradiend.util.util import to_jsonable

logger = get_logger(__name__)


def _encoder_metrics_kwargs_from_trainer(trainer: Any, encoder_df: pd.DataFrame) -> Dict[str, Any]:
    """Build optional kwargs for get_encoder_metrics_from_dataframe from trainer context."""
    target_classes = getattr(trainer, "target_classes", None) or getattr(trainer, "pair", None)
    kwargs: Dict[str, Any] = {}
    if target_classes:
        kwargs["target_classes"] = list(target_classes)
    if "data_split" in encoder_df.columns and encoder_df["data_split"].nunique(dropna=True) > 1:
        splits = encoder_df["data_split"].dropna().astype(str).tolist()
        policy = SplitPolicy.from_available(splits)
        gen_pair = policy.generalization_pair()
        if gen_pair is not None:
            kwargs["generalization_splits"] = gen_pair
    return kwargs


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
        "factual_id": [],
        "counterfactual_id": [],
        "transition_id": [],
        "input_type": [],
        "feature_class_id": [],
        "eval_group": [],
        "source_token": [],
        "factual_token": [],
        "alternative_token": [],
        "text": [],
        "template": [],
        "input_text": [],
        "masked": [],
        "display_text": [],
        "data_split": [],
        "neutral_variant": [],
    }
    for r in rows:
        data["encoded"].append(r.get("encoded"))
        data["label"].append(r.get("label", 0.0))
        data["type"].append(r.get("type", "training"))
        data["source_id"].append(r.get("source_id"))
        data["target_id"].append(r.get("target_id"))
        data["factual_id"].append(r.get("factual_id"))
        data["counterfactual_id"].append(r.get("counterfactual_id"))
        data["transition_id"].append(r.get("transition_id"))
        data["input_type"].append(r.get("input_type"))
        data["feature_class_id"].append(r.get("feature_class_id"))
        data["eval_group"].append(r.get("eval_group"))
        data["source_token"].append(r.get("source_token"))
        data["factual_token"].append(r.get("factual_token"))
        data["alternative_token"].append(r.get("alternative_token"))
        data["text"].append(r.get("text"))
        data["template"].append(r.get("template"))
        data["input_text"].append(r.get("input_text"))
        data["masked"].append(r.get("masked"))
        data["display_text"].append(r.get("display_text"))
        data["data_split"].append(r.get("data_split"))
        data["neutral_variant"].append(r.get("neutral_variant"))
    df = pd.DataFrame(data)
    for col in (
        "source_token",
        "factual_token",
        "alternative_token",
        "text",
        "template",
        "input_text",
        "masked",
        "display_text",
        "data_split",
        "neutral_variant",
    ):
        if df[col].isna().all():
            df = df.drop(columns=[col])
    return df


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
        model_with_gradiend: Any = None,
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
            model_with_gradiend: Optional model instance used for encoding. If
                omitted, ``trainer.get_model()`` is used. This is useful when the
                caller already has a loaded model and wants to avoid another
                trainer-level resolution step.
            **kwargs: Passed to trainer.create_eval_data when eval_data and encoder_df are None.

        Returns:
            Dict with keys from get_encoder_metrics_from_dataframe: n_samples, sample_counts,
            all_data, training_only, target_classes_only, boundaries, correlation,
            mean_by_class, mean_by_type; optionally neutral_mean_by_type, mean_by_feature_class,
            mean_by_eval_group, eval_group_basis, label_value_to_class_name.
        """
        use_cache = trainer._resolve_artifact_use_cache(use_cache, fallback=False)
        skip = {"eval_batch_size", "use_cache", "encoder_df", "return_df", "plot", "plot_kwargs", "model_with_gradiend"}
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
        key_kwargs = {k: create_kwargs.get(k) for k in ("split", "max_size") if create_kwargs.get(k) is not None}
        metrics_path: Optional[str] = None
        if experiment_dir and str(experiment_dir).strip():
            metrics_path = resolve_encoder_eval_result_path(experiment_dir, None, **key_kwargs)
        if use_cache and not metrics_path:
            raise ValueError(
                "evaluate_encoder(use_cache=True) requires trainer.experiment_dir so the "
                "encoder evaluation cache path can be resolved."
            )

        cache_dirs: List[str] = []
        if use_cache and hasattr(trainer, "iter_encoder_eval_cache_dirs"):
            cache_dirs = list(trainer.iter_encoder_eval_cache_dirs())
        elif use_cache and experiment_dir and str(experiment_dir).strip():
            cache_dirs = [os.path.normpath(str(experiment_dir))]

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

        if use_cache and encoder_df is None and cache_dirs:
            for cache_dir in cache_dirs:
                candidate = resolve_encoder_eval_result_path(cache_dir, None, **key_kwargs)
                if candidate and os.path.isfile(candidate):
                    raw = _load_metrics(candidate)
                    if raw is not None:
                        logger.info("Loaded cached encoder evaluation from %s", candidate)
                        return raw
                legacy_path = resolve_encoder_eval_result_path_legacy(cache_dir, **key_kwargs)
                if legacy_path and legacy_path != candidate and os.path.isfile(legacy_path):
                    raw = _load_metrics(legacy_path)
                    if raw is not None:
                        logger.info("Loaded cached encoder evaluation from legacy path %s", legacy_path)
                        return raw

        if use_cache and encoder_df is None and metrics_path and os.path.isfile(metrics_path):
            raw = _load_metrics(metrics_path)
            if raw is not None:
                logger.info("Loaded cached encoder evaluation from %s", metrics_path)
                return raw
        if use_cache and encoder_df is None and experiment_dir and str(experiment_dir).strip():
            legacy_path = resolve_encoder_eval_result_path_legacy(experiment_dir, **key_kwargs)
            if legacy_path and legacy_path != metrics_path and os.path.isfile(legacy_path):
                raw = _load_metrics(legacy_path)
                if raw is not None:
                    logger.info("Loaded cached encoder evaluation from legacy path %s", legacy_path)
                    return raw
        if use_cache and metrics_path:
            logger.info("No encoder eval cache found at %s; computing fresh results.", metrics_path)

        model_with_gradiend = model_with_gradiend if model_with_gradiend is not None else trainer.get_model()

        if encoder_df is not None:
            if encoder_df.empty:
                return {"n_samples": 0, "correlation": None}
            metrics_kw = _encoder_metrics_kwargs_from_trainer(trainer, encoder_df)
            result = get_encoder_metrics_from_dataframe(encoder_df, **metrics_kw)
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
            metrics_kw = _encoder_metrics_kwargs_from_trainer(trainer, df)
            result = get_encoder_metrics_from_dataframe(df, **metrics_kw)
            result["training_rows"] = training_rows

        if metrics_path:
            try:
                os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
                with open(metrics_path, "w") as f:
                    json.dump(to_jsonable(result), f, indent=2)
                logger.debug("Saved encoder metrics to %s", metrics_path)
            except Exception as e:
                logger.warning("Failed to save encoder metrics to %s: %s", metrics_path, e)

        return result
