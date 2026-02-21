"""
Decoder evaluation: grid search over feature_factor and lr, best training_args selection.

DecoderEvaluator runs the grid + cache + summary algorithm; it takes a trainer
(protocol: id, _get_decoder_eval_dataframe, _evaluate_model_for_decoder, _model_for_decoder_eval)
and uses it for data and model access.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import torch
from tqdm import tqdm

from gradiend.evaluator.decoder_eval_utils import convert_results_to_dict, convert_results_to_list
from gradiend.model import ModelWithGradiend
from gradiend.util.paths import resolve_decoder_grid_cache_path
from gradiend.util.logging import get_logger

logger = get_logger(__name__)


# ============================
# Feature-factor defaults
# ============================


def derive_default_feature_factor(
    trainer: Any,
    model_with_gradiend: Any = None,
    class_name: str = None,
) -> float:
    """
    Derive a single default feature factor for decoder eval.

    Rule: feature_factor = - feature_class_encoding_direction[class_name]
    when available.

    class_name is required.
    """
    model = model_with_gradiend
    if model is None and hasattr(trainer, "get_model"):
        model = trainer.get_model()
    elif isinstance(model, str):
        trust_remote_code = getattr(getattr(trainer, "_training_args", None), "trust_remote_code", False)
        model = ModelWithGradiend.from_pretrained(model, trust_remote_code=trust_remote_code)

    if class_name is None:
        raise ValueError("class_name is required to derive a default feature factor.")

    direction = getattr(model, "feature_class_encoding_direction", None) if model is not None else None
    if isinstance(direction, dict) and class_name in direction:
        return float(-direction[class_name])

    # Fallback: derive from trainer.pair when model was created before data load (e.g. in-memory after train)
    pair = getattr(trainer, "pair", None)
    if pair and len(pair) >= 2:
        class_labels = {pair[0]: 1.0, pair[1]: -1.0}
        classes = getattr(trainer, "target_classes", None) or getattr(trainer, "all_classes", None) or []
        for c in classes:
            if c not in class_labels:
                class_labels[c] = 0.0
        if class_name in class_labels:
            return float(-class_labels[class_name])

    raise ValueError(
        "Cannot derive default feature factor for class '%s': model does not have feature_class_encoding_direction (%s) or class not found in it." % (class_name, direction)
    )


def derive_feature_factor_for_class(
    trainer: Any,
    model_with_gradiend: Any = None,
    class_name: Optional[str] = None,
) -> float:
    """Derive feature factor for a specific class (push toward class_name)."""
    if class_name is None:
        raise ValueError("class_name is required to derive a feature factor.")
    return derive_default_feature_factor(trainer, model_with_gradiend, class_name=class_name)


def default_decoder_feature_factors(
    trainer: Any,
    model_with_gradiend: Any = None,
) -> List[float]:
    """Return default feature factors for all trainer target classes (feature factor such that gradient update with
    positive learning rate pushes toward counterfactual class)."""
    classes = trainer.target_classes

    if not classes:
        raise ValueError("classes must be provided to derive default feature factors.")

    return [derive_feature_factor_for_class(trainer, model_with_gradiend, cls) for cls in classes]


# ============================
# Metric selection
# ============================
CandidateId = Any  # e.g. (feature_factor, lr)


@dataclass(frozen=True)
class Candidate:
    id: CandidateId
    lms: float
    metrics: Mapping[str, float]  # scalar metrics used for selection


@dataclass(frozen=True)
class BaseContext:
    base_lms: Optional[float]


class SelectionPolicy(Protocol):
    def select(self, metric: str, candidates: Sequence[Candidate], ctx: BaseContext) -> Optional[Candidate]:
        ...


@dataclass(frozen=True)
class LMSThresholdPolicy:
    """
    Restrict to candidates with lms >= ratio * base_lms, then pick argmax(metric).

    Fallback when none pass threshold:
      pick smallest |lr| among candidates with lr != 0
      (keeps your previous behavior, but expressed as a policy).
    """
    ratio: float = 0.99
    lr_from_id: Callable[[CandidateId], float] = lambda cid: cid[1]
    require_base_lms: bool = True

    def select(self, metric: str, candidates: Sequence[Candidate], ctx: BaseContext) -> Optional[Candidate]:
        base_lms = ctx.base_lms
        if base_lms is None:
            if self.require_base_lms:
                raise ValueError("Base model lms missing; cannot apply LMSThresholdPolicy.")
            return max(candidates, key=lambda c: c.metrics.get(metric, float("-inf")), default=None)

        cutoff = self.ratio * base_lms
        passing = [c for c in candidates if c.lms >= cutoff]
        if passing:
            return max(passing, key=lambda c: c.metrics.get(metric, float("-inf")), default=None)

        non_zero = [c for c in candidates if self.lr_from_id(c.id) != 0]
        if not non_zero:
            return None
        return min(non_zero, key=lambda c: abs(self.lr_from_id(c.id)), default=None)


@dataclass(frozen=True)
class LMSTimesMetricPolicy:
    """Pick argmax(metric * lms)."""
    def select(self, metric: str, candidates: Sequence[Candidate], ctx: BaseContext) -> Optional[Candidate]:
        return max(
            candidates,
            key=lambda c: c.metrics.get(metric, float("-inf")) * c.lms,
            default=None,
        )


def default_extract_candidates(results: Mapping[Any, Mapping[str, Any]]) -> Tuple[List[Candidate], BaseContext]:
    """
    Convert current `results` format to a list of Candidates.

    Metrics convention:
      - group probs -> keys "<class_name>" (stringified, no sanitizing)
      - any scalar numeric field at top-level of entry (excluding lms/probs) -> metric with same key
    """
    base_lms = float(results["base"]["lms"]["lms"]) if "base" in results else None

    candidates: List[Candidate] = []
    for cid, entry in results.items():
        if cid == "base":
            continue

        lms = float(entry["lms"]["lms"])
        metrics: Dict[str, float] = {}

        probs = entry.get("probs") or {}
        for cls, p in probs.items():
            metrics[str(cls)] = float(p)

        for k, v in entry.items():
            if k in ("probs", "lms"):
                continue
            if isinstance(v, (int, float)):
                metrics[k] = float(v)

            candidates.append(Candidate(id=entry["id"], lms=lms, metrics=metrics))

    return candidates, BaseContext(base_lms=base_lms)


def compute_metric_summaries(
    results: Mapping[Any, Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    selector: SelectionPolicy,
    extractor: Callable[[Mapping[Any, Mapping[str, Any]]], Tuple[List[Candidate], BaseContext]] = default_extract_candidates,
    feature_factor_from_id: Callable[[CandidateId], float] = lambda cid: cid[0],
    lr_from_id: Callable[[CandidateId], float] = lambda cid: cid[1],
    empty_default_id: str = "base",
) -> Dict[str, Dict[str, Any]]:
    """
    Summarize multiple metrics in one pass.

    Returns:
      {metric: {"value", "feature_factor", "learning_rate", "id"}}
    """
    candidates, ctx = extractor(results)

    available_metrics: set = set()
    for c in candidates:
        available_metrics.update(c.metrics.keys())
    normalized_metrics: List[str] = list(metrics)

    missing = [m for m in normalized_metrics if m not in available_metrics]
    if missing:
        raise ValueError(
            "Requested metrics not present in decoder results: %s. Available metrics: %s"
            % (missing, sorted(available_metrics))
        )

    if not candidates:
        return {m: {"value": 0.0, "feature_factor": 0.0, "learning_rate": 0.0, "id": empty_default_id} for m in normalized_metrics}

    def _fallback_when_none(cands: List[Candidate]) -> Optional[Candidate]:
        """Pick candidate with lr != 0 and smallest absolute value when selector returns None."""
        non_zero = [c for c in cands if lr_from_id(c.id) != 0]
        if not non_zero:
            return None
        return min(non_zero, key=lambda c: abs(lr_from_id(c.id)))

    summary: Dict[str, Dict[str, Any]] = {}
    for metric in normalized_metrics:
        chosen = selector.select(metric, candidates, ctx)
        if chosen is None:
            chosen = _fallback_when_none(candidates)
        if chosen is None:
            summary[metric] = {"value": 0.0, "feature_factor": 0.0, "learning_rate": 0.0, "id": empty_default_id}
            continue

        summary[metric] = {
            "value": float(chosen.metrics.get(metric, 0.0)),
            "feature_factor": float(feature_factor_from_id(chosen.id)),
            "learning_rate": float(lr_from_id(chosen.id)),
            "id": chosen.id,
        }

    return summary


# ============================
# DecoderEvaluator
# ============================


class DecoderEvaluator:
    """
    Decoder evaluation: grid search over (feature_factor, lr), cache, and compute best selection summary.
    Uses trainer for eval dataframe and model evaluation.
    """

    def evaluate_decoder(
        self,
        trainer: Any,
        model_with_gradiend: Any = None,
        feature_factors: Optional[List[float]] = None,
        lrs: Optional[List[float]] = None,
        part: str = "decoder-weight",
        output_path: Optional[str] = None,
        selector: Optional[SelectionPolicy] = None,
        summary_extractor: Callable[[Mapping[Any, Mapping[str, Any]]], Tuple[List[Candidate], BaseContext]] = default_extract_candidates,
        summary_feature_factor_from_id: Callable[[CandidateId], float] = lambda cid: cid['feature_factor'] if isinstance(cid, dict) else cid[0],
        summary_lr_from_id: Callable[[CandidateId], float] = lambda cid: cid['learning_rate'] if isinstance(cid, dict) else cid[1],
        summary_empty_default_id: str = "base",
        use_cache: Optional[bool] = None,
        max_size_training_like: Optional[int] = None,
        max_size_neutral: Optional[int] = None,
        eval_batch_size: Optional[int] = None,
        training_like_df: Optional[Any] = None,
        neutral_df: Optional[Any] = None,
        summary_metrics: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run decoder grid evaluation and return summary + grid.

        Args:
            trainer: Trainer (protocol) with get_model, _model_for_decoder_eval, _get_decoder_eval_dataframe,
                     and evaluate_base_model.
            model_with_gradiend: ModelWithGradiend instance or path. If None, uses trainer.get_model().
            feature_factors: List of feature factors to test.
            lrs: List of learning rates to test.
            part: which part of GRADIEND is used to derive GRADIEND-modified models (options:  'encoder-weight' |
                'decoder-weight' | 'decoder-bias' | 'decoder-sum' | 'decoder'). All options besides `decoder` are
                independent of the feature factor (e.g., using the encoder weights as update direction), while `decoder`
                computes the update direction via dec(feature_factor) (and is the default).
            output_path: Optional explicit cache path. Overrides experiment_dir-based cache path.
            selector: SelectionPolicy, e.g. LMSThresholdPolicy(ratio=0.99) or LMSTimesMetricPolicy().
            summary_extractor: Candidate extractor for summary computation. Use a custom extractor to add
                derived metrics (e.g. bpi, fpi, mpi) to candidates; then pass summary_metrics so they are summarized.
            summary_feature_factor_from_id: Function to extract feature_factor from candidate id.
            summary_lr_from_id: Function to extract lr from candidate id.
            summary_empty_default_id: Fallback id used when no candidate is selected (for comparison with base).
                When the selector returns None, we first try the candidate with learning_rate != 0 and smallest
                absolute value; only if none exists do we use this default (representing the base model).
            use_cache: If True, use cached results when available; if False, recompute.
            max_size_training_like: Maximum size for generated training-like eval data.
            max_size_neutral: Maximum size for generated neutral eval data (and LMS text cap).
            eval_batch_size: Common eval batch size used for LMS.
            training_like_df: Optional explicit training-like DataFrame for probability scoring.
            neutral_df: Optional explicit neutral DataFrame for LMS scoring.
            summary_metrics: Optional list of metric names to summarize. If None, uses the trainer's
                target feature classes. When using a custom summary_extractor that adds metrics (e.g. bpi, fpi, mpi),
                pass them here so they appear in the summary.

        Returns:
            Dict with 'summary' and 'grid'. 'summary' is a dict keyed by metric name with values containing selected
            metric `value`, `feature_factor`, `learning_rate`, and `id`. 'grid' is a dict keyed by candidate id (e.g.
            (feature_factor, lr)) with values containing the full evaluation results for that candidate.
        """
        logger.info(f"Starting decoder evaluation with part={part}, model={model_with_gradiend}")
        use_cache = trainer._default_from_training_args(use_cache, "use_cache", fallback=False)

        if selector is None:
            selector = LMSThresholdPolicy(ratio=0.99)

        raw_model = model_with_gradiend or trainer.get_model()
        if isinstance(raw_model, str):
            trust_remote_code = getattr(getattr(trainer, "_training_args", None), "trust_remote_code", False)
            raw_model = ModelWithGradiend.from_pretrained(raw_model, trust_remote_code=trust_remote_code)

        if feature_factors is None:
            feature_factors = default_decoder_feature_factors(trainer, raw_model)

        metrics_for_summary = summary_metrics if summary_metrics is not None else trainer.get_target_feature_classes()

        model_with_gradiend = trainer._model_for_decoder_eval(raw_model)
        path = model_with_gradiend.name_or_path
        base_model = model_with_gradiend.base_model
        tokenizer = model_with_gradiend.tokenizer
        run_id = getattr(trainer, "run_id", None)
        model_id = os.path.basename(path) if path and str(path).startswith("results/models") else path

        if lrs is None:
            lrs = [1e-2, 1e-3, 1e-4, 1e-5]

        experiment_dir = trainer.experiment_dir
        cache_file = resolve_decoder_grid_cache_path(experiment_dir, explicit_path=output_path)
        if use_cache and not cache_file:
            raise ValueError(
                "evaluate_decoder(use_cache=True) requires experiment_dir on the trainer or output_path. "
                "Set experiment_dir on TrainingArguments or pass output_path= to specify the cache location."
            )
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        pairs = [(ff, lr) for ff in feature_factors for lr in lrs]
        random.shuffle(pairs)
        expected_results = len(pairs) + 1

        relevant_results: Dict[Any, Dict[str, Any]] = {}
        all_results: Dict[Any, Dict[str, Any]] = {}

        if use_cache and cache_file and os.path.isfile(cache_file):
            try:
                with open(cache_file, "r") as f:
                    payload = json.load(f)
                cached_part = payload.get("part")
                cached_feature_factors = payload.get("feature_factors")
                cached_lrs = payload.get("lrs")
                cache_matches = (
                    cached_part == part
                    and cached_feature_factors == list(feature_factors)
                    and cached_lrs == list(lrs)
                )
                if cache_matches:
                    relevant_results = convert_results_to_dict(payload.get("results", []))
                    if len(relevant_results) >= expected_results:
                        logger.info("Using cached decoder grid from %s", cache_file)
                        summary = self.compute_metric_summaries(
                            trainer,
                            relevant_results,
                            selector=selector,
                            metrics=metrics_for_summary,
                            extractor=summary_extractor,
                            feature_factor_from_id=summary_feature_factor_from_id,
                            lr_from_id=summary_lr_from_id,
                            empty_default_id=summary_empty_default_id,
                        )
                        return {"summary": summary, "grid": relevant_results}
                else:
                    logger.info("Decoder cache mismatch (part/feature_factors/lrs); recomputing.")
            except Exception as e:
                logger.warning("Error loading cached decoder results: %s", e)

        trainer_config = getattr(trainer, "config", None)
        training_args = getattr(trainer, "_training_args", None)
        if max_size_training_like is None:
            if trainer_config is not None and hasattr(trainer_config, "decoder_eval_lms_max_samples"):
                max_size_training_like = trainer_config.decoder_eval_lms_max_samples
            elif training_args is not None:
                max_size_training_like = getattr(training_args, "decoder_eval_max_size_training_like", None)
        if max_size_neutral is None:
            if trainer_config is not None and hasattr(trainer_config, "decoder_eval_lms_max_samples"):
                max_size_neutral = trainer_config.decoder_eval_lms_max_samples
            elif training_args is not None:
                max_size_neutral = getattr(training_args, "decoder_eval_max_size_neutral", None)

        if training_like_df is None or neutral_df is None:
            training_like_df, neutral_df = trainer._get_decoder_eval_dataframe(
                tokenizer,
                max_size_training_like=max_size_training_like,
                max_size_neutral=max_size_neutral,
                cached_training_like_df=training_like_df,
                cached_neutral_df=neutral_df,
            )

        _LARGE_DATASET = 10000
        if max_size_training_like is None and len(training_like_df) > _LARGE_DATASET:
            logger.warning(
                "decoder eval: max_size_training_like is not set and training data has %d rows. "
                "Computation may be slow. Set decoder_eval_max_size_training_like or max_size_training_like to cap.",
                len(training_like_df),
            )
        if max_size_neutral is None and len(neutral_df) > _LARGE_DATASET:
            logger.warning(
                "decoder eval: max_size_neutral is not set and neutral data has %d rows. "
                "LMS computation may be slow. Set decoder_eval_max_size_neutral or max_size_neutral to cap.",
                len(neutral_df),
            )

        if "base" not in relevant_results or not use_cache:
            logger.debug("Evaluating base model...")
            base_results = trainer.evaluate_base_model(
                base_model,
                tokenizer,
                use_cache=use_cache,
                training_like_df=training_like_df,
                neutral_df=neutral_df,
                max_size_training_like=max_size_training_like,
                max_size_neutral=max_size_neutral,
                eval_batch_size=eval_batch_size,
            )
            if isinstance(base_results, dict):
                base_results["id"] = "base"
            all_results["base"] = base_results
            relevant_results["base"] = base_results

        for feature_factor, lr in tqdm(pairs, desc=f"Evaluate GRADIEND {run_id or ''}", total=len(pairs)):
            id_key = (feature_factor, lr)
            if id_key in relevant_results and use_cache:
                continue

            modified_model = model_with_gradiend.rewrite_base_model(
                learning_rate=lr,
                feature_factor=feature_factor,
                part=part,
            )
            modified_results = trainer.evaluate_base_model(
                modified_model,
                tokenizer,
                use_cache=use_cache,
                cache_folder=f"{feature_factor}_{lr}",
                model_id=model_id,
                training_like_df=training_like_df,
                neutral_df=neutral_df,
                max_size_training_like=max_size_training_like,
                max_size_neutral=max_size_neutral,
                eval_batch_size=eval_batch_size,
            )
            if isinstance(modified_results, dict):
                modified_results["id"] = {"feature_factor": feature_factor, "learning_rate": lr}
            all_results[id_key] = modified_results
            relevant_results[id_key] = modified_results

            del modified_model
            torch.cuda.empty_cache()

        summary = self.compute_metric_summaries(
            trainer,
            relevant_results,
            selector=selector,
            metrics=metrics_for_summary,
            extractor=summary_extractor,
            feature_factor_from_id=summary_feature_factor_from_id,
            lr_from_id=summary_lr_from_id,
            empty_default_id=summary_empty_default_id,
        )
        if cache_file:
            try:
                payload = {
                    "part": part,
                    "feature_factors": list(feature_factors),
                    "lrs": list(lrs),
                    "results": convert_results_to_list(relevant_results),
                }
                with open(cache_file, "w") as f:
                    json.dump(payload, f, indent=2)
            except Exception as e:
                logger.warning("Error writing decoder cache %s: %s", cache_file, e)

        return {"summary": summary, "grid": relevant_results}

    def compute_metric_summaries(
            self,
            trainer: Any,
            results: Mapping[Any, Mapping[str, Any]],
            *,
            selector: SelectionPolicy,
            metrics: Optional[Sequence[str]] = None,
            extractor: Callable[
                [Mapping[Any, Mapping[str, Any]]], Tuple[List[Candidate], BaseContext]] = default_extract_candidates,
            feature_factor_from_id: Callable[[CandidateId], float] = lambda cid: cid[0],
            lr_from_id: Callable[[CandidateId], float] = lambda cid: cid[1],
            empty_default_id: str = "base",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a per-metric (i.e., target classes) summary for a decoder grid evaluation.

        This is a thin wrapper around the module-level `compute_metric_summaries` that
        resolves default metrics from the trainer and passes through selector and
        extraction behavior.

        Args:
            trainer: Trainer-like object used to resolve default metrics when
                `metrics` is None (via `get_target_feature_classes`).
            results: Mapping from candidate id to evaluation result entries. The
                expected structure is the same as produced by decoder evaluation.
            selector: Policy used to choose the best candidate per metric (e.g., LMS thresholding (LMSThresholdPolicy),
                or metric*lms (LMSTimesMetricPolicy).
            metrics: Optional list of metric names to summarize. If None, uses the
                trainer's target feature classes.
            extractor: Function that converts the raw `results` mapping into
                candidates and a base context.
            feature_factor_from_id: Function to extract feature_factor from a
                candidate id.
            lr_from_id: Function to extract learning_rate from a candidate id.
            empty_default_id: Fallback id used when no candidate is selected (for comparison with base).
                When the selector returns None, we first try the candidate with learning_rate != 0 and smallest
                absolute value; only if none exists do we use this default (representing the base model).

        Returns:
            A dict keyed by metric name with values containing selected metric
            `value`, `feature_factor`, `learning_rate`, and `id`.
        """
        metrics = list(metrics) if metrics is not None else trainer.get_target_feature_classes()
        return compute_metric_summaries(
            results,
            metrics=metrics,
            selector=selector,
            extractor=extractor,
            feature_factor_from_id=feature_factor_from_id,
            lr_from_id=lr_from_id,
            empty_default_id=empty_default_id,
        )
