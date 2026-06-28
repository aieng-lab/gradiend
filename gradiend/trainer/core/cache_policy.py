"""
Training cache policy helpers for ``TrainingArguments.use_cache``.

Besides plain bool, string modes control extra guards:

- ``"always"``: reuse any saved checkpoint (skip fingerprint matching).
- ``"only_convergent"``: reuse only when convergence requirements are met and the

  checkpoint matches the requested training configuration (pre/post-prune fingerprint).

``use_cache=True`` reuses when the checkpoint exists and matches the fingerprint
(without requiring convergence).
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Optional, Union

from gradiend.util.logging import get_logger
from gradiend.util.paths import has_saved_model

logger = get_logger(__name__)

UseCacheSetting = Union[bool, Literal["always", "only_convergent"]]
USE_CACHE_ALWAYS: Literal["always"] = "always"
USE_CACHE_ONLY_CONVERGENT: Literal["only_convergent"] = "only_convergent"

# Heuristic for legacy checkpoints: pruned BERT-scale GRADIENDs stay well below this.
STALE_PRUNED_INPUT_DIM_THRESHOLD = 25_000_000


def normalize_use_cache(value: Any) -> UseCacheSetting:
    if isinstance(value, bool):
        return value
    if value == USE_CACHE_ALWAYS:
        return USE_CACHE_ALWAYS
    if value == USE_CACHE_ONLY_CONVERGENT:
        return USE_CACHE_ONLY_CONVERGENT
    raise ValueError(
        f"use_cache must be bool, {USE_CACHE_ALWAYS!r}, or {USE_CACHE_ONLY_CONVERGENT!r}, "
        f"got {value!r}"
    )


def is_unconditional_training_cache(value: Any) -> bool:
    """Return True only for ``"always"`` (skip fingerprint and convergence checks).

    Non-empty strings are truthy in Python, so never write ``if use_cache:`` when
    ``use_cache`` may be a training policy string.
    """
    return value == USE_CACHE_ALWAYS


def coerce_artifact_use_cache(value: Any) -> bool:
    """Map ``TrainingArguments.use_cache`` to a bool for eval/annotation/CSV caches.

    ``"only_convergent"`` means that a cached convergent run may be reused,
    including its derived artifacts. The convergence check happens at the
    training-checkpoint boundary; once that run is accepted, related artifact
    caches are reusable too.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if value in (USE_CACHE_ALWAYS, USE_CACHE_ONLY_CONVERGENT):
        return True
    normalize_use_cache(value)
    return False


def load_convergence_info(model_dir: str) -> Optional[dict]:
    """Load convergence_info from training.json under a model directory."""
    if not model_dir:
        return None
    training_path = os.path.join(model_dir, "training.json")
    if not os.path.isfile(training_path):
        return None
    try:
        with open(training_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read convergence info from %s: %s", training_path, exc)
        return None
    info = payload.get("convergence_info")
    return info if isinstance(info, dict) else None


def load_training_json(model_dir: str) -> Optional[dict]:
    if not model_dir:
        return None
    training_path = os.path.join(model_dir, "training.json")
    if not os.path.isfile(training_path):
        return None
    try:
        with open(training_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read training.json from %s: %s", training_path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def load_seed_report(run_dir: str) -> Optional[dict]:
    """Load multi-seed seed_report.json from the parent of a model directory."""
    if not run_dir:
        return None
    report_path = os.path.join(run_dir, "seeds", "seed_report.json")
    if not os.path.isfile(report_path):
        parent = os.path.dirname(run_dir.rstrip("/\\"))
        report_path = os.path.join(parent, "seeds", "seed_report.json")
    if not os.path.isfile(report_path):
        return None
    try:
        with open(report_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read seed report from %s: %s", report_path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def load_saved_gradiend_input_dim(model_dir: str) -> Optional[int]:
    """Read GRADIEND input_dim from a saved model config.json."""
    if not model_dir:
        return None
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as handle:
            cfg = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read config.json from %s: %s", cfg_path, exc)
        return None
    arch = cfg.get("architecture") if isinstance(cfg, dict) else None
    if not isinstance(arch, dict):
        return None
    value = arch.get("input_dim")
    return int(value) if isinstance(value, int) else None


def _normalize_pre_prune_config(cfg: Any) -> Optional[dict]:
    if cfg is None:
        return None
    if hasattr(cfg, "__dataclass_fields__"):
        from gradiend.trainer.core.pruning import _pre_prune_config_dict

        return _pre_prune_config_dict(cfg)
    if isinstance(cfg, dict):
        out = dict(cfg)
        out["dataset"] = None
        if "topk" in out and out["topk"] is not None:
            out["topk"] = float(out["topk"])
        return out
    return None


def _normalize_post_prune_config(cfg: Any) -> Optional[dict]:
    if cfg is None:
        return None
    if hasattr(cfg, "__dataclass_fields__"):
        import dataclasses

        out = dataclasses.asdict(cfg)
        out["mask"] = None
        return out
    if isinstance(cfg, dict):
        out = dict(cfg)
        out["mask"] = None
        return out
    return None


def build_training_cache_fingerprint(training_args: Any) -> dict:
    """Build a stable fingerprint for training-cache reuse checks."""
    if hasattr(training_args, "to_dict"):
        args_dict = training_args.to_dict()
    elif isinstance(training_args, dict):
        args_dict = dict(training_args)
    else:
        args_dict = {}

    fingerprint: dict[str, Any] = {}
    pre_cfg = _normalize_pre_prune_config(args_dict.get("pre_prune_config"))
    if pre_cfg is not None:
        fingerprint["pre_prune_config"] = pre_cfg
    post_cfg = _normalize_post_prune_config(args_dict.get("post_prune_config"))
    if post_cfg is not None:
        fingerprint["post_prune_config"] = post_cfg
    if args_dict.get("reuse_pre_prune") is not None:
        fingerprint["reuse_pre_prune"] = bool(args_dict.get("reuse_pre_prune"))
    for key in ("source", "target"):
        if args_dict.get(key) is not None:
            fingerprint[key] = args_dict[key]
    return fingerprint


def load_training_cache_fingerprint(model_dir: str) -> Optional[dict]:
    payload = load_training_json(model_dir)
    if not payload:
        return None
    fp = payload.get("cache_fingerprint")
    if isinstance(fp, dict):
        return fp
    training_args = payload.get("training_args")
    if isinstance(training_args, dict):
        return build_training_cache_fingerprint(training_args)
    return None


def _pre_prune_topk_from_fingerprint(fingerprint: dict) -> Optional[float]:
    pre_cfg = fingerprint.get("pre_prune_config")
    if not isinstance(pre_cfg, dict):
        return None
    topk = pre_cfg.get("topk")
    if topk is None:
        return None
    return float(topk)


def _pre_prune_topk_from_training_args(training_args: Any) -> Optional[float]:
    if hasattr(training_args, "pre_prune_config"):
        pre_cfg = getattr(training_args, "pre_prune_config", None)
    elif isinstance(training_args, dict):
        pre_cfg = training_args.get("pre_prune_config")
    else:
        pre_cfg = None
    normalized = _normalize_pre_prune_config(pre_cfg)
    if not normalized:
        return None
    topk = normalized.get("topk")
    return float(topk) if topk is not None else None


def is_stale_pruned_checkpoint(
    model_dir: str,
    *,
    pre_topk: Optional[float] = None,
    training_args: Any = None,
) -> bool:
    """Return True when a checkpoint looks like an unpruned full model for a pruned run."""
    topk = pre_topk
    if topk is None and training_args is not None:
        topk = _pre_prune_topk_from_training_args(training_args)
    if topk is None or topk >= 1.0:
        return False
    saved_dim = load_saved_gradiend_input_dim(model_dir)
    if saved_dim is None:
        return False
    return saved_dim > STALE_PRUNED_INPUT_DIM_THRESHOLD


def checkpoint_matches_training_fingerprint(
    model_dir: str,
    training_args: Any,
    *,
    log_reason: bool = True,
) -> bool:
    """Verify a saved checkpoint matches the requested training configuration."""
    expected = build_training_cache_fingerprint(training_args)
    saved = load_training_cache_fingerprint(model_dir)

    if saved is None:
        if is_stale_pruned_checkpoint(model_dir, training_args=training_args):
            if log_reason:
                logger.warning(
                    "Rejecting stale training cache at %s: pre-prune requested (topk=%s) "
                    "but saved input_dim=%s exceeds threshold %s (legacy checkpoint without fingerprint).",
                    model_dir,
                    _pre_prune_topk_from_training_args(training_args),
                    load_saved_gradiend_input_dim(model_dir),
                    STALE_PRUNED_INPUT_DIM_THRESHOLD,
                )
            return False
        return True

    for key in ("pre_prune_config", "post_prune_config", "reuse_pre_prune", "source", "target"):
        if expected.get(key) != saved.get(key):
            if log_reason:
                logger.warning(
                    "Rejecting training cache at %s: fingerprint mismatch on %r "
                    "(expected=%r, saved=%r).",
                    model_dir,
                    key,
                    expected.get(key),
                    saved.get(key),
                )
            return False

    expected_dim = expected.get("gradiend_input_dim")
    saved_dim = saved.get("gradiend_input_dim") or load_saved_gradiend_input_dim(model_dir)
    if expected_dim is not None and saved_dim is not None and int(expected_dim) != int(saved_dim):
        if log_reason:
            logger.warning(
                "Rejecting training cache at %s: gradiend_input_dim mismatch "
                "(expected=%s, saved=%s).",
                model_dir,
                expected_dim,
                saved_dim,
            )
        return False

    if is_stale_pruned_checkpoint(model_dir, training_args=training_args):
        if log_reason:
            logger.warning(
                "Rejecting stale pruned training cache at %s: input_dim=%s with pre_topk=%s.",
                model_dir,
                saved_dim,
                _pre_prune_topk_from_training_args(training_args),
            )
        return False

    return True


def convergent_count_for_model(model_dir: str, *, min_convergent_seeds: int) -> int:
    info = load_convergence_info(model_dir) or {}
    count = info.get("convergent_count")
    if isinstance(count, int):
        return count
    if bool(info.get("converged")) and min_convergent_seeds <= 1:
        return 1
    report = load_seed_report(model_dir)
    if report is not None:
        report_count = report.get("convergent_count")
        if isinstance(report_count, int):
            return report_count
    return 0


def _correlation_mean_threshold(training_args: Any) -> Optional[float]:
    if training_args is None:
        return None
    if hasattr(training_args, "convergent_metric"):
        metric = getattr(training_args, "convergent_metric", None)
        supervised_decoder = bool(getattr(training_args, "supervised_decoder", False))
        resolved_metric = (metric or ("loss" if supervised_decoder else "correlation")).lower()
        threshold = getattr(training_args, "convergent_mean_by_class_threshold", None)
    elif isinstance(training_args, dict):
        resolved_metric = (
            training_args.get("convergent_metric")
            or ("loss" if training_args.get("supervised_decoder") else "correlation")
        ).lower()
        threshold = training_args.get("convergent_mean_by_class_threshold")
    else:
        return None
    if resolved_metric != "correlation" or threshold is None:
        return None
    return float(threshold)


def _min_target_class_abs_mean_from_training_json(model_dir: str) -> Optional[float]:
    from gradiend.trainer.core.stats import _best_step_min_target_class_abs_mean

    payload = load_training_json(model_dir)
    if not isinstance(payload, dict):
        return None
    training_stats = payload.get("training_stats") or {}
    best_score_checkpoint = payload.get("best_score_checkpoint") or {}
    return _best_step_min_target_class_abs_mean(training_stats, best_score_checkpoint)


def _convergence_info_has_required_mean(
    info: dict,
    required: float,
    *,
    model_dir: Optional[str] = None,
) -> bool:
    if not isinstance(info, dict):
        return False
    computed: Optional[float] = None
    if model_dir:
        computed = _min_target_class_abs_mean_from_training_json(model_dir)
    saved_value = info.get("convergent_min_target_class_abs_mean")
    value = computed if isinstance(computed, (int, float)) else saved_value
    if not isinstance(value, (int, float)) or float(value) < required:
        return False
    saved_threshold = info.get("convergent_mean_by_class_threshold")
    if isinstance(saved_threshold, (int, float)):
        return float(saved_threshold) >= required
    # Missing threshold metadata but per-class abs mean is proven at/above required.
    return True


def convergence_record_matches_training_args(model_dir: str, training_args: Any) -> bool:
    """Return True when saved convergence metadata satisfies current convergence guards."""
    required_mean = _correlation_mean_threshold(training_args)
    if required_mean is None:
        return True

    info = load_convergence_info(model_dir)
    if isinstance(info, dict) and bool(info.get("converged")):
        if _convergence_info_has_required_mean(info, required_mean, model_dir=model_dir):
            return True
        logger.warning(
            "Rejecting convergent cache at %s: saved convergence metadata does not prove "
            "min |mean| among target classes >= %.4f.",
            model_dir,
            required_mean,
        )
        return False

    report = load_seed_report(model_dir)
    if isinstance(report, dict):
        converged_runs = [run for run in report.get("runs") or [] if isinstance(run, dict) and bool(run.get("converged"))]
        if not converged_runs:
            return True
        if all(
            _convergence_info_has_required_mean(run, required_mean, model_dir=model_dir)
            for run in converged_runs
        ):
            return True
        logger.warning(
            "Rejecting convergent cache at %s: seed report convergence metadata does not prove "
            "min |mean| among target classes >= %.4f for all convergent seeds.",
            model_dir,
            required_mean,
        )
        return False
    return True


def seed_run_converged(model_dir: str) -> bool:
    info = load_convergence_info(model_dir) or {}
    if "converged" in info:
        return bool(info.get("converged"))
    report = load_seed_report(model_dir)
    if report is None:
        return False
    for run in report.get("runs") or []:
        if isinstance(run, dict) and os.path.normpath(str(run.get("output_dir") or "")) == os.path.normpath(model_dir):
            return bool(run.get("converged"))
    return False


def run_meets_convergence_requirement(
    model_dir: str,
    *,
    min_convergent_seeds: int,
) -> bool:
    if min_convergent_seeds is None or min_convergent_seeds <= 0:
        return True
    return convergent_count_for_model(model_dir, min_convergent_seeds=min_convergent_seeds) >= min_convergent_seeds


def should_reuse_training_cache(
    use_cache: UseCacheSetting,
    model_dir: str,
    *,
    min_convergent_seeds: int = 1,
    training_args: Any = None,
) -> bool:
    """Return True when an existing checkpoint may skip (re)training."""
    use_cache = normalize_use_cache(use_cache)
    if use_cache is False:
        return False
    if not has_saved_model(model_dir):
        return False
    if use_cache == USE_CACHE_ALWAYS:
        return True
    if use_cache is True and training_args is None:
        return True
    if use_cache is True:
        if not checkpoint_matches_training_fingerprint(model_dir, training_args):
            return False
        return True
    if not run_meets_convergence_requirement(
        model_dir,
        min_convergent_seeds=min_convergent_seeds,
    ):
        return False
    if training_args is not None and not convergence_record_matches_training_args(model_dir, training_args):
        return False
    if training_args is not None and not checkpoint_matches_training_fingerprint(
        model_dir,
        training_args,
    ):
        return False
    return True


def should_reuse_seed_training_cache(
    use_cache: UseCacheSetting,
    seed_model_dir: str,
    *,
    training_args: Any = None,
) -> bool:
    """Per-seed cache check inside a multi-seed train() loop."""
    use_cache = normalize_use_cache(use_cache)
    if use_cache is False:
        return False
    if not has_saved_model(seed_model_dir):
        return False
    if use_cache == USE_CACHE_ALWAYS:
        return True
    if use_cache is True and training_args is None:
        return True
    if use_cache is True:
        if not checkpoint_matches_training_fingerprint(seed_model_dir, training_args):
            return False
        return True
    if not seed_run_converged(seed_model_dir):
        return False
    if training_args is not None and not convergence_record_matches_training_args(seed_model_dir, training_args):
        return False
    if training_args is not None and not checkpoint_matches_training_fingerprint(
        seed_model_dir,
        training_args,
    ):
        return False
    return True


def resolve_cache_fingerprint_for_write(
    output_dir: str,
    training_args: Any,
    cache_fingerprint: Optional[dict] = None,
) -> dict:
    """Build fingerprint to persist into training.json after a run completes."""
    fp = dict(cache_fingerprint) if cache_fingerprint else build_training_cache_fingerprint(training_args)
    saved_dim = load_saved_gradiend_input_dim(output_dir)
    if saved_dim is not None:
        fp["gradiend_input_dim"] = saved_dim
    return fp
