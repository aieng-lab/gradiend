"""
Training cache policy helpers for ``TrainingArguments.use_cache``.

Besides plain bool, ``use_cache="only_convergent"`` reuses saved checkpoints only
when convergence requirements are met (per-seed or multi-seed aggregate).
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Optional, Union

from gradiend.util.logging import get_logger
from gradiend.util.paths import has_saved_model

logger = get_logger(__name__)

UseCacheSetting = Union[bool, Literal["only_convergent"]]
USE_CACHE_ONLY_CONVERGENT: Literal["only_convergent"] = "only_convergent"


def normalize_use_cache(value: Any) -> UseCacheSetting:
    if isinstance(value, bool):
        return value
    if value == USE_CACHE_ONLY_CONVERGENT:
        return USE_CACHE_ONLY_CONVERGENT
    raise ValueError(
        f"use_cache must be bool or {USE_CACHE_ONLY_CONVERGENT!r}, got {value!r}"
    )


def is_unconditional_training_cache(value: Any) -> bool:
    """Return True only for explicit bool ``True`` (never for ``"only_convergent"``).

    Non-empty strings are truthy in Python, so never write ``if use_cache:`` when
    ``use_cache`` may be a training policy string.
    """
    return value is True


def coerce_artifact_use_cache(value: Any) -> bool:
    """Map ``TrainingArguments.use_cache`` to a bool for eval/annotation/CSV caches.

    ``"only_convergent"`` is training-checkpoint policy only and must not enable
    artifact reuse via truthiness.
    """
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if value == USE_CACHE_ONLY_CONVERGENT:
        return False
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
) -> bool:
    """Return True when an existing checkpoint may skip (re)training."""
    use_cache = normalize_use_cache(use_cache)
    if use_cache is False:
        return False
    if not has_saved_model(model_dir):
        return False
    if use_cache is True:
        return True
    return run_meets_convergence_requirement(
        model_dir,
        min_convergent_seeds=min_convergent_seeds,
    )


def should_reuse_seed_training_cache(
    use_cache: UseCacheSetting,
    seed_model_dir: str,
) -> bool:
    """Per-seed cache check inside a multi-seed train() loop."""
    use_cache = normalize_use_cache(use_cache)
    if use_cache is False:
        return False
    if not has_saved_model(seed_model_dir):
        return False
    if use_cache is True:
        return True
    return seed_run_converged(seed_model_dir)
