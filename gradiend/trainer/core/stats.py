"""
Load and write training statistics for GRADIEND model directories.
"""
import os
import json
from typing import Any, Dict, Optional

from gradiend.util.logging import get_logger
from gradiend.util.paths import is_under_temp_dir

logger = get_logger(__name__)


def _best_step_abs_mean_by_type(
    training_stats: Dict[str, Any],
    best_score_checkpoint: Dict[str, Any],
) -> Dict[str, float]:
    """
    Return abs_mean_by_type at the best checkpoint step (type -> value).
    training_stats["abs_mean_by_type"] is step -> { type -> value }; we slice at best step.
    """
    best_step = best_score_checkpoint.get("global_step")
    if best_step is None:
        return {}
    abs_hist = training_stats.get("abs_mean_by_type")
    if not isinstance(abs_hist, dict):
        return {}
    # Steps may be stored as int (in-memory) or str (after JSON round-trip)
    at_step = abs_hist.get(best_step) or abs_hist.get(str(best_step))
    if isinstance(at_step, dict):
        return {k: float(v) for k, v in at_step.items() if isinstance(v, (int, float))}
    return {}


def write_training_stats(
    output_dir: str,
    training_stats: Dict[str, Any],
    best_score_checkpoint: Dict[str, Any],
    training_args: Dict[str, Any],
    time_stats: Optional[Dict[str, Any]] = None,
    losses: Optional[list] = None,
    convergence_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write full training run info to output_dir/training.json.

    Use this after training so the final directory contains the complete
    step-wise history (scores, mean_by_class, etc.), not just the stats
    that existed when the best checkpoint was saved.

    Multi-phase training (e.g. train(supervised_encoder=True) then
    train(supervised_decoder=True)): each run overwrites training.json.
    The file will contain only the last run's stats. To keep both phases,
    copy training.json after the first run (e.g. to training_encoder.json)
    before starting the second. training_args in the file includes
    supervised_encoder / supervised_decoder so consumers can tell the run type;
    for supervised_decoder runs, best_score_checkpoint may have correlation=None
    and use loss instead.

    Args:
        output_dir: Model directory (e.g. training_args.output_dir after keep_only_best).
        training_stats: Full training_stats from the end of training.
        best_score_checkpoint: Dict with correlation (or None), global_step, epoch, optionally loss.
        training_args: Training config dict (includes supervised_encoder, supervised_decoder).
        time_stats: Optional timing stats.
        losses: Optional list of per-step losses.
        convergence_info: Optional dict with convergence status (converged, convergent_count, etc.).

    Returns:
        Path to the written training.json file.
    """
    path = os.path.join(output_dir, "training.json")
    run_info = {
        "training_stats": training_stats,
        "best_score_checkpoint": best_score_checkpoint,
        "training_args": training_args,
        "time": time_stats or {},
        "losses": losses or [],
    }
    # Expose abs_mean_by_type at best step so stats["abs_mean_by_type"]["training"] is the best model's score
    best_abs = _best_step_abs_mean_by_type(training_stats, best_score_checkpoint)
    if best_abs:
        run_info["abs_mean_by_type"] = best_abs
    if convergence_info is not None:
        run_info["convergence_info"] = convergence_info
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(run_info, f, indent=2)
    if not is_under_temp_dir(path):
        logger.info("Wrote full training stats to %s", path)
    return path


def load_training_stats(model_path: str) -> Optional[dict]:
    """
    Load training statistics and metadata from a saved GRADIEND model directory.

    Use this to inspect correlation, mean_by_class, config, and best checkpoint
    after training without loading the full model.

    Args:
        model_path: Path to the saved model directory.

    Returns:
        Dict with keys:
        - training_stats: correlation, scores, mean_by_class, mean_by_feature_class (by step), etc.
        - best_score_checkpoint: correlation, global_step, epoch
        - config: training config used
        - time: timing stats (total, eval, etc.)
        Or None if model_path has no training.json.

    Example:
        stats = load_training_stats(model_path)
        ts = stats["training_stats"]
        print(ts.get("correlation"), ts.get("mean_by_class"))
        print(stats["best_score_checkpoint"])
    """
    if not isinstance(model_path, str):
        raise TypeError(f"model_path must be str, got {type(model_path).__name__}")

    training_path = os.path.join(model_path, "training.json")
    if not os.path.exists(training_path):
        return None
    try:
        with open(training_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load training stats from {training_path}: {e}")
        return None
    # Ensure stats["abs_mean_by_type"] is the best-step snapshot (type -> value) for convenience
    if "abs_mean_by_type" not in data or not _is_best_step_abs_mean_by_type(data.get("abs_mean_by_type"), data):
        best_abs = _best_step_abs_mean_by_type(
            data.get("training_stats") or {},
            data.get("best_score_checkpoint") or {},
        )
        if best_abs:
            data["abs_mean_by_type"] = best_abs
        elif "abs_mean_by_type" not in data:
            data["abs_mean_by_type"] = {}
    return data


def _is_best_step_abs_mean_by_type(
    value: Any,
    data: Dict[str, Any],
) -> bool:
    """True if value looks like a best-step dict (type -> number), not step-wise (step -> dict)."""
    if not isinstance(value, dict) or not value:
        return False
    first_val = next(iter(value.values()), None)
    # Best-step: values are numbers. Step-wise: values are dicts (type -> number).
    return isinstance(first_val, (int, float))
