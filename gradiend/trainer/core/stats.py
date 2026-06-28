"""
Load and write training statistics for GRADIEND model directories.
"""
import math
import os
import json
from typing import Any, Dict, List, Optional
from gradiend.util.logging import get_logger
from gradiend.util.util import to_jsonable
from gradiend.util.paths import is_under_temp_dir

logger = get_logger(__name__)


def _normalize_step_value_dict(value: Any) -> Dict[str, float]:
    """Normalize per-step scalar series to {step: value}. Accepts legacy list (step 1..n)."""
    if isinstance(value, dict):
        out: Dict[str, float] = {}
        for key, val in value.items():
            if isinstance(val, (int, float)):
                out[str(key)] = float(val)
        return out
    if isinstance(value, list):
        return {str(i + 1): float(v) for i, v in enumerate(value) if isinstance(v, (int, float))}
    return {}


def _normalize_training_stats_step_dicts(training_stats: Dict[str, Any]) -> Dict[str, Any]:
    """In-place normalize legacy list-shaped step series in training_stats."""
    if not isinstance(training_stats, dict):
        return training_stats
    for key in ("encoder_norms", "decoder_norms"):
        if key in training_stats:
            training_stats[key] = _normalize_step_value_dict(training_stats.get(key))
    return training_stats


def _best_step_abs_mean_by_type(
    training_stats: Dict[str, Any],
    best_score_checkpoint: Dict[str, Any],
) -> Dict[str, float]:
    """
    Return abs_mean_by_type at the best checkpoint step (type -> value).
    training_stats["abs_mean_by_type"] is step -> { type -> value }; we slice at best step.
    """
    direct = best_score_checkpoint.get("abs_mean_by_type")
    if isinstance(direct, dict):
        return {k: float(v) for k, v in direct.items() if isinstance(v, (int, float))}

    abs_hist = training_stats.get("abs_mean_by_type")
    if isinstance(abs_hist, dict) and any(isinstance(v, (int, float)) for v in abs_hist.values()):
        return {k: float(v) for k, v in abs_hist.items() if isinstance(v, (int, float))}

    best_step = best_score_checkpoint.get("global_step")
    if best_step is None:
        return {}
    if not isinstance(abs_hist, dict):
        return {}
    # Steps may be stored as int (in-memory) or str (after JSON round-trip)
    at_step = abs_hist.get(best_step) or abs_hist.get(str(best_step))
    if isinstance(at_step, dict):
        return {k: float(v) for k, v in at_step.items() if isinstance(v, (int, float))}
    return {}


def _best_checkpoint_step_is_after_initial(best_score_checkpoint: Dict[str, Any]) -> bool:
    """Return True only when the best checkpoint was selected after step 0."""
    if not isinstance(best_score_checkpoint, dict):
        return False
    best_step = best_score_checkpoint.get("global_step")
    if isinstance(best_step, bool):
        return False
    if isinstance(best_step, (int, float)):
        return best_step > 0
    if isinstance(best_step, str):
        try:
            return float(best_step) > 0
        except ValueError:
            return False
    return False


def _best_step_mean_by_class(
    training_stats: Dict[str, Any],
    best_score_checkpoint: Dict[str, Any],
) -> Dict[Any, float]:
    """
    Return mean_by_class at the best checkpoint step (label -> mean encoded value).
    training_stats["mean_by_class"] is step -> { label -> mean }.
    """
    best_step = best_score_checkpoint.get("global_step")
    if best_step is None:
        return {}
    mean_hist = training_stats.get("mean_by_class")
    if not isinstance(mean_hist, dict):
        return {}
    at_step = mean_hist.get(best_step) or mean_hist.get(str(best_step))
    if isinstance(at_step, dict):
        return {k: float(v) for k, v in at_step.items() if isinstance(v, (int, float))}
    return {}


def _nonzero_target_class_abs_means(mean_by_class: Dict[Any, float]) -> List[float]:
    """Absolute mean encoded values for non-zero (non-neutral) target classes."""
    abs_means: List[float] = []
    for label, mean_value in mean_by_class.items():
        try:
            numeric_label = float(label)
        except (TypeError, ValueError):
            continue
        if numeric_label == 0.0:
            continue
        abs_means.append(abs(float(mean_value)))
    return abs_means


def _best_step_min_target_class_abs_mean(
    training_stats: Dict[str, Any],
    best_score_checkpoint: Dict[str, Any],
) -> Optional[float]:
    """
    Return the minimum |mean encoded value| among non-zero target classes at the best checkpoint.

    Identity / neutral class 0 is ignored. Returns None when no non-zero target classes
    are available at the best step.
    """
    mean_by_class = _best_step_mean_by_class(training_stats, best_score_checkpoint)
    if not mean_by_class:
        return None
    abs_means = _nonzero_target_class_abs_means(mean_by_class)
    if not abs_means:
        return None
    return min(abs_means)


def _best_step_target_class_mean_product(
    training_stats: Dict[str, Any],
    best_score_checkpoint: Dict[str, Any],
) -> Optional[float]:
    """
    Return the product of the two non-zero target-class means at the best checkpoint.

    Identity / neutral class 0 is ignored. Returns None when exactly two non-zero
    target classes are not available at the best step.
    """
    mean_by_class = _best_step_mean_by_class(training_stats, best_score_checkpoint)
    if not mean_by_class:
        return None

    target_means = []
    for label, mean_value in mean_by_class.items():
        try:
            numeric_label = float(label)
        except (TypeError, ValueError):
            continue
        if numeric_label == 0.0:
            continue
        target_means.append(float(mean_value))

    if len(target_means) != 2:
        return None
    return target_means[0] * target_means[1]


def summarize_topk_stability(
    topk_indices_by_run: Dict[str, List[int]],
    *,
    run_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    topk: Optional[Any] = None,
    part: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Summarize top-k overlap stability across multiple runs.

    Args:
        topk_indices_by_run: Mapping run_id -> ordered list of top-k base-global indices.
        run_metadata: Optional metadata per run (e.g. seed).
        topk: Requested top-k value used to build the index sets.
        part: Importance part used to select the indices.

    Returns:
        Summary dict with pairwise overlap rows plus aggregate presence/overlap statistics.
    """
    run_metadata = run_metadata or {}
    indexed = {
        str(run_id): [int(idx) for idx in indices]
        for run_id, indices in topk_indices_by_run.items()
        if indices
    }
    if len(indexed) < 2:
        return {
            "computed": False,
            "reason": "need_at_least_two_runs",
            "num_runs": len(indexed),
            "topk": topk,
            "part": part,
        }

    sets = {run_id: set(indices) for run_id, indices in indexed.items()}
    topk_sizes = {run_id: len(indices) for run_id, indices in indexed.items()}

    pairwise_rows: List[Dict[str, Any]] = []
    pairwise_overlap_fractions: List[float] = []
    pairwise_jaccard_scores: List[float] = []
    run_ids = sorted(indexed.keys())
    for idx_a, run_a in enumerate(run_ids):
        for run_b in run_ids[idx_a + 1:]:
            set_a = sets[run_a]
            set_b = sets[run_b]
            overlap_count = len(set_a & set_b)
            denom = float(max(1, min(len(set_a), len(set_b))))
            union_size = len(set_a | set_b)
            overlap_fraction = overlap_count / denom
            jaccard = overlap_count / float(max(1, union_size))
            pairwise_overlap_fractions.append(overlap_fraction)
            pairwise_jaccard_scores.append(jaccard)
            row = {
                "run_a": run_a,
                "run_b": run_b,
                "overlap_count": overlap_count,
                "overlap_fraction": overlap_fraction,
                "jaccard": jaccard,
                "topk_size_a": len(set_a),
                "topk_size_b": len(set_b),
            }
            if run_a in run_metadata:
                row.update({f"{key}_a": value for key, value in run_metadata[run_a].items()})
            if run_b in run_metadata:
                row.update({f"{key}_b": value for key, value in run_metadata[run_b].items()})
            pairwise_rows.append(row)
    pairwise_rows.sort(
        key=lambda row: (
            -float(row["overlap_fraction"]),
            str(row["run_a"]),
            str(row["run_b"]),
        )
    )

    presence_counts: Dict[int, int] = {}
    for index_set in sets.values():
        for base_global_index in index_set:
            presence_counts[base_global_index] = presence_counts.get(base_global_index, 0) + 1

    num_runs = len(sets)
    presence_fractions = [count / float(num_runs) for count in presence_counts.values()]
    intersection = set.intersection(*sets.values())
    union = set.union(*sets.values())

    def _mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def _std(values: List[float]) -> float:
        if not values:
            return 0.0
        mean_value = _mean(values)
        return float(math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values)))

    return {
        "computed": True,
        "num_runs": num_runs,
        "run_ids": run_ids,
        "seeds": [run_metadata.get(run_id, {}).get("seed") for run_id in run_ids],
        "part": part,
        "topk": topk,
        "topk_size_by_run": topk_sizes,
        "mean_topk_size": _mean([float(size) for size in topk_sizes.values()]),
        "union_size": len(union),
        "intersection_size": len(intersection),
        "parameters_seen_once": int(sum(1 for count in presence_counts.values() if count == 1)),
        "parameters_seen_in_all_runs": int(sum(1 for count in presence_counts.values() if count == num_runs)),
        "mean_presence_fraction": _mean(presence_fractions),
        "mean_pairwise_overlap_fraction": _mean(pairwise_overlap_fractions),
        "pairwise_overlap_std": _std(pairwise_overlap_fractions),
        "min_pairwise_overlap_fraction": float(min(pairwise_overlap_fractions)) if pairwise_overlap_fractions else 1.0,
        "max_pairwise_overlap_fraction": float(max(pairwise_overlap_fractions)) if pairwise_overlap_fractions else 1.0,
        "mean_pairwise_jaccard": _mean(pairwise_jaccard_scores),
        "pairwise_jaccard_std": _std(pairwise_jaccard_scores),
        "pairwise_overlap_rows": pairwise_rows,
    }


def write_training_stats(
    output_dir: str,
    training_stats: Dict[str, Any],
    best_score_checkpoint: Dict[str, Any],
    training_args: Dict[str, Any],
    time_stats: Optional[Dict[str, Any]] = None,
    losses: Optional[list] = None,
    convergence_info: Optional[Dict[str, Any]] = None,
    seed_stability: Optional[Dict[str, Any]] = None,
    cache_fingerprint: Optional[Dict[str, Any]] = None,
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
    if seed_stability is not None:
        run_info["seed_stability"] = seed_stability
    from gradiend.trainer.core.cache_policy import resolve_cache_fingerprint_for_write

    run_info["cache_fingerprint"] = resolve_cache_fingerprint_for_write(
        output_dir,
        training_args,
        cache_fingerprint=cache_fingerprint,
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(run_info), f, indent=2)
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
    if isinstance(data.get("training_stats"), dict):
        _normalize_training_stats_step_dicts(data["training_stats"])
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
