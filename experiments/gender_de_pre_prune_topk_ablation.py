"""
Pre-prune grid for German der<->die GRADIENDs: pre_topk x source x n_samples.

Default mode (mask recall): train only the no-pre baseline per pair (oracle top-1000
decoder weights), run ``pre_prune()`` for each config, and report mask recall
|M_config ∩ T_ref| / topk_eval. No GRADIEND training per grid cell.

**--full-grid** trains all 5 der<->die pairs for every grid cell (multi-seed,
convergence enforced). Reference arm per pair: pre_topk=1.0 (no pre/post prune).
Use a separate ``--output-dir`` from the default recall study.

Full-grid primary metric — **cross_overlap** (between GRADIENDs):
  For a fixed pre-prune config, take each pair's top-1000 decoder weights (base-global
  indices) and report the mean pairwise intersection fraction across the 5 der<->die
  GRADIENDs.

Full-grid secondary metric — **ref_recall**:
  |T_config ∩ T_ref| / topk_eval comparing a pruned *trained* run to its no-pre reference.

Baseline training uses max_seeds=5, min_convergent_seeds=2, fail_on_non_convergence=True,
and use_cache="only_convergent" so non-converged checkpoints are automatically retrained
on resume.

Results and per-run top-k index files are saved incrementally (resume-safe).

Example:
  python experiments/gender_de_pre_prune_topk_ablation.py --pair masc_nom_fem_nom
  python experiments/gender_de_pre_prune_topk_ablation.py --mode plot
  python experiments/gender_de_pre_prune_topk_ablation.py --full-grid --mode run
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PrePruneConfig
from gradiend.trainer.core.cache_policy import (
    STALE_PRUNED_INPUT_DIM_THRESHOLD,
    is_stale_pruned_checkpoint,
    load_saved_gradiend_input_dim,
)
from gradiend.trainer.core.stats import load_training_stats
from gradiend.util.logging import get_logger
from gradiend.util.paths import ARTIFACT_MODEL, resolve_output_path
from gradiend.util.runtime_monitor import CudaMemorySpan

logger = get_logger(__name__)


class _AblationTrainer(TextPredictionTrainer):
    last_encoder_analysis_time_s: Optional[float] = None

    def _analyze_encoder(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return super()._analyze_encoder(*args, **kwargs)
        finally:
            self.last_encoder_analysis_time_s = time.perf_counter() - start


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _save_figure(fig: plt.Figure, output_path: str, **kwargs: Any) -> None:
    """Save a matplotlib figure; use a fallback path if the target PDF is locked."""
    _ensure_dir(os.path.dirname(output_path) or ".")
    save_kwargs = {"bbox_inches": "tight", **kwargs}
    try:
        fig.savefig(output_path, **save_kwargs)
        logger.info("Wrote %s", output_path)
        fig.show()
        return
    except PermissionError:
        root, ext = os.path.splitext(output_path)
        fallback = f"{root}_new{ext}"
        fig.savefig(fallback, **save_kwargs)
        logger.warning(
            "Could not overwrite %s (close viewer); wrote %s instead.",
            output_path,
            fallback,
        )


def _clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_path(output_dir: str, run_id: str) -> str:
    exp_dir = os.path.join(output_dir.rstrip("/\\"), str(run_id).strip("/\\"))
    path = resolve_output_path(exp_dir, None, ARTIFACT_MODEL)
    if path is None:
        raise ValueError(f"Could not resolve checkpoint path for run_id={run_id!r}")
    return path


def _format_topk(topk: float) -> str:
    return f"{topk:.6f}".replace(".", "_")


def _training_time_from_checkpoint(checkpoint_path: str) -> Optional[float]:
    stats = load_training_stats(checkpoint_path)
    if not stats:
        return None
    value = (stats.get("time") or {}).get("total")
    return float(value) if isinstance(value, (int, float)) else None


def _train_and_get_time(trainer: TextPredictionTrainer, output_dir: str, run_id: str) -> float:
    start = time.perf_counter()
    trainer.train()
    wall_time = time.perf_counter() - start
    stats_time = _training_time_from_checkpoint(_checkpoint_path(output_dir, run_id))
    return stats_time if stats_time is not None else wall_time


def _run_encoder_eval(trainer: TextPredictionTrainer, max_size: int) -> float:
    result = trainer.evaluate_encoder(max_size=max_size, use_cache=False)
    return float(result.get("correlation", 0.0)) if result else 0.0


def _run_decoder_eval(
    trainer: TextPredictionTrainer,
    pair: Tuple[str, str],
    max_size: int,
) -> Dict[str, float]:
    source_class, target_class = pair
    result = trainer.evaluate_decoder(
        lrs=[1e-3],
        max_size_training_like=max_size,
        max_size_neutral=max_size,
        use_cache=False,
    )
    if not result:
        raise ValueError("Decoder evaluation returned an empty result.")

    def _summary_value(class_name: str) -> float:
        top_level_summary = result.get(class_name)
        if isinstance(top_level_summary, dict) and isinstance(top_level_summary.get("value"), (int, float)):
            return float(top_level_summary["value"])
        nested_summary = result.get("summary", {})
        if isinstance(nested_summary, dict):
            class_summary = nested_summary.get(class_name)
            if isinstance(class_summary, dict) and isinstance(class_summary.get("value"), (int, float)):
                return float(class_summary["value"])
        raise KeyError(f"Decoder eval missing summary for class {class_name!r}")

    def _grid_rows() -> List[Dict[str, Any]]:
        if isinstance(result.get("grid"), dict):
            return [row for row in result["grid"].values() if isinstance(row, dict)]
        if isinstance(result.get("results"), list):
            return [row for row in result["results"] if isinstance(row, dict)]
        raise KeyError("Decoder eval missing grid/results rows")

    rows = _grid_rows()
    base_rows = [row for row in rows if row.get("id") == "base"]
    if len(base_rows) != 1:
        raise KeyError(f"Expected exactly one base decoder grid row, found {len(base_rows)}.")

    def _prob_by_dataset(row: Dict[str, Any], dataset_class: str, predicted_class: str) -> float:
        probs_by_dataset = row.get("probs_by_dataset")
        if not isinstance(probs_by_dataset, dict):
            raise KeyError("Decoder grid row missing probs_by_dataset")
        dataset_probs = probs_by_dataset.get(dataset_class)
        if not isinstance(dataset_probs, dict):
            raise KeyError(f"Decoder probs_by_dataset missing dataset class {dataset_class!r}")
        value = dataset_probs.get(predicted_class)
        if not isinstance(value, (int, float)):
            raise KeyError(f"Decoder probs missing {predicted_class!r} under {dataset_class!r}")
        return float(value)

    target_base_prob = _prob_by_dataset(base_rows[0], source_class, target_class)
    target_prob = _summary_value(target_class)
    return {"decoder_prob_delta": target_prob - float(target_base_prob)}


def _run_evals_with_model(
    trainer: TextPredictionTrainer,
    model: Any,
    pair: Tuple[str, str],
    max_size: int,
) -> Tuple[float, Dict[str, float]]:
    prev_instance = trainer._model_instance
    trainer._model_instance = model
    try:
        enc = _run_encoder_eval(trainer, max_size=max_size)
        dec = _run_decoder_eval(trainer, pair=pair, max_size=max_size)
        return enc, dec
    finally:
        trainer._model_instance = prev_instance


def _build_trainer(
    *,
    run_id: str,
    pair: Tuple[str, str],
    args: TrainingArguments,
) -> TextPredictionTrainer:
    return _AblationTrainer(
        model="bert-base-german-cased",
        run_id=run_id,
        data="aieng-lab/de-gender-case-articles",
        target_classes=list(pair),
        masked_col="masked",
        split_col="split",
        eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
        args=args,
    )

DER_DIE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("masc_nom", "fem_nom"),
    ("fem_nom", "fem_dat"),
    ("fem_nom", "fem_gen"),
    ("fem_acc", "fem_dat"),
    ("fem_acc", "fem_gen"),
)

PRE_SOURCES = ["factual", "alternative", "diff"]
PRE_N_SAMPLES = [1, 2, 4, 8, 16, 32, 64]
_N_SAMPLES_MARKERS: Tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X")
# Decade pre_topk grid: 1.0 = baseline (no pre-prune), then 0.1, 0.01, ...
BASELINE_PRE_TOPK = 1.0
BASELINE_REF_RECALL = 1.0
PRE_TOPK_VALUES: List[float] = [1.0, 0.1, 0.01, 0.001]


def _is_decade_pre_topk(value: float) -> bool:
    """True when value is 10**n for integer n (e.g. 1, 0.1, 0.01)."""
    if value <= 0.0 or value > 1.0:
        return False
    if math.isclose(value, 1.0):
        return True
    exponent = round(math.log10(value))
    return math.isclose(value, 10.0**exponent, rel_tol=0.0, abs_tol=1e-9)


def _decade_pre_topk_values(values: Iterable[float]) -> List[float]:
    return sorted({float(v) for v in values if _is_decade_pre_topk(float(v))}, reverse=True)


def default_pre_topk_values(*, min_topk: float = 0.01) -> List[float]:
    """Return descending decade pre_topk values from ``min_topk`` to 1.0."""
    if not (0.0 < min_topk <= 1.0):
        raise ValueError(f"min_topk must be in (0, 1], got {min_topk!r}")
    min_exp = int(math.floor(math.log10(min_topk)))
    values = {1.0}
    values.update(10.0**exponent for exponent in range(0, min_exp - 1, -1))
    return sorted(values, reverse=True)
MIN_CONVERGENT_SEEDS = 1
DEFAULT_MAX_SEEDS = 5
PRE_PRUNE_SEED = 42
TOPK_EVAL = 1000
TOPK_PART = "decoder-weight"
PAIR_ALIASES: Dict[str, Tuple[str, str]] = {
    "masc_nom_fem_nom": ("masc_nom", "fem_nom"),
    "fem_nom_fem_dat": ("fem_nom", "fem_dat"),
    "fem_nom_fem_gen": ("fem_nom", "fem_gen"),
    "fem_acc_fem_dat": ("fem_acc", "fem_dat"),
    "fem_acc_fem_gen": ("fem_acc", "fem_gen"),
}


@dataclass
class GridResult:
    pair: str
    run_id: str
    pre_topk: float
    pre_source: Optional[str] = None
    pre_n_samples: Optional[int] = None
    kept_dim: Optional[int] = None
    base_input_dim: Optional[int] = None
    ref_recall: Optional[float] = None
    ref_precision: Optional[float] = None
    topk_indices_file: Optional[str] = None
    encoder_correlation: Optional[float] = None
    decoder_prob_delta: Optional[float] = None
    training_time_s: Optional[float] = None
    pre_pruning_time_s: Optional[float] = None
    converged: Optional[bool] = None
    n_convergent_seeds: Optional[int] = None
    n_seeds_tried: Optional[int] = None
    selected_seed: Optional[int] = None
    mask_recall: bool = False
    error: Optional[str] = None


def _load_results(path: str) -> List[GridResult]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    results: List[GridResult] = []
    for row in payload:
        if row.get("error"):
            logger.warning(
                "Ignoring legacy failed row in %s: run_id=%s error=%s",
                path,
                row.get("run_id"),
                row["error"],
            )
            continue
        allowed = {field.name for field in fields(GridResult)}
        filtered = {key: value for key, value in row.items() if key in allowed}
        results.append(GridResult(**filtered))
    return results


def _load_results_many(paths: Iterable[str]) -> List[GridResult]:
    results: List[GridResult] = []
    seen: Set[Tuple[str, str]] = set()
    for path in paths:
        for row in _load_results(path):
            key = (row.pair, row.run_id)
            if key in seen:
                continue
            seen.add(key)
            results.append(row)
    return results


@dataclass
class ConfigSummary:
    pre_topk: float
    pre_source: Optional[str]
    pre_n_samples: Optional[int]
    mean_ref_recall: float
    mean_kept_dim: float
    mean_cross_overlap: float
    mean_encoder_correlation: float
    n_pairs: int = 0


def _pair_slug(pair: Tuple[str, str]) -> str:
    return f"{pair[0]}_{pair[1]}"


def _pair_child_id(pair: Tuple[str, str]) -> str:
    return f"gender_de_{_pair_slug(pair)}"


def _parse_pair(value: str) -> Tuple[str, str]:
    value = value.strip()
    if value in PAIR_ALIASES:
        return PAIR_ALIASES[value]
    if ":" in value:
        left, right = value.split(":", 1)
    elif "," in value:
        left, right = value.split(",", 1)
    else:
        raise argparse.ArgumentTypeError(
            f"Unknown pair {value!r}. Use one of {sorted(PAIR_ALIASES)} or SOURCE:TARGET."
        )
    pair = (left.strip(), right.strip())
    if not pair[0] or not pair[1]:
        raise argparse.ArgumentTypeError("Pair must have form SOURCE:TARGET.")
    return pair


def _csv_pairs(value: str) -> List[Tuple[str, str]]:
    return [_parse_pair(part) for part in value.split(",") if part.strip()]


def _default_pair_results_path(output_dir: str, pair: Tuple[str, str]) -> str:
    return os.path.join(output_dir, "pair_results", f"{_pair_slug(pair)}.json")


def _combined_results_path(output_dir: str) -> str:
    return os.path.join(output_dir, "pre_topk_grid_results.json")


def _discover_result_paths(output_dir: str, explicit_path: Optional[str]) -> List[str]:
    paths: List[str] = []
    if explicit_path:
        paths.append(explicit_path)
    else:
        pair_results_dir = os.path.join(output_dir, "pair_results")
        if os.path.isdir(pair_results_dir):
            paths.extend(
                os.path.join(pair_results_dir, name)
                for name in sorted(os.listdir(pair_results_dir))
                if name.endswith(".json")
            )
        paths.append(_combined_results_path(output_dir))
    return [path for path in paths if os.path.isfile(path)]


def _baseline_run_id(pair: Tuple[str, str]) -> str:
    return f"{_pair_child_id(pair)}/ref_pre_topk_{_format_topk(1.0)}"


def _grid_run_id(pair: Tuple[str, str], source: str, n_samples: int, topk: float) -> str:
    return (
        f"{_pair_child_id(pair)}/pre_src_{source}_n_{n_samples}_topk_{_format_topk(topk)}"
    )


def _config_key(pre_topk: float, pre_source: Optional[str], pre_n_samples: Optional[int]) -> str:
    src = pre_source or "none"
    n = pre_n_samples if pre_n_samples is not None else "none"
    return f"topk_{_format_topk(pre_topk)}__src_{src}__n_{n}"


def _resolve_max_seeds(max_seeds: int) -> int:
    if max_seeds < MIN_CONVERGENT_SEEDS:
        logger.warning(
            "max_seeds=%s is below min_convergent_seeds=%s; using %s instead.",
            max_seeds,
            MIN_CONVERGENT_SEEDS,
            DEFAULT_MAX_SEEDS,
        )
        return DEFAULT_MAX_SEEDS
    return max_seeds


def _base_args(output_dir: str, *, max_seeds: int) -> Dict[str, Any]:
    max_seeds = _resolve_max_seeds(max_seeds)
    return dict(
        experiment_dir=output_dir,
        base_gradient_batch_size=4,
        encoder_eval_max_size=100,
        decoder_eval_max_size_training_like=100,
        decoder_eval_max_size_neutral=1000,
        eval_steps=250,
        num_train_epochs=1,
        max_steps=1000,
        max_seeds=max_seeds,
        min_convergent_seeds=MIN_CONVERGENT_SEEDS,
        fail_on_non_convergence=True,
        source="alternative",
        target="diff",
        eval_batch_size=8,
        learning_rate=1e-5,
        use_cache="only_convergent",
        add_identity_for_other_classes=True,
    )


def _save_results(results: List[GridResult], path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(r) for r in results], handle, indent=2)


def _seed_pair_results_from_combined(
    *,
    output_dir: str,
    results_path: str,
    pair: Tuple[str, str],
) -> None:
    if os.path.exists(results_path):
        return
    combined_path = _combined_results_path(output_dir)
    if os.path.normcase(os.path.abspath(results_path)) == os.path.normcase(os.path.abspath(combined_path)):
        return
    pair_key = _pair_slug(pair)
    rows = [row for row in _load_results(combined_path) if row.pair == pair_key]
    if not rows:
        return
    logger.info(
        "Seeding %s with %s existing rows for pair %s from %s.",
        results_path,
        len(rows),
        pair_key,
        combined_path,
    )
    _save_results(rows, results_path)


def _append_result(results: List[GridResult], result: GridResult, path: str) -> None:
    results[:] = [r for r in results if not (r.pair == result.pair and r.run_id == result.run_id)]
    results.append(result)
    _save_results(results, path)


def _is_valid_pruned_result(row: GridResult) -> bool:
    if row.pre_topk is None or math.isclose(row.pre_topk, 1.0):
        return True
    if row.kept_dim is None:
        return False
    return row.kept_dim <= STALE_PRUNED_INPUT_DIM_THRESHOLD


def _is_definitely_stale_result(row: GridResult) -> bool:
    """True only when a row is known-bad (not merely incomplete)."""
    if row.pre_topk is None or math.isclose(row.pre_topk, 1.0):
        return False
    if row.kept_dim is not None and row.kept_dim > STALE_PRUNED_INPUT_DIM_THRESHOLD:
        return True
    return False


def _has_success(
    results: List[GridResult],
    pair: str,
    run_id: str,
    *,
    recall_only: bool = False,
) -> bool:
    if recall_only:
        return any(
            r.pair == pair
            and r.run_id == run_id
            and r.mask_recall
            and r.ref_recall is not None
            and not r.error
            and r.topk_indices_file
            and os.path.isfile(r.topk_indices_file)
            for r in results
        )
    return any(
        r.pair == pair
        and r.run_id == run_id
        and r.topk_indices_file
        and r.converged is True
        and not r.error
        and _is_valid_pruned_result(r)
        for r in results
    )


def _missing_grid_cells(
    results: List[GridResult],
    *,
    pairs: Sequence[Tuple[str, str]],
    sources: Sequence[str],
    n_samples_values: Sequence[int],
    pre_topk_values: Sequence[float],
    recall_only: bool = False,
) -> List[Dict[str, Any]]:
    missing: List[Dict[str, Any]] = []
    for pair in pairs:
        pair_key = _pair_slug(pair)
        for topk in pre_topk_values:
            if math.isclose(topk, 1.0):
                continue
            for source in sources:
                for n_samples in n_samples_values:
                    run_id = _grid_run_id(pair, source, n_samples, topk)
                    if not _has_success(results, pair_key, run_id, recall_only=recall_only):
                        missing.append(
                            {
                                "pair": pair_key,
                                "pre_topk": topk,
                                "pre_source": source,
                                "pre_n_samples": n_samples,
                                "run_id": run_id,
                            }
                        )
    return missing


def _missing_topk_counts(missing: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for cell in missing:
        key = f"{cell['pre_topk']:g}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: float(kv[0]), reverse=True))


def _invalidate_checkpoint_tree(checkpoint_path: str) -> None:
    run_dir = os.path.dirname(os.path.normpath(checkpoint_path))
    for path in (checkpoint_path, run_dir):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def migrate_stale_caches(
    *,
    output_dir: str,
    result_paths: Sequence[str],
    dry_run: bool = False,
) -> int:
    """Drop stale pruned results/checkpoints so the grid re-trains them.

    Temporary migration helper for runs produced before pre_prune_config was kept
    through multi-seed train() and before cache fingerprint validation existed.

    Only removes rows/checkpoints that are *definitely* stale (full-model kept_dim
    or on-disk input_dim). Incomplete rows (``kept_dim is None``) are kept.
    """
    invalidated = 0
    for results_path in result_paths:
        if not os.path.isfile(results_path):
            continue
        rows = _load_results(results_path)
        kept_rows: List[GridResult] = []
        changed = False
        for row in rows:
            stale_row = _is_definitely_stale_result(row)
            checkpoint_path = _checkpoint_path(output_dir, row.run_id)
            stale_ckpt = (
                not math.isclose(row.pre_topk or 1.0, 1.0)
                and os.path.isdir(checkpoint_path)
                and is_stale_pruned_checkpoint(checkpoint_path, pre_topk=row.pre_topk)
            )
            if not stale_row and not stale_ckpt:
                kept_rows.append(row)
                continue

            if not dry_run:
                if stale_ckpt or os.path.isdir(checkpoint_path):
                    _invalidate_checkpoint_tree(checkpoint_path)
                if row.topk_indices_file and os.path.isfile(row.topk_indices_file):
                    try:
                        os.remove(row.topk_indices_file)
                    except OSError:
                        pass
            logger.warning(
                "%s stale cache for %s/%s (pre_topk=%s, kept_dim=%s).",
                "Would invalidate" if dry_run else "Invalidated",
                row.pair,
                row.run_id,
                row.pre_topk,
                row.kept_dim,
            )
            invalidated += 1
            changed = True
        if changed and not dry_run:
            _save_results(kept_rows, results_path)
    return invalidated


def backfill_convergence_from_checkpoints(
    *,
    output_dir: str,
    result_paths: Sequence[str],
) -> int:
    """Update legacy rows (``converged is None``) from on-disk training.json."""
    updated = 0
    for results_path in result_paths:
        if not os.path.isfile(results_path):
            continue
        rows = _load_results(results_path)
        changed = False
        for row in rows:
            if row.converged is not None:
                continue
            checkpoint_path = _checkpoint_path(output_dir, row.run_id)
            if not os.path.isdir(checkpoint_path):
                continue
            meta = _read_run_convergence(checkpoint_path)
            row.converged = bool(meta["converged"])
            row.n_convergent_seeds = int(meta["n_convergent_seeds"])
            row.n_seeds_tried = int(meta["n_seeds_tried"])
            row.selected_seed = meta["selected_seed"]
            if not row.converged:
                row.error = (
                    f"grid cell converged {row.n_convergent_seeds}/{MIN_CONVERGENT_SEEDS} seeds"
                )
            changed = True
            updated += 1
        if changed:
            _save_results(rows, results_path)
    return updated


def _read_run_convergence(checkpoint_path: str) -> Dict[str, Any]:
    stats = load_training_stats(checkpoint_path) or {}
    info = stats.get("convergence_info") or {}
    run_dir = os.path.dirname(os.path.normpath(checkpoint_path))
    report_path = os.path.join(run_dir, "seeds", "seed_report.json")
    report: Dict[str, Any] = {}
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as handle:
            payload = json.load(handle)
            report = payload if isinstance(payload, dict) else {}

    n_convergent = info.get("convergent_count")
    if not isinstance(n_convergent, int):
        report_count = report.get("convergent_count")
        n_convergent = int(report_count) if isinstance(report_count, int) else 0

    seeds_tried = report.get("seeds_tried")
    n_tried = len(seeds_tried) if isinstance(seeds_tried, list) else info.get("max_seeds")
    if not isinstance(n_tried, int):
        n_tried = DEFAULT_MAX_SEEDS

    selected_seed = report.get("best_seed")
    selected_seed = int(selected_seed) if isinstance(selected_seed, int) else None
    converged = n_convergent >= MIN_CONVERGENT_SEEDS
    return {
        "converged": converged,
        "n_convergent_seeds": n_convergent,
        "n_seeds_tried": n_tried,
        "selected_seed": selected_seed,
    }


def _apply_convergence_fields(result: GridResult, checkpoint_path: str) -> GridResult:
    meta = _read_run_convergence(checkpoint_path)
    result.converged = bool(meta["converged"])
    result.n_convergent_seeds = int(meta["n_convergent_seeds"])
    result.n_seeds_tried = int(meta["n_seeds_tried"])
    result.selected_seed = meta["selected_seed"]
    return result


def _topk_base_global(model: Any, *, topk: int, part: str) -> Set[int]:
    local_indices = model.gradiend.get_topk_weights(part=part, topk=topk)
    if not local_indices:
        return set()
    base_map = model.gradiend._get_base_global_index_map()
    idx_t = torch.as_tensor(local_indices, dtype=torch.long)
    return {int(x) for x in base_map[idx_t].tolist()}


def _intersection_frac(a: Set[int], b: Set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _ref_recall_metrics(heuristic: Set[int], ref: Set[int]) -> Tuple[float, float]:
    """Recall/precision vs a fixed-size oracle top-k (``TOPK_EVAL``)."""
    if not ref:
        return 0.0, 0.0
    inter = len(heuristic & ref)
    recall = inter / TOPK_EVAL
    precision = inter / len(heuristic) if heuristic else 0.0
    return recall, precision


def _all_kept_base_global(model: Any) -> Set[int]:
    """Map every kept GRADIEND input dim to base-global indices."""
    local_indices = torch.arange(int(model.gradiend.input_dim), dtype=torch.long)
    base_map = model.gradiend._get_base_global_index_map()
    return {int(x) for x in base_map[local_indices].tolist()}


def _indices_path(output_dir: str, run_id: str) -> str:
    safe = run_id.replace("/", os.sep)
    return os.path.join(output_dir, "topk_indices", f"{safe}.json")


def _save_topk_indices(path: str, indices: Set[int], *, meta: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    payload = {**meta, "indices": sorted(indices)}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _load_topk_indices(path: str) -> Set[int]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return {int(x) for x in payload["indices"]}


def _evaluate_oracle_topk(
    *,
    output_dir: str,
    pair: Tuple[str, str],
    run_id: str,
    model: Any,
    pre_topk: float,
) -> Tuple[GridResult, Set[int]]:
    """Extract oracle top-k decoder weights from a trained baseline (no encoder/decoder eval)."""
    final_topk = _topk_base_global(model, topk=TOPK_EVAL, part=TOPK_PART)
    kept_dim = int(model.gradiend.input_dim)
    base_map = model.gradiend._get_base_global_index_map()
    base_input_dim = int(base_map.numel()) if base_map is not None else None

    idx_path = _indices_path(output_dir, run_id)
    _save_topk_indices(
        idx_path,
        final_topk,
        meta={
            "run_id": run_id,
            "pair": _pair_slug(pair),
            "topk_eval": TOPK_EVAL,
            "part": TOPK_PART,
            "oracle": True,
        },
    )

    result = GridResult(
        pair=_pair_slug(pair),
        run_id=run_id,
        pre_topk=pre_topk,
        kept_dim=kept_dim,
        base_input_dim=base_input_dim,
        ref_recall=1.0,
        ref_precision=1.0,
        topk_indices_file=idx_path,
    )
    return result, final_topk


def _evaluate_model(
    *,
    output_dir: str,
    pair: Tuple[str, str],
    run_id: str,
    args_base: Dict[str, Any],
    model: Any,
    ref_topk: Optional[Set[int]],
    max_size: int,
    pre_topk: float,
    pre_source: Optional[str] = None,
    pre_n_samples: Optional[int] = None,
) -> Tuple[GridResult, Set[int]]:
    eval_trainer = _build_trainer(
        run_id=run_id,
        pair=pair,
        args=TrainingArguments(**args_base),
    )
    try:
        enc_corr, decoder_metrics = _run_evals_with_model(
            eval_trainer,
            model,
            pair,
            max_size=max_size,
        )
    finally:
        eval_trainer.unload_model()

    final_topk = _topk_base_global(model, topk=TOPK_EVAL, part=TOPK_PART)
    kept_dim = int(model.gradiend.input_dim)
    base_map = model.gradiend._get_base_global_index_map()
    base_input_dim = int(base_map.numel()) if base_map is not None else None

    idx_path = _indices_path(output_dir, run_id)
    _save_topk_indices(
        idx_path,
        final_topk,
        meta={
            "run_id": run_id,
            "pair": _pair_slug(pair),
            "topk_eval": TOPK_EVAL,
            "part": TOPK_PART,
        },
    )

    ref_recall = None
    ref_precision = None
    if ref_topk is not None and final_topk:
        ref_recall, ref_precision = _ref_recall_metrics(final_topk, ref_topk)

    result = GridResult(
        pair=_pair_slug(pair),
        run_id=run_id,
        pre_topk=pre_topk,
        pre_source=pre_source,
        pre_n_samples=pre_n_samples,
        kept_dim=kept_dim,
        base_input_dim=base_input_dim,
        ref_recall=ref_recall,
        ref_precision=ref_precision,
        topk_indices_file=idx_path,
        encoder_correlation=enc_corr,
        decoder_prob_delta=decoder_metrics.get("decoder_prob_delta"),
    )
    return result, final_topk


def _run_baseline(
    *,
    output_dir: str,
    results_path: str,
    pair: Tuple[str, str],
    args_base: Dict[str, Any],
    max_size: int,
    recall_only: bool = False,
) -> Optional[Set[int]]:
    pair_key = _pair_slug(pair)
    run_id = _baseline_run_id(pair)
    results = _load_results(results_path)

    cached = next(
        (
            r
            for r in results
            if r.pair == pair_key
            and r.run_id == run_id
            and r.converged is True
            and r.topk_indices_file
        ),
        None,
    )
    if cached is not None and os.path.isfile(cached.topk_indices_file):
        logger.info("Using convergent cached reference for pair %s.", pair_key)
        return _load_topk_indices(cached.topk_indices_file)

    logger.info("Training reference (no pre-prune) for pair %s.", pair_key)
    trainer = _build_trainer(run_id=run_id, pair=pair, args=TrainingArguments(**args_base))
    checkpoint_path = _checkpoint_path(output_dir, run_id)
    try:
        with CudaMemorySpan():
            training_time = _train_and_get_time(trainer, output_dir, run_id)
    except RuntimeError as exc:
        logger.error("Reference training failed for pair %s: %s", pair_key, exc)
        failed = GridResult(
            pair=pair_key,
            run_id=run_id,
            pre_topk=1.0,
            converged=False,
            error=str(exc),
        )
        _append_result(results, failed, results_path)
        return None
    finally:
        trainer.unload_model()
        _clear_cuda_memory()

    eval_trainer = _build_trainer(
        run_id=run_id,
        pair=pair,
        args=TrainingArguments(**args_base),
    )
    model = eval_trainer.load_model(_checkpoint_path(output_dir, run_id))
    try:
        if recall_only:
            result, ref_topk = _evaluate_oracle_topk(
                output_dir=output_dir,
                pair=pair,
                run_id=run_id,
                model=model,
                pre_topk=1.0,
            )
        else:
            result, ref_topk = _evaluate_model(
                output_dir=output_dir,
                pair=pair,
                run_id=run_id,
                args_base=args_base,
                model=model,
                ref_topk=None,
                max_size=max_size,
                pre_topk=1.0,
            )
        result.training_time_s = training_time
        result.ref_recall = 1.0
        result.ref_precision = 1.0
        result = _apply_convergence_fields(result, checkpoint_path)
        if not result.converged:
            result.error = (
                f"baseline converged {result.n_convergent_seeds}/{MIN_CONVERGENT_SEEDS} seeds"
            )
    finally:
        eval_trainer.unload_model()
        del model
        _clear_cuda_memory()

    _append_result(results, result, results_path)
    if not result.converged or not result.topk_indices_file:
        return None
    return ref_topk


def _run_grid_cell(
    *,
    output_dir: str,
    results_path: str,
    pair: Tuple[str, str],
    source: str,
    n_samples: int,
    topk: float,
    args_base: Dict[str, Any],
    ref_topk: Set[int],
    max_size: int,
) -> None:
    pair_key = _pair_slug(pair)
    run_id = _grid_run_id(pair, source, n_samples, topk)
    results = _load_results(results_path)
    if _has_success(results, pair_key, run_id):
        logger.info("Skipping cached grid cell %s.", run_id)
        return

    logger.info(
        "Grid cell pair=%s source=%s n_samples=%s pre_topk=%s",
        pair_key,
        source,
        n_samples,
        f"{topk:g}",
    )
    pre_cfg = PrePruneConfig(
        n_samples=n_samples,
        topk=topk,
        source=source,
        seed=PRE_PRUNE_SEED,
    )
    train_args = TrainingArguments(
        **args_base,
        pre_prune_config=pre_cfg,
        reuse_pre_prune=True,
    )
    trainer = _build_trainer(run_id=run_id, pair=pair, args=train_args)
    checkpoint_path = _checkpoint_path(output_dir, run_id)
    try:
        with CudaMemorySpan():
            training_time_s = _train_and_get_time(trainer, output_dir, run_id)
    except RuntimeError as exc:
        logger.error("Grid cell failed convergence for %s: %s", run_id, exc)
        failed = GridResult(
            pair=pair_key,
            run_id=run_id,
            pre_topk=topk,
            pre_source=source,
            pre_n_samples=n_samples,
            converged=False,
            error=str(exc),
        )
        _append_result(results, failed, results_path)
        trainer.unload_model()
        _clear_cuda_memory()
        return
    trainer.unload_model()
    _clear_cuda_memory()

    eval_trainer = _build_trainer(
        run_id=run_id,
        pair=pair,
        args=TrainingArguments(**args_base),
    )
    model = eval_trainer.load_model(checkpoint_path)
    try:
        result, _final = _evaluate_model(
            output_dir=output_dir,
            pair=pair,
            run_id=run_id,
            args_base=args_base,
            model=model,
            ref_topk=ref_topk,
            max_size=max_size,
            pre_topk=topk,
            pre_source=source,
            pre_n_samples=n_samples,
        )
        result.training_time_s = training_time_s
        result = _apply_convergence_fields(result, checkpoint_path)
        saved_dim = load_saved_gradiend_input_dim(checkpoint_path)
        if saved_dim is not None:
            result.kept_dim = saved_dim
        if not result.converged:
            result.error = (
                f"grid cell converged {result.n_convergent_seeds}/{MIN_CONVERGENT_SEEDS} seeds"
            )
        elif not _is_valid_pruned_result(result):
            result.error = (
                f"stale full-model checkpoint (kept_dim={result.kept_dim}) for pre_topk={topk:g}"
            )
            result.converged = False
            _invalidate_checkpoint_tree(checkpoint_path)
    finally:
        eval_trainer.unload_model()
        del model
        _clear_cuda_memory()

    _append_result(results, result, results_path)


def _run_grid_cell_recall_only(
    *,
    output_dir: str,
    results_path: str,
    pair: Tuple[str, str],
    source: str,
    n_samples: int,
    topk: float,
    args_base: Dict[str, Any],
    ref_topk: Set[int],
) -> None:
    pair_key = _pair_slug(pair)
    run_id = _grid_run_id(pair, source, n_samples, topk)
    results = _load_results(results_path)
    if _has_success(results, pair_key, run_id, recall_only=True):
        logger.info("Skipping cached recall-only grid cell %s.", run_id)
        return

    logger.info(
        "Recall-only grid cell pair=%s source=%s n_samples=%s pre_topk=%s",
        pair_key,
        source,
        n_samples,
        f"{topk:g}",
    )
    pre_cfg = PrePruneConfig(
        n_samples=n_samples,
        topk=topk,
        source=source,
        seed=PRE_PRUNE_SEED,
    )
    train_args = TrainingArguments(
        **args_base,
        pre_prune_config=pre_cfg,
        reuse_pre_prune=True,
    )
    trainer = _build_trainer(run_id=run_id, pair=pair, args=train_args)
    try:
        with CudaMemorySpan():
            pre_start = time.perf_counter()
            base_model = trainer.get_model()
            base_input_dim = int(base_model.gradiend.input_dim)
            trainer.pre_prune(inplace=False)
            pre_pruning_time_s = time.perf_counter() - pre_start
        model = trainer.get_model()
        heuristic_topk = _all_kept_base_global(model)
        kept_dim = int(model.gradiend.input_dim)
        ref_recall, ref_precision = _ref_recall_metrics(heuristic_topk, ref_topk)

        idx_path = _indices_path(output_dir, run_id)
        _save_topk_indices(
            idx_path,
            heuristic_topk,
            meta={
                "run_id": run_id,
                "pair": pair_key,
                "topk_eval": TOPK_EVAL,
                "part": TOPK_PART,
                "mask_recall": True,
            },
        )

        result = GridResult(
            pair=pair_key,
            run_id=run_id,
            pre_topk=topk,
            pre_source=source,
            pre_n_samples=n_samples,
            kept_dim=kept_dim,
            base_input_dim=base_input_dim,
            ref_recall=ref_recall,
            ref_precision=ref_precision,
            topk_indices_file=idx_path,
            pre_pruning_time_s=pre_pruning_time_s,
            converged=True,
            mask_recall=True,
        )
    except Exception as exc:
        logger.error("Recall-only grid cell failed for %s: %s", run_id, exc)
        failed = GridResult(
            pair=pair_key,
            run_id=run_id,
            pre_topk=topk,
            pre_source=source,
            pre_n_samples=n_samples,
            converged=False,
            mask_recall=True,
            error=str(exc),
        )
        _append_result(results, failed, results_path)
        return
    finally:
        trainer.unload_model()
        _clear_cuda_memory()

    _append_result(results, result, results_path)


def run_full_grid(
    *,
    output_dir: str,
    results_path: str,
    pairs: Sequence[Tuple[str, str]],
    sources: Iterable[str],
    n_samples_values: Iterable[int],
    pre_topk_values: Iterable[float],
    max_seeds: int,
    max_size: int,
    recall_only: bool = True,
) -> List[GridResult]:
    args_base = _base_args(output_dir, max_seeds=max_seeds)
    ref_topk_by_pair: Dict[str, Set[int]] = {}
    pending = _missing_grid_cells(
        _load_results(results_path),
        pairs=pairs,
        sources=list(sources),
        n_samples_values=list(n_samples_values),
        pre_topk_values=list(pre_topk_values),
        recall_only=recall_only,
    )
    if pending:
        action = "pre-prune" if recall_only else "train"
        logger.info(
            "Grid has %s missing cell(s) to %s: %s",
            len(pending),
            action,
            ", ".join(f"pre_topk={k} ({n})" for k, n in _missing_topk_counts(pending).items()),
        )
    else:
        logger.info("Grid complete for requested pre_topk values (no missing cells).")

    for pair in pairs:
        ref = _run_baseline(
            output_dir=output_dir,
            results_path=results_path,
            pair=pair,
            args_base=args_base,
            max_size=max_size,
            recall_only=recall_only,
        )
        if ref is not None:
            ref_topk_by_pair[_pair_slug(pair)] = ref

    for pair in pairs:
        pair_key = _pair_slug(pair)
        ref_topk = ref_topk_by_pair.get(pair_key)
        if not ref_topk:
            continue
        for topk in pre_topk_values:
            if isinstance(topk, float) and math.isclose(topk, 1.0):
                continue
            for source in sources:
                for n_samples in n_samples_values:
                    if recall_only:
                        _run_grid_cell_recall_only(
                            output_dir=output_dir,
                            results_path=results_path,
                            pair=pair,
                            source=source,
                            n_samples=n_samples,
                            topk=topk,
                            args_base=args_base,
                            ref_topk=ref_topk,
                        )
                    else:
                        _run_grid_cell(
                            output_dir=output_dir,
                            results_path=results_path,
                            pair=pair,
                            source=source,
                            n_samples=n_samples,
                            topk=topk,
                            args_base=args_base,
                            ref_topk=ref_topk,
                            max_size=max_size,
                        )

    return _load_results(results_path)


def _row_plottable(row: GridResult, *, require_converged: bool) -> bool:
    if row.error:
        return False
    if row.converged is False:
        return False
    if require_converged and row.converged is not True:
        return False
    if not bool(row.topk_indices_file):
        return False
    if not _is_valid_pruned_result(row):
        return False
    if (
        not row.mask_recall
        and row.pre_topk is not None
        and not math.isclose(row.pre_topk, 1.0)
        and row.ref_recall is not None
        and row.ref_recall >= 0.99
    ):
        return False
    return True


def _indices_file_exists(row: GridResult) -> bool:
    path = row.topk_indices_file
    return bool(path and os.path.isfile(path))


def _summarize_configs(
    results: List[GridResult],
    *,
    require_converged: bool = True,
) -> List[ConfigSummary]:
    ok = [r for r in results if _row_plottable(r, require_converged=require_converged)]
    by_config: Dict[str, List[GridResult]] = {}
    for row in ok:
        if row.pre_topk is not None and math.isclose(row.pre_topk, 1.0):
            continue
        key = _config_key(row.pre_topk, row.pre_source, row.pre_n_samples)
        by_config.setdefault(key, []).append(row)

    summaries: List[ConfigSummary] = []
    for rows in by_config.values():
        sample = rows[0]
        topk_sets: Dict[str, Set[int]] = {}
        for row in rows:
            if _indices_file_exists(row):
                topk_sets[row.pair] = _load_topk_indices(row.topk_indices_file)

        overlaps: List[float] = []
        pair_keys = sorted(topk_sets)
        if len(pair_keys) >= 2:
            for a, b in combinations(pair_keys, 2):
                overlaps.append(_intersection_frac(topk_sets[a], topk_sets[b]))

        recalls = [r.ref_recall for r in rows if r.ref_recall is not None]
        kept = [r.kept_dim for r in rows if r.kept_dim is not None]
        encs = [r.encoder_correlation for r in rows if r.encoder_correlation is not None]
        summaries.append(
            ConfigSummary(
                pre_topk=float(sample.pre_topk),
                pre_source=sample.pre_source,
                pre_n_samples=sample.pre_n_samples,
                mean_ref_recall=float(np.mean(recalls)) if recalls else float("nan"),
                mean_kept_dim=float(np.mean(kept)) if kept else float("nan"),
                mean_cross_overlap=float(np.mean(overlaps)) if overlaps else float("nan"),
                mean_encoder_correlation=float(np.mean(encs)) if encs else float("nan"),
                n_pairs=len(rows),
            )
        )

    return sorted(
        summaries,
        key=lambda s: (s.pre_topk, s.pre_source or "", s.pre_n_samples or 0),
    )


def _summarize_baseline(results: List[GridResult], *, require_converged: bool = True) -> Optional[ConfigSummary]:
    """Aggregate no-pre-prune reference runs (pre_topk=1.0) for plot anchors."""
    baseline_rows = [
        r
        for r in results
        if _row_plottable(r, require_converged=require_converged)
        and r.pre_topk is not None
        and math.isclose(r.pre_topk, BASELINE_PRE_TOPK)
    ]
    if not baseline_rows:
        return None

    topk_sets: Dict[str, Set[int]] = {}
    for row in baseline_rows:
        if _indices_file_exists(row):
            topk_sets[row.pair] = _load_topk_indices(row.topk_indices_file)

    overlaps: List[float] = []
    pair_keys = sorted(topk_sets)
    if len(pair_keys) >= 2:
        for a, b in combinations(pair_keys, 2):
            overlaps.append(_intersection_frac(topk_sets[a], topk_sets[b]))

    kept = [r.kept_dim for r in baseline_rows if r.kept_dim is not None]
    encs = [r.encoder_correlation for r in baseline_rows if r.encoder_correlation is not None]
    return ConfigSummary(
        pre_topk=BASELINE_PRE_TOPK,
        pre_source=None,
        pre_n_samples=None,
        mean_ref_recall=BASELINE_REF_RECALL,
        mean_kept_dim=float(np.mean(kept)) if kept else float("nan"),
        mean_cross_overlap=float(np.mean(overlaps)) if overlaps else float("nan"),
        mean_encoder_correlation=float(np.mean(encs)) if encs else float("nan"),
        n_pairs=len(baseline_rows),
    )


def _summaries_with_baseline(
    results: List[GridResult],
    *,
    require_converged: bool = True,
) -> List[ConfigSummary]:
    summaries = _summarize_configs(results, require_converged=require_converged)
    baseline = _summarize_baseline(results, require_converged=require_converged)
    if baseline is not None:
        summaries = [baseline] + summaries
    return summaries


def _append_baseline_recall_anchor(
    xs: List[float],
    ys: List[float],
) -> Tuple[List[float], List[float]]:
    """Extend a pre_topk curve with the no-pre-prune reference point (1.0, recall=1)."""
    out_x = list(xs)
    out_y = list(ys)
    if out_x and math.isclose(out_x[-1], BASELINE_PRE_TOPK):
        out_y[-1] = BASELINE_REF_RECALL
        return out_x, out_y
    out_x.append(BASELINE_PRE_TOPK)
    out_y.append(BASELINE_REF_RECALL)
    return out_x, out_y


def _save_summaries(summaries: List[ConfigSummary], path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(s) for s in summaries], handle, indent=2)


def _csv_floats(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _csv_strings(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _metric_grid(
    summaries: List[ConfigSummary],
    *,
    pre_topk: float,
    sources: Sequence[str],
    n_samples_values: Sequence[int],
    metric: str,
) -> np.ndarray:
    grid = np.full((len(sources), len(n_samples_values)), np.nan, dtype=float)
    for i, source in enumerate(sources):
        for j, n_samples in enumerate(n_samples_values):
            match = next(
                (
                    s
                    for s in summaries
                    if math.isclose(s.pre_topk, pre_topk)
                    and s.pre_source == source
                    and s.pre_n_samples == n_samples
                ),
                None,
            )
            if match is not None:
                grid[i, j] = getattr(match, metric)
    return grid


def plot_metric_panels(
    summaries: List[ConfigSummary],
    *,
    metric: str,
    title: str,
    output_path: str,
    sources: Sequence[str],
    n_samples_values: Sequence[int],
    pre_topk_values: Sequence[float],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    min_pairs: int = 1,
) -> None:
    pre_topks = [v for v in pre_topk_values if not math.isclose(v, 1.0)]
    if not pre_topks:
        return

    fig, axes = plt.subplots(1, len(pre_topks), figsize=(5 * len(pre_topks), 4.5), squeeze=False)
    ims = []
    for ax, pre_topk in zip(axes.flatten(), pre_topks):
        grid = _metric_grid(
            summaries,
            pre_topk=pre_topk,
            sources=sources,
            n_samples_values=n_samples_values,
            metric=metric,
        )
        if min_pairs > 1:
            for i, source in enumerate(sources):
                for j, n_samples in enumerate(n_samples_values):
                    match = next(
                        (
                            s
                            for s in summaries
                            if math.isclose(s.pre_topk, pre_topk)
                            and s.pre_source == source
                            and s.pre_n_samples == n_samples
                        ),
                        None,
                    )
                    if match is None or match.n_pairs < min_pairs:
                        grid[i, j] = np.nan
        im = ax.imshow(
            np.ma.masked_invalid(grid),
            aspect="auto",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ims.append(im)
        for i, source in enumerate(sources):
            for j, n_samples in enumerate(n_samples_values):
                match = next(
                    (
                        s
                        for s in summaries
                        if math.isclose(s.pre_topk, pre_topk)
                        and s.pre_source == source
                        and s.pre_n_samples == n_samples
                    ),
                    None,
                )
                if match is None or math.isnan(getattr(match, metric, float("nan"))):
                    continue
                if match.n_pairs < min_pairs:
                    continue
                ax.text(
                    j,
                    i,
                    f"{match.n_pairs}",
                    ha="center",
                    va="center",
                    color="white" if getattr(match, metric) > (vmin or 0) + 0.5 * ((vmax or 1) - (vmin or 0)) else "black",
                    fontsize=7,
                )
        ax.set_xticks(range(len(n_samples_values)))
        ax.set_xticklabels([str(v) for v in n_samples_values], rotation=45, ha="right")
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources)
        ax.set_xlabel("pre n_samples")
        ax.set_ylabel("pre source")
        ax.set_title(f"pre_topk={pre_topk:g}")

    fig.suptitle(title, y=1.02)
    fig.colorbar(ims[0], ax=axes.flatten().tolist(), shrink=0.85, label=metric)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_pareto(
    summaries: List[ConfigSummary],
    *,
    output_path: str,
) -> None:
    if not summaries:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"alternative": "C0", "factual": "C1", "diff": "C2"}
    for row in summaries:
        if math.isnan(row.mean_ref_recall) or math.isnan(row.mean_kept_dim):
            continue
        is_baseline = math.isclose(row.pre_topk, BASELINE_PRE_TOPK)
        ax.scatter(
            row.mean_kept_dim,
            row.mean_ref_recall,
            c="black" if is_baseline else colors.get(row.pre_source or "", "C3"),
            s=80 if is_baseline else 30 + 5 * (row.pre_n_samples or 0),
            alpha=0.9 if is_baseline else 0.75,
            marker="*" if is_baseline else "o",
            zorder=5 if is_baseline else 3,
        )
        label = "no pre-prune" if is_baseline else (
            f"{row.pre_source} n={row.pre_n_samples} k={row.pre_topk:g}"
        )
        ax.annotate(
            label,
            (row.mean_kept_dim, row.mean_ref_recall),
            fontsize=6,
            alpha=0.8,
        )
    ax.set_xlabel("mean kept_dim (GRADIEND input size)")
    ax.set_ylabel("mean ref_recall vs no-pre reference")
    ax.set_title("Size vs top-1000 reference recall (all configs, mean over 5 pairs)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def _marker_for_n_samples(n_samples: int) -> str:
    """Distinct marker per pre-prune n_samples (redundant to line color in legends)."""
    try:
        idx = PRE_N_SAMPLES.index(n_samples)
    except ValueError:
        idx = n_samples
    return _N_SAMPLES_MARKERS[idx % len(_N_SAMPLES_MARKERS)]


def plot_ref_recall_vs_pre_topk(
    summaries: List[ConfigSummary],
    *,
    output_path: str,
    sources: Sequence[str],
    pre_topk_values: Sequence[float],
) -> None:
    pruned_topks = _decade_pre_topk_values(
        v for v in pre_topk_values if not math.isclose(v, BASELINE_PRE_TOPK)
    )
    n_samples_values = sorted(
        {s.pre_n_samples for s in summaries if s.pre_n_samples is not None}
    )
    color_by_n_samples = {
        n_samples: f"C{index % 10}"
        for index, n_samples in enumerate(n_samples_values)
    }
    fig, axes = plt.subplots(
        1,
        len(sources),
        figsize=(2 * len(sources), 3),
        squeeze=False,
        sharey=True,
    )
    for ax, source in zip(axes.flatten(), sources):
        subset = [
            s
            for s in summaries
            if s.pre_source == source
            and not math.isclose(s.pre_topk, BASELINE_PRE_TOPK)
            and _is_decade_pre_topk(s.pre_topk)
        ]
        for n_samples in sorted({s.pre_n_samples for s in subset if s.pre_n_samples is not None}):
            rows = sorted(
                [s for s in subset if s.pre_n_samples == n_samples],
                key=lambda s: s.pre_topk,
            )
            xs = [s.pre_topk for s in rows]
            ys = [s.mean_ref_recall for s in rows]
            xs, ys = _append_baseline_recall_anchor(xs, ys)
            ax.plot(
                xs,
                ys,
                marker=_marker_for_n_samples(n_samples),
                linestyle="-",
                markersize=3,
                color=color_by_n_samples[n_samples],
                label=str(n_samples),
            )
        ax.set_xscale("log")
        lo = min(pruned_topks) if pruned_topks else 0.01
        ax.set_xlim(lo * 0.85, BASELINE_PRE_TOPK * 1.05)
        ax.set_ylim(0.0, 1.03)
        ax.set_xticks(sorted(set(pruned_topks + [BASELINE_PRE_TOPK])))
        ax.set_xticklabels(
            [f"{value:g}" for value in sorted(set(pruned_topks + [BASELINE_PRE_TOPK]))]
        )
        ax.set_xlabel("Pre-pruning Top-$k$")
        ax.set_title(f"source={source}")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.supylabel("Recall", x=0.05)
    if n_samples_values:
        legend_handles = [
            Line2D([], [], linestyle="none", marker="", label="n_samples")
        ] + [
            Line2D(
                [],
                [],
                color=color_by_n_samples[n_samples],
                marker=_marker_for_n_samples(n_samples),
                linestyle="-",
                markersize=3,
                label=str(n_samples),
            )
            for n_samples in n_samples_values
        ]
        legend = fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=len(legend_handles),
            frameon=True,
            fontsize=10,
            handlelength=1.5,
            handletextpad=0.45,
            columnspacing=1.1,
        )
    fig.tight_layout(rect=(0.025, 0.0, 1.0, 0.82))
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_coverage_panels(
    summaries: List[ConfigSummary],
    *,
    output_path: str,
    sources: Sequence[str],
    n_samples_values: Sequence[int],
    pre_topk_values: Sequence[float],
    n_pairs_total: int = len(DER_DIE_PAIRS),
) -> None:
    pre_topks = [v for v in pre_topk_values if not math.isclose(v, 1.0)]
    if not pre_topks:
        return
    fig, axes = plt.subplots(1, len(pre_topks), figsize=(5 * len(pre_topks), 4.5), squeeze=False)
    for ax, pre_topk in zip(axes.flatten(), pre_topks):
        grid = _metric_grid(
            summaries,
            pre_topk=pre_topk,
            sources=sources,
            n_samples_values=n_samples_values,
            metric="n_pairs",
        )
        grid = grid / float(n_pairs_total)
        im = ax.imshow(grid, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_xticks(range(len(n_samples_values)))
        ax.set_xticklabels([str(v) for v in n_samples_values], rotation=45, ha="right")
        ax.set_yticks(range(len(sources)))
        ax.set_yticklabels(sources)
        ax.set_xlabel("pre n_samples")
        ax.set_ylabel("pre source")
        ax.set_title(f"pre_topk={pre_topk:g}")
        fig.colorbar(im, ax=ax, shrink=0.85, label=f"fraction of {n_pairs_total} pairs")
    fig.suptitle("Grid coverage (completed convergent pairs per cell)", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def plot_cross_overlap_pareto(
    summaries: List[ConfigSummary],
    *,
    output_path: str,
    min_pairs: int = 2,
) -> None:
    rows = [
        s
        for s in summaries
        if s.n_pairs >= min_pairs and not math.isnan(s.mean_cross_overlap)
    ]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"alternative": "C0", "factual": "C1", "diff": "C2"}
    for row in rows:
        ax.scatter(
            row.mean_kept_dim,
            row.mean_cross_overlap,
            c=colors.get(row.pre_source or "", "C3"),
            s=30 + 8 * row.n_pairs,
            alpha=0.75,
        )
        ax.annotate(
            f"{row.pre_source} n={row.pre_n_samples} k={row.pre_topk:g} ({row.n_pairs}p)",
            (row.mean_kept_dim, row.mean_cross_overlap),
            fontsize=6,
            alpha=0.8,
        )
    ax.set_xlabel("mean kept_dim")
    ax.set_ylabel("mean cross-GRADIEND top-1000 overlap")
    ax.set_title("Compression vs cross-pair weight overlap (partial grid OK)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(fig, output_path)
    plt.close(fig)


def write_ongoing_report(
    results: List[GridResult],
    *,
    output_path: str,
    sources: Sequence[str],
    n_samples_values: Sequence[int],
    pre_topk_values: Sequence[float],
) -> Dict[str, Any]:
    expected = 1 + len(sources) * len(n_samples_values) * (len(pre_topk_values) - 1)
    by_pair: Dict[str, Dict[str, int]] = {}
    suspicious: List[str] = []
    for pair in DER_DIE_PAIRS:
        slug = _pair_slug(pair)
        sub = [r for r in results if r.pair == slug]
        by_pair[slug] = {
            "total_rows": len(sub),
            "converged_strict": sum(1 for r in sub if _row_plottable(r, require_converged=True)),
            "expected": expected,
        }
    for row in results:
        if row.pre_topk is not None and not math.isclose(row.pre_topk, 1.0):
            stale_dim = row.kept_dim is not None and row.kept_dim > STALE_PRUNED_INPUT_DIM_THRESHOLD
            stale_recall = row.ref_recall is not None and row.ref_recall >= 0.99
            if stale_dim or stale_recall:
                suspicious.append(
                    f"{row.pair} {row.run_id} kept_dim={row.kept_dim} ref_recall={row.ref_recall} "
                    f"(pre_topk={row.pre_topk:g})"
                )

    summaries = _summaries_with_baseline(results, require_converged=True)
    top_cross = sorted(
        [s for s in summaries if s.n_pairs >= 2 and not math.isnan(s.mean_cross_overlap)],
        key=lambda s: (-s.mean_cross_overlap, -s.n_pairs),
    )[:10]

    missing = _missing_grid_cells(
        results,
        pairs=DER_DIE_PAIRS,
        sources=sources,
        n_samples_values=n_samples_values,
        pre_topk_values=pre_topk_values,
    )
    missing_by_topk = _missing_topk_counts(missing)

    payload: Dict[str, Any] = {
        "pairs": by_pair,
        "missing_cells": len(missing),
        "missing_by_pre_topk": missing_by_topk,
        "suspicious_full_dim_pruned_runs": suspicious,
        "top_cross_overlap_configs": [asdict(s) for s in top_cross],
    }
    if missing_by_topk:
        logger.warning(
            "Grid incomplete: %s missing cell(s) — %s",
            len(missing),
            ", ".join(f"pre_topk={k}: {n}" for k, n in missing_by_topk.items()),
        )
    _ensure_dir(os.path.dirname(output_path) or ".")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote ongoing report to %s", output_path)
    return payload


def plot_all(
    results: List[GridResult],
    *,
    output_dir: str,
    sources: Sequence[str],
    n_samples_values: Sequence[int],
    pre_topk_values: Sequence[float],
    require_converged: bool = True,
) -> List[ConfigSummary]:
    write_ongoing_report(
        results,
        output_path=os.path.join(output_dir, "ongoing_report.json"),
        sources=sources,
        n_samples_values=n_samples_values,
        pre_topk_values=pre_topk_values,
    )
    summaries = _summaries_with_baseline(results, require_converged=require_converged)
    summary_path = os.path.join(output_dir, "pre_topk_grid_summaries.json")
    _save_summaries(summaries, summary_path)
    logger.info("Wrote %s config summaries to %s.", len(summaries), summary_path)

    if False:
        plot_metric_panels(
            summaries,
            metric="mean_cross_overlap",
            title="Cross-GRADIEND top-1000 overlap (cell labels = n pairs; NaN = <2 pairs)",
            output_path=os.path.join(output_dir, "grid_mean_cross_overlap_heatmap.pdf"),
            sources=sources,
            n_samples_values=n_samples_values,
            pre_topk_values=pre_topk_values,
            vmin=0.0,
            vmax=1.0,
            min_pairs=2,
        )
        plot_coverage_panels(
            summaries,
            output_path=os.path.join(output_dir, "grid_coverage_heatmap.pdf"),
            sources=sources,
            n_samples_values=n_samples_values,
            pre_topk_values=pre_topk_values,
        )
        plot_cross_overlap_pareto(
            summaries,
            output_path=os.path.join(output_dir, "grid_pareto_kept_dim_vs_cross_overlap.pdf"),
        )
        plot_metric_panels(
            summaries,
            metric="mean_ref_recall",
            title="Within-GRADIEND reference recall (appendix; vs no-pre top-1000)",
            output_path=os.path.join(output_dir, "grid_mean_ref_recall_heatmap.pdf"),
            sources=sources,
            n_samples_values=n_samples_values,
            pre_topk_values=pre_topk_values,
            vmin=0.0,
            vmax=1.0,
        )
        plot_metric_panels(
            summaries,
            metric="mean_kept_dim",
            title="Mean GRADIEND kept_dim",
            output_path=os.path.join(output_dir, "grid_mean_kept_dim_heatmap.pdf"),
            sources=sources,
            n_samples_values=n_samples_values,
            pre_topk_values=pre_topk_values,
        )
        plot_pareto(
            summaries,
            output_path=os.path.join(output_dir, "grid_pareto_kept_dim_vs_ref_recall.pdf"),
        )
    plot_ref_recall_vs_pre_topk(
        summaries,
        output_path=os.path.join(output_dir, "grid_ref_recall_vs_pre_topk_by_source.pdf"),
        sources=sources,
        pre_topk_values=pre_topk_values,
    )
    logger.info("Refreshed plots in %s", output_dir)
    return summaries


DEFAULT_OUTPUT_DIR = os.path.join("runs", "gender_de_pre_topk_ablation")
FULL_GRID_OUTPUT_DIR = os.path.join("runs", "gender_de_pre_topk_ablation_full_grid")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["run", "plot", "both", "migrate"],
        default="both",
        help=(
            "'run' = run grid then refresh plots; 'plot' = figures only; "
            "'both' = same as run; "
            "'migrate' = fast stale-cache cleanup / convergence backfill only (no training)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument("--results-path", default=None)
    parser.add_argument(
        "--pair",
        type=_parse_pair,
        default=None,
        help=(
            "Run one der/die use-case, e.g. masc_nom:fem_nom or masc_nom_fem_nom. "
            "When omitted, all built-in pairs run sequentially."
        ),
    )
    parser.add_argument(
        "--pairs",
        type=_csv_pairs,
        default=None,
        help=(
            "Comma-separated use-cases to run. Values may be aliases like "
            "masc_nom_fem_nom or SOURCE:TARGET."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--recall-only",
        dest="full_grid",
        action="store_false",
        help="Mask recall via pre_prune only (default).",
    )
    mode_group.add_argument(
        "--full-grid",
        dest="full_grid",
        action="store_true",
        help=(
            "Train GRADIEND for every grid cell (multi-seed) and compute post-training "
            "ref_recall / cross_overlap."
        ),
    )
    parser.set_defaults(full_grid=False)
    parser.add_argument("--pre-n-samples", type=_csv_ints, default=PRE_N_SAMPLES)
    parser.add_argument("--pre-sources", type=_csv_strings, default=PRE_SOURCES)
    parser.add_argument("--pre-topk-values", type=_csv_floats, default=PRE_TOPK_VALUES,
                        help="Default: decade grid 1, 0.1, 0.01, 0.001.")
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=DEFAULT_MAX_SEEDS,
        help=f"Must be >= min_convergent_seeds ({MIN_CONVERGENT_SEEDS}); values below that are raised to {DEFAULT_MAX_SEEDS}.",
    )
    parser.add_argument("--eval-max-size", type=int, default=200)
    parser.add_argument(
        "--plot-include-legacy",
        action="store_true",
        help="Include legacy rows without converged=True in plots (default: convergent-only).",
    )
    parser.add_argument(
        "--migrate-stale-cache",
        action="store_true",
        help=(
            "Drop result rows and checkpoints where pre_topk<1 but kept_dim "
            "indicates an unpruned full model. One-time migration; prefer --mode migrate."
        ),
    )
    parser.add_argument(
        "--migrate-dry-run",
        action="store_true",
        help="With --mode migrate: list stale cells without deleting anything.",
    )
    parser.add_argument(
        "--backfill-convergence",
        action="store_true",
        help=(
            "With --mode migrate: set converged/n_seeds fields from existing checkpoints "
            "without retraining (skips cells with no checkpoint on disk)."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    recall_only = not args.full_grid
    if args.full_grid and args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = FULL_GRID_OUTPUT_DIR
    if args.pair is not None and args.pairs is not None:
        raise ValueError("Use either --pair or --pairs, not both.")
    pairs = [args.pair] if args.pair is not None else (args.pairs or list(DER_DIE_PAIRS))
    if args.results_path is None and len(pairs) == 1:
        results_path = _default_pair_results_path(args.output_dir, pairs[0])
    else:
        results_path = args.results_path or _combined_results_path(args.output_dir)
    _ensure_dir(args.output_dir)
    if args.mode in ("run", "both") and args.results_path is None and len(pairs) == 1:
        _seed_pair_results_from_combined(
            output_dir=args.output_dir,
            results_path=results_path,
            pair=pairs[0],
        )

    discovered_paths = _discover_result_paths(args.output_dir, args.results_path)

    if args.mode == "migrate" or args.migrate_stale_cache:
        n_invalidated = migrate_stale_caches(
            output_dir=args.output_dir,
            result_paths=discovered_paths if discovered_paths else [results_path],
            dry_run=bool(args.migrate_dry_run),
        )
        logger.info(
            "%s %s stale grid cell(s).",
            "Would invalidate" if args.migrate_dry_run else "Stale-cache migration invalidated",
            n_invalidated,
        )
        if args.backfill_convergence or args.mode == "migrate":
            n_backfill = backfill_convergence_from_checkpoints(
                output_dir=args.output_dir,
                result_paths=discovered_paths if discovered_paths else [results_path],
            )
            logger.info("Backfilled convergence metadata for %s row(s).", n_backfill)
        if args.mode == "migrate":
            return
        if args.migrate_stale_cache:
            logger.info(
                "Continuing to training. Omit --migrate-stale-cache on future runs to skip this step."
            )

    if args.mode == "plot":
        results = _load_results_many(discovered_paths)
    else:
        results = _load_results(results_path)
    if args.mode in ("run", "both"):
        max_seeds = _resolve_max_seeds(args.max_seeds)
        run_full_grid(
            output_dir=args.output_dir,
            results_path=results_path,
            pairs=pairs,
            sources=args.pre_sources,
            n_samples_values=args.pre_n_samples,
            pre_topk_values=args.pre_topk_values,
            max_seeds=max_seeds,
            max_size=args.eval_max_size,
            recall_only=recall_only,
        )
        results = _load_results_many(_discover_result_paths(args.output_dir, args.results_path))

    if args.mode in ("plot", "run", "both"):
        plot_all(
            results,
            output_dir=args.output_dir,
            sources=args.pre_sources,
            n_samples_values=args.pre_n_samples,
            pre_topk_values=args.pre_topk_values,
            require_converged=not args.plot_include_legacy,
        )


if __name__ == "__main__":
    main()
