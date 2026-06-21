"""
Full pre-prune grid for German der<->die GRADIENDs: pre_topk x source x n_samples.

Trains all 5 der<->die pairs for every grid cell. Reference arm per pair:
pre_topk=1.0 (no pre/post prune). Metrics vs reference top-1000 decoder weights
(base-global indices, heatmap-compatible):

- ref_recall: |T_final ∩ T_ref| / topk_eval
- kept_dim: GRADIEND input size after pre-prune
- cross_overlap: mean pairwise top-k intersection_frac across the 5 pairs

Results and per-run top-k index files are saved incrementally (resume-safe).

Example:
  python experiments/gender_de_pre_prune_topk_ablation.py --mode run
  python experiments/gender_de_pre_prune_topk_ablation.py --mode run --pair masc_nom_fem_nom
  python experiments/gender_de_pre_prune_topk_ablation.py --mode plot
  python experiments/gender_de_pre_prune_topk_ablation.py --mode both
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PrePruneConfig
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

PRE_SOURCES = ["alternative", "factual", "diff"]
PRE_N_SAMPLES = [1, 2, 4, 8, 16, 32, 64]
PRE_TOPK_VALUES = [1.0, 0.1, 0.01, 0.001]
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
        results.append(GridResult(**row))
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


def _base_args(output_dir: str, *, max_seeds: int) -> Dict[str, Any]:
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
        min_convergent_seeds=1,
        source="alternative",
        target="diff",
        eval_batch_size=8,
        learning_rate=1e-5,
        use_cache=True,
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


def _has_success(results: List[GridResult], pair: str, run_id: str) -> bool:
    return any(
        r.pair == pair and r.run_id == run_id and r.topk_indices_file
        for r in results
    )


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
        inter = len(final_topk & ref_topk)
        ref_recall = inter / TOPK_EVAL
        ref_precision = inter / len(final_topk)

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
) -> Optional[Set[int]]:
    pair_key = _pair_slug(pair)
    run_id = _baseline_run_id(pair)
    results = _load_results(results_path)

    cached = next(
        (r for r in results if r.pair == pair_key and r.run_id == run_id and r.topk_indices_file),
        None,
    )
    if cached is not None and os.path.isfile(cached.topk_indices_file):
        logger.info("Using cached reference for pair %s.", pair_key)
        return _load_topk_indices(cached.topk_indices_file)

    logger.info("Training reference (no pre-prune) for pair %s.", pair_key)
    trainer = _build_trainer(run_id=run_id, pair=pair, args=TrainingArguments(**args_base))
    with CudaMemorySpan():
        training_time = _train_and_get_time(trainer, output_dir, run_id)
    trainer.unload_model()
    _clear_cuda_memory()

    eval_trainer = _build_trainer(
        run_id=run_id,
        pair=pair,
        args=TrainingArguments(**args_base),
    )
    model = eval_trainer.load_model(_checkpoint_path(output_dir, run_id))
    try:
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
    finally:
        eval_trainer.unload_model()
        del model
        _clear_cuda_memory()

    _append_result(results, result, results_path)
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
    train_args = TrainingArguments(**args_base, pre_prune_config=pre_cfg)
    trainer = _build_trainer(run_id=run_id, pair=pair, args=train_args)
    pre_start = time.perf_counter()
    with CudaMemorySpan():
        trainer.pre_prune(inplace=True)
    pre_pruning_time_s = time.perf_counter() - pre_start
    trainer._training_args = TrainingArguments(**args_base)
    with CudaMemorySpan():
        training_time_s = _train_and_get_time(trainer, output_dir, run_id)
    trainer.unload_model()
    _clear_cuda_memory()

    eval_trainer = _build_trainer(
        run_id=run_id,
        pair=pair,
        args=TrainingArguments(**args_base),
    )
    model = eval_trainer.load_model(_checkpoint_path(output_dir, run_id))
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
        result.pre_pruning_time_s = pre_pruning_time_s
    finally:
        eval_trainer.unload_model()
        del model
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
) -> List[GridResult]:
    args_base = _base_args(output_dir, max_seeds=max_seeds)
    ref_topk_by_pair: Dict[str, Set[int]] = {}

    for pair in pairs:
        ref = _run_baseline(
            output_dir=output_dir,
            results_path=results_path,
            pair=pair,
            args_base=args_base,
            max_size=max_size,
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


def _summarize_configs(results: List[GridResult]) -> List[ConfigSummary]:
    ok = [r for r in results if r.topk_indices_file]
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
            if row.topk_indices_file and os.path.isfile(row.topk_indices_file):
                topk_sets[row.pair] = _load_topk_indices(row.topk_indices_file)

        overlaps: List[float] = []
        pair_keys = sorted(topk_sets)
        for a, b in combinations(pair_keys, 2):
            overlaps.append(_intersection_frac(topk_sets[a], topk_sets[b]))

        recalls = [r.ref_recall for r in rows if r.ref_recall is not None]
        kept = [r.kept_dim for r in rows if r.kept_dim is not None]
        summaries.append(
            ConfigSummary(
                pre_topk=float(sample.pre_topk),
                pre_source=sample.pre_source,
                pre_n_samples=sample.pre_n_samples,
                mean_ref_recall=float(np.mean(recalls)) if recalls else float("nan"),
                mean_kept_dim=float(np.mean(kept)) if kept else float("nan"),
                mean_cross_overlap=float(np.mean(overlaps)) if overlaps else float("nan"),
                n_pairs=len(rows),
            )
        )

    return sorted(
        summaries,
        key=lambda s: (s.pre_topk, s.pre_source or "", s.pre_n_samples or 0),
    )


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
        im = ax.imshow(grid, aspect="auto", origin="lower", vmin=vmin, vmax=vmax)
        ims.append(im)
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
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight")
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
        ax.scatter(
            row.mean_kept_dim,
            row.mean_ref_recall,
            c=colors.get(row.pre_source or "", "C3"),
            s=30 + 5 * (row.pre_n_samples or 0),
            alpha=0.75,
        )
        ax.annotate(
            f"{row.pre_source} n={row.pre_n_samples} k={row.pre_topk:g}",
            (row.mean_kept_dim, row.mean_ref_recall),
            fontsize=6,
            alpha=0.8,
        )
    ax.set_xlabel("mean kept_dim (GRADIEND input size)")
    ax.set_ylabel("mean ref_recall vs no-pre reference")
    ax.set_title("Size vs top-1000 reference recall (all configs, mean over 5 pairs)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_ref_recall_vs_pre_topk(
    summaries: List[ConfigSummary],
    *,
    output_path: str,
    sources: Sequence[str],
) -> None:
    pre_topks = sorted({s.pre_topk for s in summaries if not math.isclose(s.pre_topk, 1.0)})
    fig, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 4), squeeze=False)
    for ax, source in zip(axes.flatten(), sources):
        subset = [s for s in summaries if s.pre_source == source]
        for n_samples in sorted({s.pre_n_samples for s in subset if s.pre_n_samples is not None}):
            rows = sorted(
                [s for s in subset if s.pre_n_samples == n_samples],
                key=lambda s: s.pre_topk,
            )
            xs = [s.pre_topk for s in rows]
            ys = [s.mean_ref_recall for s in rows]
            ax.plot(xs, ys, marker="o", label=f"n={n_samples}")
        ax.set_xscale("log")
        ax.set_xlabel("pre_topk")
        ax.set_ylabel("mean ref_recall")
        ax.set_title(f"source={source}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Reference top-1000 recall vs pre_topk", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_all(
    results: List[GridResult],
    *,
    output_dir: str,
    sources: Sequence[str],
    n_samples_values: Sequence[int],
    pre_topk_values: Sequence[float],
) -> List[ConfigSummary]:
    summaries = _summarize_configs(results)
    summary_path = os.path.join(output_dir, "pre_topk_grid_summaries.json")
    _save_summaries(summaries, summary_path)
    logger.info("Wrote %s config summaries to %s.", len(summaries), summary_path)

    plot_metric_panels(
        summaries,
        metric="mean_ref_recall",
        title="Mean reference top-1000 recall (5 der<->die pairs)",
        output_path=os.path.join(output_dir, "grid_mean_ref_recall_heatmap.pdf"),
        sources=sources,
        n_samples_values=n_samples_values,
        pre_topk_values=pre_topk_values,
        vmin=0.0,
        vmax=1.0,
    )
    plot_metric_panels(
        summaries,
        metric="mean_cross_overlap",
        title="Mean pairwise top-1000 overlap across 5 pairs (heatmap proxy)",
        output_path=os.path.join(output_dir, "grid_mean_cross_overlap_heatmap.pdf"),
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
    )
    return summaries


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["run", "plot", "both"],
        default="both",
        help="'run' = train grid; 'plot' = figures from results; 'both' = run then plot. No post-prune ablation.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("runs", "gender_de_pre_topk_ablation"),
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
    parser.add_argument("--pre-n-samples", type=_csv_ints, default=PRE_N_SAMPLES)
    parser.add_argument("--pre-sources", type=_csv_strings, default=PRE_SOURCES)
    parser.add_argument("--pre-topk-values", type=_csv_floats, default=PRE_TOPK_VALUES)
    parser.add_argument("--max-seeds", type=int, default=1)
    parser.add_argument("--eval-max-size", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
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

    if args.mode == "plot":
        results = _load_results_many(_discover_result_paths(args.output_dir, args.results_path))
    else:
        results = _load_results(results_path)
    if args.mode in ("run", "both"):
        results = run_full_grid(
            output_dir=args.output_dir,
            results_path=results_path,
            pairs=pairs,
            sources=args.pre_sources,
            n_samples_values=args.pre_n_samples,
            pre_topk_values=args.pre_topk_values,
            max_seeds=args.max_seeds,
            max_size=args.eval_max_size,
        )
        if args.mode == "both":
            results = _load_results_many(_discover_result_paths(args.output_dir, args.results_path))

    if args.mode in ("plot", "both"):
        plot_all(
            results,
            output_dir=args.output_dir,
            sources=args.pre_sources,
            n_samples_values=args.pre_n_samples,
            pre_topk_values=args.pre_topk_values,
        )


if __name__ == "__main__":
    main()
