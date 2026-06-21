"""
German DE pruning analysis over a pre_topk x post_topk pruning grid.

Goal: find a good default topk proportion balancing efficiency vs recall.
- Pre-prune heuristic quality: compare pre-prune mask vs weight-based top-k mask (recall/precision/F1).
- Encoder metric: correlation from trainer.evaluate_encoder().
- Decoder metric: <target_class> from trainer.evaluate_decoder().

For efficiency, each pre_topk model is trained once without post-pruning. The saved
checkpoint is then loaded repeatedly and post-pruned for every post_topk value.
topk=1.0 means "no pruning" on either axis.
Pairs are filtered before training so the final pre+post GRADIEND input space
keeps at least 10 represented base-model weights.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PrePruneConfig, PostPruneConfig
from gradiend.trainer.core.stats import load_training_stats
from gradiend.trainer.core.pruning import post_prune
from gradiend.util.logging import get_logger
from gradiend.util.paths import ARTIFACT_MODEL, resolve_output_path

logger = get_logger(__name__)


REQUIRED_RESULT_FIELDS = (
    "decoder_prob_delta",
    "decoder_other_prob",
    "decoder_other_prob_delta",
    "post_pruning_time_s",
)


class PruningAnalysisTrainer(TextPredictionTrainer):
    """Experiment-only trainer with local encoder-analysis timing."""

    last_encoder_analysis_time_s: Optional[float] = None

    def _analyze_encoder(self, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return super()._analyze_encoder(*args, **kwargs)
        finally:
            self.last_encoder_analysis_time_s = time.perf_counter() - start


@dataclass
class RunResult:
    pre_topk: float
    post_topk: float
    encoder_correlation: float
    decoder_prob: float
    decoder_prob_delta: Optional[float] = None
    decoder_other_prob: Optional[float] = None
    decoder_other_prob_delta: Optional[float] = None
    gradiend_num_params: Optional[int] = None
    base_model_num_params: Optional[int] = None
    training_time_s: Optional[float] = None
    post_pruning_time_s: Optional[float] = None
    encoding_inference_time_s: Optional[float] = None
    heuristic_recall: Optional[float] = None
    heuristic_precision: Optional[float] = None
    heuristic_f1: Optional[float] = None


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _clear_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _checkpoint_path(output_dir: str, run_id: str) -> str:
    """Resolve model checkpoint path for a run (matches trainer output layout)."""
    exp_dir = os.path.join(output_dir.rstrip("/\\"), str(run_id).strip("/\\"))
    path = resolve_output_path(exp_dir, None, ARTIFACT_MODEL)
    if path is None:
        raise ValueError(f"Could not resolve checkpoint path for run_id={run_id!r}")
    return path


def _format_topk(topk: float) -> str:
    """Stable topk token for run IDs; keeps tiny proportions distinct."""
    return f"{topk:.6f}".replace(".", "_")


def _pre_run_id(pre_topk: float) -> str:
    return f"prune_pre_topk_{_format_topk(pre_topk)}"


def _kept_dim(input_dim: int, topk: float) -> int:
    """Return kept GRADIEND input dimensions for the repo's topk semantics."""
    if isinstance(topk, float) and topk == 1.0:
        return int(input_dim)
    if isinstance(topk, float):
        return int(math.ceil(topk * input_dim))
    return min(int(topk), int(input_dim))


def _final_dim_after_pre_post(base_dim: int, pre_topk: float, post_topk: float) -> int:
    pre_dim = _kept_dim(base_dim, pre_topk)
    return _kept_dim(pre_dim, post_topk)


def _format_topk_list(values: List[float]) -> str:
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"


def _suitable_topk_pairs(
    topk_values: List[float],
    *,
    base_dim: int,
    min_final_dim: int = 10,
) -> Dict[float, List[float]]:
    return {
        pre_topk: [
            post_topk
            for post_topk in topk_values
            if _final_dim_after_pre_post(base_dim, pre_topk, post_topk) >= min_final_dim
        ]
        for pre_topk in topk_values
        if _kept_dim(base_dim, pre_topk) >= min_final_dim
    }


def _topk_mask(importance: torch.Tensor, topk: float) -> torch.Tensor:
    """topk=1.0 (float) means keep all (return all True). topk int = keep top-k dims."""
    if not torch.is_tensor(importance):
        raise TypeError("importance must be a torch.Tensor")
    flat = importance.detach().flatten()
    n = flat.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)
    if isinstance(topk, float) and topk == 1.0:
        return torch.ones(n, dtype=torch.bool)
    if isinstance(topk, float):
        if not (0.0 < topk <= 1.0):
            raise ValueError("topk must be in (0,1] when float (use 1.0 for no pruning)")
        k = int(math.ceil(topk * n))
    else:
        k = int(topk)
    k = max(1, min(k, n))
    _, idx = torch.topk(flat, k=k, largest=True, sorted=False)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx] = True
    return mask


def _mask_metrics(heuristic_mask: torch.Tensor, oracle_mask: torch.Tensor) -> Tuple[float, float, float]:
    h = heuristic_mask.flatten().to(dtype=torch.bool)
    o = oracle_mask.flatten().to(dtype=torch.bool)
    if h.numel() != o.numel():
        raise ValueError("Mask sizes do not match")
    n_heuristic = int(h.sum().item())
    n_oracle = int(o.sum().item())
    if n_oracle == 0:
        return 0.0, 0.0, 0.0
    tp = int((h & o).sum().item())
    recall = tp / n_oracle if n_oracle > 0 else 0.0
    precision = tp / n_heuristic if n_heuristic > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return recall, precision, f1


def _run_encoder_eval(
    trainer: TextPredictionTrainer, max_size: int = 200, use_cache: bool = True
) -> float:
    result = trainer.evaluate_encoder(max_size=max_size, use_cache=use_cache)
    return float(result.get("correlation", 0.0)) if result else 0.0


def _timed_run_encoder_eval(
    trainer: TextPredictionTrainer,
    max_size: int = 200,
    use_cache: bool = True,
) -> Tuple[float, float]:
    if hasattr(trainer, "last_encoder_analysis_time_s"):
        trainer.last_encoder_analysis_time_s = None
    start = time.perf_counter()
    result = _run_encoder_eval(trainer, max_size=max_size, use_cache=use_cache)
    elapsed = time.perf_counter() - start
    trainer_elapsed = getattr(trainer, "last_encoder_analysis_time_s", None)
    return result, float(trainer_elapsed) if isinstance(trainer_elapsed, (int, float)) else elapsed


def _run_decoder_eval(
    trainer: TextPredictionTrainer,
    pair: Tuple[str, str],
    max_size: int = 200,
    use_cache: bool = True,
) -> Dict[str, float]:
    source_class, target_class = pair
    result = trainer.evaluate_decoder(
        max_size_training_like=max_size,
        max_size_neutral=max_size,
        use_cache=use_cache,
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

        raise KeyError(
            "Decoder eval did not contain selected summary value for class %r. Top-level keys: %s"
            % (class_name, sorted(str(k) for k in result.keys()))
        )

    def _grid_rows() -> List[Dict[str, Any]]:
        if isinstance(result.get("grid"), dict):
            return [row for row in result["grid"].values() if isinstance(row, dict)]
        if isinstance(result.get("results"), list):
            return [row for row in result["results"] if isinstance(row, dict)]
        raise KeyError(
            "Decoder eval did not contain grid/results rows needed for base probabilities. Top-level keys: %s"
            % sorted(str(k) for k in result.keys())
        )

    rows = _grid_rows()
    base_rows = [row for row in rows if row.get("id") == "base"]
    if len(base_rows) != 1:
        raise KeyError(f"Expected exactly one base decoder grid row, found {len(base_rows)}.")

    base_probs_by_dataset = base_rows[0].get("probs_by_dataset")
    if not isinstance(base_probs_by_dataset, dict):
        raise KeyError("Base decoder grid row does not contain a 'probs_by_dataset' dict.")

    def _prob_by_dataset(row: Dict[str, Any], dataset_class: str, predicted_class: str) -> float:
        probs_by_dataset = row.get("probs_by_dataset")
        if not isinstance(probs_by_dataset, dict):
            raise KeyError("Decoder grid row does not contain a 'probs_by_dataset' dict.")
        dataset_probs = probs_by_dataset.get(dataset_class)
        if not isinstance(dataset_probs, dict):
            raise KeyError(
                "Decoder probs_by_dataset missing dataset class %r. Available dataset keys: %s"
                % (dataset_class, sorted(str(k) for k in probs_by_dataset.keys()))
            )
        value = dataset_probs.get(predicted_class)
        if not isinstance(value, (int, float)):
            raise KeyError(
                "Decoder probs_by_dataset[%r] missing predicted class %r. Available predicted keys: %s"
                % (dataset_class, predicted_class, sorted(str(k) for k in dataset_probs.keys()))
            )
        return float(value)

    # For source -> target, compare P(target | source articles) against the base
    # model's P(target | source articles). The reverse metric is P(source | target articles).
    target_base_prob = _prob_by_dataset(base_rows[0], source_class, target_class)
    other_base_prob = _prob_by_dataset(base_rows[0], target_class, source_class)

    target_prob = _summary_value(target_class)
    other_prob = _summary_value(source_class)
    return {
        "decoder_prob": target_prob,
        "decoder_prob_delta": target_prob - float(target_base_prob),
        "decoder_other_prob": other_prob,
        "decoder_other_prob_delta": other_prob - float(other_base_prob),
    }


def _run_evals_with_model(
    trainer: TextPredictionTrainer,
    model: Any,
    pair: Tuple[str, str],
    max_size: int = 200,
) -> Tuple[float, Dict[str, float], float]:
    """Run encoder and decoder eval using the given model (injected into trainer)."""
    prev_instance = trainer._model_instance
    trainer._model_instance = model
    try:
        enc, enc_time = _timed_run_encoder_eval(trainer, max_size=max_size, use_cache=False)
        dec = _run_decoder_eval(trainer, pair=pair, max_size=max_size, use_cache=False)
        return enc, dec, enc_time
    finally:
        trainer._model_instance = prev_instance


def _num_params(module: Any) -> Optional[int]:
    if module is None or not hasattr(module, "parameters"):
        return None
    return int(sum(p.numel() for p in module.parameters()))


def _model_param_counts(model: Any) -> Tuple[Optional[int], Optional[int]]:
    gradiend = getattr(model, "gradiend", None)
    base_model = getattr(model, "base_model", None)
    return _num_params(gradiend), _num_params(base_model)


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


def _build_trainer(
    *,
    run_id: str,
    pair: Tuple[str, str],
    args: TrainingArguments,
) -> TextPredictionTrainer:
    return PruningAnalysisTrainer(
        model="bert-base-german-cased",
        #model="deepset/gbert-base",
        run_id=run_id,
        data="aieng-lab/de-gender-case-articles",
        target_classes=list(pair),
        masked_col="masked",
        split_col="split",
        eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
        args=args,
    )


def _compute_oracle_mask(
    trainer: TextPredictionTrainer,
    topk: float,
    part: str = "decoder-weight",
) -> torch.Tensor:
    """Compute the weight-based top-k mask (oracle) for pre-prune heuristic quality."""
    model = trainer.get_model()
    importance = model.get_weight_importance(part=part).to("cpu")
    return _topk_mask(importance, topk)


def _load_existing_results(results_path: str) -> List[RunResult]:
    """Load existing results for restartability; returns [] if file missing."""
    if not os.path.isfile(results_path):
        return []
    with open(results_path, "r") as f:
        payload = json.load(f)
    out = []
    for r in payload:
        enc = r.get("encoder_correlation")
        dec = r.get("decoder_prob")
        if enc is None:
            enc = float("nan")
        if dec is None:
            dec = float("nan")
        out.append(
            RunResult(
                pre_topk=r.get("pre_topk", 1.0 if r.get("mode") == "post" else r.get("topk")),
                post_topk=r.get("post_topk", 1.0 if r.get("mode") == "pre" else r.get("topk")),
                encoder_correlation=enc,
                decoder_prob=dec,
                decoder_prob_delta=r.get("decoder_prob_delta"),
                decoder_other_prob=r.get("decoder_other_prob"),
                decoder_other_prob_delta=r.get("decoder_other_prob_delta"),
                gradiend_num_params=r.get("gradiend_num_params"),
                base_model_num_params=r.get("base_model_num_params"),
                training_time_s=r.get("training_time_s"),
                post_pruning_time_s=r.get("post_pruning_time_s"),
                encoding_inference_time_s=r.get("encoding_inference_time_s"),
                heuristic_recall=r.get("heuristic_recall"),
                heuristic_precision=r.get("heuristic_precision"),
                heuristic_f1=r.get("heuristic_f1"),
            )
        )
    return out


def _has_result(results: List[RunResult], pre_topk: float, post_topk: float) -> bool:
    return any(
        math.isclose(r.pre_topk, pre_topk) and math.isclose(r.post_topk, post_topk)
        and all(getattr(r, field) is not None for field in REQUIRED_RESULT_FIELDS)
        for r in results
    )


def _append_and_save(
    results: List[RunResult],
    result: RunResult,
    results_path: str,
) -> None:
    results[:] = [
        existing
        for existing in results
        if not (
            math.isclose(existing.pre_topk, result.pre_topk)
            and math.isclose(existing.post_topk, result.post_topk)
        )
    ]
    results.append(result)
    _save_results(results, results_path)
    logger.info(
        "Saved result pre_topk=%s post_topk=%s encoder_correlation=%.6g "
        "decoder_prob=%.6g decoder_delta=%.6g decoder_other_prob=%.6g decoder_other_delta=%.6g "
        "gradiend_params=%s base_model_params=%s training_time_s=%.2f post_pruning_time_s=%.3f encoding_time_s=%.2f",
        f"{result.pre_topk:g}",
        f"{result.post_topk:g}",
        result.encoder_correlation,
        result.decoder_prob,
        result.decoder_prob_delta if result.decoder_prob_delta is not None else float("nan"),
        result.decoder_other_prob if result.decoder_other_prob is not None else float("nan"),
        result.decoder_other_prob_delta if result.decoder_other_prob_delta is not None else float("nan"),
        result.gradiend_num_params,
        result.base_model_num_params,
        result.training_time_s if result.training_time_s is not None else float("nan"),
        result.post_pruning_time_s if result.post_pruning_time_s is not None else float("nan"),
        result.encoding_inference_time_s if result.encoding_inference_time_s is not None else float("nan"),
    )


def run_analysis(
    *,
    topk_values: List[float],
    pair: Tuple[str, str],
    output_dir: str,
    results_path: Optional[str] = None,
) -> List[RunResult]:
    _ensure_dir(output_dir)
    results_path = results_path or os.path.join(output_dir, "pruning_analysis_grid_results.json")
    results = _load_existing_results(results_path)
    logger.info(
        "Loaded %s existing pruning-analysis result(s) from %s.",
        len(results),
        results_path,
    )

    base_args = dict(
        experiment_dir=output_dir,
        base_gradient_batch_size=8,
        encoder_eval_max_size=100,
        decoder_eval_max_size_training_like=100,
        decoder_eval_max_size_neutral=1000,
        eval_steps=250,
        num_train_epochs=1,
        max_steps=1000,
        max_seeds=10,
        source="alternative",
        target="diff",
        eval_batch_size=8,
        learning_rate=1e-5,
        use_cache=False,
        add_identity_for_other_classes=True,
    )

    max_size = 200

    probe_trainer = _build_trainer(
        run_id="prune_size_probe",
        pair=pair,
        args=TrainingArguments(**base_args),
    )
    base_dim = int(probe_trainer.get_model().gradiend.input_dim)
    probe_trainer.unload_model()
    del probe_trainer
    _clear_cuda_memory()
    suitable_pairs = _suitable_topk_pairs(topk_values, base_dim=base_dim, min_final_dim=10)
    total_pairs = sum(len(post_values) for post_values in suitable_pairs.values())
    skipped_pairs = len(topk_values) * len(topk_values) - total_pairs
    logger.info(
        "Base GRADIEND input_dim=%s; running %s suitable pre/post topk pair(s) with final input_dim >= 10; "
        "skipping %s unsuitable pair(s).",
        base_dim,
        total_pairs,
        skipped_pairs,
    )
    for pre_topk, post_topks in suitable_pairs.items():
        logger.info(
            "Suitable post_topk values for pre_topk=%s (pre_dim=%s): %s",
            f"{pre_topk:g}",
            _kept_dim(base_dim, pre_topk),
            _format_topk_list(post_topks),
        )

    for pre_topk, post_topks in suitable_pairs.items():
        if all(_has_result(results, pre_topk, post_topk) for post_topk in post_topks):
            logger.info(
                "Skipping pre_topk=%s because all %s suitable post_topk cell(s) are already cached.",
                f"{pre_topk:g}",
                len(post_topks),
            )
            continue

        run_id = _pre_run_id(pre_topk)
        pre_cfg = None
        if not (isinstance(pre_topk, float) and pre_topk == 1.0):
            pre_cfg = PrePruneConfig(n_samples=32, topk=pre_topk, source="diff")
        args = TrainingArguments(**base_args, pre_prune_config=pre_cfg)
        trainer = _build_trainer(run_id=run_id, pair=pair, args=args)

        checkpoint_path = _checkpoint_path(output_dir, run_id)
        checkpoint_exists = os.path.isdir(checkpoint_path)
        logger.info(
            "Training/cache step for pre_topk=%s (pre_dim=%s, run_id=%s, checkpoint_exists=%s); "
            "remaining suitable post_topk values: %s",
            f"{pre_topk:g}",
            _kept_dim(base_dim, pre_topk),
            run_id,
            checkpoint_exists,
            _format_topk_list([post_topk for post_topk in post_topks if not _has_result(results, pre_topk, post_topk)]),
        )
        training_time = _train_and_get_time(trainer, output_dir, run_id)
        trainer.unload_model()
        _clear_cuda_memory()
        logger.info(
            "Finished training/cache step for pre_topk=%s; checkpoint=%s; training_time_s=%.2f.",
            f"{pre_topk:g}",
            checkpoint_path,
            training_time,
        )
        eval_trainer = _build_trainer(
            run_id=run_id,
            pair=pair,
            args=TrainingArguments(**base_args),
        )

        for post_topk in post_topks:
            if _has_result(results, pre_topk, post_topk):
                logger.info(
                    "Skipping cached evaluation cell pre_topk=%s post_topk=%s.",
                    f"{pre_topk:g}",
                    f"{post_topk:g}",
                )
                continue
            logger.info(
                "Evaluating cell pre_topk=%s post_topk=%s (pre_dim=%s final_dim=%s).",
                f"{pre_topk:g}",
                f"{post_topk:g}",
                _kept_dim(base_dim, pre_topk),
                _final_dim_after_pre_post(base_dim, pre_topk, post_topk),
            )
            model = None
            pruned = None
            try:
                model = eval_trainer.load_model(checkpoint_path)
                post_prune_start = time.perf_counter()
                pruned = post_prune(
                    model,
                    PostPruneConfig(topk=post_topk, part="decoder-weight", inplace=True),
                )
                post_pruning_time = time.perf_counter() - post_prune_start
                enc_corr, decoder_metrics, enc_time = _run_evals_with_model(
                    eval_trainer,
                    pruned,
                    pair,
                    max_size=max_size,
                )
                gradiend_num_params, base_model_num_params = _model_param_counts(pruned)
                _append_and_save(
                    results,
                    RunResult(
                        pre_topk=pre_topk,
                        post_topk=post_topk,
                        encoder_correlation=enc_corr,
                        decoder_prob=decoder_metrics["decoder_prob"],
                        decoder_prob_delta=decoder_metrics["decoder_prob_delta"],
                        decoder_other_prob=decoder_metrics["decoder_other_prob"],
                        decoder_other_prob_delta=decoder_metrics["decoder_other_prob_delta"],
                        gradiend_num_params=gradiend_num_params,
                        base_model_num_params=base_model_num_params,
                        training_time_s=training_time + post_pruning_time,
                        post_pruning_time_s=post_pruning_time,
                        encoding_inference_time_s=enc_time,
                    ),
                    results_path,
                )
            finally:
                eval_trainer.unload_model()
                del pruned
                del model
                _clear_cuda_memory()

    return results


def _result_value(result: RunResult, metric: str) -> Optional[float]:
    value = getattr(result, metric)
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


def _metric_norm(numeric_grid: List[List[float]], metric: str) -> Optional[LogNorm]:
    if metric != "gradiend_num_params":
        return None
    positive_values = [
        value
        for row in numeric_grid
        for value in row
        if not math.isnan(value) and value > 0
    ]
    if not positive_values:
        return None
    return LogNorm(vmin=min(positive_values), vmax=max(positive_values))


def _plot_metric_heatmap(
    results: List[RunResult],
    *,
    metric: str,
    title: str,
    output_path: str,
) -> None:
    pre_values = sorted({r.pre_topk for r in results})
    post_values = sorted({r.post_topk for r in results})
    if not pre_values or not post_values:
        return

    grid = [
        [
            next(
                (
                    _result_value(r, metric)
                    for r in results
                    if r.pre_topk == pre_topk and r.post_topk == post_topk
                ),
                None,
            )
            for post_topk in post_values
        ]
        for pre_topk in pre_values
    ]
    if all(value is None for row in grid for value in row):
        return

    numeric_grid = [
        [float("nan") if value is None else value for value in row]
        for row in grid
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(numeric_grid, aspect="auto", origin="lower", norm=_metric_norm(numeric_grid, metric))
    ax.set_xticks(range(len(post_values)))
    ax.set_xticklabels([f"{v:g}" for v in post_values], rotation=45, ha="right")
    ax.set_yticks(range(len(pre_values)))
    ax.set_yticklabels([f"{v:g}" for v in pre_values])
    ax.set_xlabel("post_topk")
    ax.set_ylabel("pre_topk")
    ax.set_title(title)
    colorbar_label = metric.replace("_", " ")
    if metric == "gradiend_num_params":
        colorbar_label += " (log scale)"
    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_grid(
    results: List[RunResult],
    output_path: str,
    *,
    pair: Tuple[str, str],
) -> None:
    metrics = [
        ("gradiend_num_params", "GRADIEND parameters"),
        ("training_time_s", "Training time (s)"),
        ("encoding_inference_time_s", "Encoding inference time (s)"),
        ("encoder_correlation", "Encoder correlation"),
        ("decoder_prob_delta", f"Decoder prob {pair[1]}"),
        ("decoder_other_prob_delta", f"Decoder prob {pair[0]}"),
    ]
    pre_values = sorted({r.pre_topk for r in results})
    post_values = sorted({r.post_topk for r in results})
    if not pre_values or not post_values:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes_flat = axes.flatten()
    for ax, (metric, title) in zip(axes_flat, metrics):
        grid = [
            [
                next(
                    (
                        _result_value(r, metric)
                        for r in results
                        if r.pre_topk == pre_topk and r.post_topk == post_topk
                    ),
                    None,
                )
                for post_topk in post_values
            ]
            for pre_topk in pre_values
        ]
        if all(value is None for row in grid for value in row):
            ax.set_axis_off()
            continue
        numeric_grid = [
            [float("nan") if value is None else value for value in row]
            for row in grid
        ]
        im = ax.imshow(numeric_grid, aspect="auto", origin="lower", norm=_metric_norm(numeric_grid, metric))
        ax.set_title(title)
        ax.set_xlabel("post_topk")
        ax.set_ylabel("pre_topk")
        ax.set_xticks(range(len(post_values)))
        ax.set_xticklabels([f"{v:g}" for v in post_values], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pre_values)))
        ax.set_yticklabels([f"{v:g}" for v in pre_values], fontsize=7)
        colorbar_label = metric.replace("_", " ")
        if metric == "gradiend_num_params":
            colorbar_label += " (log scale)"
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=colorbar_label)
    fig.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_pre_prune_heuristics(results: List[RunResult], output_path: str) -> None:
    pre = [
        r
        for r in results
        if r.post_topk == 1.0
        and r.heuristic_recall is not None
        and not (isinstance(r.heuristic_recall, float) and math.isnan(r.heuristic_recall))
    ]
    if not pre:
        return
    pre_by_topk = {r.pre_topk: r for r in pre}
    xs = sorted(pre_by_topk)
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [pre_by_topk[x].heuristic_recall for x in xs], marker="o", label="recall")
    plt.plot(xs, [pre_by_topk[x].heuristic_precision for x in xs], marker="o", label="precision")
    plt.plot(xs, [pre_by_topk[x].heuristic_f1 for x in xs], marker="o", label="f1")
    plt.xscale("log")
    plt.xlabel("pre_topk")
    plt.ylabel("score")
    plt.title("Pre-prune heuristic quality vs topk")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _save_results(results: List[RunResult], output_path: str) -> None:
    def _to_serializable(d: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in d.items():
            if v is not None and isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf")):
                out[k] = None  # nan/inf not valid JSON
            else:
                out[k] = v
        return out

    payload = [_to_serializable(r.__dict__) for r in results]
    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    topk_values = [
        1.0,
        0.5, 0.2, 0.1,
        0.05, 0.02, 0.01,
        0.005, 0.002, 0.001,
        0.0005, 0.0002, 0.0001,
        0.00005, 0.00002, 0.00001,
        0.000005, 0.000002, 0.000001,
        0.0000005, 0.0000002, 0.0000001,
    ]

    multipliers = [1, 0.7, 0.5, 0.3, 0.2]

    topk_values = [
        float(m * 10 ** e)
        for e in range(0, -8, -1)
        for m in multipliers
    ]



    topk_values_ = [
        1.0,
        0.1,
        0.01,
        0.001,
        0.0001,
        0.00001,
        0.000001,
        0.0000001,
    ]

    pair = ("masc_nom", "fem_nom")
    output_dir = os.path.join("runs", "pruning_analysis", "german_de_v7")

    results = run_analysis(topk_values=topk_values, pair=pair, output_dir=output_dir)

    _save_results(results, os.path.join(output_dir, "pruning_analysis_grid_results.json"))
    _plot_metric_heatmap(
        results,
        metric="encoder_correlation",
        title="Encoder correlation by pre/post topk",
        output_path=os.path.join(output_dir, "encoder_correlation_heatmap.pdf"),
    )
    _plot_metric_heatmap(
        results,
        metric="decoder_prob_delta",
        title=f"Decoder {pair[1]} on {pair[0]} articles: delta to base",
        output_path=os.path.join(output_dir, "decoder_prob_heatmap.pdf"),
    )
    _plot_metric_heatmap(
        results,
        metric="decoder_other_prob_delta",
        title=f"Decoder {pair[0]} on {pair[1]} articles: delta to base",
        output_path=os.path.join(output_dir, "decoder_other_prob_heatmap.pdf"),
    )
    _plot_metric_heatmap(
        results,
        metric="gradiend_num_params",
        title="GRADIEND parameters by pre/post topk",
        output_path=os.path.join(output_dir, "gradiend_params_heatmap.pdf"),
    )
    _plot_metric_heatmap(
        results,
        metric="training_time_s",
        title="Training time by pre/post topk",
        output_path=os.path.join(output_dir, "training_time_heatmap.pdf"),
    )
    _plot_metric_heatmap(
        results,
        metric="encoding_inference_time_s",
        title="Encoding inference time by pre/post topk",
        output_path=os.path.join(output_dir, "encoding_inference_time_heatmap.pdf"),
    )
    _plot_metric_grid(
        results,
        output_path=os.path.join(output_dir, "pruning_analysis_metric_grid.pdf"),
        pair=pair,
    )
    _plot_pre_prune_heuristics(
        results,
        output_path=os.path.join(output_dir, "pre_prune_heuristics_vs_topk.pdf"),
    )
