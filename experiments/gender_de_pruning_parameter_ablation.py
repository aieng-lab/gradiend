"""
Small German DE pruning-parameter ablations.

Default pre-prune mode (mask recall): train one no-pre baseline, run ``pre_prune()``
per config, and report mask recall |M_config INTERSECT T_ref| / topk_eval vs the
baseline oracle top-1000 decoder weights. No GRADIEND training per pre-prune cell.

**--full-grid** pre mode trains and evaluates encoder/decoder for every cell instead.

Also supports post-prune part ablation (--mode post-part): part x post_topk, trained once
then post-pruned.

This is the lightweight first pass before the large pre_topk x post_topk grid in
``gender_de_pruning_analysis.py``. Multi-pair recall screening with all der<->die pairs
lives in ``gender_de_pre_prune_topk_ablation.py``.

Results are saved incrementally so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from matplotlib import pyplot as plt

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PostPruneConfig, PrePruneConfig
from gradiend.trainer.core.pruning import post_prune
from gradiend.trainer.core.stats import load_training_stats
from gradiend.util.logging import get_logger
from gradiend.util.paths import ARTIFACT_MODEL, resolve_output_path
from gradiend.util.runtime_monitor import CudaMemorySpan

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pre_prune_mask_recall import (  # noqa: E402
    TOPK_EVAL,
    TOPK_PART,
    all_kept_base_global as _all_kept_base_global,
    ref_recall_metrics as _ref_recall_metrics,
    topk_base_global as _topk_base_global,
)

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


def _run_encoder_eval(
    trainer: TextPredictionTrainer,
    max_size: int = 200,
    use_cache: bool = True,
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
        lrs=[1e-3],
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


PRE_SOURCES = ["alternative", "factual", "diff"]
PRE_N_SAMPLES = [1, 2, 4, 8, 16, 32, 64]
PRE_TOPK_VALUES = [1.0, 0.1, 0.01, 0.001]
PRE_PRUNE_SEED = 42
POST_PARTS = ["decoder-weight", "decoder-bias", "decoder-sum", "encoder-weight"]
POST_TOPK_VALUES = [1.0, 0.1, 0.01, 0.001, 0.0001]


@dataclass
class AblationResult:
    ablation: str
    run_id: str
    pre_topk: Optional[float] = None
    pre_source: Optional[str] = None
    pre_n_samples: Optional[int] = None
    post_topk: Optional[float] = None
    post_part: Optional[str] = None
    encoder_correlation: Optional[float] = None
    decoder_prob: Optional[float] = None
    decoder_prob_delta: Optional[float] = None
    decoder_other_prob: Optional[float] = None
    decoder_other_prob_delta: Optional[float] = None
    gradiend_num_params: Optional[int] = None
    base_model_num_params: Optional[int] = None
    pre_pruning_time_s: Optional[float] = None
    training_time_s: Optional[float] = None
    post_pruning_time_s: Optional[float] = None
    encoding_inference_time_s: Optional[float] = None
    pre_pruning_max_gpu_allocated_bytes: Optional[int] = None
    pre_pruning_max_gpu_reserved_bytes: Optional[int] = None
    training_max_gpu_allocated_bytes: Optional[int] = None
    training_max_gpu_reserved_bytes: Optional[int] = None
    post_pruning_max_gpu_allocated_bytes: Optional[int] = None
    post_pruning_max_gpu_reserved_bytes: Optional[int] = None
    eval_max_gpu_allocated_bytes: Optional[int] = None
    eval_max_gpu_reserved_bytes: Optional[int] = None
    ref_recall: Optional[float] = None
    ref_precision: Optional[float] = None
    mask_recall: bool = False
    error: Optional[str] = None


def _record_cuda_memory_span(result: AblationResult, phase: str, span: CudaMemorySpan) -> None:
    setattr(result, f"{phase}_max_gpu_allocated_bytes", span.max_allocated_bytes)
    setattr(result, f"{phase}_max_gpu_reserved_bytes", span.max_reserved_bytes)


def _base_args(output_dir: str) -> Dict[str, Any]:
    return dict(
        experiment_dir=output_dir,
        base_gradient_batch_size=4,
        encoder_eval_max_size=100,
        decoder_eval_max_size_training_like=100,
        decoder_eval_max_size_neutral=1000,
        eval_steps=250,
        num_train_epochs=1,
        max_steps=1000,
        max_seeds=5,
        source="alternative",
        target="diff",
        eval_batch_size=8,
        learning_rate=1e-5,
        use_cache=False,
        add_identity_for_other_classes=True,
    )


def _load_results(path: str) -> List[AblationResult]:
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        payload = json.load(f)
    return [AblationResult(**row) for row in payload]


def _save_results(results: List[AblationResult], path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def _append_result(results: List[AblationResult], result: AblationResult, path: str) -> None:
    results[:] = [r for r in results if r.run_id != result.run_id or r.ablation != result.ablation]
    results.append(result)
    _save_results(results, path)


def _has_success(
    results: List[AblationResult],
    ablation: str,
    run_id: str,
    *,
    recall_only: bool = True,
) -> bool:
    for row in results:
        if row.ablation != ablation or row.run_id != run_id or row.error:
            continue
        if recall_only:
            if row.mask_recall and row.ref_recall is not None:
                return True
            continue
        if row.mask_recall:
            continue
        return row.encoder_correlation is not None
    return False


def _baseline_oracle_topk(model: Any) -> Set[int]:
    return _topk_base_global(model, topk=TOPK_EVAL, part=TOPK_PART)


def _train_pre_baseline(
    *,
    output_dir: str,
    pair: Tuple[str, str],
    args_base: Dict[str, Any],
) -> Tuple[Any, float]:
    run_id = _pre_baseline_run_id()
    trainer = _build_trainer(
        run_id=run_id,
        pair=pair,
        args=TrainingArguments(**args_base),
    )
    try:
        with CudaMemorySpan():
            training_time = _train_and_get_time(trainer, output_dir, run_id)
    finally:
        trainer.unload_model()
        _clear_cuda_memory()

    eval_trainer = _build_trainer(
        run_id=run_id,
        pair=pair,
        args=TrainingArguments(**args_base),
    )
    try:
        model = eval_trainer.load_model(_checkpoint_path(output_dir, run_id))
        return model, training_time
    finally:
        eval_trainer.unload_model()


def _ensure_baseline_oracle(
    *,
    output_dir: str,
    results_path: str,
    pair: Tuple[str, str],
    args_base: Dict[str, Any],
    results: List[AblationResult],
) -> Optional[Set[int]]:
    run_id = _pre_baseline_run_id()
    checkpoint_path = _checkpoint_path(output_dir, run_id)
    if os.path.isdir(checkpoint_path):
        eval_trainer = _build_trainer(
            run_id=run_id,
            pair=pair,
            args=TrainingArguments(**args_base),
        )
        try:
            model = eval_trainer.load_model(checkpoint_path)
            return _baseline_oracle_topk(model)
        finally:
            eval_trainer.unload_model()
            _clear_cuda_memory()

    logger.info("Training pre ablation baseline for mask-recall oracle.")
    model = None
    try:
        model, training_time = _train_pre_baseline(
            output_dir=output_dir,
            pair=pair,
            args_base=args_base,
        )
        ref_topk = _baseline_oracle_topk(model)
        gradiend_num_params, base_model_num_params = _model_param_counts(model)
        baseline = AblationResult(
            ablation="pre",
            run_id=run_id,
            pre_topk=1.0,
            post_topk=1.0,
            ref_recall=1.0,
            ref_precision=1.0,
            gradiend_num_params=gradiend_num_params,
            base_model_num_params=base_model_num_params,
            training_time_s=training_time,
        )
        _append_result(results, baseline, results_path)
        return ref_topk
    except Exception as exc:
        logger.exception("Pre ablation baseline run failed: %s", run_id)
        failed = AblationResult(
            ablation="pre",
            run_id=run_id,
            pre_topk=1.0,
            post_topk=1.0,
            error=str(exc),
        )
        _append_result(results, failed, results_path)
        return None
    finally:
        del model
        _clear_cuda_memory()


def _csv_floats(value: str) -> List[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _csv_ints(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _csv_strings(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _pre_run_id(source: str, n_samples: int, topk: float) -> str:
    return f"pre_src_{source}_n_{n_samples}_topk_{_format_topk(topk)}"


def _pre_baseline_run_id() -> str:
    return f"pre_topk_{_format_topk(1.0)}"


def _post_part_run_id(part: str, topk: float) -> str:
    return f"post_part_{part.replace('-', '_')}_topk_{_format_topk(topk)}"


def run_pre_ablation(
    *,
    output_dir: str,
    results_path: str,
    pair: Tuple[str, str],
    n_samples_values: Iterable[int],
    sources: Iterable[str],
    pre_topk_values: Iterable[float],
    max_size: int = 200,
    recall_only: bool = True,
) -> List[AblationResult]:
    results = _load_results(results_path)
    args_base = _base_args(output_dir)

    if recall_only:
        ref_topk = _ensure_baseline_oracle(
            output_dir=output_dir,
            results_path=results_path,
            pair=pair,
            args_base=args_base,
            results=results,
        )
        if not ref_topk:
            return results

        for topk in pre_topk_values:
            if isinstance(topk, float) and math.isclose(topk, 1.0):
                continue
            for source in sources:
                for n_samples in n_samples_values:
                    run_id = _pre_run_id(source, n_samples, topk)
                    if _has_success(results, "pre", run_id, recall_only=True):
                        logger.info("Skipping cached mask-recall pre ablation run %s.", run_id)
                        continue

                    logger.info(
                        "Mask-recall pre ablation: source=%s n_samples=%s pre_topk=%s.",
                        source,
                        n_samples,
                        f"{topk:g}",
                    )
                    result = AblationResult(
                        ablation="pre",
                        run_id=run_id,
                        pre_topk=topk,
                        pre_source=source,
                        pre_n_samples=n_samples,
                        post_topk=1.0,
                        mask_recall=True,
                    )
                    trainer = None
                    try:
                        pre_cfg = PrePruneConfig(
                            n_samples=n_samples,
                            topk=topk,
                            source=source,
                            seed=PRE_PRUNE_SEED,
                        )
                        args = TrainingArguments(
                            **args_base,
                            pre_prune_config=pre_cfg,
                            reuse_pre_prune=True,
                        )
                        trainer = _build_trainer(run_id=run_id, pair=pair, args=args)
                        pre_prune_start = time.perf_counter()
                        with CudaMemorySpan() as memory_span:
                            base_input_dim = int(trainer.get_model().gradiend.input_dim)
                            trainer.pre_prune(inplace=False)
                        result.pre_pruning_time_s = time.perf_counter() - pre_prune_start
                        _record_cuda_memory_span(result, "pre_pruning", memory_span)
                        model = trainer.get_model()
                        heuristic_topk = _all_kept_base_global(model)
                        ref_recall, ref_precision = _ref_recall_metrics(heuristic_topk, ref_topk)
                        gradiend_num_params, base_model_num_params = _model_param_counts(model)
                        result.ref_recall = ref_recall
                        result.ref_precision = ref_precision
                        result.gradiend_num_params = gradiend_num_params
                        result.base_model_num_params = base_model_num_params
                    except Exception as exc:
                        logger.exception("Mask-recall pre ablation run failed: %s", run_id)
                        result.error = str(exc)
                    finally:
                        if trainer is not None:
                            trainer.unload_model()
                        _clear_cuda_memory()
                    _append_result(results, result, results_path)
        return results

    for topk in pre_topk_values:
        if isinstance(topk, float) and math.isclose(topk, 1.0):
            run_id = _pre_baseline_run_id()
            if _has_success(results, "pre", run_id, recall_only=False):
                logger.info("Skipping cached pre ablation baseline (no pre-prune).")
                continue

            logger.info("Running pre ablation baseline: pre_topk=1.0 (no pre-prune).")
            result = AblationResult(
                ablation="pre",
                run_id=run_id,
                pre_topk=1.0,
                post_topk=1.0,
            )
            trainer = None
            eval_trainer = None
            model = None
            try:
                trainer = _build_trainer(
                    run_id=run_id,
                    pair=pair,
                    args=TrainingArguments(**args_base),
                )
                with CudaMemorySpan() as memory_span:
                    training_time = _train_and_get_time(trainer, output_dir, run_id)
                _record_cuda_memory_span(result, "training", memory_span)
                trainer.unload_model()
                _clear_cuda_memory()
                checkpoint_path = _checkpoint_path(output_dir, run_id)

                eval_trainer = _build_trainer(
                    run_id=run_id,
                    pair=pair,
                    args=TrainingArguments(**args_base),
                )
                model = eval_trainer.load_model(checkpoint_path)
                with CudaMemorySpan() as memory_span:
                    enc_corr, decoder_metrics, enc_time = _run_evals_with_model(
                        eval_trainer,
                        model,
                        pair,
                        max_size=max_size,
                    )
                _record_cuda_memory_span(result, "eval", memory_span)
                gradiend_num_params, base_model_num_params = _model_param_counts(model)
                result.encoder_correlation = enc_corr
                result.decoder_prob = decoder_metrics["decoder_prob"]
                result.decoder_prob_delta = decoder_metrics["decoder_prob_delta"]
                result.decoder_other_prob = decoder_metrics["decoder_other_prob"]
                result.decoder_other_prob_delta = decoder_metrics["decoder_other_prob_delta"]
                result.gradiend_num_params = gradiend_num_params
                result.base_model_num_params = base_model_num_params
                result.training_time_s = training_time
                result.encoding_inference_time_s = enc_time
            except Exception as exc:
                logger.exception("Pre ablation baseline run failed: %s", run_id)
                result.error = str(exc)
            finally:
                if eval_trainer is not None:
                    eval_trainer.unload_model()
                if trainer is not None:
                    trainer.unload_model()
                del model
                _clear_cuda_memory()
            _append_result(results, result, results_path)
            continue

        for source in sources:
            for n_samples in n_samples_values:
                run_id = _pre_run_id(source, n_samples, topk)
                if _has_success(results, "pre", run_id, recall_only=False):
                    logger.info("Skipping cached pre ablation run %s.", run_id)
                    continue

                logger.info(
                    "Running pre ablation: source=%s n_samples=%s pre_topk=%s.",
                    source,
                    n_samples,
                    f"{topk:g}",
                )
                result = AblationResult(
                    ablation="pre",
                    run_id=run_id,
                    pre_topk=topk,
                    pre_source=source,
                    pre_n_samples=n_samples,
                    post_topk=1.0,
                )
                trainer = None
                eval_trainer = None
                model = None
                try:
                    pre_cfg = PrePruneConfig(
                        n_samples=n_samples,
                        topk=topk,
                        source=source,
                        seed=PRE_PRUNE_SEED,
                    )
                    args = TrainingArguments(**args_base, pre_prune_config=pre_cfg)
                    trainer = _build_trainer(run_id=run_id, pair=pair, args=args)
                    pre_prune_start = time.perf_counter()
                    with CudaMemorySpan() as memory_span:
                        trainer.pre_prune(inplace=True)
                    result.pre_pruning_time_s = time.perf_counter() - pre_prune_start
                    _record_cuda_memory_span(result, "pre_pruning", memory_span)
                    trainer._training_args = TrainingArguments(**args_base)
                    with CudaMemorySpan() as memory_span:
                        training_time = _train_and_get_time(trainer, output_dir, run_id)
                    _record_cuda_memory_span(result, "training", memory_span)
                    trainer.unload_model()
                    _clear_cuda_memory()
                    checkpoint_path = _checkpoint_path(output_dir, run_id)

                    eval_trainer = _build_trainer(
                        run_id=run_id,
                        pair=pair,
                        args=TrainingArguments(**args_base),
                    )
                    model = eval_trainer.load_model(checkpoint_path)
                    with CudaMemorySpan() as memory_span:
                        enc_corr, decoder_metrics, enc_time = _run_evals_with_model(
                            eval_trainer,
                            model,
                            pair,
                            max_size=max_size,
                        )
                    _record_cuda_memory_span(result, "eval", memory_span)
                    gradiend_num_params, base_model_num_params = _model_param_counts(model)
                    result.encoder_correlation = enc_corr
                    result.decoder_prob = decoder_metrics["decoder_prob"]
                    result.decoder_prob_delta = decoder_metrics["decoder_prob_delta"]
                    result.decoder_other_prob = decoder_metrics["decoder_other_prob"]
                    result.decoder_other_prob_delta = decoder_metrics["decoder_other_prob_delta"]
                    result.gradiend_num_params = gradiend_num_params
                    result.base_model_num_params = base_model_num_params
                    result.training_time_s = training_time
                    result.encoding_inference_time_s = enc_time
                except Exception as exc:
                    logger.exception("Pre ablation run failed: %s", run_id)
                    result.error = str(exc)
                finally:
                    if eval_trainer is not None:
                        eval_trainer.unload_model()
                    if trainer is not None:
                        trainer.unload_model()
                    del model
                    _clear_cuda_memory()
                _append_result(results, result, results_path)

    return results


def run_post_part_ablation(
    *,
    output_dir: str,
    results_path: str,
    pair: Tuple[str, str],
    parts: Iterable[str],
    post_topk_values: Iterable[float],
    max_size: int = 200,
) -> List[AblationResult]:
    results = _load_results(results_path)
    args_base = _base_args(output_dir)
    base_run_id = "post_part_base"
    base_args = TrainingArguments(**args_base)
    base_trainer = _build_trainer(run_id=base_run_id, pair=pair, args=base_args)
    base_training_memory_span = CudaMemorySpan()
    with base_training_memory_span:
        training_time = _train_and_get_time(base_trainer, output_dir, base_run_id)
    base_trainer.unload_model()
    _clear_cuda_memory()
    checkpoint_path = _checkpoint_path(output_dir, base_run_id)

    eval_trainer = _build_trainer(
        run_id=base_run_id,
        pair=pair,
        args=TrainingArguments(**args_base),
    )

    baseline_done = False
    for topk in post_topk_values:
        for part in parts:
            if isinstance(topk, float) and topk == 1.0:
                if baseline_done:
                    continue
                run_id = _post_part_run_id("baseline", topk)
                result_part = "baseline"
                baseline_done = True
            else:
                run_id = _post_part_run_id(part, topk)
                result_part = part

            if _has_success(results, "post_part", run_id):
                logger.info("Skipping cached post-part ablation run %s.", run_id)
                continue

            logger.info(
                "Running post-part ablation: part=%s post_topk=%s.",
                result_part,
                f"{topk:g}",
            )
            result = AblationResult(
                ablation="post_part",
                run_id=run_id,
                pre_topk=1.0,
                post_topk=topk,
                post_part=result_part,
                training_time_s=training_time,
            )
            _record_cuda_memory_span(result, "training", base_training_memory_span)
            model = None
            pruned = None
            try:
                model = eval_trainer.load_model(checkpoint_path)
                post_prune_start = time.perf_counter()
                with CudaMemorySpan() as memory_span:
                    if isinstance(topk, float) and topk == 1.0:
                        pruned = model
                    else:
                        pruned = post_prune(
                            model,
                            PostPruneConfig(topk=topk, part=part, inplace=True),
                        )
                post_pruning_time = time.perf_counter() - post_prune_start
                _record_cuda_memory_span(result, "post_pruning", memory_span)
                with CudaMemorySpan() as memory_span:
                    enc_corr, decoder_metrics, enc_time = _run_evals_with_model(
                        eval_trainer,
                        pruned,
                        pair,
                        max_size=max_size,
                    )
                _record_cuda_memory_span(result, "eval", memory_span)
                gradiend_num_params, base_model_num_params = _model_param_counts(pruned)
                result.encoder_correlation = enc_corr
                result.decoder_prob = decoder_metrics["decoder_prob"]
                result.decoder_prob_delta = decoder_metrics["decoder_prob_delta"]
                result.decoder_other_prob = decoder_metrics["decoder_other_prob"]
                result.decoder_other_prob_delta = decoder_metrics["decoder_other_prob_delta"]
                result.gradiend_num_params = gradiend_num_params
                result.base_model_num_params = base_model_num_params
                result.post_pruning_time_s = post_pruning_time
                result.encoding_inference_time_s = enc_time
            except Exception as exc:
                logger.exception("Post-part ablation run failed: %s", run_id)
                result.error = str(exc)
            finally:
                eval_trainer.unload_model()
                del pruned
                del model
                _clear_cuda_memory()
            _append_result(results, result, results_path)

    return results


def _result_value(result: AblationResult, metric: str) -> Optional[float]:
    value = getattr(result, metric)
    if not isinstance(value, (int, float)):
        return None
    value = float(value)
    if math.isnan(value):
        return None
    return value


def _numeric_or_nan(value: Optional[float]) -> float:
    return float("nan") if value is None else float(value)


def plot_pre_metric(
    results: List[AblationResult],
    *,
    metric: str,
    output_path: str,
    mask_recall_only: bool = False,
) -> None:
    rows = [r for r in results if r.ablation == "pre" and r.error is None]
    if mask_recall_only:
        rows = [r for r in rows if r.mask_recall or (r.pre_topk is not None and math.isclose(r.pre_topk, 1.0))]
    n_samples_values = sorted({r.pre_n_samples for r in rows if r.pre_n_samples is not None})
    sources = sorted({r.pre_source for r in rows if r.pre_source is not None})
    topks = sorted({r.pre_topk for r in rows if r.pre_topk is not None}, reverse=True)
    if not rows or not n_samples_values or not sources or not topks:
        return

    fig, axes = plt.subplots(1, len(topks), figsize=(6 * len(topks), 4), squeeze=False)
    axes_flat = axes.flatten()
    for ax, topk in zip(axes_flat, topks):
        if isinstance(topk, float) and math.isclose(topk, 1.0):
            baseline = next(
                (
                    r
                    for r in rows
                    if r.pre_topk is not None and math.isclose(r.pre_topk, 1.0)
                ),
                None,
            )
            if baseline is not None:
                value = _result_value(baseline, metric)
                if value is not None:
                    ax.axhline(value, linestyle="--", color="black", label="no pre-prune")
            ax.set_xscale("log", base=2)
            ax.set_xticks(n_samples_values)
            ax.set_xticklabels([str(v) for v in n_samples_values])
            ax.set_xlabel("pre n_samples")
            ax.set_ylabel(metric.replace("_", " "))
            ax.set_title(f"pre_topk={topk:g}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            continue

        for source in sources:
            values = []
            for n_samples in n_samples_values:
                match = next(
                    (
                        r
                        for r in rows
                        if r.pre_source == source
                        and r.pre_n_samples == n_samples
                        and r.pre_topk is not None
                        and math.isclose(r.pre_topk, topk)
                    ),
                    None,
                )
                values.append(float("nan") if match is None else _numeric_or_nan(_result_value(match, metric)))
            ax.plot(n_samples_values, values, marker="o", label=source)
        ax.set_xscale("log", base=2)
        ax.set_xticks(n_samples_values)
        ax.set_xticklabels([str(v) for v in n_samples_values])
        ax.set_xlabel("pre n_samples")
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(f"pre_topk={topk:g}")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_post_part_metric(
    results: List[AblationResult],
    *,
    metric: str,
    output_path: str,
) -> None:
    rows = [r for r in results if r.ablation == "post_part" and r.error is None]
    parts = sorted({r.post_part for r in rows if r.post_part != "baseline"})
    topks = sorted({r.post_topk for r in rows if r.post_topk is not None})
    if not rows or not topks:
        return
    display_parts = ["baseline"] + parts
    grid = []
    for part in display_parts:
        row = []
        for topk in topks:
            match = next(
                (
                    r
                    for r in rows
                    if r.post_part == part
                    and r.post_topk is not None
                    and math.isclose(r.post_topk, topk)
                ),
                None,
            )
            row.append(float("nan") if match is None else _numeric_or_nan(_result_value(match, metric)))
        grid.append(row)

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, aspect="auto", origin="lower")
    ax.set_xticks(range(len(topks)))
    ax.set_xticklabels([f"{v:g}" for v in topks], rotation=45, ha="right")
    ax.set_yticks(range(len(display_parts)))
    ax.set_yticklabels(display_parts)
    ax.set_xlabel("post_topk")
    ax.set_ylabel("post part")
    ax.set_title(metric.replace("_", " "))
    fig.colorbar(im, ax=ax, label=metric.replace("_", " "))
    fig.tight_layout()
    _ensure_dir(os.path.dirname(output_path))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["pre", "post-part", "both", "plot"], default="pre")
    parser.add_argument("--output-dir", default=os.path.join("runs", "pruning_parameter_ablation", "german_de_v1"))
    parser.add_argument("--results-path", default=None)
    parser.add_argument("--pair", default="masc_nom:fem_nom")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--recall-only",
        dest="recall_only",
        action="store_true",
        help="Mask recall via pre_prune only (default for --mode pre).",
    )
    mode_group.add_argument(
        "--full-grid",
        dest="recall_only",
        action="store_false",
        help="Train and evaluate encoder/decoder for every pre-prune cell.",
    )
    parser.set_defaults(recall_only=True)
    parser.add_argument("--pre-n-samples", type=_csv_ints, default=PRE_N_SAMPLES)
    parser.add_argument("--pre-sources", type=_csv_strings, default=PRE_SOURCES)
    parser.add_argument("--pre-topk-values", type=_csv_floats, default=PRE_TOPK_VALUES)
    parser.add_argument("--post-parts", type=_csv_strings, default=POST_PARTS)
    parser.add_argument("--post-topk-values", type=_csv_floats, default=POST_TOPK_VALUES)
    parser.add_argument("--eval-max-size", type=int, default=200)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    pair = tuple(args.pair.split(":", 1))
    if len(pair) != 2 or not pair[0] or not pair[1]:
        raise ValueError("--pair must have form source_class:target_class")
    results_path = args.results_path or os.path.join(args.output_dir, "pruning_parameter_ablation_results.json")

    results: List[AblationResult] = _load_results(results_path)
    if args.mode in ("pre", "both"):
        results = run_pre_ablation(
            output_dir=args.output_dir,
            results_path=results_path,
            pair=(pair[0], pair[1]),
            n_samples_values=args.pre_n_samples,
            sources=args.pre_sources,
            pre_topk_values=args.pre_topk_values,
            max_size=args.eval_max_size,
            recall_only=args.recall_only,
        )
    if args.mode in ("post-part", "both"):
        results = run_post_part_ablation(
            output_dir=args.output_dir,
            results_path=results_path,
            pair=(pair[0], pair[1]),
            parts=args.post_parts,
            post_topk_values=args.post_topk_values,
            max_size=args.eval_max_size,
        )

    if args.mode in ("pre", "both", "plot"):
        if args.recall_only:
            plot_pre_metric(
                results,
                metric="ref_recall",
                output_path=os.path.join(args.output_dir, "pre_ref_recall.pdf"),
                mask_recall_only=True,
            )
            plot_pre_metric(
                results,
                metric="ref_precision",
                output_path=os.path.join(args.output_dir, "pre_ref_precision.pdf"),
                mask_recall_only=True,
            )
            plot_pre_metric(
                results,
                metric="pre_pruning_time_s",
                output_path=os.path.join(args.output_dir, "pre_pruning_time.pdf"),
            )
        else:
            plot_pre_metric(
                results,
                metric="decoder_prob_delta",
                output_path=os.path.join(args.output_dir, "pre_decoder_prob_delta.pdf"),
            )
            plot_pre_metric(
                results,
                metric="encoder_correlation",
                output_path=os.path.join(args.output_dir, "pre_encoder_correlation.pdf"),
            )
            plot_pre_metric(
                results,
                metric="pre_pruning_time_s",
                output_path=os.path.join(args.output_dir, "pre_pruning_time.pdf"),
            )
            plot_pre_metric(
                results,
                metric="training_max_gpu_allocated_bytes",
                output_path=os.path.join(args.output_dir, "pre_training_max_gpu_allocated.pdf"),
            )

    if args.mode in ("post-part", "both", "plot"):
        plot_post_part_metric(
            results,
            metric="decoder_prob_delta",
            output_path=os.path.join(args.output_dir, "post_part_decoder_prob_delta.pdf"),
        )
        plot_post_part_metric(
            results,
            metric="encoder_correlation",
            output_path=os.path.join(args.output_dir, "post_part_encoder_correlation.pdf"),
        )
        plot_post_part_metric(
            results,
            metric="eval_max_gpu_allocated_bytes",
            output_path=os.path.join(args.output_dir, "post_part_eval_max_gpu_allocated.pdf"),
        )


if __name__ == "__main__":
    main()
