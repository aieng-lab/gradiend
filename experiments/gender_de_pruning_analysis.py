"""
German DE pruning analysis (pre-prune, post-prune, and combined).

Goal: find a good default topk proportion balancing efficiency vs recall.
- Pre-prune heuristic quality: compare pre-prune mask vs weight-based top-k mask (recall/precision/F1).
- Encoder metric: correlation from trainer.evaluate_encoder().
- Decoder metric: <target_class> from trainer.evaluate_decoder().

Post-pruning is applied outside the trainer: we always save the model without post-pruning,
then load, apply post_prune, and analyze. This keeps the trainer focused on training only.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from matplotlib import pyplot as plt

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PrePruneConfig, PostPruneConfig
from gradiend.trainer.core.pruning import post_prune, pre_prune
from gradiend.util.paths import ARTIFACT_MODEL, resolve_output_path


@dataclass
class RunResult:
    topk: float
    mode: str
    encoder_correlation: float
    decoder_prob: float
    heuristic_recall: Optional[float] = None
    heuristic_precision: Optional[float] = None
    heuristic_f1: Optional[float] = None
    error: Optional[str] = None  # Set when run failed (e.g. empty model); ensures we skip on restart


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _checkpoint_path(output_dir: str, run_id: str) -> str:
    """Resolve model checkpoint path for a run (matches trainer output layout)."""
    exp_dir = os.path.join(output_dir.rstrip("/\\"), str(run_id).strip("/\\"))
    path = resolve_output_path(exp_dir, None, ARTIFACT_MODEL)
    if path is None:
        raise ValueError(f"Could not resolve checkpoint path for run_id={run_id!r}")
    return path


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


def _run_decoder_eval(
    trainer: TextPredictionTrainer,
    target_class: str,
    max_size: int = 200,
    use_cache: bool = True,
) -> float:
    result = trainer.evaluate_decoder(
        lrs=[1e-3],
        max_size_training_like=max_size,
        max_size_neutral=max_size,
        use_cache=use_cache,
    )
    summary = result.get("summary", {}) if result else {}
    if target_class in summary:
        return float(summary[target_class].get("value", 0.0))
    return 0.0


def _run_evals_with_model(
    trainer: TextPredictionTrainer,
    model: Any,
    target_class: str,
    max_size: int = 200,
) -> Tuple[float, float]:
    """Run encoder and decoder eval using the given model (injected into trainer)."""
    prev_instance = trainer._model_instance
    trainer._model_instance = model
    try:
        enc = _run_encoder_eval(trainer, max_size=max_size, use_cache=False)
        dec = _run_decoder_eval(trainer, target_class=target_class, max_size=max_size, use_cache=False)
        return enc, dec
    finally:
        trainer._model_instance = prev_instance


def _build_trainer(
    *,
    run_id: str,
    pair: Tuple[str, str],
    args: TrainingArguments,
) -> TextPredictionTrainer:
    return TextPredictionTrainer(
        model="bert-base-german-cased",
        run_id=run_id,
        data="aieng-lab/de-gender-case-articles",
        target_classes=list(pair),
        masked_col="masked",
        split_col="split",
        eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
        args=args,
    )


def _compute_pre_prune_heuristic_mask(
    trainer: TextPredictionTrainer,
    topk: float,
) -> torch.Tensor:
    model = trainer.get_model()
    dataset = trainer.create_training_data(model, batch_size=1)
    pre_cfg = PrePruneConfig(
        n_samples=32,
        topk=topk,
        source="diff",
        batch_size=4,
    )
    _, heuristic_mask = pre_prune(
        model,
        dataset,
        pre_cfg,
        definition=trainer,
        inplace=False,
        return_mask=True,
    )
    return heuristic_mask


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
                topk=r["topk"],
                mode=r["mode"],
                encoder_correlation=enc,
                decoder_prob=dec,
                heuristic_recall=r.get("heuristic_recall"),
                heuristic_precision=r.get("heuristic_precision"),
                heuristic_f1=r.get("heuristic_f1"),
                error=r.get("error"),
            )
        )
    return out


def _has_result(results: List[RunResult], topk: float, mode: str) -> bool:
    return any(r.topk == topk and r.mode == mode for r in results)


def _append_and_save(
    results: List[RunResult],
    result: RunResult,
    results_path: str,
) -> None:
    results.append(result)
    _save_results(results, results_path)


def run_analysis(
    *,
    topk_values: List[float],
    pair: Tuple[str, str],
    output_dir: str,
    results_path: Optional[str] = None,
) -> List[RunResult]:
    _ensure_dir(output_dir)
    results_path = results_path or os.path.join(output_dir, "pruning_analysis_results.json")
    results = _load_existing_results(results_path)

    base_args = dict(
        experiment_dir=output_dir,
        train_batch_size=8,
        encoder_eval_max_size=100,
        decoder_eval_max_size_training_like=100,
        decoder_eval_max_size_neutral=1000,
        eval_steps=1000,
        num_train_epochs=1,
        max_steps=1000,
        source="alternative",
        target="diff",
        eval_batch_size=8,
        learning_rate=1e-4,
        use_cache=True,
        add_identity_for_other_classes=True,
    )

    target_class = pair[1]
    max_size = 200


    def _record_failed(topk: float, mode: str, err: BaseException) -> None:
        _append_and_save(
            results,
            RunResult(
                topk=topk,
                mode=mode,
                encoder_correlation=float("nan"),
                decoder_prob=float("nan"),
                error=str(err),
            ),
            results_path,
        )

    # ---- Post-only: train once without post-pruning, then apply post_prune per topk ----
    post_baseline_run_id = "prune_post_baseline"
    args_post_baseline = TrainingArguments(**base_args)
    trainer_post = _build_trainer(run_id=post_baseline_run_id, pair=pair, args=args_post_baseline)
    trainer_post.train()
    checkpoint_path = _checkpoint_path(output_dir, post_baseline_run_id)

    for topk in topk_values:
        if _has_result(results, topk, "post"):
            continue
        try:
            model = trainer_post.load_model(checkpoint_path)
            pruned = post_prune(model, PostPruneConfig(topk=topk, part="decoder-weight", inplace=False))
            enc_corr, dec_prob = _run_evals_with_model(trainer_post, pruned, target_class, max_size=max_size)
            _append_and_save(
                results,
                RunResult(topk=topk, mode="post", encoder_correlation=enc_corr, decoder_prob=dec_prob),
                results_path,
            )
        except NotImplementedError as e:
            _record_failed(topk, "post", e)

    # ---- Pre-only and pre+post: per topk ----
    for topk in topk_values:
        # Pre-prune only (no post-pruning in trainer)
        if not _has_result(results, topk, "pre"):
            try:
                args = TrainingArguments(
                    **base_args,
                    pre_prune_config=PrePruneConfig(n_samples=32, topk=topk, source="diff"),
                )
                # Must be unique per topk: .4f would collapse 0.000001..0.00005 to "0.0000"; .6f + replace keeps dirs distinct
                run_id = f"prune_pre_only_topk_{topk:.6f}".replace(".", "_")
                trainer = _build_trainer(run_id=run_id, pair=pair, args=args)
                heuristic_mask = _compute_pre_prune_heuristic_mask(trainer, topk=topk)
                trainer.train()
                oracle_mask = _compute_oracle_mask(trainer, topk=topk)
                recall, precision, f1 = _mask_metrics(heuristic_mask, oracle_mask)
                enc_corr = _run_encoder_eval(trainer, max_size=max_size)
                dec_prob = _run_decoder_eval(trainer, target_class=target_class, max_size=max_size)
                _append_and_save(
                    results,
                    RunResult(
                        topk=topk,
                        mode="pre",
                        encoder_correlation=enc_corr,
                        decoder_prob=dec_prob,
                        heuristic_recall=recall,
                        heuristic_precision=precision,
                        heuristic_f1=f1,
                    ),
                    results_path,
                )
            except Exception as e:
                _record_failed(topk, "pre", e)

        # Pre + post: train with pre-prune only, then load and apply post_prune before eval
        if not _has_result(results, topk, "pre_post"):
            try:
                args = TrainingArguments(
                    **base_args,
                    pre_prune_config=PrePruneConfig(n_samples=32, topk=topk, source="diff"),
                )
                # Unique per topk (see pre-only comment)
                run_id = f"prune_pre_post_topk_{topk:.6f}".replace(".", "_")
                trainer = _build_trainer(run_id=run_id, pair=pair, args=args)
                trainer.train()
                checkpoint_path_pre = _checkpoint_path(output_dir, run_id)
                model = trainer.load_model(checkpoint_path_pre)
                pruned = post_prune(model, PostPruneConfig(topk=topk, part="decoder-weight", inplace=False))
                enc_corr, dec_prob = _run_evals_with_model(trainer, pruned, target_class, max_size=max_size)
                _append_and_save(
                    results,
                    RunResult(topk=topk, mode="pre_post", encoder_correlation=enc_corr, decoder_prob=dec_prob),
                    results_path,
                )
            except Exception as e:
                _record_failed(topk, "pre_post", e)

    return results


def _plot_metric(
    results: List[RunResult],
    *,
    metric: str,
    title: str,
    output_path: str,
) -> None:
    modes = sorted({r.mode for r in results})
    plt.figure(figsize=(8, 5))
    for mode in modes:
        pts = [(r.topk, getattr(r, metric)) for r in results if r.mode == mode]
        pts = [(x, y) for x, y in pts if not (isinstance(y, float) and math.isnan(y))]
        if not pts:
            continue
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="o", label=mode)
    plt.xscale("log")
    plt.xlabel("topk")
    plt.ylabel(metric.replace("_", " "))
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    _ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _plot_pre_prune_heuristics(results: List[RunResult], output_path: str) -> None:
    pre = [
        r
        for r in results
        if r.mode == "pre"
        and r.heuristic_recall is not None
        and not (isinstance(r.heuristic_recall, float) and math.isnan(r.heuristic_recall))
    ]
    if not pre:
        return
    xs = [r.topk for r in pre]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, [r.heuristic_recall for r in pre], marker="o", label="recall")
    plt.plot(xs, [r.heuristic_precision for r in pre], marker="o", label="precision")
    plt.plot(xs, [r.heuristic_f1 for r in pre], marker="o", label="f1")
    plt.xscale("log")
    plt.xlabel("topk")
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
        0.000001, 0.000002, 0.000005,
        0.00001, 0.00002, 0.00005,
        0.0001, 0.0002, 0.0005,
        0.001, 0.002, 0.005,
        0.01, 0.02, 0.05,
        0.1, 0.2, 0.5,
        1.0
    ]
    pair = ("fem_nom", "fem_dat")
    output_dir = os.path.join("runs", "pruning_analysis", "german_de")

    results = run_analysis(topk_values=topk_values, pair=pair, output_dir=output_dir)

    _save_results(results, os.path.join(output_dir, "pruning_analysis_results.json"))
    _plot_metric(
        results,
        metric="encoder_correlation",
        title="Encoder correlation vs topk",
        output_path=os.path.join(output_dir, "encoder_correlation_vs_topk.pdf"),
    )
    _plot_metric(
        results,
        metric="decoder_prob",
        title=f"Decoder {pair[1]} vs topk",
        output_path=os.path.join(output_dir, "decoder_prob_vs_topk.pdf"),
    )
    _plot_pre_prune_heuristics(
        results,
        output_path=os.path.join(output_dir, "pre_prune_heuristics_vs_topk.pdf"),
    )
