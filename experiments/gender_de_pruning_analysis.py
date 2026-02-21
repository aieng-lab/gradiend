"""
German DE pruning analysis (pre-prune, post-prune, and combined).

Goal: find a good default topk proportion balancing efficiency vs recall.
- Pre-prune heuristic quality: compare pre-prune mask vs weight-based top-k mask (recall/precision/F1).
- Encoder metric: correlation from trainer.evaluate_encoder().
- Decoder metric: <target_class> from trainer.evaluate_decoder().

Suggested settings (per user request): 1000 steps, eval_steps=500.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from matplotlib import pyplot as plt

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PrePruneConfig, PostPruneConfig
from gradiend.trainer.core.pruning import pre_prune


@dataclass
class RunResult:
    topk: float
    mode: str
    encoder_correlation: float
    decoder_prob: float
    heuristic_recall: Optional[float] = None
    heuristic_precision: Optional[float] = None
    heuristic_f1: Optional[float] = None


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def _topk_mask(importance: torch.Tensor, topk: float) -> torch.Tensor:
    if not torch.is_tensor(importance):
        raise TypeError("importance must be a torch.Tensor")
    flat = importance.detach().flatten()
    n = flat.numel()
    if n == 0:
        return torch.zeros(0, dtype=torch.bool)
    if isinstance(topk, float):
        if not (0.0 < topk <= 1.0):
            raise ValueError("topk must be in (0,1] when float")
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


def _run_encoder_eval(trainer: TextPredictionTrainer, max_size: int = 200) -> float:
    result = trainer.evaluate_encoder(max_size=max_size, use_cache=True)
    return float(result.get("correlation", 0.0)) if result else 0.0



def _run_decoder_eval(
    trainer: TextPredictionTrainer, target_class: str, max_size: int = 200
) -> float:
    result = trainer.evaluate_decoder(
        lrs=[1e-3],
        max_size_training_like=max_size,
        max_size_neutral=max_size,
        use_cache=True,
    )
    summary = result.get("summary", {}) if result else {}
    key = target_class
    if key in summary:
        return float(summary[key].get("value", 0.0))
    return 0.0


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
    """
    Compute the weight-based top-k mask (oracle) for pre-prune heuristic quality.

    Used to compare the pre-prune heuristic (gradient-mean based) against the
    post-training weight-based selection. Recall/precision/F1 measure how well
    the gradient heuristic approximates the ideal weight-based top-k.
    """
    model = trainer.get_model()
    importance = model.get_weight_importance(part=part).to("cpu")
    return _topk_mask(importance, topk)


def run_analysis(
    *,
    topk_values: List[float],
    pair: Tuple[str, str],
    output_dir: str,
) -> List[RunResult]:
    _ensure_dir(output_dir)

    base_args = dict(
        experiment_dir=output_dir,
        train_batch_size=32,
        encoder_eval_max_size=500,
        decoder_eval_max_size_training_like=500,
        decoder_eval_max_size_neutral=1000,
        eval_steps=1000,
        num_train_epochs=1,
        max_steps=5000,
        source="alternative",
        target="diff",
        eval_batch_size=8,
        learning_rate=1e-5,
        use_cache=True,
        add_identity_for_other_classes=True,
    )

    results: List[RunResult] = []
    target_class = pair[1]

    for topk in topk_values:
        # Pre-prune only
        args = TrainingArguments(
            **base_args,
            pre_prune_config=PrePruneConfig(n_samples=32, topk=topk, source="diff"),
        )
        run_id = f"prune_pre_only_topk_{topk:.4f}"
        trainer = _build_trainer(run_id=run_id, pair=pair, args=args)
        heuristic_mask = _compute_pre_prune_heuristic_mask(trainer, topk=topk)
        trainer.train()
        oracle_mask = _compute_oracle_mask(trainer, topk=topk)
        recall, precision, f1 = _mask_metrics(heuristic_mask, oracle_mask)
        enc_corr = _run_encoder_eval(trainer)
        dec_prob = _run_decoder_eval(trainer, target_class=target_class)
        results.append(
            RunResult(
                topk=topk,
                mode="pre",
                encoder_correlation=enc_corr,
                decoder_prob=dec_prob,
                heuristic_recall=recall,
                heuristic_precision=precision,
                heuristic_f1=f1,
            )
        )

        # Post-prune only
        args = TrainingArguments(
            **base_args,
            post_prune_config=PostPruneConfig(topk=topk, part="decoder-weight"),
        )
        run_id = f"prune_post_only_topk_{topk:.4f}"
        trainer = _build_trainer(run_id=run_id, pair=pair, args=args)
        trainer.train()
        enc_corr = _run_encoder_eval(trainer)
        dec_prob = _run_decoder_eval(trainer, target_class=target_class)
        results.append(
            RunResult(
                topk=topk,
                mode="post",
                encoder_correlation=enc_corr,
                decoder_prob=dec_prob,
            )
        )

        # Combined pre + post (same topk)
        args = TrainingArguments(
            **base_args,
            pre_prune_config=PrePruneConfig(n_samples=32, topk=topk, source="diff"),
            post_prune_config=PostPruneConfig(topk=topk, part="decoder-weight"),
        )
        run_id = f"prune_pre_post_topk_{topk:.4f}"
        trainer = _build_trainer(run_id=run_id, pair=pair, args=args)
        trainer.train()
        enc_corr = _run_encoder_eval(trainer)
        dec_prob = _run_decoder_eval(trainer, target_class=target_class)
        results.append(
            RunResult(
                topk=topk,
                mode="pre_post",
                encoder_correlation=enc_corr,
                decoder_prob=dec_prob,
            )
        )

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
        xs = [r.topk for r in results if r.mode == mode]
        ys = [getattr(r, metric) for r in results if r.mode == mode]
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
    pre = [r for r in results if r.mode == "pre" and r.heuristic_recall is not None]
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
    payload = [r.__dict__ for r in results]
    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    topk_values = [0.00001, 0.0002, 0.0005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.0]
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
