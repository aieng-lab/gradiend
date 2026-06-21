"""
Multi-seed stability demo on the race bias feature (HuggingFace data).

Uses aieng-lab/gradiend_race_data (white vs black) and aieng-lab/biasneutral.
Trains until two seeds converge, keeps convergent checkpoints, then compares:

- single-seed evaluation on the selected best model
- multi-seed evaluation (mean correlation + seeds.stats dispersion)

Run from repo root:

    python -m gradiend.examples.train_multi_seed_stability

See docs/design/multi-seed-analysis.md for the full design.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from gradiend import (
    TextPredictionTrainer,
    TrainingArguments,
    compute_grouped_similarity_matrices,
    compute_similarity_matrix,
    plot_comparison_heatmap,
)

MODEL = "distilbert-base-cased"
DATASET = "aieng-lab/gradiend_race_data"
NEUTRAL = "aieng-lab/biasneutral"
TARGET_CLASSES = ("white", "black")
RUN_ID = "race_white_black"
EXPERIMENT_DIR = "runs/examples/train_multi_seed_stability"


def _mean_off_diagonal(matrix) -> float:
    values = [
        float(value)
        for row_idx, row in enumerate(matrix)
        for col_idx, value in enumerate(row)
        if row_idx != col_idx
    ]
    return sum(values) / len(values) if values else 0.0


def _plot_layerwise_seed_similarity(trainer: TextPredictionTrainer, view) -> str | None:
    seed_paths = view.seed_paths()
    models = view.load_models()
    if len(models) < 2:
        print("  layer-wise similarity skipped: need at least two saved convergent seed models")
        return None

    labels = [Path(path).name for path in seed_paths]
    models_by_seed = {label: model for label, model in zip(labels, models)}
    grouped = compute_grouped_similarity_matrices(
        models_by_seed,
        measure="cosine",
        part="decoder-weight",
        topk=1000,
        group_by="layer",
    )
    layer_scores = sorted(
        (
            (name, _mean_off_diagonal(data["matrix"]))
            for name, data in grouped.items()
            if name.startswith("layer_")
        ),
        key=lambda item: int(item[0].split("_", 1)[1]),
    )
    if not layer_scores:
        print("  layer-wise similarity skipped: no layer-specific parameter groups found")
        return None

    comparison_data = {
        "measure": "layerwise_similarity",
        "part": "decoder-weight",
        "model_ids": ["mean"],
        "column_ids": [name.split("_", 1)[1] for name, _ in layer_scores],
        "matrix": [[value for _, value in layer_scores]],
        "row_labels": {"mean": "mean pairwise cosine"},
    }
    output_path = os.path.join(EXPERIMENT_DIR, RUN_ID, "layerwise_seed_similarity.png")
    plot_comparison_heatmap(
        comparison_data,
        output_path=output_path,
        title="Layer-wise seed similarity",
        show=False,
        vmin=0.0,
        vmax=1.0,
    )
    return output_path


def _plot_seed_comparison_examples(view) -> dict[str, str]:
    seed_paths = view.seed_paths()
    models = view.load_models()
    if len(models) < 2:
        print("  seed comparisons skipped: need at least two saved convergent seed models")
        return {}

    labels = [Path(path).name for path in seed_paths]
    models_by_seed = {label: model for label, model in zip(labels, models)}
    output_dir = os.path.join(EXPERIMENT_DIR, RUN_ID)
    comparisons = {
        "topk_overlap": compute_similarity_matrix(
            models_by_seed,
            measure="topk_overlap",
            part="decoder-weight",
            topk=1000,
        ),
        "decoder_cosine": compute_similarity_matrix(
            models_by_seed,
            measure="cosine",
            part="decoder-weight",
            topk=1000,
        ),
        "mass_overlap": compute_similarity_matrix(
            models_by_seed,
            measure="mass_overlap",
            part="decoder-weight",
            topk=1000,
        ),
    }
    paths: dict[str, str] = {}
    for name, comparison_data in comparisons.items():
        output_path = os.path.join(output_dir, f"seed_comparison_{name}.png")
        plot_comparison_heatmap(
            comparison_data,
            output_path=output_path,
            title=f"Seed comparison: {name.replace('_', ' ')}",
            show=False,
            vmin=0.0,
            vmax=1.0,
        )
        paths[name] = output_path

    grouped_components = compute_grouped_similarity_matrices(
        models_by_seed,
        measure="cosine",
        part="decoder-weight",
        topk=1000,
        group_by="component",
    )
    component_scores = sorted(
        (
            (name, _mean_off_diagonal(data["matrix"]))
            for name, data in grouped_components.items()
            if name != "other"
        ),
        key=lambda item: item[0],
    )
    if component_scores:
        comparison_data = {
            "measure": "component_similarity",
            "part": "decoder-weight",
            "model_ids": ["mean"],
            "column_ids": [name for name, _ in component_scores],
            "matrix": [[value for _, value in component_scores]],
            "row_labels": {"mean": "mean pairwise cosine"},
        }
        output_path = os.path.join(output_dir, "component_seed_similarity.png")
        plot_comparison_heatmap(
            comparison_data,
            output_path=output_path,
            title="Component-wise seed similarity",
            show=False,
            vmin=0.0,
            vmax=1.0,
        )
        paths["component_similarity"] = output_path
    return paths

if __name__ == "__main__":
    args = TrainingArguments(
        experiment_dir=EXPERIMENT_DIR,
        train_batch_size=8,
        encoder_eval_max_size=10,
        decoder_eval_max_size_training_like=50,
        decoder_eval_max_size_neutral=50,
        eval_steps=25,
        max_steps=100,
        source="alternative",
        target="diff",
        learning_rate=1e-4,
        use_cache=True,
        max_seeds=3,
        min_convergent_seeds=2,
        fail_on_non_convergence=True,
        analyze_seed_stability=True,
        saved_seed_runs="all_convergent",
    )

    trainer = TextPredictionTrainer(
        model=MODEL,
        run_id=RUN_ID,
        data=DATASET,
        target_classes=TARGET_CLASSES,
        masked_col="masked",
        eval_neutral_data=NEUTRAL,
        img_format="png",
        args=args,
    )

    print(f"=== Race ({TARGET_CLASSES[0]} vs {TARGET_CLASSES[1]}): multi-seed training ===")
    trainer.train()
    print(f"Best model: {trainer.model_path}")

    report = trainer.get_seed_report() or {}
    print(
        f"Convergent seeds: {report.get('convergent_seeds')} "
        f"(count={report.get('convergent_count')})"
    )

    eval_kwargs = {"split": "test", "max_size": 50, "plot": False, "use_cache": True}

    print("\n=== Single-seed eval (best model) ===")
    single = trainer.evaluate_encoder(**eval_kwargs)
    print(f"  correlation = {single.get('correlation')!r}")

    print("\n=== Multi-seed eval (trainer.multi_seed()) ===")
    view = trainer.multi_seed(selection="all_convergent", dispersion="std")
    print(f"  seed paths: {view.seed_paths()}")
    multi = view.evaluate_encoder(**eval_kwargs)
    corr_stats = multi.get("seeds", {}).get("stats", {}).get("correlation", {})
    print(f"  mean correlation = {multi.get('correlation')!r}")
    print(f"  seeds.stats.correlation = {json.dumps(corr_stats, indent=2)}")

    print("\n=== Multi-seed encoder plots (one figure per convergent seed) ===")
    plot_result = view.plot_encoder_distributions(show=False, max_size=50)
    print(f"  plot paths: {plot_result.get('paths')}")

    print("\n=== Layer-wise similarity across convergent seeds ===")
    layer_plot = _plot_layerwise_seed_similarity(trainer, view)
    print(f"  layer-wise plot: {layer_plot}")

    print("\n=== Seed comparison heatmaps ===")
    comparison_plots = _plot_seed_comparison_examples(view)
    print(f"  comparison plots: {json.dumps(comparison_plots, indent=2)}")

    print("\n=== Done ===")
    print("Single-seed API is unchanged; use trainer.multi_seed() for stability analysis.")
