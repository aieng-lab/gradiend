"""
End-to-end multi-seed analysis demo (matches docs/guides/multi-seed.md).

Uses aieng-lab/gradiend_race_data (white vs black) and aieng-lab/biasneutral.
Trains until two seeds converge, then exercises:

- single-seed vs multi-seed evaluation (encoder, decoder, full evaluate)
- MultiSeedTrainerView options (selection, aggregate, dispersion, return_per_seed)
- get_model() / get_model(gradiend_only=True) -> SeedModelGroup
- encoder-by-target plots (per-seed grid, combined strip, errorbar ± std, interactive)
- per-seed encoder distributions, scatter, probability shifts, training convergence
- seed comparison heatmaps (top-k overlap, decoder cosine, layer-wise, component)

Run from repo root:

    python -m gradiend.examples.train_multi_seed_stability
    python -m gradiend.examples.train_multi_seed_stability --plot-only
    python -m gradiend.examples.train_multi_seed_stability --write-docs-images

Verify without full training:

    pytest tests/test_multi_seed_view.py tests/test_multi_seed_analysis_integration.py -q
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from gradiend import (
    TextPredictionTrainer,
    TrainingArguments,
    compute_grouped_similarity_matrices,
    compute_similarity_matrix,
    plot_comparison_heatmap,
)
from gradiend.trainer.core.seed_models import SeedModelGroup

MODEL = "distilbert-base-cased"
DATASET = "aieng-lab/gradiend_race_data"
NEUTRAL = "aieng-lab/biasneutral"
TARGET_CLASSES = ("white", "black")
RUN_ID = "race_white_black"
EXPERIMENT_DIR = "runs/examples/train_multi_seed_stability"
PLOT_DIR = os.path.join(EXPERIMENT_DIR, RUN_ID, "plots")

DOCS_IMAGES: dict[str, str] = {
    "encoder_by_target_seeds": "docs/img/multi_seed_encoder_by_target_seeds.png",
    "encoder_by_target_combined": "docs/img/multi_seed_encoder_by_target_combined.png",
    "encoder_by_target_errorbar": "docs/img/multi_seed_encoder_by_target_errorbar.png",
    "encoder_distributions": "docs/img/multi_seed_encoder_distributions.png",
    "encoder_scatter": "docs/img/multi_seed_encoder_scatter.png",
    "probability_shifts": "docs/img/multi_seed_probability_shifts.png",
    "training_convergence": "docs/img/multi_seed_training_convergence.png",
    "layerwise_similarity": "docs/img/multi_seed_layerwise_similarity.png",
    "seed_comparison_topk_overlap": "docs/img/seed_comparison_topk_overlap.png",
    "seed_comparison_decoder_cosine": "docs/img/seed_comparison_decoder_cosine.png",
    "component_similarity": "docs/img/multi_seed_component_similarity.png",
}


def _plot_path(name: str, *, ext: str = "png") -> str:
    os.makedirs(PLOT_DIR, exist_ok=True)
    return os.path.join(PLOT_DIR, f"{name}.{ext}")


def _mean_off_diagonal(matrix) -> float:
    values = [
        float(value)
        for row_idx, row in enumerate(matrix)
        for col_idx, value in enumerate(row)
        if row_idx != col_idx
    ]
    return sum(values) / len(values) if values else 0.0


def _copy_docs_image(source: str | None, docs_key: str) -> None:
    if not source or not os.path.isfile(source):
        return
    dest = DOCS_IMAGES[docs_key]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(source, dest)
    print(f"  copied -> {dest}")  # copied -> docs/img/multi_seed_encoder_by_target_seeds.png


def build_trainer() -> TextPredictionTrainer:
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
        analyze_seed_stability=True,  # required for multi_seed(); sets saved_seed_runs to "all_convergent"
        saved_seed_runs="all_convergent",
    )
    return TextPredictionTrainer(
        model=MODEL,
        run_id=RUN_ID,
        data=DATASET,
        target_classes=TARGET_CLASSES,
        masked_col="masked",
        eval_neutral_data=NEUTRAL,
        img_format="png",
        args=args,
    )


def run_training(trainer: TextPredictionTrainer) -> None:
    print(f"=== Race ({TARGET_CLASSES[0]} vs {TARGET_CLASSES[1]}): multi-seed training ===")
    trainer.train()
    print(f"Best model: {trainer.model_path}")  # Best model: runs/examples/train_multi_seed_stability/race_white_black/model

    report = trainer.get_seed_report() or {}
    print(
        f"Convergent seeds: {report.get('convergent_seeds')} "
        f"(count={report.get('convergent_count')})"
    )  # Convergent seeds: [42, 7] (count=2)


def run_single_seed_eval(trainer: TextPredictionTrainer, eval_kwargs: dict[str, Any]) -> dict[str, Any]:
    print("\n=== Single-seed eval (best model) ===")
    single = trainer.evaluate_encoder(**eval_kwargs)
    print(f"  correlation = {single.get('correlation')!r}")  # correlation = 0.8123
    return single


def run_multi_seed_eval(trainer: TextPredictionTrainer, eval_kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    print("\n=== Multi-seed view (selection, aggregate, dispersion) ===")
    view = trainer.multi_seed(
        selection="all_convergent",  # best | all_convergent | all_tried
        aggregate="mean",            # mean | median | min | max
        dispersion="std",            # none | std | range | minmax
    )
    print(f"  seed paths: {view.seed_paths()}")  # seed paths: ['.../seeds/seed_42', '.../seeds/seed_7']
    print(f"  seed values: {view.seed_values()}")  # seed values: [42, 7]

    print("\n=== Multi-seed encoder eval ===")
    multi = view.evaluate_encoder(**eval_kwargs)
    corr_stats = multi.get("seeds", {}).get("stats", {}).get("correlation", {})
    print(f"  mean correlation = {multi.get('correlation')!r}")  # mean correlation = 0.7981
    print(f"  seeds.n = {multi.get('seeds', {}).get('n')!r}")  # seeds.n = 2
    print(f"  seeds.values = {multi.get('seeds', {}).get('values')!r}")  # seeds.values = [42, 7]
    print(f"  seeds.stats.correlation = {json.dumps(corr_stats, indent=2)}")
    # seeds.stats.correlation = {
    #   "mean": 0.7981,
    #   "std": 0.0142,
    #   "min": 0.7839,
    #   "max": 0.8123,
    #   "n": 2
    # }

    print("\n=== Multi-seed evaluate() / evaluate_decoder() ===")
    full_eval = view.evaluate(**eval_kwargs)
    enc_corr = full_eval.get("encoder", {}).get("correlation")
    dec_corr = full_eval.get("decoder", {}).get("correlation")
    print(f"  evaluate().encoder.correlation = {enc_corr!r}")  # evaluate().encoder.correlation = 0.7981
    print(f"  evaluate().decoder.correlation = {dec_corr!r}")  # evaluate().decoder.correlation = 0.65

    print("\n=== return_per_seed=True ===")
    per_seed = view.evaluate_encoder(return_per_seed=True, **eval_kwargs)
    per_seed_keys = sorted((per_seed.get("seeds") or {}).get("per_seed", {}).keys())
    print(f"  per_seed keys = {per_seed_keys}")  # per_seed keys = [42, 7]

    print("\n=== evaluate_encoder(return_df=True) ===")
    enc_df = view.evaluate_encoder(return_df=True, **eval_kwargs)
    frame = enc_df.get("encoder_df")
    n_rows = len(frame) if frame is not None else 0
    print(f"  encoder_df rows = {n_rows}")  # encoder_df rows = 120
    print(f"  encoder_df columns include source_token = {'source_token' in getattr(frame, 'columns', [])}")  # True

    return view, multi


def run_model_loading(view) -> None:
    print("\n=== get_model() — full checkpoints ===")
    group = view.get_model()
    if isinstance(group, SeedModelGroup):
        print(f"  SeedModelGroup len = {len(group)}")  # SeedModelGroup len = 2
        print(f"  SeedModelGroup.primary = {type(group.primary).__name__}")  # TextPredictionModelWithGradiend
        print(f"  list(group) -> {len(list(group))} models")  # list(group) -> 2 models
    else:
        print(f"  single model type = {type(group).__name__}")  # TextPredictionModelWithGradiend

    print("\n=== get_model(gradiend_only=True) — weight-space comparison ===")
    gradiend_group = view.get_model(gradiend_only=True)
    if isinstance(gradiend_group, SeedModelGroup):
        print(f"  gradiend_only SeedModelGroup len = {len(gradiend_group)}")  # gradiend_only SeedModelGroup len = 2
    else:
        print(f"  gradiend_only model type = {type(gradiend_group).__name__}")  # _GradiendOnlyModel


def run_encoder_by_target_plots(view, *, write_docs_images: bool) -> dict[str, str | None]:
    print("\n=== plot_encoder_by_target (multi-seed variants) ===")
    plot_kwargs = {"split": "test", "max_size": 50, "show": False}
    paths: dict[str, str | None] = {}

    paths["encoder_by_target_seeds"] = view.plot_encoder_by_target(
        output=_plot_path("encoder_by_target_seeds"),
        **plot_kwargs,
    ).get("path")
    print(f"  per-seed grid: {paths['encoder_by_target_seeds']}")
    # per-seed grid: runs/examples/train_multi_seed_stability/race_white_black/plots/encoder_by_target_seeds.png

    paths["encoder_by_target_combined"] = view.plot_encoder_by_target(
        output=_plot_path("encoder_by_target_combined"),
        combine_seed_rows=True,
        **plot_kwargs,
    ).get("path")
    print(f"  combined strip: {paths['encoder_by_target_combined']}")

    paths["encoder_by_target_errorbar"] = view.plot_encoder_by_target(
        output=_plot_path("encoder_by_target_errorbar"),
        plot_style="errorbar",
        error_stat="std",
        show_seed_points=True,
        **plot_kwargs,
    ).get("path")
    print(f"  errorbar ± std: {paths['encoder_by_target_errorbar']}")

    try:
        paths["encoder_by_target_interactive"] = view.plot_encoder_by_target(
            output=_plot_path("encoder_by_target_interactive", ext="html"),
            interactive=True,
            **plot_kwargs,
        ).get("path")
        print(f"  interactive html: {paths['encoder_by_target_interactive']}")
    except ImportError as exc:
        print(f"  interactive html skipped: {exc}")  # interactive html skipped: install gradiend[plot]

    if write_docs_images:
        print("  copying encoder-by-target doc images...")
        _copy_docs_image(paths.get("encoder_by_target_seeds"), "encoder_by_target_seeds")
        _copy_docs_image(paths.get("encoder_by_target_combined"), "encoder_by_target_combined")
        _copy_docs_image(paths.get("encoder_by_target_errorbar"), "encoder_by_target_errorbar")

    return paths


def run_other_encoder_plots(view, *, write_docs_images: bool) -> dict[str, Any]:
    print("\n=== Per-seed plot methods (paths list) ===")
    eval_plot_kwargs = {"split": "test", "max_size": 50, "show": False, "use_cache": True}
    outputs: dict[str, Any] = {}

    dist = view.plot_encoder_distributions(**eval_plot_kwargs)
    outputs["encoder_distributions"] = dist
    print(f"  plot_encoder_distributions paths: {dist.get('paths')}")
    # plot_encoder_distributions paths: ['.../seed_42/encoder_distributions.png', '.../seed_7/encoder_distributions.png']

    scatter = view.plot_encoder_scatter(**eval_plot_kwargs)
    outputs["encoder_scatter"] = scatter
    print(f"  plot_encoder_scatter paths: {scatter.get('paths')}")

    conv = view.plot_training_convergence(show=False)
    outputs["training_convergence"] = conv
    print(f"  plot_training_convergence paths: {conv.get('paths')}")

    prob = view.plot_probability_shifts(show=False, use_cache=True)
    outputs["probability_shifts"] = prob
    print(f"  plot_probability_shifts paths: {prob.get('paths')}")

    if write_docs_images:
        print("  copying per-seed doc images (first path per plot)...")
        first_dist = (dist.get("paths") or [None])[0]
        first_scatter = (scatter.get("paths") or [None])[0]
        first_conv = (conv.get("paths") or [None])[0]
        first_prob = (prob.get("paths") or [None])[0]
        _copy_docs_image(first_dist, "encoder_distributions")
        _copy_docs_image(first_scatter, "encoder_scatter")
        _copy_docs_image(first_conv, "training_convergence")
        _copy_docs_image(first_prob, "probability_shifts")

    return outputs


def run_seed_comparison_heatmaps(view, *, write_docs_images: bool) -> dict[str, str]:
    print("\n=== Seed comparison heatmaps ===")
    seed_paths = view.seed_paths()
    models = view.load_models()
    if len(models) < 2:
        print("  skipped: need at least two convergent seed checkpoints")  # skipped: need at least two convergent seed checkpoints
        return {}

    labels = [Path(path).name for path in seed_paths]
    models_by_seed = {label: model for label, model in zip(labels, models)}
    paths: dict[str, str] = {}

    layer_path = _plot_layerwise_seed_similarity(models_by_seed)
    if layer_path:
        paths["layerwise_similarity"] = layer_path
        print(f"  layer-wise: {layer_path}")
    paths.update(_plot_pairwise_seed_comparisons(models_by_seed))

    if write_docs_images:
        print("  copying seed-comparison doc images...")
        _copy_docs_image(paths.get("layerwise_similarity"), "layerwise_similarity")
        _copy_docs_image(paths.get("topk_overlap"), "seed_comparison_topk_overlap")
        _copy_docs_image(paths.get("decoder_cosine"), "seed_comparison_decoder_cosine")
        _copy_docs_image(paths.get("component_similarity"), "component_similarity")

    print(f"  comparison plots: {json.dumps(paths, indent=2)}")
    return paths


def _plot_layerwise_seed_similarity(models_by_seed: dict[str, Any]) -> str | None:
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
        print("  layer-wise skipped: no layer groups found")
        return None

    comparison_data = {
        "measure": "layerwise_similarity",
        "part": "decoder-weight",
        "model_ids": ["mean"],
        "column_ids": [name.split("_", 1)[1] for name, _ in layer_scores],
        "matrix": [[value for _, value in layer_scores]],
        "row_labels": {"mean": "mean pairwise cosine"},
    }
    output_path = _plot_path("layerwise_seed_similarity")
    plot_comparison_heatmap(
        comparison_data,
        output_path=output_path,
        title="Layer-wise seed similarity",
        show=False,
        vmin=0.0,
        vmax=1.0,
    )
    return output_path


def _plot_pairwise_seed_comparisons(models_by_seed: dict[str, Any]) -> dict[str, str]:
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
    }
    paths: dict[str, str] = {}
    for name, comparison_data in comparisons.items():
        output_path = _plot_path(f"seed_comparison_{name}")
        plot_comparison_heatmap(
            comparison_data,
            output_path=output_path,
            title=f"Seed comparison: {name.replace('_', ' ')}",
            show=False,
            vmin=0.0,
            vmax=1.0,
        )
        paths[name] = output_path
        print(f"  {name}: {output_path}")

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
        output_path = _plot_path("component_seed_similarity")
        plot_comparison_heatmap(
            comparison_data,
            output_path=output_path,
            title="Component-wise seed similarity",
            show=False,
            vmin=0.0,
            vmax=1.0,
        )
        paths["component_similarity"] = output_path
        print(f"  component_similarity: {output_path}")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-seed analysis demo (docs/guides/multi-seed.md).")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training; run analysis/plots from cached checkpoints.",
    )
    parser.add_argument(
        "--write-docs-images",
        action="store_true",
        help="Copy generated plots into docs/img/ for documentation.",
    )
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    trainer = build_trainer()
    eval_kwargs = {"split": "test", "max_size": 50, "plot": False, "use_cache": True}

    if not cli.plot_only:
        run_training(trainer)
    else:
        print("=== Plot-only mode (using cached checkpoints) ===")

    run_single_seed_eval(trainer, eval_kwargs)
    view, _multi = run_multi_seed_eval(trainer, eval_kwargs)
    run_model_loading(view)
    run_encoder_by_target_plots(view, write_docs_images=cli.write_docs_images)
    run_other_encoder_plots(view, write_docs_images=cli.write_docs_images)
    run_seed_comparison_heatmaps(view, write_docs_images=cli.write_docs_images)

    print("\n=== Done ===")
    print("Single-seed API unchanged; use trainer.multi_seed() for stability analysis.")
    # Single-seed API unchanged; use trainer.multi_seed() for stability analysis.


if __name__ == "__main__":
    main()
