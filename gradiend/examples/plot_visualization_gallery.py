"""Generate representative visualization artifacts for manual approval.

This script uses synthetic data and tiny stand-ins where possible. It is meant
for visual QA of plot layout/styling, not for validating model quality.

Run:
    python gradiend/examples/plot_visualization_gallery.py

Outputs are written to runs/examples/plot_visualization_gallery by default.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from gradiend.visualizer import (
    plot_comparison_heatmap,
    plot_encoder_by_target,
    plot_encoder_distributions,
    plot_encoder_scatter,
    plot_encoder_strip_by_split,
    plot_topk_overlap_heatmap,
    plot_topk_overlap_venn,
    plot_training_convergence,
)
from gradiend.visualizer.probability_shifts import plot_probability_shifts


class _GalleryTrainer:
    run_id = "plot_visualization_gallery"
    pair = ("positive", "negative")
    target_classes = ["positive", "negative"]
    all_classes = ["positive", "negative"]
    _id2label = {}
    config = SimpleNamespace(img_format="png")
    training_args = SimpleNamespace(highlight_non_convergence=False)

    def get_model(self):
        return None


class _DummyTopKModel:
    def __init__(self, weights, *, total_count=None):
        self._weights = list(weights)
        self._total_count = int(total_count) if total_count is not None else len(self._weights)
        self.name_or_path = "gallery-model"

    def get_topk_weights(self, part="decoder-weight", topk=100):
        if isinstance(topk, float):
            k = max(1, int(math.ceil(topk * self._total_count)))
        else:
            k = int(topk)
        return self._weights[:k]


def _make_encoder_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    targets = {
        "positive": ["good", "nice", "helpful", "calm"],
        "negative": ["bad", "awful", "harsh", "cold"],
    }
    split_shift = {"train": 0.18, "validation": 0.0, "test": -0.18}
    for cls, tokens in targets.items():
        sign = 1.0 if cls == "positive" else -1.0
        other = "negative" if cls == "positive" else "positive"
        for split, shift in split_shift.items():
            for token in tokens:
                for rep in range(4):
                    value = sign * (0.55 + shift + rng.normal(0.0, 0.08))
                    rows.append(
                        {
                            "encoded": value,
                            "label": sign,
                            "source_id": cls,
                            "target_id": other,
                            "feature_class": cls,
                            "factual_token": token,
                            "target_token": token,
                            "data_split": split,
                            "type": "training",
                            "text": f"The review was [MASK] ({token}, {split}, #{rep}).",
                        }
                    )
    for split in ("validation", "test"):
        for rep in range(8):
            rows.append(
                {
                    "encoded": rng.normal(0.0, 0.12),
                    "label": 0.0,
                    "source_id": "neutral",
                    "target_id": "neutral",
                    "feature_class": "neutral",
                    "factual_token": "neutral",
                    "target_token": "neutral",
                    "data_split": split,
                    "type": "neutral_dataset",
                    "text": f"Neutral held-out sentence {rep}.",
                }
            )
    return pd.DataFrame(rows)


def _make_training_stats() -> dict:
    return {
        "training_stats": {
            "mean_by_class": {
                0: {"1": 0.05, "-1": -0.04, "0": 0.0},
                1: {"1": 0.22, "-1": -0.20, "0": 0.01},
                2: {"1": 0.45, "-1": -0.42, "0": 0.02},
                3: {"1": 0.64, "-1": -0.59, "0": 0.01},
            },
            "mean_by_feature_class": {
                0: {"positive": 0.05, "negative": -0.04, "neutral": 0.0},
                1: {"positive": 0.22, "negative": -0.20, "neutral": 0.01},
                2: {"positive": 0.45, "negative": -0.42, "neutral": 0.02},
                3: {"positive": 0.64, "negative": -0.59, "neutral": 0.01},
            },
            "scores": {0: 0.12, 1: 0.42, 2: 0.71, 3: 0.83},
        },
        "best_score_checkpoint": {"global_step": 3},
    }


def _make_decoder_plot_data():
    lrs = [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    grid = {
        "base": {
            "probs_by_dataset": {
                "positive": {"positive": 0.70, "negative": 0.30},
                "negative": {"positive": 0.28, "negative": 0.72},
            },
            "lms": {"lms": 0.56},
        }
    }
    for lr in lrs[1:]:
        gain = min(0.22, math.log10(lr * 10000 + 1) * 0.055)
        grid[(-1.0, lr)] = {
            "id": {"feature_factor": -1.0, "learning_rate": lr},
            "probs_by_dataset": {
                "positive": {"positive": 0.70 + gain, "negative": 0.30 - gain},
                "negative": {"positive": 0.28 + gain * 0.35, "negative": 0.72 - gain * 0.35},
            },
            "lms": {"lms": 0.56 - gain * 0.45},
        }
    decoder_results = {
        "positive": {"learning_rate": 0.1, "feature_factor": -1.0, "value": 0.89},
        "negative": {"learning_rate": 0.1, "feature_factor": -1.0, "value": 0.40},
        "grid": {},
    }
    return decoder_results, {"plotting_data": grid}


def _make_comparison_data() -> dict:
    return {
        "matrix": [
            [1.00, 0.72, 0.35, 0.28],
            [0.72, 1.00, 0.31, 0.26],
            [0.35, 0.31, 1.00, 0.81],
            [0.28, 0.26, 0.81, 1.00],
        ],
        "model_ids": ["sent_pos_neg", "sent_pos_neu", "gender_sg_pl", "gender_masc_fem"],
        "measure": "cosine",
        "row_labels": {
            "sent_pos_neg": "sentiment pos/neg",
            "sent_pos_neu": "sentiment pos/neutral",
            "gender_sg_pl": "pronoun sg/pl",
            "gender_masc_fem": "gender masc/fem",
        },
    }


def _make_layerwise_similarity_data() -> dict:
    return {
        "measure": "layerwise_similarity",
        "model_ids": ["mean"],
        "column_ids": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
        "matrix": [
            [0.18, 0.21, 0.26, 0.35, 0.48, 0.61, 0.72, 0.76, 0.69, 0.55, 0.42, 0.31],
        ],
        "row_labels": {
            "mean": "mean pairwise cosine",
        },
    }


def generate_gallery(output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer = _GalleryTrainer()
    encoder_df = _make_encoder_df()
    written: list[Path] = []

    paths = {
        "training": output_dir / "training_convergence.png",
        "encoder_distribution": output_dir / "encoder_distributions_facet.png",
        "encoder_by_target_strip": output_dir / "encoder_by_target_strip.png",
        "encoder_by_target_box": output_dir / "encoder_by_target_box.png",
        "encoder_by_target_violin": output_dir / "encoder_by_target_violin.png",
        "encoder_by_target_interactive": output_dir / "encoder_by_target_strip_interactive.html",
        "encoder_strip": output_dir / "encoder_strip_by_split.png",
        "encoder_scatter": output_dir / "encoder_scatter.html",
        "probability": output_dir / "probability_shifts.png",
        "comparison": output_dir / "comparison_heatmap_grouped.png",
        "comparison_rectangular": output_dir / "comparison_heatmap_rectangular.png",
        "comparison_row_metric": output_dir / "comparison_heatmap_row_metric.png",
        "comparison_dispersion": output_dir / "comparison_heatmap_dispersion.png",
        "layerwise_similarity": output_dir / "layerwise_similarity_heatmap.png",
        "topk_heatmap": output_dir / "topk_overlap_heatmap.png",
        "topk_venn": output_dir / "topk_overlap_venn.png",
    }

    plot_training_convergence(training_stats=_make_training_stats(), output=str(paths["training"]), show=False)
    written.append(paths["training"])

    plot_encoder_distributions(
        trainer,
        encoder_df=encoder_df,
        output=str(paths["encoder_distribution"]),
        split_plot_mode="facet",
        include_neutral=True,
        target_and_neutral_only=False,
        show=False,
    )
    written.append(paths["encoder_distribution"])

    for style, key in (
        ("strip", "encoder_by_target_strip"),
        ("box", "encoder_by_target_box"),
        ("violin", "encoder_by_target_violin"),
    ):
        plot_encoder_by_target(
            encoder_df=encoder_df,
            plot_style=style,
            output=str(paths[key]),
            legend_loc="lower left",
            show=False,
            title=f"Encoder by target ({style})",
        )
        written.append(paths[key])

    interactive_by_target = plot_encoder_by_target(
        encoder_df=encoder_df,
        plot_style="strip",
        interactive=True,
        output=str(paths["encoder_by_target_interactive"]),
        show=False,
        title="Interactive encoder by target",
    )
    if interactive_by_target is not None and paths["encoder_by_target_interactive"].exists():
        written.append(paths["encoder_by_target_interactive"])

    plot_encoder_strip_by_split(
        encoder_df=encoder_df,
        output=str(paths["encoder_strip"]),
        label_points="outliers+sample",
        label_sample_per_group=2,
        show=False,
        title="Encoder strip by split",
    )
    written.append(paths["encoder_strip"])

    scatter_result = plot_encoder_scatter(
        trainer=trainer,
        encoder_df=encoder_df,
        output_path=str(paths["encoder_scatter"]),
        show=False,
        title="Encoder scatter",
        max_points=80,
    )
    if paths["encoder_scatter"].exists():
        written.append(paths["encoder_scatter"])
    elif scatter_result is None:
        print("Skipped encoder scatter: Plotly is not installed or no scatter figure was returned.")

    decoder_results, plotting_data = _make_decoder_plot_data()
    plot_probability_shifts(
        trainer=trainer,
        decoder_results=decoder_results,
        plotting_data=plotting_data,
        class_ids=["positive", "negative"],
        target_class="positive",
        output=str(paths["probability"]),
        show=False,
    )
    written.append(paths["probability"])

    plot_comparison_heatmap(
        _make_comparison_data(),
        pretty_groups={
            "Sentiment": ["sent_pos_neg", "sent_pos_neu"],
            "Morphology": ["gender_sg_pl", "gender_masc_fem"],
        },
        output_path=str(paths["comparison"]),
        show=False,
        title="Grouped comparison heatmap",
    )
    written.append(paths["comparison"])

    rectangular = {
        "measure": "gradiend_feature_cross_encoding_mean",
        "model_ids": ["sent_pos_neg", "gender_sg_pl", "gender_masc_fem"],
        "column_ids": ["positive", "negative", "3SG", "3PL", "masc", "fem"],
        "matrix": [
            [0.71, -0.64, 0.08, -0.02, 0.01, -0.01],
            [0.02, -0.04, 0.66, -0.61, 0.12, -0.10],
            [0.01, -0.03, 0.10, -0.08, 0.58, -0.55],
        ],
        "row_labels": {
            "sent_pos_neg": "sentiment pos/neg",
            "gender_sg_pl": "pronoun sg/pl",
            "gender_masc_fem": "gender masc/fem",
        },
    }
    plot_comparison_heatmap(
        rectangular,
        output_path=str(paths["comparison_rectangular"]),
        show=False,
        title="Rectangular cross-encoding heatmap",
    )
    written.append(paths["comparison_rectangular"])

    plot_comparison_heatmap(
        _make_comparison_data(),
        row_metric={
            "sent_pos_neg": 0.84,
            "sent_pos_neu": 0.73,
            "gender_sg_pl": 0.79,
            "gender_masc_fem": 0.62,
        },
        row_metric_label="corr",
        output_path=str(paths["comparison_row_metric"]),
        show=False,
        title="Comparison heatmap with row metric",
    )
    written.append(paths["comparison_row_metric"])

    dispersion = _make_comparison_data()
    dispersion["cell_stats"] = [
        [{"std": 0.00}, {"std": 0.05}, {"std": 0.09}, {"std": 0.07}],
        [{"std": 0.05}, {"std": 0.00}, {"std": 0.06}, {"std": 0.08}],
        [{"std": 0.09}, {"std": 0.06}, {"std": 0.00}, {"std": 0.04}],
        [{"std": 0.07}, {"std": 0.08}, {"std": 0.04}, {"std": 0.00}],
    ]
    dispersion["dispersion"] = "std"
    dispersion["global_n"] = 3
    plot_comparison_heatmap(
        dispersion,
        output_path=str(paths["comparison_dispersion"]),
        show=False,
        title="Comparison heatmap with seed dispersion",
        dispersion_display="stacked",
        seed_annotation=True,
    )
    written.append(paths["comparison_dispersion"])

    plot_comparison_heatmap(
        _make_layerwise_similarity_data(),
        output_path=str(paths["layerwise_similarity"]),
        show=False,
        title="Layer-wise similarity overview",
        vmin=0.0,
        vmax=1.0,
    )
    written.append(paths["layerwise_similarity"])

    topk_models = {
        "A": _DummyTopKModel([1, 2, 3, 4, 5, 6]),
        "B": _DummyTopKModel([3, 4, 5, 6, 7, 8]),
        "C": _DummyTopKModel([5, 6, 7, 8, 9, 10]),
    }
    plot_topk_overlap_heatmap(
        topk_models,
        topk=6,
        value="intersection_frac",
        output_path=str(paths["topk_heatmap"]),
        show=False,
        title="Top-k overlap heatmap",
    )
    written.append(paths["topk_heatmap"])

    try:
        plot_topk_overlap_venn(
            topk_models,
            topk=6,
            output_path=str(paths["topk_venn"]),
            show=False,
            title="Top-k overlap Venn",
        )
        written.append(paths["topk_venn"])
    except ImportError as exc:
        print(f"Skipped Venn plot: {exc}")

    return [path for path in written if path.exists()]


if __name__ == "__main__":
    written = generate_gallery(Path("runs") / "examples" / "plot_visualization_gallery")
    print("Generated visualization approval artifacts:")
    for path in written:
        print(f"  {path}")