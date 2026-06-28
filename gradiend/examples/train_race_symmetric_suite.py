"""
SymmetricTrainerSuite example: three race GRADIENDs (asian, black, white).

Runnable counterpart to ``docs/guides/trainer-suites.md``. Trains one GRADIEND
per unordered pair, then writes suite comparison plots.

Documentation images (``--write-docs-images`` copies into ``docs/img/``):

- ``symmetric_suite_topk_overlap.png`` — from ``plot_topk_overlap_heatmap``
- ``symmetric_suite_cross_encoding.png`` — from ``plot_cross_encoding_heatmap``

Requires: pip install gradiend[data]

Run:
    python -m gradiend.examples.train_race_symmetric_suite
    python -m gradiend.examples.train_race_symmetric_suite --plot-only --write-docs-images
    python -m gradiend.examples.train_race_symmetric_suite --write-docs-images
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from gradiend.util.hf_env import configure_hf_download_env

configure_hf_download_env()

from gradiend import (
    SymmetricTrainerSuite,
    TextPredictionTrainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "runs" / "examples" / "race_symmetric_suite"
DOCS_IMG_DIR = PROJECT_ROOT / "docs" / "img"

RACE_CLASSES = ["asian", "black", "white"]
TOPK_OVERLAP_PLOT = "suite_topk_overlap.png"
CROSS_ENCODING_PLOT = "suite_cross_encoding_counterfactual.png"
DOCS_TOPK_OVERLAP = "symmetric_suite_topk_overlap.png"
DOCS_CROSS_ENCODING = "symmetric_suite_cross_encoding.png"


def build_suite(*, max_steps: int = 150) -> SymmetricTrainerSuite:
    return SymmetricTrainerSuite(
        TextPredictionTrainer,
        model="distilbert-base-cased",
        data="aieng-lab/gradiend_race_data",
        eval_neutral_data="aieng-lab/biasneutral",
        target_classes=RACE_CLASSES,
        retain_models_in_memory=False,
        args=TrainingArguments(
            experiment_dir=str(EXPERIMENT_DIR),
            train_batch_size=8,
            eval_batch_size=16,
            eval_steps=25,
            max_steps=max_steps,
            learning_rate=1e-4,
            fail_on_non_convergence=True,
            use_cache="only_convergent",
            encoder_eval_max_size=100,
            encoder_eval_train_max_size=50,
            seed=0,
        ),
    )


def plot_suite_outputs(
    suite: SymmetricTrainerSuite,
    *,
    write_docs_images: bool = False,
) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    topk_output = EXPERIMENT_DIR / TOPK_OVERLAP_PLOT
    print(f"\n=== Plotting top-k overlap heatmap to {topk_output} ===")
    suite.plot_topk_overlap_heatmap(
        topk=1000,
        value="intersection_frac",
        output_path=str(topk_output),
        show=False,
    )

    cross_encoding_output = EXPERIMENT_DIR / CROSS_ENCODING_PLOT
    print(f"\n=== Plotting oriented cross-encoding heatmap to {cross_encoding_output} ===")
    suite.evaluate_encoder(split="test", plot=False, full_eval=True, max_size=100)
    suite.plot_cross_encoding_heatmap(
        RACE_CLASSES,
        split="test",
        alignment="counterfactual",
        run_evaluation=False,
        output_path=str(cross_encoding_output),
        show=False,
    )

    if write_docs_images:
        DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)
        docs_topk = DOCS_IMG_DIR / DOCS_TOPK_OVERLAP
        docs_cross = DOCS_IMG_DIR / DOCS_CROSS_ENCODING
        shutil.copy2(topk_output, docs_topk)
        shutil.copy2(cross_encoding_output, docs_cross)
        print(f"\n=== Wrote documentation images ===")
        print(f"  {docs_topk}")
        print(f"  {docs_cross}")

    print(f"\nWrote suite outputs under {EXPERIMENT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a symmetric race TrainerSuite.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training; plot from cached checkpoints when available.",
    )
    parser.add_argument(
        "--write-docs-images",
        action="store_true",
        help=f"Also copy plots to docs/img/{DOCS_TOPK_OVERLAP} and {DOCS_CROSS_ENCODING}.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Training steps per child GRADIEND (default: 150).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    suite = build_suite(max_steps=cli.max_steps)

    print("=== Symmetric race suite pairs ===")
    for child_id, pair in suite.pair_by_id.items():
        print(f"  {child_id}: {pair[0]} <-> {pair[1]}")

    if not cli.plot_only:
        print("\n=== Training suite ===")
        suite.train()

    plot_suite_outputs(suite, write_docs_images=cli.write_docs_images)
