"""
PositiveTrainerSuite example: several positive-vs-negative sentiment word pairs.

Runnable counterpart to ``docs/guides/trainer-suites.md``. Trains one GRADIEND
per pair, then writes suite comparison plots.

Documentation images (``--write-docs-images`` copies into ``docs/img/``):

- ``suite_similarity_heatmap.png`` — from ``plot_similarity_heatmap``
- ``suite_cross_encoding_heatmap.png`` — from ``plot_cross_encoding_heatmap``

Requires: pip install gradiend[data]

Run:
    python -m gradiend.examples.train_sentiment_positive_suite
    python -m gradiend.examples.train_sentiment_positive_suite --plot-only --write-docs-images
    python -m gradiend.examples.train_sentiment_positive_suite --write-docs-images
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from gradiend.util.hf_env import configure_hf_download_env

configure_hf_download_env()

from datasets import load_dataset

from gradiend import (
    PostPruneConfig,
    PrePruneConfig,
    PositiveFeatureDefinition,
    PositiveTrainerSuite,
    TextFilterConfig,
    TextPredictionDataCreator,
    TextPredictionTrainer,
    TrainingArguments,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = PROJECT_ROOT / "runs" / "examples" / "sentiment_positive_suite"
DOCS_IMG_DIR = PROJECT_ROOT / "docs" / "img"

SENTIMENT_PAIRS = [
    ("good", "bad"),
    ("happy", "sad"),
    ("love", "hate"),
    ("fast", "slow"),
    ("excited", "bored"),
]

TWEET_EVAL_DATASET = "cardiffnlp/tweet_eval"
TWEET_EVAL_SPLITS = ("train", "validation", "test")

SIMILARITY_PLOT = "suite_similarity_heatmap.png"
CROSS_ENCODING_PLOT = "suite_cross_encoding_heatmap.png"
DOCS_SIMILARITY = "suite_similarity_heatmap.png"
DOCS_CROSS_ENCODING = "suite_cross_encoding_heatmap.png"


def _load_tweet_eval_texts():
    texts = []
    for split in TWEET_EVAL_SPLITS:
        df = load_dataset(TWEET_EVAL_DATASET, "sentiment", split=split).to_pandas()
        texts.extend(df["text"].dropna().astype(str).tolist())
    return texts


def create_data(*, max_size_per_class: int = 500, neutral_max_size: int = 200):
    words = [word for pair in SENTIMENT_PAIRS for word in pair]
    creator = TextPredictionDataCreator(
        base_data=_load_tweet_eval_texts(),
        feature_targets=[
            TextFilterConfig(target=word, id=word, min_left_context_words=0)
            for word in words
        ],
        seed=0,
    )
    training = creator.generate_training_data(
        max_size_per_class=max_size_per_class,
        format="per_class",
        min_rows_per_class_for_split=5,
    )
    neutral = creator.generate_neutral_data(
        additional_excluded_words=words,
        max_size=neutral_max_size,
    )
    return training, neutral


def build_suite(training, neutral, *, max_steps: int = 500):
    return PositiveTrainerSuite(
        TextPredictionTrainer,
        model="bert-base-uncased",
        data=training,
        eval_neutral_data=neutral,
        retain_models_in_memory=False,
        positive_feature_definitions=[
            PositiveFeatureDefinition(
                positive_feature_class=positive,
                negative_feature_class=negative,
            )
            for positive, negative in SENTIMENT_PAIRS
        ],
        args=TrainingArguments(
            experiment_dir=str(EXPERIMENT_DIR),
            train_batch_size=4,
            eval_batch_size=8,
            eval_steps=50,
            max_steps=max_steps,
            learning_rate=1e-4,
            fail_on_non_convergence=True,
            use_cache="only_convergent",
            encoder_eval_max_size=100,
            encoder_eval_train_max_size=50,
            pre_prune_config=PrePruneConfig(n_samples=16, topk=0.01, source="diff"),
            post_prune_config=PostPruneConfig(),
            seed=0,
        ),
    )


def plot_suite_outputs(
    suite: PositiveTrainerSuite,
    *,
    write_docs_images: bool = False,
) -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Evaluating encoders ===")
    suite.evaluate_encoder(split="test", plot=False, max_size=100)

    similarity_output = EXPERIMENT_DIR / SIMILARITY_PLOT
    print(f"\n=== Plotting cosine model similarity heatmap to {similarity_output} ===")
    suite.plot_similarity_heatmap(
        measure="cosine",
        output_path=str(similarity_output),
        show=False,
    )

    cross_encoding_output = EXPERIMENT_DIR / CROSS_ENCODING_PLOT
    print(f"\n=== Plotting cross-encoding heatmap to {cross_encoding_output} ===")
    suite.plot_cross_encoding_heatmap(
        run_evaluation=False,
        output_path=str(cross_encoding_output),
        show=False,
    )

    if write_docs_images:
        DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)
        docs_similarity = DOCS_IMG_DIR / DOCS_SIMILARITY
        docs_cross = DOCS_IMG_DIR / DOCS_CROSS_ENCODING
        shutil.copy2(similarity_output, docs_similarity)
        shutil.copy2(cross_encoding_output, docs_cross)
        print("\n=== Wrote documentation images ===")
        print(f"  {docs_similarity}")
        print(f"  {docs_cross}")

    print(f"\nWrote suite outputs under {EXPERIMENT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a positive sentiment TrainerSuite.")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip training; plot from cached checkpoints when available.",
    )
    parser.add_argument(
        "--write-docs-images",
        action="store_true",
        help=f"Also copy plots to docs/img/{DOCS_SIMILARITY} and {DOCS_CROSS_ENCODING}.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Training steps per child GRADIEND (default: 500).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    training_data, neutral_data = create_data()
    suite = build_suite(training_data, neutral_data, max_steps=cli.max_steps)

    print("=== Positive sentiment suite pairs ===")
    for child_id, definition in suite.pair_definitions.items():
        print(f"  {child_id}: {definition.label}")

    if not cli.plot_only:
        print("\n=== Training suite ===")
        suite.train()

    plot_suite_outputs(suite, write_docs_images=cli.write_docs_images)
