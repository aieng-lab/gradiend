"""
PositiveTrainerSuite ``mode="all_but_one"`` example.

**Purpose.** Train one GRADIEND per *leave-one-out* holdout: each child learns a
single positive-vs-negative axis built from *all other* sentiment word pairs.
That answers questions like: "Does a model trained only on non-valence contrasts
still encode valence?" or "How similar are union-of-features GRADIENDs when
different groups are excluded?"

Compare with ``train_sentiment_positive_suite.py`` (default ``mode="single"``):
that script trains one GRADIEND per word pair. This script trains one GRADIEND
per held-out *group* (default) or per held-out *pair* (``--holdout feature``).

Five word pairs are grouped into four ``feature_class_group`` values. ``valence``
contains both ``happy``/``sad`` and ``excited``/``bored``, so holding out
``valence`` removes two pairs at once.

Requires: pip install gradiend[data]

Run:
    python -m gradiend.examples.train_sentiment_positive_suite_all_but_one
    python -m gradiend.examples.train_sentiment_positive_suite_all_but_one --holdout feature
"""

from __future__ import annotations

import argparse
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
EXPERIMENT_DIR = PROJECT_ROOT / "runs" / "examples" / "sentiment_positive_all_but_one"

# (positive, negative, feature_class_group)
SENTIMENT_FEATURES = [
    ("good", "bad", "quality"),
    ("happy", "sad", "valence"),
    ("excited", "bored", "valence"),
    ("love", "hate", "affection"),
    ("fast", "slow", "pace"),
]

TWEET_EVAL_DATASET = "cardiffnlp/tweet_eval"
TWEET_EVAL_SPLITS = ("train", "validation", "test")


def _load_tweet_eval_texts() -> list[str]:
    texts: list[str] = []
    for split in TWEET_EVAL_SPLITS:
        df = load_dataset(TWEET_EVAL_DATASET, "sentiment", split=split).to_pandas()
        texts.extend(df["text"].dropna().astype(str).tolist())
    return texts


def create_data(*, max_size_per_class: int = 500, neutral_max_size: int = 200):
    words = [word for positive, negative, _group in SENTIMENT_FEATURES for word in (positive, negative)]
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


def build_suite(training, neutral, *, holdout_by_group: bool):
    positive_feature_definitions = [
        PositiveFeatureDefinition(
            positive_feature_class=positive,
            negative_feature_class=negative,
            feature_class_group=group if holdout_by_group else None,
        )
        for positive, negative, group in SENTIMENT_FEATURES
    ]
    return PositiveTrainerSuite(
        TextPredictionTrainer,
        model="bert-base-uncased",
        data=training,
        eval_neutral_data=neutral,
        mode="all_but_one",
        retain_models_in_memory=False,
        positive_feature_definitions=positive_feature_definitions,
        args=TrainingArguments(
            experiment_dir=str(EXPERIMENT_DIR),
            train_batch_size=4,
            eval_batch_size=8,
            eval_steps=50,
            max_steps=500,
            learning_rate=1e-4,
            fail_on_non_convergence=True,
            use_cache="only_convergent",
            pre_prune_config=PrePruneConfig(n_samples=16, topk=0.01, source="diff"),
            post_prune_config=PostPruneConfig(),
            seed=0,
        ),
    )


def _print_holdout_plan(suite: PositiveTrainerSuite, *, holdout_by_group: bool) -> None:
    unit = "group" if holdout_by_group else "pair"
    print(f"=== all_but_one holdout plan ({unit}) ===")
    print(
        "Each child trains positive vs negative super-classes from every pair "
        f"except the held-out {unit}.\n"
    )
    for child_id, definition in suite.pair_definitions.items():
        merge = definition.class_merge_map or {}
        positive = merge.get("positive", [])
        negative = merge.get("negative", [])
        print(f"  {child_id}")
        print(f"    label: {definition.label}")
        print(f"    merged positive: {positive}")
        print(f"    merged negative: {negative}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout",
        choices=("group", "feature"),
        default="group",
        help="Hold out one feature_class_group (default) or one word pair.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    holdout_by_group = cli_args.holdout == "group"

    training_data, neutral_data = create_data()
    suite = build_suite(training_data, neutral_data, holdout_by_group=holdout_by_group)

    _print_holdout_plan(suite, holdout_by_group=holdout_by_group)

    print("=== Training suite ===")
    suite.train()

    print("\n=== Evaluating encoders ===")
    suite.evaluate_encoder(split="test", plot=True)

    similarity_output = EXPERIMENT_DIR / "all_but_one_similarity_heatmap.png"
    print(f"\n=== Plotting cosine similarity heatmap to {similarity_output} ===")
    suite.plot_similarity_heatmap(
        measure="cosine",
        output_path=str(similarity_output),
        show=True,
    )

    print(
        "\nNote: suite.plot_cross_encoding_heatmap() is not used here. Cross-encoding "
        "columns are defined by each child's target_classes; in all_but_one every child "
        "trains on the same synthetic pair (positive/negative), so the matrix would have "
        "identical values across each row. Use mode='single' for cross-encoding heatmaps."
    )
    print(f"\nWrote suite outputs under {EXPERIMENT_DIR}")
