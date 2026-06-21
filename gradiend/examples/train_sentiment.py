"""
Sentiment GRADIEND workflow: positive vs negative via masked emotion words.

Single script: generates tweet_eval-based training/neutral CSVs and trains a
TextPredictionTrainer for positive <-> negative.

Mask targets come from the NRC Emotion Lexicon (Mohammad & Turney, 2013), loaded
from Hugging Face (``vladinc/nrc``), filtered to **adjectives** attested in tweet_eval
(canonical lemma per inflection).

Requires: pip install gradiend[data]

On minimal Linux/HPC images without system CA certs, install certifi and set
before running (or use scripts/prefetch_hf_datasets.py to warm the cache)::

    export HF_HUB_DISABLE_XET=1
    export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
    export REQUESTS_CA_BUNDLE=$SSL_CERT_FILE
    export HF_HOME=/path/to/shared/hf-cache

Run:
    python -m gradiend.examples.train_sentiment

After training, the script evaluates encoder stability across train/validation/test
splits (vocabulary-held-out by emotion word) and saves target-grouped strip/box
plots plus a labeled class-level strip overview.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from gradiend.util.hf_env import configure_hf_download_env

configure_hf_download_env()

from datasets import load_dataset

from gradiend import (
    PostPruneConfig,
    TextFilterConfig,
    TextPredictionConfig,
    TextPredictionTrainer,
    TrainingArguments,
)
from gradiend.data.core import apply_split_group_key
from gradiend.examples.nrc_sentiment_lexicon import (
    NRC_CITATION,
    build_sentiment_lexicon_for_corpus,
    load_nrc_sentiment_words,
)
from gradiend.util.encoder_splits import order_split_names

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sentiment_tweets"
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "runs" / "examples" / "sentiment"
DEFAULT_MODEL = "google-bert/bert-base-multilingual-cased"
TARGET_CLASSES = ("positive", "negative")
DEFAULT_LEXICON_WORDS_PER_CLASS = 10
DEFAULT_MAX_SIZE_PER_CLASS = 3000
DEFAULT_MIN_OCCURRENCES_PER_TARGET = 50
TWEET_EVAL_SPLITS = ("train", "validation", "test")
_TARGET_KEY = [str.strip, str.casefold]


def _canonical_target_word(word: str) -> str:
    return apply_split_group_key(word, _TARGET_KEY)


def _canonical_target_words(words: List[str]) -> List[str]:
    unique: dict[str, str] = {}
    for word in words:
        key = _canonical_target_word(word)
        if key:
            unique.setdefault(key, key)
    return sorted(unique.values())


def _normalize_training_labels(df):
    if df is None or df.empty or "label" not in df.columns:
        return df
    out = df.copy()
    out["label"] = out["label"].map(lambda word: _canonical_target_word(str(word)))
    return out


def _load_tweet_eval_texts(*, splits: Tuple[str, ...] = TWEET_EVAL_SPLITS) -> List[str]:
    texts: List[str] = []
    for split in splits:
        df = load_dataset("tweet_eval", "sentiment", split=split).to_pandas()
        df = df[df["label"].isin([0, 2])].copy()
        texts.extend(df["text"].astype(str).tolist())
    return texts


def _feature_targets(
    *,
    positive_words: List[str],
    negative_words: List[str],
) -> List[TextFilterConfig]:
    adj_tags = {"pos": "ADJ"}
    return [
        TextFilterConfig(
            targets=positive_words,
            spacy_tags=adj_tags,
            use_lemma=True,
            id="positive",
        ),
        TextFilterConfig(
            targets=negative_words,
            spacy_tags=adj_tags,
            use_lemma=True,
            id="negative",
        ),
    ]


def generate_data(
    *,
    output_dir: Path = DEFAULT_DATA_DIR,
    max_size_per_class: int = DEFAULT_MAX_SIZE_PER_CLASS,
    neutral_max_size: int = 1000,
    lexicon_words_per_class: int = DEFAULT_LEXICON_WORDS_PER_CLASS,
    min_occurrences_per_target: int = DEFAULT_MIN_OCCURRENCES_PER_TARGET,
    seed: int = 42,
) -> Tuple[Path, Path]:
    from gradiend import TextPredictionDataCreator

    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training.csv"
    neutral_path = output_dir / "neutral.csv"

    base_texts = _load_tweet_eval_texts()
    positive_words, negative_words, lex_stats = build_sentiment_lexicon_for_corpus(
        base_texts,
        max_words_per_class=lexicon_words_per_class,
        min_count_per_word=min_occurrences_per_target,
    )
    positive_words = _canonical_target_words(positive_words)
    negative_words = _canonical_target_words(negative_words)
    if not positive_words or not negative_words:
        raise ValueError(
            "No NRC sentiment words from the lexicon were attested in tweet_eval. "
            "Increase lexicon_words_per_class or check the base corpus."
        )

    creator = TextPredictionDataCreator(
        base_data=base_texts,
        spacy_model="en_core_web_sm",
        feature_targets=_feature_targets(
            positive_words=positive_words,
            negative_words=negative_words,
        ),
        seed=seed,
        output_dir=str(output_dir),
        use_cache=False,
        download_if_missing=True,
        split_group_col="label",
        split_group_key=[str.strip, str.casefold],
    )

    print("=== Sentiment data (tweet_eval + NRC Emotion Lexicon) ===")
    print(f"  lexicon source: {NRC_CITATION}")
    print(
        "  NRC sentiment words:",
        f"positive={lex_stats['nrc_positive']}, negative={lex_stats['nrc_negative']}",
    )
    print(
        "  attested as ADJ (canonical lemma) in tweet_eval:",
        f"positive={lex_stats['corpus_positive']}, negative={lex_stats['corpus_negative']}",
    )
    print(f"  minimum corpus occurrences per target word: {min_occurrences_per_target}")

    training_df = creator.generate_training_data(
        max_size_per_class=max_size_per_class,
        format="unified",
        balance="strict",
        min_rows_per_class_for_split=100,
        min_rows_per_target_for_balance=min_occurrences_per_target,
        output=str(training_path),
    )
    training_df = _normalize_training_labels(training_df)
    if len(training_df):
        training_df.to_csv(training_path, index=False)
    for class_id in TARGET_CLASSES:
        n = int((training_df["label_class"] == class_id).sum()) if len(training_df) else 0
        print(f"  {class_id}: {n} rows")
    if len(training_df):
        print("  vocabulary (canonical lemma per class):")
        for class_id in TARGET_CLASSES:
            labels = training_df.loc[training_df["label_class"] == class_id, "label"].dropna().astype(str)
            canonical = sorted({_canonical_target_word(word) for word in labels})
            print(f"    {class_id}: {len(canonical)} distinct words")
            if canonical:
                preview = ", ".join(canonical[:8])
                suffix = "..." if len(canonical) > 8 else ""
                print(f"      e.g. {preview}{suffix}")
            target_counts = labels.map(_canonical_target_word).value_counts()
            if len(target_counts):
                min_count = int(target_counts.min())
                print(f"      rows per target: min={min_count}, max={int(target_counts.max())}")
                if min_count < min_occurrences_per_target:
                    raise ValueError(
                        f"{class_id} has target words with only {min_count} generated rows after balancing "
                        f"(required >= {min_occurrences_per_target}). "
                            "Lower DEFAULT_MIN_OCCURRENCES_PER_TARGET, lower DEFAULT_LEXICON_WORDS_PER_CLASS, "
                            "or increase DEFAULT_MAX_SIZE_PER_CLASS. "
                            "Lexicon filtering counts matching sentences; if this persists after updating "
                            "gradiend, some targets may have too few maskable sentences in tweet_eval."
                    )
        sample = training_df[training_df["label_class"] == "positive"].head(1)
        if len(sample):
            row = sample.iloc[0]
            print("  example masked:", row["masked"])
            print("  example label: ", row["label"])

    nrc_positive, nrc_negative = load_nrc_sentiment_words()
    neutral_df = creator.generate_neutral_data(
        additional_excluded_words=nrc_positive + nrc_negative,
        max_size=neutral_max_size,
        output=str(neutral_path),
    )
    print(f"  wrote {training_path} ({len(training_df)} rows)")
    print(f"  wrote {neutral_path} ({len(neutral_df)} rows)")
    return training_path, neutral_path


def _print_vocabulary_splits(trainer: TextPredictionTrainer) -> None:
    df = trainer.combined_data
    if df is None:
        return
    print("\n--- Vocabulary-held-out splits (canonical word -> split) ---")
    seen: set[str] = set()
    for tok in sorted(df["factual"].dropna().unique(), key=str):
        key = _canonical_target_word(tok)
        if key in seen:
            continue
        seen.add(key)
        sub = df[df["factual"].map(lambda x: _canonical_target_word(str(x))) == key]
        splits = sorted(sub["split"].astype(str).unique().tolist())
        examples = sorted(sub["factual"].unique().tolist())
        print(f"  {key!r}: split={splits[0]!r}  (tokens={examples})")


def _evaluate_split_stability(trainer: TextPredictionTrainer, experiment_dir: Path) -> None:
    plot_dir = experiment_dir / "split_stability"
    plot_dir.mkdir(parents=True, exist_ok=True)

    trainer._ensure_data()
    _print_vocabulary_splits(trainer)

    print("\n=== Multi-split encoder stability (split='all', full training data) ===")
    enc = trainer.evaluate_encoder(
        split="all",
        max_size=None,
        return_df=True,
        plot=False,
        use_cache=False,
        include_other_classes=False,
    )
    enc_df = enc["encoder_df"]
    train_df = enc_df[enc_df["type"] == "training"]
    if "data_split" in train_df.columns:
        print(
            "  training splits:",
            order_split_names(train_df["data_split"].dropna().astype(str).unique().tolist()),
        )

    sg = enc.get("split_generalization")
    if sg:
        print(json.dumps(sg, indent=2))

    print("\n=== Faceted encoder distribution by split ===")
    facet_path = trainer.plot_encoder_distributions(
        encoder_df=enc_df,
        split_plot_mode="facet",
        include_neutral=True,
        target_and_neutral_only=False,
        title=None,
        output=str(plot_dir / "encoder_distributions_facet.pdf"),
        legend_loc="lower left",
        show=True,
    )
    print(f"  facet: {facet_path}")

    print("\n=== By-target plots (grouped by feature class, hue = split) ===")
    for style in ("strip", "box", "violin"):
        path = trainer.plot_encoder_by_target(
            encoder_df=enc_df,
            plot_style=style,
            title=None,
            output=str(plot_dir / f"encoder_by_target_{style}.pdf"),
            legend_loc="lower left",
            show=True,
        )
        print(f"  {style}: {path}")
    interactive = trainer.plot_encoder_by_target(
        encoder_df=enc_df,
        plot_style="strip",
        interactive=True,
        title=None,
        output=str(plot_dir / "encoder_by_target_strip_interactive.html"),
        show=True,
    )
    print(f"  interactive strip: {plot_dir / 'encoder_by_target_strip_interactive.html' if interactive is not None else None}")

    print("\n=== Class-level scatter (optional overview) ===")
    scatter_path = trainer.plot_encoder_strip_by_split(
        encoder_df=enc_df,
        title=None,
        output=str(plot_dir / "encoder_scatter_by_class.pdf"),
        label_points="outliers+sample",
        label_sample_per_group=2,
        adjust_labels=True,
        show=True,
        point_size=4,
    )
    print(f"  saved {scatter_path or plot_dir / 'encoder_scatter_by_class.pdf'}")


def train(
    *,
    training_path: Path,
    neutral_path: Path,
    model: str = DEFAULT_MODEL,
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    max_steps: int = 150,
    train_batch_size: int = 8,
    learning_rate: float = 1e-4,
    use_cache: bool = True,
) -> TextPredictionTrainer:
    config = TextPredictionConfig(
        run_id="sentiment_positive_negative",
        data=training_path,
        target_classes=list(TARGET_CLASSES),
        eval_neutral_data=neutral_path,
        split_col=None,
        split_group_key=[str.strip, str.casefold],
        split_ratios=(0.6, 0.2, 0.2),
        img_format="png",
    )
    args = TrainingArguments(
        experiment_dir=str(experiment_dir),
        train_batch_size=train_batch_size,
        eval_batch_size=8,
        eval_steps=100,
        num_train_epochs=3,
        max_steps=max_steps,
        source="alternative",
        target="diff",
        encoder_eval_train_max_size=None,
        learning_rate=learning_rate,
        post_prune_config=PostPruneConfig(topk=0.001, part="decoder-weight"),
        use_cache=use_cache,
        fail_on_non_convergence=True,
    )

    print("\n=== Sentiment training (positive <-> negative) ===")
    trainer = TextPredictionTrainer(model=model, config=config, args=args)
    trainer.train()
    trainer.plot_training_convergence()

    stats = trainer.get_training_stats()
    ts = stats.get("training_stats", {}) if stats else {}
    print(f"  correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")

    print("\n=== Encoder evaluation ===")
    trainer.evaluate_encoder(plot=True)
    print(f"  {trainer.get_encoder_metrics(use_cache=True)}")

    print("\n=== Decoder evaluation ===")
    dec = trainer.evaluate_decoder(plot=True)
    for cls in TARGET_CLASSES:
        if cls in dec:
            print(f"  decoder {cls}: {dec[cls]}")

    _evaluate_split_stability(trainer, experiment_dir)

    return trainer


if __name__ == "__main__":
    training_path = DEFAULT_DATA_DIR / "training.csv"
    neutral_path = DEFAULT_DATA_DIR / "neutral.csv"

    if not (training_path.is_file() and neutral_path.is_file()):
        training_path, neutral_path = generate_data(
            output_dir=DEFAULT_DATA_DIR,
            max_size_per_class=DEFAULT_MAX_SIZE_PER_CLASS,
            neutral_max_size=1000,
            lexicon_words_per_class=DEFAULT_LEXICON_WORDS_PER_CLASS,
            min_occurrences_per_target=DEFAULT_MIN_OCCURRENCES_PER_TARGET,
            seed=42,
        )
    else:
        print(f"=== Sentiment data: using existing CSVs in {DEFAULT_DATA_DIR} ===")
        print(f"  {training_path}")
        print(f"  {neutral_path}")
        cached = _normalize_training_labels(pd.read_csv(training_path))
        cached.to_csv(training_path, index=False)

    train(
        training_path=training_path,
        neutral_path=neutral_path,
        model=DEFAULT_MODEL,
        experiment_dir=DEFAULT_EXPERIMENT_DIR,
        max_steps=150,
        train_batch_size=8,
        learning_rate=1e-4,
        use_cache=False,
    )
    print("\n=== Done ===")
