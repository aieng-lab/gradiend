"""
TextPredictionDataCreator demo: English pronoun data creation.

Creates training data for pronoun groups:

- 1SG: I
- 1PL: we
- 2: you (singular/plural not distinguished)
- 3SG: he, she, it (single group with multiple targets)
- 3PL: they

Uses English Wikipedia via Hugging Face (``wikimedia/wikipedia``, config ``20231101.en``).
Demonstrates hf_config for dataset subset,
non-overlapping sentence windows, TextFilterConfig with targets=[...] for
multi-keyword groups, and balance="try".

Requires: pip install gradiend[data]
Set download_if_missing=True to auto-download the spacy model.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from gradiend import TextFilterConfig, TextPreprocessConfig, TextPredictionDataCreator

DEFAULT_OUTPUT_DIR = "data/english_pronouns"
TRAINING_BASENAME = "training"
NEUTRAL_BASENAME = "neutral"
GENERATION_CONFIG_BASENAME = "generation_config.json"
GENERATION_CONFIG_VERSION = 3
MAX_SIZE_PER_CLASS = 1000
NEUTRAL_MAX_SIZE = 1000
MIN_LEFT_CONTEXT_WORDS = 5
# Parquet-based HF dataset (datasets>=4); legacy ``wikipedia`` loading scripts are unsupported.
WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_HF_CONFIG = "20231101.en"
WIKIPEDIA_BASE_MAX_SIZE = 50_000

# English pronouns for neutral data exclusion (example-specific)
NEUTRAL_EXCLUDE_ENGLISH_PRONOUNS = [
    "i", "we", "you", "he", "she", "it", "they",
    "me", "us", "him", "her", "them",
]

# All pronoun classes required by the multilingual demo (C(5,2) = 10 binary GRADIENDs).
PRONOUN_CLASSES: List[str] = ["1SG", "1PL", "2SGPL", "3SG", "3PL"]
MIN_ROWS_PER_CLASS_FOR_SPLIT = 5


def english_pronoun_generation_config() -> Dict[str, Any]:
    return {
        "version": GENERATION_CONFIG_VERSION,
        "base_data": WIKIPEDIA_DATASET,
        "hf_config": WIKIPEDIA_HF_CONFIG,
        "base_max_size": WIKIPEDIA_BASE_MAX_SIZE,
        "split_to_sentences": 2,
        "min_chars": 20,
        "max_chars": 200,
        "min_left_context_words": MIN_LEFT_CONTEXT_WORDS,
        "max_size_per_class": MAX_SIZE_PER_CLASS,
        "balance": "try",
        "neutral_max_size": NEUTRAL_MAX_SIZE,
        "neutral_excluded_words": NEUTRAL_EXCLUDE_ENGLISH_PRONOUNS,
    }


def build_english_pronoun_data_creator(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    use_cache: bool = False,
) -> TextPredictionDataCreator:
    return TextPredictionDataCreator(
        base_data=WIKIPEDIA_DATASET,
        text_column="text",
        hf_config=WIKIPEDIA_HF_CONFIG,
        base_max_size=WIKIPEDIA_BASE_MAX_SIZE,
        preprocess=TextPreprocessConfig(
            split_to_sentences=2,
            min_chars=20,
            max_chars=200,
        ),
        spacy_model="en_core_web_sm",
        feature_targets=[
            TextFilterConfig(target="I", spacy_tags={"pos": "PRON"}, id="1SG"),
            TextFilterConfig(target="we", spacy_tags={"pos": "PRON"}, id="1PL"),
            TextFilterConfig(target="you", spacy_tags={"pos": "PRON"}, id="2SGPL"),
            TextFilterConfig(
                targets=["he", "she", "it"],
                spacy_tags={"pos": "PRON"},
                id="3SG",
            ),
            TextFilterConfig(target="they", spacy_tags={"pos": "PRON"}, id="3PL"),
        ],
        min_left_context_words=MIN_LEFT_CONTEXT_WORDS,
        output_dir=output_dir,
        training_basename=TRAINING_BASENAME,
        neutral_basename=NEUTRAL_BASENAME,
        use_cache=use_cache,
    )


def _generation_config_path(output_dir: str) -> Path:
    return Path(output_dir) / GENERATION_CONFIG_BASENAME


def _data_paths(output_dir: str) -> Tuple[Path, Path]:
    base = Path(output_dir)
    return base / f"{TRAINING_BASENAME}.csv", base / f"{NEUTRAL_BASENAME}.csv"


def _incomplete_classes_sidecar_path(training_path: Path) -> Path:
    return training_path.with_name(f"{training_path.stem}_incomplete_classes{training_path.suffix}")


def _pronoun_training_has_all_classes(training_path: Path) -> bool:
    if not training_path.is_file():
        return False
    try:
        df = pd.read_csv(training_path, usecols=["label_class"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    present = {str(value) for value in df["label_class"].dropna().unique().tolist()}
    return set(PRONOUN_CLASSES).issubset(present)


def pronoun_training_data_is_complete(output_dir: str = DEFAULT_OUTPUT_DIR) -> bool:
    """Return True when every pronoun class is in training.csv and no incomplete sidecar exists."""
    training_path, _ = _data_paths(output_dir)
    if _incomplete_classes_sidecar_path(training_path).is_file():
        return False
    return _pronoun_training_has_all_classes(training_path)


def _has_current_english_pronoun_data(output_dir: str) -> bool:
    training_path, neutral_path = _data_paths(output_dir)
    if not training_path.is_file() or not neutral_path.is_file():
        return False
    config_path = _generation_config_path(output_dir)
    if not config_path.is_file():
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return existing == english_pronoun_generation_config()


def ensure_english_pronoun_data(
    *,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> Tuple[Path, Path]:
    """Generate English pronoun training/neutral CSVs if missing or stale.

    Staleness is checked via ``generation_config.json`` so older cached CSVs
    created with sentence-level segments or without left-context filtering are
    regenerated automatically.

    Also regenerates when ``training_incomplete_classes.csv`` exists or when
    ``training.csv`` is missing any class in :data:`PRONOUN_CLASSES`. Without
    all five classes, TrainerSuite silently skips binary pairs involving the
    incomplete classes (leaving only three among 1SG/3SG/3PL).
    """
    training_path, neutral_path = _data_paths(output_dir)
    if (
        not force
        and _has_current_english_pronoun_data(output_dir)
        and pronoun_training_data_is_complete(output_dir)
    ):
        return training_path, neutral_path

    creator = build_english_pronoun_data_creator(output_dir=output_dir, use_cache=False)
    incomplete_sidecar = _incomplete_classes_sidecar_path(training_path)
    if incomplete_sidecar.is_file():
        incomplete_sidecar.unlink()

    creator.generate_training_data(
        max_size_per_class=MAX_SIZE_PER_CLASS,
        format="per_class",
        balance="try",
        min_rows_per_class_for_split=MIN_ROWS_PER_CLASS_FOR_SPLIT,
        raise_on_incomplete_classes=True,
    )
    if incomplete_sidecar.is_file():
        incomplete_sidecar.unlink()
    if not _pronoun_training_has_all_classes(training_path):
        raise ValueError(
            f"English pronoun training data at {training_path} is still missing classes "
            f"after regeneration. Expected all of {PRONOUN_CLASSES}."
        )
    creator.generate_neutral_data(
        additional_excluded_words=NEUTRAL_EXCLUDE_ENGLISH_PRONOUNS,
        max_size=NEUTRAL_MAX_SIZE,
    )

    config_path = _generation_config_path(output_dir)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(english_pronoun_generation_config(), handle, indent=2, sort_keys=True)
        handle.write("\n")

    return training_path, neutral_path


def main():
    creator = build_english_pronoun_data_creator(output_dir=DEFAULT_OUTPUT_DIR, use_cache=False)

    print("=== Training data (per_class, balance='try') ===")
    training = creator.generate_training_data(
        max_size_per_class=MAX_SIZE_PER_CLASS, # limit for demo; set higher or None for full data
        format="per_class",
        balance="try",
    )
    for class_id, df in training.items():
        print(f"  {class_id}: {len(df)} rows")
        for _, row in df.head(2).iterrows():
            print(f"    masked: {row['masked']}")
            print(f"    label:  {row['label']}")

    print("\n=== Neutral data (pronouns excluded) ===")
    neutral = creator.generate_neutral_data(
        additional_excluded_words=NEUTRAL_EXCLUDE_ENGLISH_PRONOUNS,
        max_size=NEUTRAL_MAX_SIZE,
    )
    config_path = _generation_config_path(DEFAULT_OUTPUT_DIR)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(english_pronoun_generation_config(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(neutral.head(10).to_string())
    print(f"\n  Total neutral sentences: {len(neutral)}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
