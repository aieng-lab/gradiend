"""
TextPredictionDataCreator demo: English pronoun data creation.

Creates training data for pronoun groups:
- 1SG: I
- 1PL: we
- 2: you (singular/plural not distinguished)
- 3SG: he, she, it (single group with multiple targets)
- 3PL: they

Uses English Wikipedia via HuggingFace. Demonstrates hf_config for dataset subset,
split_to_sentences, TextFilterConfig with targets=[...] for multi-keyword groups,
and balance="try".

Requires: pip install gradiend[data]
Set download_if_missing=True to auto-download the spacy model.
"""

from gradiend import TextFilterConfig, TextPreprocessConfig, TextPredictionDataCreator

# English pronouns for neutral data exclusion (example-specific)
NEUTRAL_EXCLUDE_ENGLISH_PRONOUNS = [
    "i", "we", "you", "he", "she", "it", "they",
    "me", "us", "him", "her", "them",
]


def main():
    creator = TextPredictionDataCreator(
        base_data="wikipedia",
        text_column="text",
        hf_config="20220301.en",
        preprocess=TextPreprocessConfig(
            split_to_sentences=True,
            min_chars=20,
            max_chars=200,
        ),
        trust_remote_code=True,
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
        output_dir='data/english_pronouns',
    )

    print("=== Training data (per_class, balance='try') ===")
    training = creator.generate_training_data(
        max_size_per_class=1000,
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
        max_size=1000,
    )
    print(neutral.head(10).to_string())
    print(f"\n  Total neutral sentences: {len(neutral)}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
