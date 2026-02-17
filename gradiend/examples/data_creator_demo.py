"""
TextPredictionDataCreator demo: generate training and neutral data from base texts.

Demonstrates spacy-based filtering for German articles (der/die/das) to handle
syncretism: "der" can be nominative masculine OR dative feminine; spacy tags
disambiguate. Requires: pip install gradiend[data]

Set download_if_missing=True to auto-download the spacy model if not installed.
"""

from gradiend import TextFilterConfig, TextPreprocessConfig, TextPredictionDataCreator

# German gendered articles (example-specific; define per use case)
NEUTRAL_EXCLUDE_GERMAN_ARTICLES = [
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einer", "einem", "einen", "eines",
    "kein", "keine", "keiner", "keinem", "keinen", "keines",
]

# Sentences illustrating syncretism: "der" as masc nom vs fem dat
SENTENCES = [
    "Der Mann ging in die Stadt.",
    "Die Frau kam am Mittag.",
    "Das Kind spielte im Garten.",
    "Der Hund bellte laut.",
    "Irren ist menschlich.",
    "Die Katze schlief auf dem Sofa.",
    "Das Buch lag auf dem Tisch.",
    "Der Lehrer half der Schülerin.",
    "Die Schülerin las dem Lehrer das Gedicht vor.",
    "Das Auto stand vor der Tür.",
    "Der Arzt half der Patientin.",
    "Was weiß ich schon?",
    "Die Ärztin untersuchte den Patienten.",
    "Das Haus gehörte der Familie.",
    "Der Vater schenkte der Mutter Blumen.",
    "Die Mutter kümmerte sich um das Kind.",
    "Das Fenster ging zur Strasse hinaus.",
    "Ein Vogel sang im Baum.",
    "Zwei Hunde rannten über die Wiese.",
    "Alles wird gut."
]


def main():
    creator = TextPredictionDataCreator(
        base_data=SENTENCES,
        text_column="text",
        spacy_model="de_core_news_sm",
        feature_targets=[
            TextFilterConfig(
                target="der",
                spacy_tags={"pos": "DET", "Case": "Nom", "Gender": "Masc", "Number": "Sing"},
                id="masc_nom",
            ),
            TextFilterConfig(
                target="der",
                spacy_tags={"pos": "DET", "Case": "Dat", "Gender": "Fem", "Number": "Sing"},
                id="fem_dat",
            ),
            TextFilterConfig(
                target="die",
                spacy_tags={"pos": "DET", "Case": "Nom", "Gender": "Fem", "Number": "Sing"},
                id="fem_nom",
            ),
            TextFilterConfig(
                target="die",
                spacy_tags={"pos": "DET", "Case": "Acc", "Gender": "Fem", "Number": "Sing"},
                id="fem_acc",
            ),
            TextFilterConfig(
                target="das",
                spacy_tags={"pos": "DET", "Case": "Nom", "Gender": "Neut", "Number": "Sing"},
                id="neut_nom",
            ),
            TextFilterConfig(
                target="das",
                spacy_tags={"pos": "DET", "Case": "Acc", "Gender": "Neut", "Number": "Sing"},
                id="neut_acc",
            ),
        ],
        seed=42,
        download_if_missing=True,
    )

    print("=== Training data (per_class) ===")
    training = creator.generate_training_data(max_size_per_class=5, format="per_class")
    for class_id, df in training.items():
        print(f"\n  {class_id} ({len(df)} rows):")
        for _, row in df.head(3).iterrows():
            print(f"    masked: {row['masked']}")
            print(f"    label:  {row['label']}")
        if len(df) > 3:
            print(f"    ... and {len(df) - 3} more")

    print("\n=== Training data (minimal) ===")
    minimal = creator.generate_training_data(format="minimal")
    print(minimal.head(20).to_string())

    print("\n=== Neutral data (target words + articles + DET/3rd-person PRON excluded) ===")
    neutral = creator.generate_neutral_data(
        additional_excluded_words=NEUTRAL_EXCLUDE_GERMAN_ARTICLES,
        excluded_spacy_tags=[
            {"pos": "DET"},
            {"pos": "PRON", "Person": "3"},
        ],
        max_size=5,
    )

    print(neutral.to_string())
    print(f"\n  Total neutral sentences: {len(neutral)}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
