# Tutorial: Data generation

This tutorial is **Part 1** of the detailed workflow. 
You will create training and neutral data from raw text using `TextPredictionDataCreator`. 
The output is later fed into `TextPredictionTrainer` in [Tutorial: Training](training.md) (see also [Data handling](../guides/data-handling.md)).

We use **German definite articles** as the running example: their form depends on **gender** and **case** (e.g. masculine nominative, feminine dative). 
For a study of how language models encode these distinctions (and whether they rely on rules or memorization), see [Understanding or Memorizing? A Case Study of German Definite Articles in Language Models](https://arxiv.org/abs/2601.09313), which uses GRADIEND on this kind of data.

---

## Goal: filter by grammatical role, not just by word

Suppose we want to extract sentences where the word *der* appears **only** in the role “masculine nominative” (e.g. *der Mann* “the man” in subject position). In many languages, the same **surface form** can correspond to different **grammatical roles**—this is called **syncretism**. In German, *der* can be masculine nominative, but also feminine dative or genitive plural. If we select all sentences containing the string *der*, we mix these roles and our training data no longer represents a single, well-defined feature. So we need to filter by **morphology** (gender, case, number, part-of-speech), not by the raw token alone.

String-based matching (as in [Start here](../start.md) with *he*/*she*/*they*) is enough when the token uniquely identifies the feature. For a full pronoun workflow (data creation from Wikipedia → training 3SG vs 3PL → evaluation), see the [english_pronouns.ipynb](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/english_pronouns.ipynb) notebook. When the token does not uniquely identify the feature (e.g., German articles), we need morphological constraints. This package supports that via **[spaCy](https://spacy.io/)** by providing expected `spacy_tags` (e.g., Case, Gender, POS) to **TextFilterConfig**, in addition to the target string(s) that we already saw in [Start here](../start.md). 

!!! tip "Optional dependency: data generation"
    Creating training and neutral data with **spaCy**-based filtering (as in this tutorial) requires the **data** extra. If you did not install it with GRADIEND, install it with:

    ```bash
    pip install gradiend[data]
    ```

    This also installs Hugging Face (HF) datasets, so you can pass HF dataset ids directly. Depending on your use case, you may need a spaCy language model (e.g. `de_core_news_sm` for German).

---

## Defining a filter for: “der” = nominative masculine singular

To filter specific gender-case-number combination occurences of German articles, e.g., "der" as nominative masculine singular article, we can specify the according [spacy tags](https://spacy.io/usage/linguistic-features) in the filter configuration.

```python
from gradiend.data import TextFilterConfig

TextFilterConfig(
    targets=["der"], # the string match (we still need this, but it is not enough on its own)
    spacy_tags={
        "pos": "DET", # part-of-speech determiner (article)
        "Case": "Nom", # nominative case
        "Gender": "Masc", # masculine gender
        "Number": "Sing", # singular number
    },
    id="masc_nom", # the feature class id used later
)
```

- **targets**: The surface form(s) to match (*der*). These strings are matched with regex on word-level.
- **spacy_tags**: Morphological constraints from [spaCy](https://spacy.io/usage/linguistic-features) (part-of-speech DET, Case Nom, Gender Masc, Number Sing). Only tokens that satisfy **both** the form and these tags are kept.

So we no longer match “any *der*”; we get only *der* in the nominative-masculine singular role.

---

## Full paradigm: one filter per gender–case cell

German definite singular articles form a **paradigm**: 3 genders × 4 cases = 12 cells (e.g. masc_nom, masc_acc, fem_nom, fem_dat, …). By defining one **TextFilterConfig** per cell, we can generate data for each cell separately. 

```python
feature_targets = [
    TextFilterConfig(
            targets=["der"],
            spacy_tags={"pos": "DET", "Case": "Nom", "Gender": "Masc", "Number": "Sing"},
            id="masc_nom",
        ),
        TextFilterConfig(
            targets=["der"], # different "der" as above!
            spacy_tags={"pos": "DET", "Case": "Dat", "Gender": "Fem", "Number": "Sing"},
            id="fem_dat",
        ),
        TextFilterConfig(
            targets=["die"],
            spacy_tags={"pos": "DET", "Case": "Nom", "Gender": "Fem", "Number": "Sing"},
            id="fem_nom",
        ),
        # ... further cells (e.g. masc_dat, masc_gen, fem_acc, ...)
]
```

## Base Data

The `TextPredictionTrainer` supports a variety of base data inputs:

- Python list of strings (e.g., `TextPredictionDataCreator(base_data=["This is a sentence.", "This is another sentence."])`)
- CSV file path (e.g., `TextPredictionDataCreator(base_data="path/to/texts.csv", text_column="text")`)
- Hugging Face dataset id (e.g., `TextPredictionDataCreator(base_data="wikipedia", hf_config="20220301.en", text_column="text")`; requires `datasets` library)

The raw texts might be too long (e.g. full articles) or too short (e.g. tweets), so we can use `TextPreprocessConfig` to split into sentences and set character limits. If you omit `preprocess` (or pass `None`), no preprocessing is applied and texts are used as-is.

```python
from gradiend.data import TextPreprocessConfig
preprocessor = TextPreprocessConfig(split_to_sentences=True, min_chars=50, max_chars=500)
```

## Generating training data

Combining all the above, we can create a `TextPredictionDataCreator` with the base data, preprocessing config, and the list of filters for each gender–case cell.

```python
from gradiend.data import TextPredictionDataCreator

creator = TextPredictionDataCreator(
    base_data="wikipedia",
    hf_config="20220301.en",
    preprocess=preprocessor,
    spacy_model="de_core_news_sm",   # auto-downloaded if missing (download_if_missing=True by default)
    feature_targets=feature_classes, # TextFilterConfig list as defined above
    seed=42,
)

training_data = creator.generate_training_data(max_size_per_class=5000, format="per_class")
```

**Saving to disk with `output_dir`.** Set `output_dir` on the creator to have `generate_training_data` and `generate_neutral_data` write files when you omit the `output=` argument. The directory is created if needed, and default filenames are used (e.g. `training.csv` and `neutral.csv`, depending on `output_format`). Use `training_basename` and `neutral_basename` to customize the base names. If you pass an explicit `output=` to either method, it overrides the default path for that call.

```python
creator = TextPredictionDataCreator(
    ...,
    output_dir="data/german_articles",   # writes training.csv, neutral.csv here
    training_basename="training",        # default
    neutral_basename="neutral",          # default
    output_format="csv",                 # or "parquet", "hf"
)
training_data = creator.generate_training_data(max_size_per_class=5000)  # saves to output_dir
neutral = creator.generate_neutral_data(max_size=5000)                   # saves to output_dir
```

This creates per-class training data: each key in the dict corresponds to one gender–case cell. Pass it to the trainer as `data=training`; the trainer detects dict input and treats it as per-class automatically. See [Data handling](../guides/data-handling.md).

---

## Neutral data

For feature-independent evaluation (e.g. language-model score on text without feature-related targets), we need **neutral** sentences. We exclude the target forms and optionally other related words (e.g., indefinite articles), and we can exclude by morphology (e.g. any determiner, or third-person pronouns):

```python
EXCLUDED_WORDS = ["der", "die", "das", "den", "dem", "des", "ein", "eine"]

neutral = creator.generate_neutral_data(
    additional_excluded_words=EXCLUDED_WORDS,
    excluded_spacy_tags=[
        {"pos": "DET"},
        {"pos": "PRON", "Person": "3"},
    ],
    max_size=5000,
)
```

## Next Step: Training
Pass `training` and `neutral` to `TextPredictionTrainer` as `data=training` and `eval_neutral_data=neutral`, see [Tutorial: Training](training.md).

---

## Next steps

- **[Tutorial: Training](training.md)** — Configure the trainer, pruning, convergence plot, and multi-seed.
- **[Tutorial: Evaluation (intra-model)](evaluation-intra-model.md)** — Encoder and decoder evaluation and decoder config selection.
- **[Tutorial: Model Rewrite](model-rewrite.md)** — Apply decoder-selected rewrites and save changed checkpoints.
- **[Tutorial: Evaluation (inter-model)](evaluation-inter-model.md)** — Comparing multiple runs (top-k overlap, heatmap).
- **[Data handling](../guides/data-handling.md)** — All supported data formats and column names.
