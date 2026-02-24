# Data handling (text prediction)

This guide describes how to provide text data to `TextPredictionTrainer` and which columns your datasets must have. It is specific to text prediction.

---

## Overview: three input paths

`TextPredictionTrainer` accepts data in three main ways:

| Path | Use when |
|------|----------|
| **Per-class HuggingFace** | Your dataset is on HuggingFace with one config/subset per class (e.g. `aieng-lab/de-gender-case-articles`, `aieng-lab/gradiend_race_data`). Pass the dataset ID to `data`. |
| **Per-class dict** | You have a Python `dict` of DataFrames keyed by class name, or output from `TextPredictionDataCreator.generate_training_data(format="per_class")`. Pass it to `data`. |
| **Merged** | You have a single table (DataFrame, CSV, Parquet, or HF dataset) where each row has `label_class` and `label`. Pass a DataFrame or file path to `data`; or pass an HF dataset ID to `hf_dataset`. |

---

## 1. Per-class HuggingFace format

Use when your dataset is on HuggingFace and has **one config/subset per class** (e.g. `masc_nom`, `fem_nom` or `white`, `black`, `asian`).

### How to specify

```python
trainer = TextPredictionTrainer(
    model="bert-base-uncased",
    data="aieng-lab/de-gender-case-articles",   # HuggingFace dataset ID (string, not a file path)
    target_classes=["masc_nom", "fem_nom"],     # The pair to train on
    masked_col="masked",
    split_col="split",
    # Optional:
    all_classes=["masc_nom", "fem_nom", "neut_nom"],  # Limit which configs to load; default: all
    hf_splits=["train", "validation", "test"],        # Splits to include; default: all
)
```

### Required columns (per subset/config)

Each config/subset of the dataset must have:

| Column | Required | Description |
|--------|----------|-------------|
| `masked` | ✅ | Sentence with `[MASK]` at the target position. |
| `split` | ✅ | Dataset split: `train`, `validation` (or `val`), `test`. |
| Factual token | ✅ | One column holding the token that fills the mask for this class. See below. |

**Factual token column:** The trainer looks for a column in this order:

1. A column with the same name as the class (e.g. `masc_nom` in subset `masc_nom`) — when `use_class_names_as_columns=True` (default)
2. Otherwise `label` or `token`

### Two variants of per-class HuggingFace data

Per-class datasets come in two shapes. Both are valid; the trainer handles them automatically.

#### Variant A: Single token per row (one factual column)

Each subset has one factual column. The alternative token is derived by joining with the other target class’s subset (same `masked` + `split`).

**Example:** [aieng-lab/de-gender-case-articles](https://huggingface.co/datasets/aieng-lab/de-gender-case-articles)

- Configs: `masc_nom`, `fem_nom`, `neut_nom`, etc.
- Each config has: `masked`, `split`, `label` (the factual token for that class)
- No columns for other classes — alternatives come from the other config’s rows

```python
# gender_de example
trainer = TextPredictionTrainer(
    model="bert-base-german-cased",
    data="aieng-lab/de-gender-case-articles",
    target_classes=["masc_nom", "fem_nom"],
    masked_col="masked",
    split_col="split",
)
```

#### Variant B: All tokens per row (one column per class)

Each subset has columns for every class, so factual and alternative tokens are in the same row (i.e., alternative tokens are explicitly defined; this can be useful if different multiple target words are considered per target class that are not pairwise interchangeable).

**Example:** [aieng-lab/gradiend_race_data](https://huggingface.co/datasets/aieng-lab/gradiend_race_data), [aieng-lab/gradiend_religion_data](https://huggingface.co/datasets/aieng-lab/gradiend_religion_data)

- Configs: `white`, `black`, `asian` (or `christian`, `jewish`, `muslim`)
- Each config has: `masked`, `split`, `white`, `black`, `asian` (columns = class names)
- The column matching the subset holds the factual token; other columns hold alternatives

```python
# race_religion example
trainer = TextPredictionTrainer(
    model="distilbert-base-cased",
    data="aieng-lab/gradiend_race_data",
    target_classes=["white", "black"],
    masked_col="masked",
    split_col="split",
)
```

---

## 2. Per-class dict format

Use when you have a Python `dict` mapping class name → DataFrame, e.g. from `TextPredictionDataCreator.generate_training_data(format="per_class")` or your own preparation.

### How to specify

```python
per_class_data = {
    "masc_nom": df_masc,   # DataFrame for class masc_nom
    "fem_nom": df_fem,     # DataFrame for class fem_nom
}

trainer = TextPredictionTrainer(
    model="bert-base-uncased",
    data=per_class_data,
    target_classes=["masc_nom", "fem_nom"],
    masked_col="masked",
    split_col="split",
)
```

### Required columns (per DataFrame)

Each DataFrame in the dict must have:

| Column | Required | Description |
|--------|----------|-------------|
| `masked` | ✅ | Sentence with `[MASK]` at the target position. |
| `split` | ✅ | Dataset split. Defaults to `"train"` if missing. |
| Factual token | ✅ | One column with the factual token: class-name column, `label`, `token`, or `source`/`target` (see below). |

**Factual token column** (lookup order):

1. Class-name column (e.g. `masc_nom` when key is `"masc_nom"`) — when `use_class_names_as_columns=True` (default)
2. `label`, `token`, or `source` (for pre-paired data)

**Alternative tokens:**

- If the DataFrame has columns for other classes: the alternative is taken from that column in the same row.
- If not (single-token-per-class): the alternative is derived from the other class’s DataFrame (pair must be set via `target_classes`).

**Pre-paired rows (source/target):** If a DataFrame has `source` and `target` columns, each row is treated as a factual/alternative pair. Optional: `source_id`, `target_id` for class names.

### Configurable column names

Override defaults with `TextPredictionConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `masked_col` | `"masked"` | Column for masked sentences |
| `split_col` | `"split"` | Column for split |
| `use_class_names_as_columns` | `True` | Use class names as token column names when present |

---

## 3. Merged format

Use when you have a **single table** where each row already has a class and label. Can be a DataFrame, CSV/Parquet path, or HuggingFace dataset (via `hf_dataset`).

### How to specify

**DataFrame or file path:**

```python
# DataFrame in memory
trainer = TextPredictionTrainer(
    model="bert-base-uncased",
    data=df,  # DataFrame with masked, split, label_class, label
    target_classes=["A", "B"],
    masked_col="masked",
    split_col="split",
    label_col="label",
    label_class_col="label_class",
)

# Local file
trainer = TextPredictionTrainer(
    model="bert-base-uncased",
    data="path/to/data.csv",   # or .parquet
    target_classes=["A", "B"],
    ...
)
```

**HuggingFace (merged):**

```python
trainer = TextPredictionTrainer(
    model="bert-base-uncased",
    hf_dataset="org/merged-dataset",   # Use hf_dataset for merged format
    hf_subset="subset_name",           # Optional: specific config
    hf_splits=["train", "validation", "test"],
    target_classes=["A", "B"],
    masked_col="masked",
    split_col="split",
    label_col="label",
    label_class_col="label_class",
)
```

### Required columns

| Column | Required | Description |
|--------|----------|-------------|
| `masked` | ✅ | Sentence with `[MASK]` at the target position |
| `split` | ✅ | Dataset split (`train`, `validation`, `test`, or `val`) |
| `label_class` | ✅ | The factual (source) class for this row (e.g. `"masc_nom"`) |
| `label` | ✅ | The factual token that fills the mask |

### Optional: explicit alternative columns

If your table has explicit factual/alternative pairs per row:

| Column | Required | Description |
|--------|----------|-------------|
| `alternative` | Optional | The counterfactual token for this row |
| `alternative_class` | Optional | The counterfactual class (used with `alternative`) |

When both `alternative` and `alternative_class` are present, the trainer uses them directly. Otherwise it infers the alternative from the other class in the pair (requires `target_classes` with exactly two classes and a single token per class in the pair).

### Configurable column names

| Parameter | Default | Description |
|-----------|---------|-------------|
| `masked_col` | `"masked"` | Column for masked sentences |
| `split_col` | `"split"` | Column for split |
| `label_col` | `"label"` | Column for factual token |
| `label_class_col` | `"label_class"` | Column for factual class |
| `alternative_col` | `"alternative"` | Column for alternative token (merged only) |
| `alternative_class_col` | `"alternative_class"` | Column for alternative class (merged only) |

---

## Import flow

The trainer decides how to load and convert data using this resolution order:

1. If **hf_dataset** is set: load that dataset (with `hf_subset`, `hf_splits`), then convert using merged column names.
2. Else if **data** is a string and not a file path: treat as HuggingFace ID → per-class load (multiple configs/subsets).
3. Else if **data** is a **dict**: per-class DataFrames; keys = class names.
4. Else if **data** is a **file path** (str or Path): load CSV/parquet as merged format.
5. Else **data** is a **DataFrame**: use as merged format in memory.

```
Data source (HF id / path / DataFrame / dict)
    → load or use in memory
    → merged_to_unified() or per_class_dict_to_unified()
    → unified DataFrame (pair-filtered)
    → TextTrainingDataset / GradientTrainingDataset
    → training batches
```

---

## Training pair and target classes

Training uses transitions between **two** classes (e.g. masculine vs feminine). Specify them with `target_classes`:

```python
target_classes=["masc_nom", "fem_nom"]   # Pair for training (currently: must have exactly two elements!)
```

- When `all_classes` (usually determined by data automatically) has exactly two elements, the trainer uses these two classes automatically as `target_classes`.
- **Merged classes**: Use `class_merge_map` to merge base classes (e.g. `{"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]}`); then `target_classes` refers to merged names. With exactly two merged keys, `target_classes` can be omitted.

---

## Data balancing and size caps

Balancing and caps rely on a feature-class identifier. After conversion, the trainer infers this from the class labels in your data.

- `TrainingArguments.train_max_size`: Caps training samples; applied per feature class when available.
- `TrainingArguments.encoder_eval_max_size`, `decoder_eval_max_size_training_like`, etc.: Similar per-class caps for evaluation.
- When class information is available, the scheduler oversamples smaller classes and `train_max_size` applies per-class downsampling.

---

## Splits

Datasets should provide `train`, `validation` (or `val`), and `test` splits. The trainer normalizes split names (e.g. `"val"` → `"validation"`). Support for fewer splits may be added later.

---

---

## Optional: neutral evaluation data (`eval_neutral_data`)

`TextPredictionTrainer` accepts **optional** neutral data via `eval_neutral_data` (DataFrame, path, or HuggingFace dataset ID). This is used for:

- **Encoder evaluation** — a separate `neutral_dataset` variant for encoding gradients from feature-independent text.
- **Decoder evaluation (LMS)** — text to compute language modeling score (perplexity) without feature-related targets.

**When `eval_neutral_data` is omitted or empty:**

- Encoder evaluation still runs (no `neutral_dataset` variant).
- Decoder evaluation falls back to **training-like data** (test split): each row's `text` is built by filling the mask with the factual token. Target tokens are automatically added to `decoder_eval_ignore_tokens` so they are ignored in LMS. This works for quick runs; for best practice, provide true neutral data when available (e.g. `TextPredictionDataCreator.generate_neutral_data()` or HuggingFace datasets like `aieng-lab/wortschatz-leipzig-de-grammar-neutral`).

See [Evaluation (intra-model)](../tutorials/evaluation-intra-model.md#neutral-data-for-decoder-evaluation-lms) for details.

---

## Generating data from raw text

To **create** training data from base corpora (Wikipedia, CSV, HuggingFace, or lists of strings), use `TextPredictionDataCreator` from `gradiend.data`. See the [Data generation](../tutorials/data-generation.md) tutorial for full usage.
