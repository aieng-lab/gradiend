# Tutorial: Detailed workflow (overview)

This page is the **map** of the detailed workflow. After [Start here](../start.md), the full pipeline has three main parts: **data**, **training**, and **evaluation**. Each part has its own tutorial where the real detail lives.

- **[Tutorial: Data generation](data-generation.md)** — Build training and neutral data (e.g. with spaCy and morphology). Output feeds into the trainer.
- **[Tutorial: Training](training.md)** — Experiment layout, pruning, multi-seed, convergence plot, and how to configure a real run.
- **[Tutorial: Evaluation (intra-model)](evaluation-intra-model.md)** — Encoder and decoder evaluation, selecting/saving the changed model.
- **[Tutorial: Evaluation (inter-model)](evaluation-inter-model.md)** — Comparing multiple runs (top-k overlap, heatmap).

Use this page to see how the pieces connect in one run; use the part tutorials when you need to understand or customize a step.

---

## Data: precomputed or generate yourself

You can either **use precomputed data** (e.g. a Hugging Face dataset) or **generate data yourself** from raw text with [Tutorial: Data generation](data-generation.md). The trainer accepts both; see [Data handling](../guides/data-handling.md) for all formats.

### Option A: Precomputed data (e.g. Hugging Face)

Pass a dataset id as `data` and, for neutral evaluation, as `eval_neutral_data`:

```python
trainer = TextPredictionTrainer(
    model="bert-base-german-cased",
    run_id="masc_nom_fem_nom",
    data="aieng-lab/de-gender-case-articles",
    target_classes=["masc_nom", "fem_nom"],
    masked_col="masked",
    split_col="split",
    eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
    args=args,
)
```

### Option B: Generate data (e.g. German gender–case, 12 classes)

For the full German definite singular article paradigm there are **12** gender–case cells (3 genders × 4 cases). You define one **TextFilterConfig** per cell and pass them to `TextPredictionDataCreator`. Here we show **five** cells; the rest follow the same pattern (see [Data generation](data-generation.md) for syncretism and the full list).

```python
from gradiend.data import TextFilterConfig, TextPreprocessConfig, TextPredictionDataCreator

feature_targets = [
    TextFilterConfig(targets=["der"], spacy_tags={"pos": "DET", "Case": "Nom", "Gender": "Masc", "Number": "Sing"}, id="masc_nom"),
    TextFilterConfig(targets=["die"], spacy_tags={"pos": "DET", "Case": "Nom", "Gender": "Fem", "Number": "Sing"}, id="fem_nom"),
    TextFilterConfig(targets=["den"], spacy_tags={"pos": "DET", "Case": "Acc", "Gender": "Masc", "Number": "Sing"}, id="masc_acc"),
    TextFilterConfig(targets=["der"], spacy_tags={"pos": "DET", "Case": "Dat", "Gender": "Fem", "Number": "Sing"}, id="fem_dat"),
    TextFilterConfig(targets=["des"], spacy_tags={"pos": "DET", "Case": "Gen", "Gender": "Neut", "Number": "Sing"}, id="neut_gen"),
    # ... e.g. masc_dat, masc_gen, fem_acc, fem_gen, neut_nom, neut_acc, neut_dat for all 12
]
creator = TextPredictionDataCreator(
    base_data="path/to/texts.csv",
    text_column="text",
    preprocess=TextPreprocessConfig(split_to_sentences=True, min_chars=50, max_chars=500),
    spacy_model="de_core_news_sm",
    feature_targets=feature_targets,
)
training = creator.generate_training_data(max_size_per_class=5000, format="per_class")
neutral = creator.generate_neutral_data(additional_excluded_words=["der", "die", "das", "den", "dem", "des"], max_size=5000)
```

Then pass `training` and `neutral` to the trainer as `data=training`, `eval_neutral_data=neutral`, and set **target_classes** to the **pair** you want for this run (e.g. `["masc_nom", "fem_nom"]`).

---

## One run, end to end

Below: one run using **precomputed** data (Option A). Replace the `data` and `eval_neutral_data` with the result of Option B if you generated data yourself. Training and evaluation steps are the same.

```python
from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer import PrePruneConfig, PostPruneConfig

# --- Data: set from Option A (precomputed) or Option B (generated) ---
data = "aieng-lab/de-gender-case-articles"  # Option A; or use training from Option B
eval_neutral_data = "aieng-lab/wortschatz-leipzig-de-grammar-neutral"  # Option A; or neutral from Option B

# --- Training args and trainer (see Tutorial: Training) ---
args = TrainingArguments(
    experiment_dir="runs/gender_de_detailed",
    run_id="masc_nom_fem_nom",
    train_batch_size=8,
    max_steps=500,
    eval_steps=100,
    learning_rate=5e-5,
    pre_prune_config=PrePruneConfig(n_samples=16, topk=0.01, source="diff"),
    post_prune_config=PostPruneConfig(topk=0.05, part="decoder-weight"),
    use_cache=True,
    add_identity_for_other_classes=True,
)
trainer = TextPredictionTrainer(
    model="bert-base-german-cased",
    data=data,  # or training from Option B
    eval_neutral_data=eval_neutral_data,  # or neutral from Option B
    target_classes=["masc_nom", "fem_nom"],
    masked_col="masked",
    split_col="split",
    args=args,
)

# --- Train and evaluate (see Tutorial: Evaluation) ---
trainer.train()
trainer.plot_training_convergence()
enc_eval = trainer.evaluate_encoder(max_size=100, return_df=True, plot=True)
dec_results = trainer.evaluate_decoder()
changed_model = trainer.rewrite_base_model(decoder_results=dec_results, metric_key="masc_nom")
```

For **why** each option matters and what to change when, follow the part tutorials: [Data generation](data-generation.md) → [Training](training.md) → [Evaluation (intra-model)](evaluation-intra-model.md) and [Evaluation (inter-model)](evaluation-inter-model.md).

---

## Where to go next

- **[Data generation](data-generation.md)** — Syncretism, spaCy tags, one filter per gender–case cell.
- **[Training](training.md)** — `experiment_dir` and `run_id`, source/target, pre- and post-pruning, multi-seed, convergence plot.
- **[Evaluation (intra-model)](evaluation-intra-model.md)** — Encoder vs decoder, caching, `metric_key`, saving changed models.
- **[Evaluation (inter-model)](evaluation-inter-model.md)** — Top-k overlap and heatmap for comparing runs.
- **[Data handling](../guides/data-handling.md)** — All supported data formats and column names.
- **[Decoder-only models](../guides/decoder-only.md)** — Causal LMs and optional MLM head.
