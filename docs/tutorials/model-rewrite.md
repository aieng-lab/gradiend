# Tutorial: Model Rewrite

This tutorial describes how to **export a persistently modified checkpoint** whose behavior is shifted along the learned feature direction (e.g., higher token probabilities for one target class). You can either use **assisted parameter selection** via decoder evaluation (recommended) or supply **manual** feature factor and learning rate.

For running decoder evaluation itself, see [Tutorial: Evaluation (intra-model)](evaluation-intra-model.md).

---

## Prerequisites

Before rewriting, you need:

- A trained GRADIEND model (i.e. `trainer.train()` has been run).
- Either decoder evaluation results from `trainer.evaluate_decoder(...)`, or you will pass manual `feature_factor` and `learning_rate` (see below).
- A target class id to strengthen or weaken (e.g. `"masc_nom"`).

---

## Parameter selection: assisted vs manual

The rewrite applies an update of the form *base model + learning_rate × decoder(feature_factor)*. The strength and direction of the effect depend on **feature factor** and **learning rate**.

- **Manual parameters:** You can call the model’s `rewrite_base_model(learning_rate=..., feature_factor=...)` (on a `ModelWithGradiend` instance) with any values. This gives full control but the **outcome is ambiguous**—different choices can over-strengthen, under-strengthen, or harm other classes.
- **Assisted parameter selection (recommended):** Run `trainer.evaluate_decoder(...)` to sweep a grid of `(feature_factor, learning_rate)` and score each candidate (e.g. by target-class probability or combined metric). The trainer then uses the **best** config per class when you call `trainer.rewrite_base_model(decoder_results=..., target_class=...)`. This way, the chosen parameters are driven by your evaluation data and metric.

Use the trainer’s `rewrite_base_model` with `decoder_results` when you want data-driven parameters; use the model’s `rewrite_base_model` with explicit `learning_rate` and `feature_factor` when you want to experiment manually despite ambiguous outcomes.

---

## What `trainer.rewrite_base_model(...)` does

When you pass `decoder_results` (and optionally `target_class`), the trainer looks up the best `(learning_rate, feature_factor)` from the decoder evaluation for the chosen class and direction, then applies that update to the base model and returns the rewritten model (e.g. `BertForMaskedLM` or a causal LM).

- **Strengthen (default):** `increase_target_probabilities=True` — increases probabilities for the target tokens of `target_class`.
- **Weaken:** `increase_target_probabilities=False` — applies the weakening config; you must have run decoder evaluation with `increase_target_probabilities=False` first so that the corresponding summary exists (e.g. `"masc_nom_weaken"`).

---

## Basic usage (assisted parameters)

```python
# 1) Run decoder evaluation (strengthen direction by default)
dec_results = trainer.evaluate_decoder()

# 2) Rewrite for one class using the best config from the evaluation
changed_model = trainer.rewrite_base_model(
    decoder_results=dec_results,
    target_class="masc_nom",
)
```

You can then use `changed_model` for inference or further evaluation.

---

## Strengthen vs weaken

**Strengthen** (default):

```python
changed_model = trainer.rewrite_base_model(
    decoder_results=dec_results,
    target_class="masc_nom",
)
```

**Weaken:** run decoder evaluation in weaken direction first, then pass the same flag to rewrite:

```python
# Evaluate weaken direction (produces keys like "masc_nom_weaken")
dec_results_weaken = trainer.evaluate_decoder(increase_target_probabilities=False)

# Rewrite using the weaken config
changed_model_weaken = trainer.rewrite_base_model(
    decoder_results=dec_results_weaken,
    target_class="masc_nom",
    increase_target_probabilities=False,
)
```

---

## Choosing target classes

- `target_class="masc_nom"` — one rewritten model for that class.
- `target_class=["masc_nom", "fem_nom"]` — one rewritten model per class.

Use class ids that match your setup (e.g. from `target_classes` or your dataset’s feature class ids).

---

## Using cached decoder results

If `experiment_dir` is set and decoder evaluation was run with `use_cache=True`, you can omit `decoder_results`; the trainer will load decoder stats from cache when available:

```python
changed_model = trainer.rewrite_base_model(target_class="masc_nom")
```

---

## Saving rewritten model(s)

To save the rewritten checkpoint(s), pass `output_dir`:

```python
trainer.rewrite_base_model(
    decoder_results=dec_results,
    target_class="masc_nom",
    output_dir="./output/masc_nom_rewrite",
)
```

With multiple `target_class` entries, `experiment_dir` is used to derive paths when saving.

---

## Minimal end-to-end snippet

```python
trainer.train()
dec_results = trainer.evaluate_decoder()

# In-memory rewritten model (uses best config from decoder evaluation)
changed_model = trainer.rewrite_base_model(
    decoder_results=dec_results,
    target_class="masc_nom",
)

# Save rewritten model to disk
trainer.rewrite_base_model(
    decoder_results=dec_results,
    target_class="masc_nom",
    output_dir="./output/masc_nom_rewrite",
)
```

---

## See also

- [Tutorial: Evaluation (intra-model)](evaluation-intra-model.md) — Encoder and decoder evaluation, including decoder config selection.
- [Tutorial: Evaluation (inter-model)](evaluation-inter-model.md) — Comparing runs (e.g. top-k overlap, heatmaps).
- [Core classes and use cases](../guides/core-classes.md) — `ModelWithGradiend` and rewrite APIs.
- [API reference](../api/index.md) — `TextPredictionTrainer`, `ModelWithGradiend`.
