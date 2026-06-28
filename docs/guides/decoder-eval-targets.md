# Decoder evaluation targets

Decoder evaluation scores **target-token probabilities** across a grid of decoder updates.
This guide covers **`decoder_eval_targets`**: which tokens are scored per class, or per
training example when a token means different things in different rows.

## Where to set it

Set **`decoder_eval_targets` when you create the trainer** — it is a field on
[`TextPredictionConfig`][gradiend.trainer.text.prediction.trainer.TextPredictionConfig],
passed either as `config=TextPredictionConfig(...)` or as a
[`TextPredictionTrainer`][gradiend.trainer.text.prediction.trainer.TextPredictionTrainer]
keyword argument:

```python
trainer = TextPredictionTrainer(
    model="distilbert-base-uncased",
    data=df,
    target_classes=["3SG", "3PL"],
    decoder_eval_targets=None,  # default
    args=TrainingArguments(experiment_dir="runs/pronouns"),
)
```

[`evaluate_decoder()`][gradiend.trainer.trainer.Trainer.evaluate_decoder] does **not** take
`decoder_eval_targets`; it reads the value stored on the trainer config. Grid size and
sample caps for decoder eval (`decoder_eval_lrs`, `decoder_eval_feature_factors`,
`decoder_eval_max_size_training_like`, …) live on
[`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments] instead.

**Default:** `decoder_eval_targets=None`. Leave it unset (or pass `None`) and GRADIEND
infers targets from your training data when decoder evaluation runs depending on the data:

- **Disjoint tokens** (usual case; e.g. `he` and `He` for 3SG vs `they` and `They` in 3PL): one shared token list per class
- **Shared surface forms** (e.g. `+` in both commutative and non-commutative rows): score
  **that row's** factual token (`label`) against **that row's** alternative token
  (`alternative`) — not one global list per class.

Note that `decoder_eval_targets` is independent of `prediction_objective` (training gradient source).

---

## Decision guide

| Situation                                                                    | Setting |
|------------------------------------------------------------------------------|---------|
| Disjoint token sets per class (e.g. 3SG vs 3PL pronouns)                     | default (`None`) — auto-infer one list per class |
| Same token, different class meaning (e.g. `*` vs `+` in math)                | default (`None`, auto per-row) or `"label"` |
| Always use each row's `label` / `alternative` tokens (targets depend on row) | `"label"` |
| Fixed token list per class, no per-row semantics                             | `{class_name: [tokens]}` dict |

---

## 1. Default (`None`): infer from data

When you do **not** pass `decoder_eval_targets` to the trainer (default `None`), GRADIEND
collects factual and alternative tokens per class from training data at evaluation time.

- **No token overlap across classes** → build one token list per class (e.g. `3SG → [he, He]`,
  `3PL → [they, They]`). Every row in a class is scored against that class's list.
- **Overlap** (same token in more than one class) → score **per training row**: compare
  P(`label`) vs P(`alternative`) for that row only. GRADIEND logs an info message when it
  switches to this mode.

Use the default for disjoint pronoun sets. Overlap is detected automatically when classes
share operators like `+` or `*`.

The automatically inferred decoder eval targets are logged.

---

## 2. Per-row scoring: `decoder_eval_targets="label"`

Force per-row scoring even when classes do not overlap. For **each training row**:

- **P(dataset class)** = probability of the row's **factual** token (`label`)
- **P(other class)** = probability of the row's **alternative** token (`alternative`)

Use this when the same surface form has **different meanings** per row (commutative math
is the usual example).

```python
trainer = TextPredictionTrainer(
    model="distilbert-base-uncased",
    data=commutative_df,
    label_col="label",
    label_class_col="label_class",
    alternative_col="alternative",
    alternative_class_col="alternative_class",
    masked_col="masked",
    decoder_eval_targets="label",
    args=TrainingArguments(experiment_dir="runs/math", max_steps=60),
)
trainer.train()
dec = trainer.evaluate_decoder(plot=True, target_class="commutative")
```

There is no dedicated example script for commutative overlap — inline snippet only. Overlap detection is covered in
`tests/test_trainer_data_inputs.py` (`test_infer_decoder_eval_targets_marks_overlapping_tokens_for_row_wise_fallback`).

<!-- DOC_PLOT: docs/img/decoder_eval_commutative_probability_shifts.png
Optional: run the snippet above in a notebook or short script; copy decoder plot from experiment_dir.
-->

---

## 3. Class-based static lists

```python
decoder_eval_targets = {
    "3SG": ["he", "He"],
    "3PL": ["they", "They"],
}
```

[:material-file-code-outline: `start_workflow.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/start_workflow.py)

Keys must match class names. The same token in multiple classes emits a warning — use
`decoder_eval_targets="label"` (per-row scoring) when token meaning depends on the row.

---

## Exporting per-row scores

When per-row scoring is active (`decoder_eval_targets="label"`, or default with overlapping
tokens), `decoder_eval_export_row_wise_csv=True` writes
`experiment_dir/decoder_row_wise_scores.csv` with per-row `P_factual`, `P_alternative`,
class ids, and masked text.

---

## See also

- [Core classes: TextPredictionConfig](core-classes.md)
- [Tutorial: Evaluation (intra-model)](../tutorials/evaluation-intra-model.md)
- [Token prediction methods](token-prediction-methods.md)
