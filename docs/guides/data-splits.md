# Data splits

GRADIEND uses train / validation / test splits in two places:

1. **Training** — fit on train; monitor convergence on validation-like data.
2. **Evaluation** — encoder and decoder metrics on held-out rows when available.

For text prediction, the main choice is **row-level** splits vs **vocabulary-held-out**
splits (group by target token). The second is stricter: it tests whether a feature
generalizes to **target words never seen during training**. The data creator (`split_by_target=True`) or the trainer (`split_col="heldout"`) can make that split, see below.

---

## Use an existing split column

When data already has a split column (or you created row-level splits externally):

```python
trainer = TextPredictionTrainer(
    model="bert-base-uncased",
    data="training.csv",
    masked_col="masked",
    split_col="split",
)
```

[:material-file-code-outline: `train_sentiment.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_sentiment.py)

Split values are normalized: `valid`, `validation`, and `val` → validation split.

**When to use:** external benchmark protocols, maximum training rows, or classes with
too few target surface forms for vocabulary holdout.

---

## Hold out target words (vocabulary split)

```python
trainer = TextPredictionTrainer(
    model="bert-base-uncased",
    data=training_data,
    split_col="heldout",                         # explicitly request vocabulary holdout
    split_group_key=[str.strip, str.casefold],   # what counts as one "word"
)
```

**What happens internally:**

1. GRADIEND groups rows by the **factual target token** (after `split_group_key`).
2. Whole groups are assigned to train, validation, or test — a lemma never spans splits.
3. Example: all rows with target `great` → train; all with `excellent` → test.

For data with paired targets (see [`PairedTextFilterConfig`][gradiend.data.text.filter_config.PairedTextFilterConfig]), the groups are the pairs: the two words of a pair always share a split.

This avoids testing on the same adjectives used to fit gradients (a common memorization trap).

**Tune `split_group_key` carefully** — it defines the generalization unit. Grouping
`Good` and `good` together is usually right; grouping unrelated lemmas is not.

**Minimum groups:** for train + validation + test, each class needs **≥3 distinct target
groups**. Otherwise GRADIEND raises a validation error. Fix by adding targets, using
fewer splits, or switching to row-level `split_col`.

---

## Hold out words when creating the data

The data creator can assign the splits itself. With `split_by_target=True`, every target word is placed in exactly one of train, validation and test, so validation and test contain words the model has never seen. The result has the usual `split` column, so the trainer needs no extra setting:

```python
training = creator.generate_training_data(
    max_size_per_class=1500, split_by_target=True, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
)
trainer = TextPredictionTrainer(model="bert-base-uncased", data=training, target_classes=["BE", "AE"])  # split_col="split"
```

With [`PairedTextFilterConfig`][gradiend.data.text.filter_config.PairedTextFilterConfig] targets the unit is the pair, so *colour* and *color* always share a split. With plain targets the unit is the word; casing and surrounding whitespace are ignored (*Bad* and *bad* are one word) unless you pass `split_group_key`. `split_by_target` can be set on the creator or per call, and cannot be combined with `split_group_col`.

Both ways hold out whole words. They differ in when the words are chosen:

| | `split_by_target=True` (creator) | `split_col="heldout"` (trainer) |
|---|---|---|
| Held-out words are chosen | once, when the data is created, and stored in the `split` column | when training starts |
| Different held-out words per seed | no | yes, with `split_resplit_per_seed=True` |
| The two words of a pair | share a split | share a split |
| Grouping of words | `split_group_key` (optional) | `split_group_key` (optional) |

---

## Verify the split worked

After training, check that train and test targets are disjoint:

```python
enc = trainer.evaluate_encoder(split="test", return_df=True)
trainer.plot_encoder_by_target(
    encoder_df=enc["encoder_df"],
    hue_col="data_split",       # color by train / val / test
    plot_style="strip",
    split="test",               # optional: forwarded to analyze_encoder if needed
)
```

You should see **different x-axis tokens** per split hue — test targets must not
appear in the train hue. Strong class separation on test-only targets supports a
generalization claim.

<!-- DOC_PLOT: docs/img/data_splits_encoder_by_target_test.png
Regenerate: gradiend/examples/train_sentiment.py
Copy: runs/examples/sentiment/bert-base-uncased/split_stability/encoder_by_target_strip.pdf -> docs/img/
-->

Runnable example: [train_sentiment.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_sentiment.py)
(vocabulary-held-out splits via `split_col="heldout"`; by-target plots under `split_stability/`).
