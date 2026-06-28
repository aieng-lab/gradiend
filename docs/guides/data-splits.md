# Data splits

GRADIEND uses train / validation / test splits in two places:

1. **Training** — fit on train; monitor convergence on validation-like data.
2. **Evaluation** — encoder and decoder metrics on held-out rows when available.

For text prediction, the main choice is **row-level** splits vs **vocabulary-held-out**
splits (group by target token). The second is stricter: it tests whether a feature
generalizes to **target words never seen during training**.

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
    split_col=None,                              # let GRADIEND assign splits
    split_group_key=[str.strip, str.casefold],   # what counts as one "word"
)
```

**What happens internally:**

1. GRADIEND groups rows by the **factual target token** (after `split_group_key`).
2. Whole groups are assigned to train, validation, or test — a lemma never spans splits.
3. Example: all rows with target `great` → train; all with `excellent` → test.

This avoids testing on the same adjectives used to fit gradients (a common memorization trap).

**Tune `split_group_key` carefully** — it defines the generalization unit. Grouping
`Good` and `good` together is usually right; grouping unrelated lemmas is not.

**Minimum groups:** for train + validation + test, each class needs **≥3 distinct target
groups**. Otherwise GRADIEND raises a validation error. Fix by adding targets, using
fewer splits, or switching to row-level `split_col`.

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
(vocabulary-held-out splits via `split_col=None`; by-target plots under `split_stability/`).
