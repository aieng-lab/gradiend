# Tutorial: Evaluation (intra-model)

This tutorial covers **intra-model** evaluation: encoder and decoder evaluation for a **single** trained GRADIEND model, i.e., evaluating whether the encoder separates the feature classes, whether the decoder can shift the base model’s behavior, and how we can derive a GRADIEND-modified model with probabilities shifted towards the targeted feature class. 

To run code yourself, you should have a trained GRADIEND model (e.g. after [Tutorial: Training](training.md)). 
For **comparing multiple runs** (top-k overlap, heatmaps), see [Tutorial: Evaluation (inter-model)](evaluation-inter-model.md).

---

## Encoder Evaluation

### What encoder evaluation does

**Encoder evaluation** answers the questions: *Do the encoded gradients separate the feature classes? And how are non-target but related gradients encoded?* 

The trainer runs your evaluation data (and optionally neutral data) through the GRADIEND encoder and computes a **correlation** between the encoded values and the class labels. High correlation means the encoder has learned to distinguish the two classes (e.g. masc_nom vs fem_nom) from their gradients.

**evaluate_encoder(...)** runs the full pipeline: encode the data, compute unified encoder metrics, and optionally produce a distribution plot. It returns a dict with all metrics (see below). Use `return_df=True` to include the full `encoder_df` key.

```python
enc_eval = trainer.evaluate_encoder(max_size=100, split='test', return_df=True, plot=True)
```
Options:
- **max_size**: Maximum number of examples to encode per *input class* (for speed; set to `None` for all). Input class are defined by feature class ids (i.e., per training transition) and the up to two neutral datasets. 
- **split**: Which split to evaluate on (e.g. `test`, `val`).
- **return_df**: Whether to include the full DataFrame of encoded values and metadata in the returned dict (can be large).
- **plot**: Whether to create a distribution plot of encoded values by class (see below).
- **plot_kwargs**: Optional dict of kwargs to customize the plot (e.g. `legend_name_mapping` to make class labels human-readable).
- **use_cache**: Whether to load from cache if available (skips re-encoding).
- ... (much more options; see API reference for details).

Crucially, this method computes cache files depending on `max_size` and `split` such that you can later easily compare different encodings on different splits and sizes without re-running the encoding analysis.

---

### Encoder Datasets

We consider different types of datasets/ gradient as GRADIEND encoder inputs:

- **training**: data as seen during training (i.e., the target class transitions and optionally identity transitions for other classes).
- **neutral_training_masked**: an automated approach to derive a neutral-like dataset for training-like data, by first creating un-masked texts (replace [MASK] by the factual label), and then only use non-feature related target tokens to derive *neutral* gradients (see )
- **neutral_dataset**: a separate dataset of neutral examples that are not seen during training. Pass this dataset as `eval_neutral_dataset` to the trainer.


### Encoder Distribution Plot

An easy overview of the encoder’s behavior is the **distribution plot**: a violin plot showing how the encoded values distribute for 

*Run encoder evaluation with `plot=True` (or `trainer.plot_encoder_distributions()`) to generate the distribution plot.*

We expect that the target training classes encode to polarly different values  (+-1), while the neutral variants (if present) should cluster around 0.
To visually highlight the special role of target classes, we mark their violins and legend entries in **bold**.


This plot has a lot of options to customize, e.g., to make the class labels human-readable (e.g. `masc_nom` → “masc nominative”) or to group classes into broader categories (e.g. all neutral variants into “neutral”). See [Evaluation & visualization](../guides/evaluation-visualization.md#2-encoder-distribution-plot) for full customization options.


### Encoder Metrics 

Another option to interpret the encoder’s behavior is the returned dict of metrics (i.e., quantified properties). The exact keys depend on the options you pass to `evaluate_encoder()`, but here is a general overview of the main keys and their interpretation:


| Key | Type | Interpretation                                                                                                                                                                                                                      |
|-----|------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `n_samples` | int | Total number of encoded examples.                                                                                                                                                                                                   |
| `sample_counts` | dict | Breakdown by `by_type` (training, neutral_training_masked, neutral_dataset) and optionally `training_by_feature_class`.                                                                                                             |
| `all_data` | dict | Metrics over **all** rows. Contains `correlation` (Pearson) and `accuracy` (ternary classification using neg/pos boundaries).                                                                                                       |
| `training_only` | dict | Same keys as `all_data`, but computed only over **training** rows (excludes type neutral; compared to `target_classes_only` this may include identity transitions if enabled with label 0). Uses ternary classification (-1, 0, 1). |
| `target_classes_only` | dict | Same keys, but over target class transitions only (excludes type neutral and identity, label 0). Uses **binary** classification (pred ≥ neutral_boundary → 1, else -1).                                                             |
| `boundaries` | dict | Thresholds used for accuracy computation: `neg_boundary`, `pos_boundary`, `neutral_boundary` (configurable via parameters in `evaluate_encoder()`).                                                                                 |
| `correlation` | float | Pearson correlation of training_only (typically used as main training score). |
| `mean_by_class` | dict | Label value (−1, 0, 1) → mean encoded value. Uses **non-neutral-type rows only** (training + identity). For class separation on target/identity transitions. |
| `mean_by_type` | dict | Type (`training`, `neutral_training_masked`, `neutral_dataset`) → mean encoded value. Covers **all** rows including neutral variants; use for final eval when neutral means matter. |
| `neutral_mean_by_type` | dict | Subset of `mean_by_type` for neutral variants only (`neutral_training_masked`, `neutral_dataset`). Convenience for comparing encoder behavior on neutral data. |
| `mean_by_feature_class` | dict | Feature class id (e.g. source_id) → mean encoded value. When present. |
| `label_value_to_class_name` | dict | Label value → human-readable class name: **0 → "neutral"**; other labels → `source_id` when available (e.g. `1 → "masc_nom"`, `-1 → "fem_nom"`). |
| `encoder_df` | DataFrame | Included only when `evaluate_encoder(return_df=True)`. Full per-row data (encoded, label, type, source_id, target_id).                                                                                                              |


---

### Running encoder evaluation and plots

Typical usage after training:

```python
# Full run: encode, correlate, optionally get DataFrame and plot
enc_eval = trainer.evaluate_encoder(max_size=1000, return_df=True, plot=True)
print("Correlation:", enc_eval.get("correlation"))

# Later: reuse from cache (no re-encoding)
# Importantly: you must provide same max_size and split to load the same cache file (or omit these both times to use defaults)
enc_eval = trainer.evaluate_encoder(max_size=1000, use_cache=True)
```

---

## Decoder Evaluation

### What decoder evaluation does

**Decoder evaluation** answers: *Can we change the base model’s behavior by applying the learned decoder (before negatively affecting general language modeling capabilities)?* 
It tries different positive **learning rates** and **feature factors** (usually -1 and +1) using the update formula "base model + learning rate * decoder(feature factor)", and aims to select the *best* option at the end.

Each run measures how much the model’s predictions shift (i.e. probability of target tokens and impact on language modeling performance). 
The result is a grid of scores; the “best” configuration is the one that maximizes a chosen metric (e.g. probability shift toward the target class while satisfying a language-model constraint), see *policies* below.

**evaluate_decoder()** runs this grid (or loads it from cache when **use_cache=True**). It returns a dict with **summary** (per-class or combined best configs), raw results, and metadata.

---

### Selecting the “changed” model (metric_key)

The decoder results dict has one or more **metric keys**:  per-class keys (e.g. your `target_classes` ids like `"masc_nom"`, `"fem_nom"`). Each key has a recommended **feature_factor** and **learning_rate**.

- **select_changed_model(decoder_results=..., metric_key="masc_nom")** builds the modified model **in memory** only: it takes the trainer’s trained model and applies the decoder with the best config for that key. It returns the modified model (e.g. a `BertForMaskedLM`) that you can use for inference or further evaluation. It does **not** save to disk.
- **metric_key** can be a single string (one model) or a list of strings (one model per key). Use the class id when you want the model biased toward that class (e.g. `metric_key="masc_nom"` for “stronger masculine nominative”).
- If you set **experiment_dir** and have already run **evaluate_decoder(use_cache=True)**, you can omit **decoder_results**: the trainer will load the cached decoder stats from disk when available (so you can call **select_changed_model(metric_key="masc_nom")** without keeping the large dict in memory). This requires **use_cache=True** when running **evaluate_decoder** so that the decoder results cache is populated.



---

### Saving the changed model to disk

**select_and_save_changed_model(...)** does the same selection as **select_changed_model** but then **saves** the modified model to disk. You need a place to save: either **experiment_dir** in `TrainingArguments` (then the path is derived from the run and metric key) or **output_dir** for a single metric key.

---

### Policies for selecting the best config

The “best” config is the one that optimizes a chosen metric, which is defined by `SelectionPolicy`:

- **`LMSThresholdPolicy`**: Restrict to candidates whose LMS is at least `ratio` × base LMS (default `ratio=0.99`), then pick the one that maximizes the metric. If none pass the threshold, fall back to the candidate with smallest learning rate (least decoder impact).
- **`LMSTimesMetricPolicy`**: Pick the candidate that maximizes `metric × lms` (product of metric and LMS). Balances probability shift with language modeling preservation in one score.

---

### Minimal intra-model evaluation snippet

```python
# After trainer.train() and trainer.plot_training_convergence()
enc_eval = trainer.evaluate_encoder(max_size=100, return_df=True, plot=True)
# Reuse from cache when re-running:
# enc_eval = trainer.evaluate_encoder(use_cache=True)

dec_results = trainer.evaluate_decoder()
# Optional: dec_results = trainer.evaluate_decoder(use_cache=True) when re-running

changed_model = trainer.select_changed_model(decoder_results=dec_results, metric_key="masc_nom")
# Or save: trainer.select_and_save_changed_model(decoder_results=dec_results, metric_key="masc_nom")
```

---

## See also

- [Tutorial: Evaluation (inter-model)](evaluation-inter-model.md) — Top-k overlap and heatmaps for comparing multiple runs.
- [Evaluation & visualization](../guides/evaluation-visualization.md) — Plot customization (convergence, encoder, heatmap, Venn).
- [API reference](../api-reference.md) — `Evaluator`, `EncoderEvaluator`, `DecoderEvaluator`.
