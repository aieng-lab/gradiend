# Tutorial: Training

This tutorial explains how to train a GRADIEND model with [`TextPredictionTrainer`][gradiend.trainer.text.prediction.trainer.TextPredictionTrainer]: the main concepts, usage, what gets saved, and how to inspect results. You should have data ready (e.g. from [Data generation](data-generation.md)) before running the trainer.

!!! tip "Optional dependency: plotting"
    The convergence plot ([`plot_training_convergence()`][gradiend.visualizer.convergence.plot_training_convergence]) and other training-related plots require the **plot** extra. If you did not install it with GRADIEND, install it with:

    ```bash
    pip install gradiend[plot]
    ```

---

## Overview

**Goal:** Train a model that learns to separate feature classes (e.g. masculine vs feminine forms, i.e, feature *gender*) in model parameter space and can shift the base model’s behavior by rewriting weights.

**What you can do:**

- Configure training via **[`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments]** (HF-like API: `num_train_epochs`, `learning_rate`, `eval_steps`, etc.). This tutorial covers the most important args and crucial concepts (e.g., *pruning*), but for detailed documentation of each argument, see the [Training Arguments guide](../guides/training-arguments.md).
- Provide data in various formats. Most importantly, you can pass your own generated data from [Data generation](data-generation.md) (dict of per-class DataFrames, unified DataFrame, or path). See [Data handling](../guides/data-handling.md) for all supported formats.

**Concepts to understand:** source/target, factual/counterfactual, pre-pruning, post-pruning, multi-seed training, caching. These are covered in the sections below.

---

## Key concepts

### HF-like Trainer API

[`TextPredictionTrainer`][gradiend.trainer.text.prediction.trainer.TextPredictionTrainer] and [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments] follow [Hugging Face Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) conventions where applicable: `num_train_epochs`, `learning_rate`, `eval_steps`, `train_batch_size`, etc. 
See the [Training arguments guide](../guides/training-arguments.md) for the full list of configurable arguments.
In addition to [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments] describing the GRADIEND training, the trainer object gets additional parameter describing the *feature* to be learnt, e.g., via data or the base model (e.g., `bert-base-cased`).

### Target Classes

GRADIEND learns to separate between two orthogonal classes belonging to the same feature (e.g., male and female belong to feature gender).
These two classes are called *`target_classes`*.
Besides the `target_classes`, non-binary features have other classes that may be used for training and evaluation (e.g., feature *race* with classes *Asian*, *Black*, *White*, ...). 
Hence, if more than two classes are provided via `data`, `target_classes` becomes a required Trainer init argument. 

### Source and target (factual/alternative/diff)

Each training example has a **factual** token (what appears in the text at the mask) and an **alternative** (counterfactual) token. GRADIEND is trained on gradients derived from these.

- **source** — Which gradient feeds the encoder: `"factual"`, `"alternative"`, or `"diff"`. Common choice: `"alternative"`.
- **target** — What the decoder is trained to predict: `"factual"`, `"alternative"`, or `"diff"`. Common choice: `"diff"`.

The default `source="alternative"` and `target="diff"` works well for “change the model toward the alternative” use cases (e.g. debiasing).

### Target and Identity Transitions

*Target transitions* are those where the target classes are involved and the alternative is the counterfactual: the other target class’s token (or a randomly weighted sample from the surface realisations of the other target classes).

In contrast, *identity transitions* are those where the alternative equals the factual token, i.e., the same token as in the text. Identity transitions are used for non-target classes when **add_identity_for_other_classes=True** (default: `False`) is passed to Trainer, which can help the model learn to keep non-target classes unchanged.

### Experiment directory and caching (`use_cache`)

**experiment_dir** ([`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments]) is the root under which the trainer writes checkpoints, caches, and plots - if provided. With **run_id** set ([`Trainer`][gradiend.trainer.trainer.Trainer] argument), outputs go under `experiment_dir/run_id/`.
This enables reusing the same [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments] across different [`Trainer`][gradiend.trainer.trainer.Trainer]s.

**use_cache** controls whether an existing training checkpoint may be reused:

| Value | Behavior |
|-------|----------|
| `False` | Always train (or retrain) even when a saved model exists. |
| `True` | Reuse a saved model when it exists **and** matches the current training fingerprint (see below). |
| `"always"` | Reuse any saved model at the output path (skip fingerprint and convergence checks). |
| `"only_convergent"` | Same fingerprint check as `True`, but only reuse checkpoints that met convergence requirements (`min_convergent_seeds`, convergence metric/threshold). |

When a checkpoint is reused, **evaluate_encoder** / **evaluate_decoder** may also skip recomputation when their own caches exist (evaluator `use_cache` is separate).

**Training cache fingerprint:** After each run, `training.json` stores a `cache_fingerprint` derived from pruning and gradient settings. On reuse, the trainer compares this fingerprint to the current [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments]. A mismatch logs a warning and forces retraining. Checked fields include:

- `pre_prune_config` (all [`PrePruneConfig`][gradiend.trainer.core.pruning.PrePruneConfig] fields, e.g. `n_samples`, `topk`, `source`)
- `post_prune_config`
- `reuse_pre_prune`
- `source`, `target`
- `gradiend_input_dim` (saved model size after pruning)

Legacy checkpoints without `cache_fingerprint` are rejected when pre-pruning is requested but the saved `input_dim` looks like an unpruned full model.

> **Incomplete coverage:** Fingerprinting is a best-effort guard, not a full equivalence check on [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments]. It currently covers pruning and source/target settings only. Many arguments that affect training are **not** compared (e.g. `learning_rate`, `max_steps`, `train_batch_size`, `params`, `target_classes`, data splits). Changing those can still reuse an old checkpoint when the fingerprint matches. Treat `use_cache=True` as convenient for iterative analysis, not as proof that two runs used identical settings. Set `use_cache=False`, use `use_cache="always"` only when you intentionally want to skip fingerprint checks, or use a new `experiment_dir` / `run_id`, when you need a guaranteed fresh train.

Evaluator caches key on different arguments (e.g. `split` and `max_size` in [`evaluate_encoder()`][gradiend.trainer.trainer.Trainer.evaluate_encoder]), so evaluating on different subsets can coexist.

Use `use_cache=True` when iterating on analysis or plots with matching pruning/source settings; use `"always"` to force reuse regardless of fingerprint; use `False` when you want to force recomputation or retrain.

### Pruning

By default, a GRADIEND model is trained over all core base model parameters (i.e., all model parameters except for the final prediction layers, like MLM head). To use only specific layers (e.g. first two encoder layers), set **[`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments].params** to a list of parameter names or wildcards (e.g. `params=["bert.encoder.layer.0.*", "bert.encoder.layer.1.*"]`). See [Training arguments: Model parameters (which layers to use)](../guides/training-arguments.md#model-parameters-which-layers-to-use) for details. 
However, this means that the default GRADIEND model has about three times as many parameters as the base model. 
This is not only computationally exhaustive (OOM GPU error), but also requires substantial disk space for checkpoints.
At the same time, many of these parameters are not important for the feature being learnt and can be pruned away without hurting performance.
By using weight absolute value as an importance score, we can prune the least important weights *after* training (*post-pruning*). 
However, to speed up training, we aim to prune as many weights as possible *before* training (*pre-pruning*), based on a heuristic (gradient statistics over a small sample set).
As the heuristic is less precise and we want to ensure (almost) perfect recall of the selection mechanims, pre-pruning is recommended to be more conservative (e.g., only prune 99% of the weights, which typically ensures perfect recall!).

Pruning can be applied automatically before/after training by providing [`PrePruneConfig`][gradiend.trainer.core.pruning.PrePruneConfig] and [`PostPruneConfig`][gradiend.trainer.core.pruning.PostPruneConfig] in [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments], see [Pruning](../guides/pruning-guide.md) for details.
Alternatively, you can also apply manual masks to the model (see [`ParamMappedGradiendModel`][gradiend.model.param_mapped.ParamMappedGradiendModel].prune())

### Multi-seed training

For more stable results, you can run **multiple seeds** and let the trainer pick the best one. 
[`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments] provides the following options:

- `max_seeds` (default `3`): the maximum number of seeds to run.
- `min_convergent_seeds` (default `1`): as soon as this many seeds have converged (i.e., reached the convergence threshold on the convergence metric), stop training more seeds and select the best among the converged ones.
- `convergence_metric` (default `"correlation"`): the metric to check for convergence (see [Encoder evaluation](evaluation-inter-model.md) for details on correlation computation)
- `convergence_threshold` (default `0.6` for `correlation`): the threshold for convergence on the convergence metric.

**Best-model selection:** Within a single run, the trainer keeps the best checkpoint by the convergence metric, and picks the best model at the end. 
Across seeds, it runs [`evaluate_encoder(split="val")`][gradiend.trainer.trainer.Trainer.evaluate_encoder] (capped by **seed_selection_eval_max_size**) per seed and selects the **best_seed**; the final model is that seed’s output.

See the [Training arguments guide](../guides/training-arguments.md#seed-report-format) for the seed report format.

---

## Using the trainer

Create a trainer with your model, data, and arguments, then call `train()`:

```python
from gradiend import TextPredictionTrainer, TrainingArguments, PrePruneConfig, PostPruneConfig

args = TrainingArguments(
    experiment_dir="runs/my_experiment",
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
    run_id="masc_nom_fem_nom",
    model="bert-base-uncased",
    data=training, # generated before
    eval_neutral_data=neutral,
    target_classes=["masc_nom", "fem_nom"],
    args=args,
)

trainer.train()
```

[:material-file-code-outline: `train_gender_de_detailed.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de_detailed.py)

---

## What is saved by train()

When **experiment_dir** is set, `train()` writes:

- **Model** — Final model at `experiment_dir/model` (or `experiment_dir/run_id/model`). By default the best checkpoint is promoted into that path; with **max_seeds** &gt; 1, the best seed’s model is copied there. Optional **save_strategy="steps"** also writes `model_step_{step}` (e.g. `model_step_500`).
- **Seed report** — When `max_seeds` &gt; 1, `experiment_dir/seeds/seed_report.json` and per-seed runs under `experiment_dir/seeds/`.
- **Training stats** — `experiment_dir/training_stats.json` with training-time stats (e.g. loss, encoder correlation) per step and per seed.

**Caching:** With **use_cache=True** (or `"only_convergent"`), an existing model at the output path skips training when the cache fingerprint matches; per-seed caching applies when using multiple seeds. See `training.json` → `cache_fingerprint` under each seed directory.

---

## Accessing training statistics

After training, use `get_training_stats()` to load the training statistics:

```python
stats = trainer.get_training_stats()
# e.g. stats["training_stats"]["correlation"], stats["training_stats"]["mean_by_class"]
```

This reads from the saved checkpoint directory (when `experiment_dir` is set), otherwise a GRADIEND model path can be passed to `get_training_stats(model_path=...)` to load stats from a specific model.

---

## Plotting the convergence

Plot loss and encoder correlation over steps:

```python
trainer.plot_training_convergence()
```

*Run [`trainer.plot_training_convergence()`][gradiend.trainer.trainer.Trainer.plot_training_convergence] to generate this plot; use the `output` option to save to a file.*

Options (passed as kwargs):

- **output** — Path to save the plot (default: under `experiment_dir` when set).
- **show** — Whether to display the plot (default: `True`).
- **label_name_mapping** — Dict to rename class ids in the legend (*pretty labels*, e.g. `{"masc_nom": "Masc. Nom."}`).
- **class_spread** — Optional shaded band behind class means: `"minmax"`, `"iqr"`, or `"ci95"` (95% confidence interval).
- **img_format** — Image format for saving (e.g. `"pdf"`, `"png"`).

See the Python docstring for [`plot_training_convergence`][gradiend.visualizer.convergence.plot_training_convergence] and the [evaluation visualization guide](../guides/evaluation-visualization.md#training-convergence) for the full list of options.

If `experiment_dir` is set, the plot is saved automatically under that directory (`[MODEL_DIR]/training_convergence.pdf`).

---

### Seed stability analysis

When you need to study variance across convergent seeds (not just pick the best one), enable stability mode and use [`trainer.multi_seed()`][gradiend.trainer.trainer.Trainer.multi_seed] after training:

```python
args = TrainingArguments(
    max_seeds=5,
    min_convergent_seeds=3,
    analyze_seed_stability=True,
)
trainer.train()

view = trainer.multi_seed()  # selection, aggregate, dispersion kwargs optional
# the view object contains the same methods as the single-seed best-model trainer, but now with aggregated results across seeds

enc = view.evaluate_encoder(split="test")
print(enc["correlation"])                 # mean across seeds
print(enc["seeds"]["stats"]["correlation"])  # std, min, max, n

view.plot_encoder_distributions(show=False)
```

Single-seed APIs ([`trainer.evaluate_encoder`][gradiend.trainer.trainer.Trainer.evaluate_encoder], [`trainer.get_model()`][gradiend.trainer.trainer.Trainer.get_model]) are unchanged and always use the selected best model. See [Multi-seed analysis](../guides/multi-seed.md).

---

## Accessing the GRADIEND model

After training, you typically work with a **[`ModelWithGradiend`][gradiend.model.model_with_gradiend.ModelWithGradiend]**: it wraps the base model plus the trained GRADIEND encoder/decoder. You can obtain it in two ways:

1. **From the trainer** — Use [`trainer.get_model()`][gradiend.trainer.trainer.Trainer.get_model] to get the trainer’s cached ModelWithGradiend.
2. **From disk** — Load from a checkpoint path with `ModelWithGradiend.from_pretrained(load_directory)`.

```python
# From the trainer
model = trainer.get_model()

# Or load from a saved checkpoint
from gradiend import ModelWithGradiend
model = ModelWithGradiend.from_pretrained("experiment_dir/run_id/model")
```

**Model variants:**

| Class | Purpose |
|-------|---------|
| **[`ModelWithGradiend`][gradiend.model.model_with_gradiend.ModelWithGradiend]** | Base model + GRADIEND. Use for evaluation, encoding gradients, or applying decoder updates via `rewrite_base_model()`. |
| **[`ParamMappedGradiendModel`][gradiend.model.param_mapped.ParamMappedGradiendModel]** | GRADIEND encoder/decoder with parameter mapping (dict I/O). Access via `model.gradiend` on a [`ModelWithGradiend`][gradiend.model.model_with_gradiend.ModelWithGradiend]. |
| **[`GradiendModel`][gradiend.model.model.GradiendModel]** | Weights-only encoder/decoder (no base model). For low-level access or saving/loading GRADIEND weights independently. |

All three are exported from `gradiend`:

```python
from gradiend import GradiendModel, ParamMappedGradiendModel, ModelWithGradiend
```

For more detail, see [Core classes](../guides/core-classes.md) and the [API reference](../api/index.md).

---

## Next steps

- **[Example Code](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de_detailed.py)** — See an example of training several GRADIENDs for the German Article paradigm, and plotting venn diagrams and heatmap.
- **[Tutorial: Evaluation (intra-model)](evaluation-intra-model.md)** — Encoder and decoder evaluation, decoder config selection.
- **[Tutorial: Model Rewrite](model-rewrite.md)** — Apply decoder-selected rewrites and save changed checkpoints.
- **[Tutorial: Evaluation (inter-model)](evaluation-inter-model.md)** — Comparing multiple runs, i.e., different target classes (top-k overlap, heatmaps).
- **[Multi-seed analysis](../guides/multi-seed.md)** — Evaluate and plot across convergent seed checkpoints.

---

## See also

- [Training arguments guide](../guides/training-arguments.md) — Detailed documentation of each argument.
- [Pruning](../guides/pruning-guide.md) — Manual masks and full pruning API.
- [API reference](../api/index.md) — [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments], [`PrePruneConfig`][gradiend.trainer.core.pruning.PrePruneConfig], [`PostPruneConfig`][gradiend.trainer.core.pruning.PostPruneConfig].
