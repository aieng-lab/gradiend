# Tutorial: Training

This tutorial explains how to train a GRADIEND model with `TextPredictionTrainer`: the main concepts, usage, what gets saved, and how to inspect results. You should have data ready (e.g. from [Data generation](data-generation.md)) before running the trainer.

---

## Overview

**Goal:** Train a model that learns to separate feature classes (e.g. masculine vs feminine forms, i.e, feature *gender*) in model parameter space and can shift the base model’s behavior by rewriting weights.

**What you can do:**

- Configure training via **TrainingArguments** (HF-like API: `num_train_epochs`, `learning_rate`, `eval_steps`, etc.). This tutorial covers the most important args and crucial concepts (e.g., *pruning*), but for detailed documentation of each argument, see the [Training Arguments guide](../guides/training-arguments.md).
- Provide data in various formats. Most importantly, you can pass your own generated data from [Data generation](data-generation.md) (dict of per-class DataFrames, unified DataFrame, or path). See [Data handling](../guides/data-handling.md) for all supported formats.

**Concepts to understand:** source/target, factual/counterfactual, pre-pruning, post-pruning, multi-seed training, caching. These are covered in the sections below.

---

## Key concepts

### HF-like Trainer API

`TextPredictionTrainer` and `TrainingArguments` follow [Hugging Face Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer) conventions where applicable: `num_train_epochs`, `learning_rate`, `eval_steps`, `train_batch_size`, etc. 
See the [Training arguments guide](../guides/training-arguments.md) for the full list of configurable arguments.
In addition to `TrainingArguments` describing the GRADIEND training, the trainer object gets additional parameter describing the *feature* to be learnt, e.g., via data or the base model (e.g., `bert-base-cased`).

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

**experiment_dir** (`TrainingArguments`) is the root under which the trainer writes checkpoints, caches, and plots - if provided. With **run_id** set (`Trainer` argument), outputs go under `experiment_dir/run_id/`.
This enables reusing the same `TrainingArguments` across different `Trainer`s.

When **experiment_dir** is set, **use_cache=True** lets the trainer reuse existing results:

- **train()** skips training if a saved model already exists at the output path.
- Re-running **evaluate_encoder** reuses existing cache when available.
- Re-running **evaluate_decoder** reuses the cached decoder grid.

> Warning: While `use_cache` is a powerful tool to simplify training checkpointing, it does *not* appropriately handle changes in training arguments. 
> For example, if you change the learning rate or pruning configuration but keep `use_cache=True`, the trainer will reuse the old model (if existing in cache folder) and not retrain with the new settings. 
> Always set `use_cache=False` when you want to force retraining with changed arguments. 
> Only a few arguments are part of the caching path, such as `split` and `max_size` in `Trainer.evaluate_encoder()` (as evaluating on different subsets for different purposes makes sense, and therefore these different versions are kept)

Use `use_cache=True` when iterating on analysis or plots; use `False` when you want to force recomputation or retrain.


### Pruning

By default, a GRADIEND model is trained over all core base model parameters (i.e., all model parameters except for the final prediction layers, like MLM head). 
However, this means that the default GRADIEND model has about three times as many parameters as the base model. 
This is not only computationally exhaustive (OOM GPU error), but also requires substantial disk space for checkpoints.
At the same time, many of these parameters are not important for the feature being learnt and can be pruned away without hurting performance.
By using weight absolute value as an importance score, we can prune the least important weights *after* training (*post-pruning*). 
However, to speed up training, we aim to prune as many weights as possible *before* training (*pre-pruning*), based on a heuristic (gradient statistics over a small sample set).
As the heuristic is less precise and we want to ensure (almost) perfect recall of the selection mechanims, pre-pruning is recommended to be more conservative (e.g., only prune 99% of the weights, which typically ensures perfect recall!).

Pruning can be applied automatically before/after training by providing `PrePruneConfig` and `PostPruneConfig` in `TrainingArguments`, see [Pruning](../guides/pruning-guide.md) for details.
Alternatively, you can also apply manual masks to the model (see `ParamMappedGradiendModel.prune()`)


### Multi-seed training

For more stable results, you can run **multiple seeds** and let the trainer pick the best one. 
`TrainingArguments` provides the following options:

- `max_seeds` (default `3`): the maximum number of seeds to run.
- `min_convergent_seeds` (default `1`): as soon as this many seeds have converged (i.e., reached the convergence threshold on the convergence metric), stop training more seeds and select the best among the converged ones.
- `convergence_metric` (default `"correlation"`): the metric to check for convergence (see [Encoder evaluation](evaluation-inter-model.md) for details on correlation computation)
- `convergence_threshold` (default `0.6` for `correlation`): the threshold for convergence on the convergence metric.

**Best-model selection:** Within a single run, the trainer keeps the best checkpoint by the convergence metric, and picks the best model at the end. 
Across seeds, it runs `evaluate_encoder(split="val")` (capped by **seed_selection_eval_max_size**) per seed and selects the **best_seed**; the final model is that seed’s output.

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

---

## What is saved by train()

When **experiment_dir** is set, `train()` writes:

- **Model** — Final model at `experiment_dir/model` (or `experiment_dir/run_id/model`). By default the best checkpoint is promoted into that path; with **max_seeds** &gt; 1, the best seed’s model is copied there. Optional **save_strategy="steps"** also writes `model_step_{step}` (e.g. `model_step_500`).
- **Seed report** — When `max_seeds` &gt; 1, `experiment_dir/seeds/seed_report.json` and per-seed runs under `experiment_dir/seeds/`.
- **Training stats** — `experiment_dir/training_stats.json` with training-time stats (e.g. loss, encoder correlation) per step and per seed.

**Caching:** With **use_cache=True**, an existing model at the output path skips training; per-seed caching applies when using multiple seeds.

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



![Training Convergence Plot](img/training_convergence_example.png)

Options (passed as kwargs):

- **output** — Path to save the plot (default: under `experiment_dir` when set).
- **show** — Whether to display the plot (default: `True`).
- **label_name_mapping** — Dict to rename class ids in the legend (*pretty labels*, e.g. `{"masc_nom": "Masc. Nom."}`).
- **img_format** — Image format for saving (e.g. `"pdf"`, `"png"`).

See the Python docstring for `plot_training_convergence` for the full list of options.

If `experiment_dir` is set, the plot is saved automatically under that directory (`[MODEL_DIR]/training_convergence.pdf`).

---

## Next steps

- **[Example Code](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/gender_de_detailed.py)** — See an example of training several GRADIENDs for the German Article paradigm, and plotting venn diagrams and heatmap.
- **[Tutorial: Evaluation (intra-model)](evaluation-intra-model.md)** — Encoder and decoder evaluation, selecting the changed model.
- **[Tutorial: Evaluation (inter-model)](evaluation-inter-model.md)** — Comparing multiple runs, i.e., different target classes (top-k overlap, heatmaps).

---

## See also

- [Training arguments guide](../guides/training-arguments.md) — Detailed documentation of each argument.
- [Pruning](../guides/pruning-guide.md) — Manual masks and full pruning API.
- [API reference](../api-reference.md) — `TrainingArguments`, `PrePruneConfig`, `PostPruneConfig`.
