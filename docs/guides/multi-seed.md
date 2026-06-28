# Multi-seed analysis

GRADIEND training can differ across random seeds. Most workflows train until **one**
converging seed is found (`min_convergent_seeds=1`) and use that checkpoint for
everything else.

**Multi-seed analysis** trains several *converging* seeds, keeps each checkpoint on
disk, and lets you report **mean metrics with dispersion** (typically std) instead of
a single lucky run.

This guide covers **training**, the [`trainer.multi_seed()`][gradiend.trainer.trainer.Trainer.multi_seed] view, **aggregated eval**,
and **encoder plots with mean ± std**. Pairwise **checkpoint similarity** (top-k overlap,
cosine heatmaps, `gradiend_only=True`) is [cross-model comparison](cross-model-comparison.md#comparing-seeds) — same seeds, different question.

**Runnable example** (covers every section below):

```bash
python -m gradiend.examples.train_multi_seed_stability
python -m gradiend.examples.train_multi_seed_stability --plot-only          # cached checkpoints
python -m gradiend.examples.train_multi_seed_stability --write-docs-images  # refresh docs/img/
```

Quick verification without full training: `pytest tests/test_multi_seed_view.py tests/test_multi_seed_analysis_integration.py -q`

---

## Quick start

```python
args = TrainingArguments(
    experiment_dir="runs/my_feature",
    max_seeds=5,
    min_convergent_seeds=3,            # must be <= max_seeds; training ends after this many converging seeds
    analyze_seed_stability=True,       # saves all convergent seeds; required for analysis
    fail_on_non_convergence=True,      # scripts should not silently continue
)

trainer = TextPredictionTrainer(..., args=args)
trainer.train()

# Single-seed API — always the selected *best* convergent checkpoint
single = trainer.evaluate_encoder(split="test")
print(single["correlation"])  # 0.8123

# Multi-seed API — aggregate over convergent checkpoints
view = trainer.multi_seed(selection="all_convergent", dispersion="std")
multi = view.evaluate_encoder(split="test")
print(multi["correlation"])                          # 0.7981  (mean across seeds)
print(multi["seeds"]["stats"]["correlation"])        # {"mean": 0.7981, "std": 0.014, "n": 2, ...}

view.plot_encoder_by_target(
    split="test",
    plot_style="errorbar",
    error_stat="std",
    show_seed_points=True,
)
```

[:material-file-code-outline: `train_multi_seed_stability.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_multi_seed_stability.py)

---

## Training settings

```python
args = TrainingArguments(
    experiment_dir="runs/my_feature",
    max_seeds=5,
    min_convergent_seeds=3,            # must be <= max_seeds; training ends after this many converging seeds
    saved_seed_runs="best_only",
    analyze_seed_stability=True,       # required for multi_seed(); sets saved_seed_runs to "all_convergent"
    fail_on_non_convergence=True,      # scripts should not silently continue
)
```

| Argument | Meaning |
|----------|---------|
| `max_seeds` | Maximum seeds to try |
| `min_convergent_seeds` | Stop once this many converge; `None` = run all `max_seeds` |
| `analyze_seed_stability` | Save every convergent checkpoint for later analysis. Sets `saved_seed_runs` to `"all_convergent"` (forbidden with `"best_only"`) |
| `saved_seed_runs` | Which runs are kept: `best_only`, `all_convergent`, or `all_tried` |
| `fail_on_non_convergence` | Raise if fewer than `min_convergent_seeds` converge |

> See `<experiment_dir>/seeds/seed_report.json` for per-seed training details.

Checkpoints live under `<experiment_dir>/seeds/seed_<N>/`. The best seed is still
copied to `<experiment_dir>/model/` for the default single-seed API.

---

## Single-seed vs multi-seed API

| Call | Uses | Returns |
|------|------|---------|
| [`trainer.evaluate_encoder(...)`][gradiend.trainer.trainer.Trainer.evaluate_encoder] | Best convergent checkpoint only | One correlation, one plot |
| [`view.evaluate_encoder(...)`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView.evaluate_encoder] | Every selected seed | Mean metric + `seeds.stats` dispersion |
| [`trainer.get_model()`][gradiend.trainer.trainer.Trainer.get_model] | Best checkpoint (full base + GRADIEND) | [`ModelWithGradiend`][gradiend.model.model_with_gradiend.ModelWithGradiend] |
| `view.get_model()` | Every selected seed | `SeedModelGroup` when N > 1 |

[`trainer.multi_seed()`][gradiend.trainer.trainer.Trainer.multi_seed] returns a [`MultiSeedTrainerView`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView]. It exposes the same
eval/plot method names as [`Trainer`][gradiend.trainer.trainer.Trainer], but runs them per seed and aggregates.

To reuse an existing analysis script, rebind after training:

```python
trainer.train()
view = trainer.multi_seed(dispersion="std")  # analysis mode: rebind for scripts

enc = view.evaluate_encoder(split="test", return_df=True)  # mean encoder_df + seeds.stats
```

---

## View options

```python
view = trainer.multi_seed(
    selection="all_convergent",  # best | all_convergent | all_tried
    aggregate="mean",            # mean | median | min | max
    dispersion="std",            # none | std | range | minmax
    return_per_seed=False,       # True -> full per-seed payloads under seeds.per_seed
)
```

| Option | When to use |
|--------|-------------|
| `selection="all_convergent"` | Stability analysis (default when `analyze_seed_stability=True`) |
| `selection="best"` | Same as single-seed, but through the view API |
| `aggregate="mean"` | Tables and summary plots |
| `dispersion="std"` | Report spread in `seeds.stats` — use for paper claims |
| `return_per_seed=True` | Debug or custom downstream aggregation |

When `dispersion` is omitted, it defaults to `"std"` if `analyze_seed_stability=True`,
else `"none"`.

---

## Evaluation results

```python
enc = view.evaluate_encoder(split="test", return_df=True)

enc["correlation"]                    # aggregated scalar (mean by default)
enc["seeds"]["n"]                     # how many seeds were aggregated
enc["seeds"]["values"]                # seed integers
enc["seeds"]["stats"]["correlation"]  # {"mean", "std", "min", "max", "n", ...}
enc["encoder_df"]                     # mean encoder_df across seeds (when return_df=True)
```

The same aggregation applies to `evaluate()`, `evaluate_decoder()`, and other
dict-returning eval methods on the view.

---

## Plots

All plots below are produced by
[`train_multi_seed_stability.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_multi_seed_stability.py).
Regenerate doc figures with `--write-docs-images`.

### Encoder by target (mean ± std)

Best plot for vocabulary-held-out stability. [`view.plot_encoder_by_target()`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView.plot_encoder_by_target] evaluates
each seed, then renders a multi-seed figure.

**Default — one row per seed** (compare seeds visually):

```python
view.plot_encoder_by_target(split="test", output="encoder_by_target_seeds.pdf")
```

![Encoder by target — one row per seed](../img/multi_seed_encoder_by_target_seeds.png)

<!-- DOC_PLOT: docs/img/multi_seed_encoder_by_target_seeds.png
Regenerate: python -m gradiend.examples.train_multi_seed_stability --write-docs-images
-->

**Combined strip — all seeds on one row** (overlay points, shared x-axis):

```python
view.plot_encoder_by_target(
    split="test",
    combine_seed_rows=True,
    output="encoder_by_target_combined.pdf",
)
```

![Encoder by target — combined strip](../img/multi_seed_encoder_by_target_combined.png)

<!-- DOC_PLOT: docs/img/multi_seed_encoder_by_target_combined.png
Regenerate: python -m gradiend.examples.train_multi_seed_stability --write-docs-images
-->

**Error bar summary — mean ± std per target** (publication-friendly):

```python
view.plot_encoder_by_target(
    split="test",
    plot_style="errorbar",
    error_stat="std",        # or "sem"
    show_seed_points=True,   # faint individual seed points behind bars
    output="encoder_by_target_errorbar.pdf",
)
```

![Encoder by target — mean ± std error bars](../img/multi_seed_encoder_by_target_errorbar.png)

<!-- DOC_PLOT: docs/img/multi_seed_encoder_by_target_errorbar.png
Regenerate: python -m gradiend.examples.train_multi_seed_stability --write-docs-images
-->

**Interactive** (Plotly HTML, requires `gradiend[plot]`):

```python
view.plot_encoder_by_target(split="test", interactive=True, output="encoder_by_target.html")
```

### Other encoder plots

These methods run **once per convergent seed** and return `result["paths"]` (one file
per seed) plus `result["path"]` (first file, convenience):

| Method | Figure (one seed shown) |
|--------|-------------------------|
| [`view.plot_encoder_distributions(...)`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView.plot_encoder_distributions] | ![Encoder distributions](../img/multi_seed_encoder_distributions.png) |
| [`view.plot_encoder_scatter(...)`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView.plot_encoder_scatter] | ![Encoder scatter](../img/multi_seed_encoder_scatter.png) |
| [`view.plot_training_convergence(...)`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView.plot_training_convergence] | ![Training convergence](../img/multi_seed_training_convergence.png) |
| [`view.plot_probability_shifts(...)`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView.plot_probability_shifts] | ![Decoder probability shifts](../img/multi_seed_probability_shifts.png) |

<!-- DOC_PLOT: docs/img/multi_seed_encoder_distributions.png
Regenerate: python -m gradiend.examples.train_multi_seed_stability --write-docs-images
-->
<!-- DOC_PLOT: docs/img/multi_seed_encoder_scatter.png
Regenerate: python -m gradiend.examples.train_multi_seed_stability --write-docs-images
-->
<!-- DOC_PLOT: docs/img/multi_seed_training_convergence.png
Regenerate: python -m gradiend.examples.train_multi_seed_stability --write-docs-images
-->
<!-- DOC_PLOT: docs/img/multi_seed_probability_shifts.png
Regenerate: python -m gradiend.examples.train_multi_seed_stability --write-docs-images
-->

Implementation: [`run_other_encoder_plots()`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_multi_seed_stability.py) in `train_multi_seed_stability.py`.

There is no built-in mean±std violin overlay for distributions — use
[`plot_encoder_by_target(..., plot_style="errorbar")`][gradiend.visualizer.encoder_by_target.plot_encoder_by_target] for aggregated target-level
summaries, or compare the per-seed distribution files side by side.

---

## Trainer suites

If a child trainer uses `analyze_seed_stability=True`, [`TrainerSuite`][gradiend.trainer.suite.base.TrainerSuite].train() replaces
that child with a multi-seed view automatically. Suite heatmaps then include all
convergent seeds for that child without extra wiring.

Only enable multi-seed on suites after each child converges reliably on its own —
suites multiply failure modes.

See [Trainer suites](trainer-suites.md).

---

## See also

| Topic | Guide |
|-------|--------|
| Aggregated metrics + dispersion | This page ([`trainer.multi_seed()`][gradiend.trainer.trainer.Trainer.multi_seed], `seeds.stats`) |
| Pairwise seed **checkpoint** similarity (top-k, cosine heatmaps) | [Cross-model comparison — Comparing seeds](cross-model-comparison.md#comparing-seeds) |
| `gradiend_only=True`, checkpoint paths | [Saving and loading](saving-loading.md) |

---

## Example script

[`train_multi_seed_stability.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_multi_seed_stability.py)
covers multi-seed training, aggregated eval, and encoder plots from this guide.
Seed checkpoint similarity heatmaps are documented under
[Comparing seeds](cross-model-comparison.md#comparing-seeds) (same script, `run_seed_comparison_heatmaps()`).

| Script | Extra focus |
|--------|-------------|
| [`train_sentiment.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_sentiment.py) (`RUN_MODE="multi_seed_heldout"`) | Vocabulary-held-out target rotation across seeds |
| [`train_gender_de_detailed.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de_detailed.py) | Minimal multi-seed eval on gender DE feature |
