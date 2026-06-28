# Saving & loading (modality-agnostic)

This guide explains where GRADIEND writes artifacts, what is inside a checkpoint, and how to reload models and training metadata. It applies across modalities (text today; same layout for future trainers). For the training workflow that produces these files, see [Tutorial: Training](../tutorials/training.md).

---

## Where artifacts live

[`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments].experiment_dir is the root output directory. When the trainer has a `run_id`, paths are resolved under `experiment_dir/run_id/` (unless `run_id` is already the leaf directory name).

The trained GRADIEND model is written to **`output_dir`**, which defaults to `<experiment_dir>/model` when `experiment_dir` is set and `output_dir` is omitted. You can override with [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments].output_dir.

```text
experiment_dir/                  # or experiment_dir/run_id/
├── model/                       # final promoted checkpoint (default output_dir)
├── model_best/                  # transient during training when save_only_best=True
├── model_step_500/              # optional periodic checkpoints (save_strategy="steps")
├── seeds/
│   ├── seed_report.json         # multi-seed summary (best seed, convergent seeds, …)
│   ├── seed_42/                 # per-seed checkpoint directory
│   └── seed_123/
├── decoder_mlm_head/            # optional auxiliary head (decoder-only MLM mode)
├── custom_prediction_head/      # optional custom prediction head
├── cache/                       # pre-prune cache, gradient cache, …
├── encoded_values*.csv          # encoder evaluation cache
├── decoder_analysis/            # decoder evaluation artifacts
├── training_convergence.pdf     # convergence plot
└── <target_class>/              # rewritten base model(s) from rewrite_base_model
```

**Trainer suites** nest one trainer run per child under the suite’s experiment directory, e.g. `experiment_dir/<run_id>/<child_id>/model`. See [Trainer suites](trainer-suites.md).

If neither `experiment_dir` nor `output_dir` is set, `train()` uses a temporary directory and logs a warning — copy or call `save_pretrained` yourself if you need the checkpoint after the process exits.

---

## GRADIEND checkpoint contents

A loadable GRADIEND directory contains at least weights, `config.json` (with an `architecture` block), and `gradiend_context.json`.

| File | Purpose |
|------|---------|
| `model.safetensors` or `pytorch_model.bin` | GRADIEND encoder/decoder weights (safetensors preferred when available) |
| `config.json` | Architecture (`input_dim`, `latent_dim`, activations), parameter **mapping** (required for pruned models), and metadata (e.g. `base_model`, `tokenizer` ids) |
| `gradiend_context.json` | Gradient **source** / **target** and optional `feature_class_encoding_direction` |
| `training.json` | Step-wise training stats, best checkpoint step, `training_args`, `cache_fingerprint`, convergence info |

**Important:** a standard GRADIEND checkpoint stores the **GRADIEND weights and mapping**, not a full copy of the base model. On load, the base model is fetched again from the Hugging Face id recorded in checkpoint metadata (e.g. `bert-base-uncased`). Ensure that id is reachable (network or local cache) when loading on a new machine.

Evaluation caches (encoder CSV/JSON, decoder stats, plots) live alongside the model under `experiment_dir` but are separate from the checkpoint itself. Evaluator `use_cache` is independent of training `use_cache`; see [Tutorial: Training](../tutorials/training.md#experiment-directory-and-caching-use_cache).

---

## What `train()` writes

When `experiment_dir` (or an explicit `output_dir`) is set, `train()` persists:

| Artifact | Location | Notes |
|----------|----------|-------|
| **Final model** | `output_dir` (default `…/model`) | Best checkpoint by convergence metric; with multi-seed, the selected best seed is copied here |
| **Seed report** | `seeds/seed_report.json` | When `max_seeds` > 1 |
| **Per-seed runs** | `seeds/seed_<N>/` | Full checkpoint per seed tried |
| **Periodic checkpoints** | `model_step_<N>/` | When `save_strategy="steps"` and `save_steps=N` |
| **Training metadata** | `model/training.json` | Written/overwritten at end of training |

Checkpointing is controlled by [TrainingArguments](../guides/training-arguments.md#checkpointing-and-saving):

| Argument | Default | Effect |
|----------|---------|--------|
| `save_strategy` | `"best"` | `"best"` tracks best-by-metric checkpoints; `"steps"` also saves every `save_steps`; `"no"` disables saving |
| `save_steps` | `5000` | Step interval when `save_strategy="steps"` |
| `save_only_best` | `True` | Promote `model_best/` to `model/` at end; drop non-best weight snapshots |
| `delete_models` | `False` | After training, delete `.bin`/`.safetensors` in `output_dir` (metadata kept; rarely needed) |

During training with `save_only_best=True`, intermediate best snapshots are written to `<output_dir>_best` and promoted to `<output_dir>` when training finishes. Step-0 evaluation is tracked but **not** saved as a selectable best checkpoint.

---

## Reusing checkpoints (`use_cache`)

When a complete checkpoint already exists at `output_dir`, `train()` can skip training depending on [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments].use_cache:

| Value | Behavior | When to use |
|-------|----------|-------------|
| `False` | Always retrain (default) | You want a guaranteed fresh train — e.g. after changing data, `learning_rate`, `max_steps`, or other settings not covered by the fingerprint. |
| `True` | Reuse when a checkpoint exists and `cache_fingerprint` in `training.json` matches current pruning/source settings | Reasonable default for iterative work: skips retraining when pruning and source/target are unchanged, but retrains when those key settings change. |
| `"always"` | Reuse any saved checkpoint (no fingerprint check) | Exploratory or large-scale analysis where you want to keep progress and re-run plots/eval quickly; reproducibility across setting changes is not the priority yet. |
| `"only_convergent"` | Same fingerprint check as `True`, plus convergence metadata must satisfy `min_convergent_seeds` | Multi-seed runs where some seeds failed: retry without retraining seeds that already converged (e.g. after relaxing `learning_rate` or convergence thresholds). |

Fingerprinting is **partial** — it does not compare most hyperparameters or data. See [Tutorial: Training](../tutorials/training.md#experiment-directory-and-caching-use_cache) and [Pruning guide](pruning-guide.md#training-cache-fingerprint-prepost-prune). When in doubt, set `use_cache=False` or use a new `experiment_dir` / `run_id`.

To check whether a directory is a complete GRADIEND checkpoint before training:

```python
from gradiend.util.paths import has_saved_model

has_saved_model("experiment_dir/run_id/model")  # True when weights + config.json architecture exist
```

---

## Loading a trained model

### From the trainer

After `train()`, the trainer knows the output path as [`trainer.model_path`][gradiend.trainer.trainer.Trainer.model_path]. Use `get_model()` for the trainer’s current (best) checkpoint:

```python
model = trainer.get_model()
```

[:material-file-code-outline: `start_workflow.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/start_workflow.py)

| Method | Use when |
|--------|----------|
| [`trainer.get_model()`][gradiend.trainer.trainer.Trainer.get_model] | Load (or return cached) model from `model_path`; default after training |
| [`trainer.get_model(load_directory="path/to/checkpoint")`][gradiend.trainer.trainer.Trainer.get_model] | Load a specific checkpoint into the trainer cache |
| [`trainer.load_model("path/to/checkpoint")`][gradiend.trainer.trainer.Trainer.load_model] | Load a **different** checkpoint without replacing `model_path` (comparisons, ablations, other seeds) |
| [`trainer.unload_model()`][gradiend.trainer.trainer.Trainer.unload_model] | Free GPU memory; evaluation will reload from disk with a warning |

`get_model()` always caches the loaded instance in memory. Disk reuse is controlled by `use_cache` on `train()` / evaluators, not by `get_model()`.

For text models, pass the trainer (or `feature_definition=self`) implicitly so `feature_class_encoding_direction` is restored from the trainer’s pair/classes when not stored in the checkpoint:

```python
from gradiend import TextPredictionTrainer

trainer = TextPredictionTrainer(...)  # same data/classes as training
model = trainer.get_model(load_directory="experiment_dir/run_id/model")
```

### From disk (standalone)

Load without a trainer via the modality’s [`ModelWithGradiend`][gradiend.model.model_with_gradiend.ModelWithGradiend] subclass (for text: [`TextPredictionTrainer`][gradiend.trainer.text.prediction.trainer.TextPredictionTrainer].default_model_with_gradiend_cls, typically `TextPredictionModelWithGradiend`):

```python
from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend

model = TextPredictionModelWithGradiend.from_pretrained(
    "experiment_dir/run_id/model",
    require_gradiend_model=True,
    feature_definition=trainer,  # optional: restore class encoding directions
)
```

**Base model path vs GRADIEND checkpoint:** if `load_directory` is a Hugging Face model id (or a path **without** `gradiend_context.json`), `from_pretrained` loads the base model and **creates a fresh, untrained** GRADIEND on top. Pass `require_gradiend_model=True` to raise instead when the path is not a GRADIEND checkpoint.

**Device placement** (optional kwargs): `device_encoder`, `device_decoder`, `device_base_model`, `base_model_device_map`, `base_model_max_memory`, `trust_remote_code`. [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments] passed as `training_args=` are merged into load kwargs (`params`, `torch_dtype`, pruning configs, etc.).

### GRADIEND-only loading (`gradiend_only=True`)

For weight-space comparisons (heatmaps, top-k overlap) that do not run forward passes through the base model, load only GRADIEND weights:

```python
model = trainer.get_model(gradiend_only=True)
```

This returns a lightweight wrapper around [`ParamMappedGradiendModel`][gradiend.model.param_mapped.ParamMappedGradiendModel] (or a `SeedModelGroup` in multi-seed analysis). Suite comparison uses the same flag: `suite.get_models(gradiend_only=True)`. See [Multi-seed analysis](multi-seed.md) and [Cross-model comparison](cross-model-comparison.md).

---

## Training metadata without loading weights

Inspect correlation, best step, and config without loading tensors:

```python
# From trainer (uses model_path when path omitted)
stats = trainer.get_training_stats()

# Standalone
from gradiend import load_training_stats

stats = load_training_stats("experiment_dir/run_id/model")
print(stats["best_score_checkpoint"]["global_step"])
print(stats["training_stats"]["correlation"])
print(stats.get("cache_fingerprint"))
```

Returns `None` if `training.json` is missing. Keys include `training_stats`, `best_score_checkpoint`, `training_args`, `losses`, `time`, `convergence_info`, `cache_fingerprint`, and (multi-seed) seed stability summaries.

For multi-seed runs, read `seeds/seed_report.json` (or [`trainer.get_seed_report()`][gradiend.trainer.trainer.Trainer.get_seed_report]) for convergent seeds, selected best seed, and per-run metrics before loading individual `seeds/seed_<N>/` directories.

---

## Saving manually

### Full ModelWithGradiend checkpoint

```python
model = trainer.get_model()
model.save_pretrained("path/to/output")
```

Writes `gradiend_context.json`, modality hooks (tokenizer ids for text), and GRADIEND weights/config. Optional `training=` kwargs are persisted to `training.json`.

### GRADIEND weights only

Access the inner module via `model.gradiend` ([`ParamMappedGradiendModel`][gradiend.model.param_mapped.ParamMappedGradiendModel] or [`GradiendModel`][gradiend.model.model.GradiendModel]):

```python
model.gradiend.save_pretrained("path/to/gradiend_only", use_safetensors=True)
loaded = ParamMappedGradiendModel.from_pretrained("path/to/gradiend_only")
```

Use `use_safetensors=False` to force PyTorch `.bin` format. Pruned models require the `mapping` block in `config.json` — always save and load through [`ParamMappedGradiendModel`][gradiend.model.param_mapped.ParamMappedGradiendModel] for pruned checkpoints.

After manual pruning, save the full wrapper so `gradiend_context.json` and base-model references stay consistent:

```python
model.prune_gradiend(topk=0.05, inplace=True)
model.save_pretrained("path/to/pruned")
```

See [Pruning guide](pruning-guide.md).

---

## Rewritten base models (different format)

`rewrite_base_model()` applies decoder-selected updates to a **copy** of the base model and returns a plain Hugging Face model (not a [`ModelWithGradiend`][gradiend.model.model_with_gradiend.ModelWithGradiend]). Saved rewrites are standard HF checkpoints (full base weights + config), not GRADIEND directories:

```python
trainer.rewrite_base_model(
    decoder_results=dec_results,
    target_class="masc_nom",
    output_dir="./output/masc_nom_rewrite",  # single class: explicit path
)
```

With multiple targets and `experiment_dir` set, paths default to `<experiment_dir>/<target_class>/`. Details: [Tutorial: Model Rewrite](../tutorials/model-rewrite.md).

---

## Auxiliary checkpoints

Some pipelines save **helper** models under the experiment directory. These are not GRADIEND checkpoints but are loaded automatically when present:

| Directory | Role |
|-----------|------|
| `decoder_mlm_head/` | Lightweight MLM head for decoder-only models (`clm_mlm_head`); trained via `train_decoder_only_mlm_head()` |
| `custom_prediction_head/` | Generic custom prediction head |

`trainer.resolve_model_path(base_model_id)` returns the auxiliary head path when training **before** a GRADIEND checkpoint exists, but leaves GRADIEND checkpoint paths unchanged (the encoder needs the full GRADIEND model). See [Decoder-only models](decoder-only.md).

Check for a saved decoder MLM head with `has_saved_decoder_mlm_head(dir)` from `gradiend.util.paths`.

---

## Multi-seed and suites (pointers)

| Scenario | Checkpoint layout | Loading |
|----------|-------------------|---------|
| Multi-seed training | `seeds/seed_<N>/`, best copied to `model/` | `get_model()` uses best; `load_model("…/seeds/seed_42")` for a specific seed; [`trainer.multi_seed()`][gradiend.trainer.trainer.Trainer.multi_seed] for aggregated analysis |
| Seed stability analysis | Same, plus `seed_report.json` | `get_model(gradiend_only=True)` on a multi-seed view |
| Trainer suite | `<experiment_dir>/<run_id>/<child_id>/model` | `suite.get_trainer(child_id).get_model()` or `suite.get_models()` |

See [Multi-seed analysis](multi-seed.md) and [Trainer suites](trainer-suites.md).

---

## Related docs

- [Tutorial: Training](../tutorials/training.md) — experiment directory, caching, multi-seed training
- [Training arguments](training-arguments.md) — `save_strategy`, `use_cache`, `output_dir`
- [Tutorial: Model Rewrite](../tutorials/model-rewrite.md) — saving modified base models
- [Pruning guide](pruning-guide.md) — pruned checkpoints and cache fingerprints
- [Core classes](core-classes.md) — [`ModelWithGradiend`][gradiend.model.model_with_gradiend.ModelWithGradiend], [`ParamMappedGradiendModel`][gradiend.model.param_mapped.ParamMappedGradiendModel], [`GradiendModel`][gradiend.model.model.GradiendModel]
- [`load_training_stats`][gradiend.trainer.core.stats.load_training_stats]
