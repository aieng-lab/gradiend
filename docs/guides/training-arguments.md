# Training Arguments

This guide explains `TrainingArguments` in detail. For a conceptual overview and minimal usage, see [Tutorial: Training](../tutorials/training.md). For the full API (including defaults), see [API reference](../api/index.md).

`TrainingArguments` follows Hugging Face Trainer conventions where applicable (e.g. `num_train_epochs`, `learning_rate`, `eval_steps`).

---

## Output and paths

| Argument | Default | Description |
|----------|---------|-------------|
| **experiment_dir** | `None` | Root directory for this experiment. When set, checkpoints, caches, and plots use subpaths under it. Required for saving models and using encoder/decoder cache. With `run_id`, output goes under `experiment_dir/run_id/`. |
| **output_dir** | `None` | Directory to save the trained model. If `None` and `experiment_dir` is set, uses `experiment_dir/model` (or `experiment_dir/run_id/model`). |
| **use_cache** | `False` | When `True`, skip training if a saved model exists at the output path; skip encoder/decoder recomputation when cache exists. Use `False` to force retrain or recompute. |
| **add_identity_for_other_classes** | `False` | When `True`, add identity examples (factual == alternative) for classes not in the target pair. Important for multi-class data so extra classes do not push the model arbitrarily. |

---

## GRADIEND interpretation (source / target)

| Argument | Default | Description |
|----------|---------|-------------|
| **source** | `"alternative"` | Which gradient feeds the encoder: `"factual"`, `"alternative"`, or `"diff"`. Common: `"alternative"` so the encoder sees the alternative gradient. |
| **target** | `"diff"` | Which quantity the decoder predicts: `"factual"`, `"alternative"`, or `"diff"`. Common: `"diff"` so the decoder predicts the difference; at inference you combine base model + decoder(diff). |

---

## Training loop (HF-style)

| Argument | Default | Description |
|----------|---------|-------------|
| **train_batch_size** | `32` | Batch size for training. |
| **train_max_size** | `None` | If set, cap training samples per feature class (for speed/memory). `None` = use all data. |
| **learning_rate** | `1e-5` | Peak learning rate. |
| **num_train_epochs** | `3` | Number of training epochs. Ignored when `max_steps > 0`. |
| **max_steps** | `-1` | If `> 0`, total training steps; overrides `num_train_epochs`. `-1` = use epochs. |
| **weight_decay** | `1e-2` | Weight decay for the optimizer. |
| **adam_epsilon** | `1e-8` | Epsilon for Adam/AdamW. |
| **optim** | `"adamw"` | Optimizer: `"adamw"` or `"adam"`. |

---

## Evaluation during training

| Argument | Default | Description |
|----------|---------|-------------|
| **eval_strategy** | `"steps"` | When to run evaluation: `"steps"` (every `eval_steps`) or `"no"`. |
| **eval_steps** | `250` | Run evaluation every `eval_steps` (when `eval_strategy == "steps"`). |
| **do_eval** | `True` | Whether to run evaluation during training. |
| **encoder_eval_train_max_size** | `None` | Max samples for in-training encoder evaluation. `None` = use `encoder_eval_max_size`. |
| **encoder_eval_max_size** | `None` | Max samples for encoder evaluation outside training (e.g. `evaluate_encoder`, analysis). `None` = use all. |
| **encoder_eval_balance** | `True` | If `True`, balance encoder eval data per feature class. |
| **seed_selection_eval_max_size** | `None` | Max samples for encoder evaluation when selecting the best seed. `None` = use `encoder_eval_max_size`. |
| **decoder_eval_max_size_training_like** | `None` | Max samples for decoder training-like evaluation. |
| **decoder_eval_max_size_neutral** | `None` | Max samples for decoder neutral evaluation (LMS). |
| **eval_batch_size** | `32` | Batch size for evaluation. |

---

## Checkpointing and saving

| Argument | Default | Description |
|----------|---------|-------------|
| **save_strategy** | `"best"` | `"best"` = keep only best checkpoint; `"steps"` = also save every `save_steps`; `"no"` = no checkpointing. |
| **save_steps** | `5000` | Save every `save_steps` when `save_strategy == "steps"`. |
| **save_only_best** | `True` | If `True`, keep only the best checkpoint (by evaluation metric). |
| **delete_models** | `False` | If `True`, delete intermediate model files at end (to save disk). |

---

## Multi-seed training

| Argument | Default | Description |
|----------|---------|-------------|
| **max_seeds** | `3` | Maximum number of seeds to try. |
| **min_convergent_seeds** | `1` | Stop once this many seeds have converged. `None` = run `max_seeds`. |
| **convergent_metric** | `"correlation"` | Metric for convergence: `"correlation"` (encoder) or `"loss"`. |
| **convergent_score_threshold** | `0.5` | Threshold; a seed is “convergent” if its metric passes this. For `correlation`, typical 0.5–0.6. |
| **seed** | `None` | Base seed; seeds used are `seed+i` for `i = 0..max_seeds-1`. `None` = 0..max_seeds-1. |
| **seed_runs_dir** | `None` | Directory for per-seed runs. Default: `experiment_dir/seeds`. |
| **keep_seed_runs** | `False` | If `True`, keep all per-seed model directories; else delete model files, keep metrics only. |

### Seed report format

When `max_seeds > 1`, `Trainer.train()` writes a JSON report to `<experiment_dir>/seeds/seed_report.json` summarizing how each seed performed and which seed was selected.

**Top-level keys:** `convergence_metric`, `threshold`, `min_convergent_seeds`, `max_seeds`, `seeds_tried`, `convergent_count`, `best_seed`, `best_selection_score`, `early_stop_reason`, `runs`.

**Per-seed entries in `runs`:** `seed`, `output_dir`, `trained` / `used_cache`, `training_score`, `eval_correlation`, `selection_score`, `convergence_metric`, `convergence_metric_value`, `threshold`, `converged`.

- `training_score`: Best training-time score (for `correlation` = best encoder correlation; for `loss` = negative loss, higher is better).
- `eval_correlation`: Post-hoc `evaluate_encoder(split="val")` correlation used for seed selection; may be `null` when skipped.
- `selection_score`: Score used to pick best seed (`eval_correlation` when available, else `training_score`).
- `convergence_metric_value`: Raw metric (for `loss` = un-negated loss, lower is better for threshold checks).

This report helps debug why a particular seed was chosen and how training-time metrics compare to validation metrics.

---

## Pre- and post-pruning

| Argument | Default | Description |
|----------|---------|-------------|
| **pre_prune_config** | `None` | If set, pre-pruning runs before training (gradient-based importance, reduces input dimension). See [Pruning](pruning-guide.md). |
| **post_prune_config** | `None` | If set, post-pruning runs after training (weight-based, keeps top-k dimensions). See [Pruning](pruning-guide.md). |

---

## Advanced

| Argument | Default | Description |
|----------|---------|-------------|
| **trust_remote_code** | `False` | Pass to Hugging Face when loading models. |
| **params** | `None` | If set, only these parameter names/wildcards in the GRADIEND param map. |
| **normalize_gradiend** | `True` | Normalize encodings (first target class → +1, second → -1). |
| **torch_dtype** | `torch.float32` | Model dtype. |
| **supervised_encoder** | `False` | If `True`, train only the encoder (baseline mode). |
| **supervised_decoder** | `False` | If `True`, train only the decoder (baseline mode). |
| **use_cached_gradients** | `False` | Use cached gradients if available (faster, higher memory/disk). |
