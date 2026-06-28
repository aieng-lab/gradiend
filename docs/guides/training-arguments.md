# Training Arguments

This guide documents every field of [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments]. For the conceptual
training workflow, see [Tutorial: Training](../tutorials/training.md). For the
generated API page, see [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments].

[`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments] follows Hugging Face Trainer conventions where they fit
(`learning_rate`, `num_train_epochs`, `max_steps`, `eval_steps`), but it also
contains GRADIEND-specific options for gradient batching, convergence, pruning,
prediction objectives, and multi-seed analysis.

---

## Output and caching

| Argument | Default | Description |
|----------|---------|-------------|
| **experiment_dir** | `None` | Root directory for this experiment. With `run_id`, trainer artifacts go under `experiment_dir/run_id/`. |
| **output_dir** | `None` | Directory for the trained model. If omitted and `experiment_dir` is set, GRADIEND derives a model path under the experiment directory. |
| **use_cache** | `False` | Training checkpoint reuse policy: `False`, `True`, `"always"`, or `"only_convergent"`. `True` and `"only_convergent"` require a matching `cache_fingerprint` in `training.json` (pruning config, `source`/`target`, `reuse_pre_prune`, `gradiend_input_dim`). `"always"` reuses any saved checkpoint without fingerprint checks. `"only_convergent"` additionally requires convergence metadata. Fingerprinting is **partial** — it does not compare most hyperparameters, data, or model-selection settings; see [Tutorial: Training](../tutorials/training.md#experiment-directory-and-caching-use_cache). Evaluator/visualizer cache arguments are separate. |
| **add_identity_for_other_classes** | `False` | Add identity examples (`factual == alternative`) for non-target classes so they are not pushed arbitrarily. |
| **metadata** | `{}` | Free-form metadata serialized with the training arguments. |

---

## Source and target

| Argument | Default | Description |
|----------|---------|-------------|
| **source** | `"alternative"` | Which gradient feeds the encoder: `"factual"`, `"alternative"`, or `"diff"`. |
| **target** | `"diff"` | Which gradient quantity the decoder predicts: `"factual"`, `"alternative"`, or `"diff"`. |

The common setting is `source="alternative", target="diff"`: the encoder sees
the alternative gradient and the decoder learns the difference to apply.

---

## Training loop

| Argument | Default | Description |
|----------|---------|-------------|
| **train_batch_size** | `32` | Alias/default for `base_gradient_batch_size`. |
| **base_gradient_batch_size** | `None` | Number of raw training examples merged into one base-model loss/backward call. `None` uses `train_batch_size`. |
| **gradiend_batch_size** | `None` | Number of base-gradient vectors stacked into one GRADIEND optimizer step. `None` becomes `1`. |
| **precompute_gradient_batches** | `None` | Whether to precompute the next gradient row asynchronously. `None` auto-enables only for suitable sharded/multi-CUDA setups. |
| **precompute_gradient_buffer_size** | `1` | Number of already-computed gradient rows kept in the async queue. |
| **gradient_timing_steps** | `0` | If `> 0`, log gradient-row timing every N rows. |
| **train_max_size** | `None` | Cap training samples per feature class. `None` uses all data. |
| **learning_rate** | `1e-5` | Peak learning rate. |
| **num_train_epochs** | `3` | Number of epochs. Ignored when `max_steps > 0`. |
| **max_steps** | `-1` | Total training steps when `> 0`; overrides `num_train_epochs`. |
| **weight_decay** | `1e-2` | Weight decay for the optimizer. |
| **adam_epsilon** | `1e-8` | Epsilon for Adam/AdamW. |
| **optim** | `"adamw"` | Optimizer: `"adamw"` or `"adam"`. |
| **criterion** | `None` | Loss function. `None` becomes `torch.nn.MSELoss()`. Advanced use only. |

---

## Evaluation during training

| Argument | Default | Description |
|----------|---------|-------------|
| **eval_strategy** | `"steps"` | Evaluation schedule: `"steps"` or `"no"`. |
| **eval_steps** | `250` | Run evaluation every N steps when `eval_strategy="steps"`. |
| **encoder_eval_train_max_size** | `None` | Max samples for in-training encoder evaluation. `None` uses `encoder_eval_max_size`. |
| **encoder_eval_max_size** | `None` | Max samples for encoder evaluation outside training. |
| **encoder_eval_balance** | `True` | Balance encoder evaluation data per feature class. |
| **include_other_classes** | `False` | Include all class transitions in the evaluation split (not only the trained pair) when `len(all_classes) > 2`. Affects encoder evaluation, encoder plots, and cross-encoding with `full_eval=True`. See also [`TransitionSpec`][gradiend.trainer.core.transition_selection.TransitionSpec] for explicit transition lists. |
| **seed_selection_eval_max_size** | `None` | Max samples for post-hoc seed selection evaluation. `None` uses `encoder_eval_max_size`. |
| **decoder_eval_max_size_training_like** | `None` | Max samples for decoder training-like evaluation data. |
| **decoder_eval_max_size_neutral** | `None` | Max samples for decoder neutral evaluation and LMS text. |
| **decoder_eval_lrs** | `None` | Learning-rate grid for decoder evaluation. `None` uses evaluator defaults (1/2/5 grid from 100 to 1e-3). |
| **decoder_eval_feature_factors** | `None` | Feature-factor grid for decoder evaluation. `None` derives factors from trainer target classes. |
| **eval_batch_size** | `32` | Batch size for evaluation. |
| **do_eval** | `True` | Whether to evaluate during training. |
| **evaluate_fn** | `None` | Custom in-training evaluation callable. `None` uses the default encoder-correlation evaluation. |

---

## Checkpointing and saving

| Argument | Default | Description |
|----------|---------|-------------|
| **save_strategy** | `"best"` | `"best"` keeps only the best checkpoint; `"steps"` also saves periodic checkpoints; `"no"` disables checkpointing. |
| **save_steps** | `5000` | Save every N steps when `save_strategy="steps"`. |
| **save_only_best** | `True` | Keep only the best checkpoint by evaluation metric. |
| **delete_models** | `False` | Delete intermediate model files at the end to save disk. Does not delete the whole model directory. |

---

## Model loading and device placement

| Argument | Default | Description |
|----------|---------|-------------|
| **trust_remote_code** | `False` | Pass `trust_remote_code=True` when loading Hugging Face models/tokenizers. |
| **dataset_trust_remote_code** | `None` | Optional `trust_remote_code` value for `datasets.load_dataset`. `None` means the keyword is not passed. |
| **model_use_cache** | `False` | Whether decoder models use transformer KV cache during training. Keep `False` for training. |
| **torch_dtype** | `None` | Model dtype. `None` becomes `torch.float32`. |
| **base_model_device_map** | `None` | Hugging Face `device_map` for the base model. `False` disables device mapping; strings/dicts are passed through. |
| **base_model_max_memory** | `None` | Optional Hugging Face `max_memory` map for base-model placement. |
| **encoder_decoder_same_device** | `False` | Place GRADIEND encoder and decoder on the same GPU where possible, leaving more devices for the base model. |

### Device Placement {#device-placement}

Device placement is automatic when these fields are omitted:

| GPUs | Placement |
|------|-----------|
| 1 | encoder, decoder, base model all on `cuda:0` |
| 2 | encoder + base model on `cuda:0`, decoder on `cuda:1` |
| >=3 | encoder on `cuda:0`, decoder on `cuda:1`, base model on `cuda:2` |
| 0 | all on CPU |

---

## Prediction objective

| Argument | Default | Description |
|----------|---------|-------------|
| **prediction_objective** | `"auto"` | Token prediction objective for text-gradient training and decoder probability scoring. See [Token prediction methods](token-prediction-methods.md). |
| **decoder_mlm_head_epochs** | `5` | Epochs for auxiliary decoder MLM-head training when `prediction_objective="clm_mlm_head"`. |
| **decoder_mlm_head_batch_size** | `4` | Batch size for auxiliary decoder MLM-head training. |
| **decoder_mlm_head_lr** | `1e-4` | Learning rate for auxiliary decoder MLM-head training. |
| **decoder_mlm_head_max_size** | `None` | Optional per-label cap for auxiliary decoder MLM-head training data. |
| **decoder_sequence_cloze_rhs_window** | `-1` | Right-context token window for `clm_sequence_cloze` and `seq2seq_decoder_sequence_cloze`. `-1` uses the full RHS. |

Supported objectives are `auto`, `mlm_mask_token`, `clm_next_token`,
`clm_mlm_head`, `clm_sequence_cloze`, `seq2seq_decoder`,
`seq2seq_decoder_sequence_cloze`, and `seq2seq_encoder_mlm`.

---

## GRADIEND model shape

| Argument | Default | Description |
|----------|---------|-------------|
| **params** | `None` | Optional list of parameter names or wildcard patterns included in the GRADIEND parameter map. `None` includes all backbone parameters. |
| **activation_encoder** | `None` | Encoder activation, e.g. `"tanh"`, `"gelu"`, or `"relu"`. `None` uses the model default. |
| **activation_decoder** | `None` | Decoder activation, e.g. `"id"` or `"tanh"`. `None` uses the model default. |
| **bias_decoder** | `None` | Whether the decoder linear layer has a bias. `None` uses the model default. |
| **latent_dim** | `None` | GRADIEND latent dimension. `None` uses the model default, normally one feature dimension. |
| **normalize_gradiend** | `True` | Normalize encodings so the first target class maps toward `+1` and the second toward `-1`. |
| **positive_class** | `None` | Optional canonical positive class for binary cross-encoding comparisons. Normal training usually leaves this unset. |

### Model Parameters (Which Layers To Use) {#model-parameters-which-layers-to-use}

`params` accepts exact parameter names or simple wildcard patterns:

```python
args = TrainingArguments(
    params=["bert.encoder.layer.0.*", "bert.encoder.layer.1.*"],
)
```

Only backbone parameters are included. Prediction heads such as MLM heads are
excluded by the model-loading logic.

---

## Multi-seed training

| Argument | Default | Description |
|----------|---------|-------------|
| **max_seeds** | `3` | Maximum number of seeds to try. |
| **min_convergent_seeds** | `1` | Stop once this many seeds have converged. `None` runs all `max_seeds`. |
| **convergent_metric** | `None` | `"correlation"` or `"loss"`. `None` defaults to `"correlation"` unless `supervised_decoder=True`. |
| **convergent_score_threshold** | `None` | Score threshold for convergence. `None` becomes `0.5` for correlation; required for loss. |
| **convergent_mean_by_class_threshold** | `None` | Additional convergence threshold: every non-zero target class must have \|mean encoded\| ≥ this value at the best step. For correlation mode, `None` becomes `0.5`. |
| **split_resplit_per_seed** | `False` | When `split_col=None`, redraw vocabulary-held-out splits per training seed. |
| **split_resplit_strategy** | `"random"` | Strategy for per-seed resplitting: `"random"` or `"balanced_cycle"`. |
| **seed** | `0` | Base seed. Multi-seed runs use `seed+i`; `None` requests non-deterministic runs. |
| **seed_runs_dir** | `None` | Directory for per-seed runs. Defaults to `experiment_dir/seeds`. |
| **saved_seed_runs** | `"all_convergent"` | Retention policy: `"best_only"`, `"all_convergent"`, or `"all_tried"`. |
| **seed_stability_topk** | `1000` | Top-k selection used for stability summaries. `None` disables the report. |
| **seed_stability_part** | `"decoder-weight"` | Importance part used for seed-stability top-k summaries. |
| **analyze_seed_stability** | `False` | Require saved convergent seeds for later [`trainer.multi_seed()`][gradiend.trainer.trainer.Trainer.multi_seed] analysis. Forbids `saved_seed_runs="best_only"`. |
| **fail_on_non_convergence** | `False` | Raise if training finishes with fewer than `min_convergent_seeds` convergent seeds. |
| **highlight_non_convergence** | `True` | Append a non-convergence marker to plot/tick labels for non-converged runs. |

See [Multi-seed analysis](multi-seed.md) for [`trainer.multi_seed()`][gradiend.trainer.trainer.Trainer.multi_seed] usage,
seed selection, aggregation, and dispersion.

### Seed report format

When `max_seeds > 1`, [`Trainer`][gradiend.trainer.trainer.Trainer].train() writes:

```text
<experiment_dir>/seeds/seed_report.json
```

Top-level keys include `convergence_metric`, `threshold`,
`min_convergent_seeds`, `max_seeds`, `seeds_tried`, `convergent_count`,
`best_seed`, `best_selection_score`, `early_stop_reason`, and `runs`.

---

## Runtime monitor

| Argument | Default | Description |
|----------|---------|-------------|
| **runtime_monitor** | `False` | Write persistent JSONL runtime diagnostics under the training output directory. |
| **runtime_monitor_interval** | `5.0` | Seconds between runtime-monitor heartbeat samples. |
| **runtime_monitor_system_stats** | `True` | Include CPU/GPU memory stats in heartbeat samples when available. |

---

## Pruning

| Argument | Default | Description |
|----------|---------|-------------|
| **pre_prune_config** | `None` | Run pre-pruning before training. See [Pruning](pruning-guide.md). |
| **reuse_pre_prune** | `False` | Cache pre-prune `keep_idx` under `experiment_dir/cache/pre_prune/` for reuse across seeds in one `train()` call. |
| **post_prune_config** | `None` | Run post-pruning after training. See [Pruning](pruning-guide.md). |

---

## Baseline and gradient-cache modes

| Argument | Default | Description |
|----------|---------|-------------|
| **supervised_encoder** | `False` | Train only the encoder against labels as a baseline. Cannot be combined with `supervised_decoder`. |
| **supervised_decoder** | `False` | Train only the decoder against target gradients as a baseline. Cannot be combined with `supervised_encoder`. |
| **use_cached_gradients** | `False` | Use cached gradients when available. Faster, but can use substantial memory or disk. |

---

## Deprecated arguments (0.2.0)

| Deprecated | Replacement | Notes |
|------------|-------------|-------|
| Heatmap **`fmt`** | **`annot_fmt`** | Applies to [`plot_comparison_heatmap`][gradiend.visualizer.heatmaps.base.plot_comparison_heatmap], [`plot_similarity_heatmap`][gradiend.visualizer.heatmaps.similarity.plot_similarity_heatmap], [`plot_topk_overlap_heatmap`][gradiend.visualizer.topk.pairwise_heatmap.plot_topk_overlap_heatmap], and related wrappers. |
| **`use_all_transitions`** (method kwarg) | **`include_other_classes`** | Same behavior: broaden encoder evaluation to all transitions in the split when `len(all_classes) > 2`. Prefer [`TrainingArguments.include_other_classes`][gradiend.trainer.core.arguments.TrainingArguments] as the default, or pass `include_other_classes=True` to [`evaluate_encoder()`][gradiend.trainer.trainer.Trainer.evaluate_encoder]. For explicit control, use [`transition_selection`][gradiend.trainer.core.transition_selection.TransitionSpec]. |
