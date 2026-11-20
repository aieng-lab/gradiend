# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.3.0] - 2026-11-20

Minor release with breaking changes for decoder-only models: the default label rule (`label_token_protocol`), which models count as Instruct models and their chat format, and `torch_dtype` now takes effect. Models trained with 0.2.5 or earlier keep loading and keep the rules they were trained with; retrain to get the new behaviour, or see the entries below to reproduce 0.2.5.

### Added

- Python 3.12 and 3.13 are supported and tested in CI (Python 3.9 stays supported).
- Divergence rows (shared token prefix) drop a leading BOS and truncate from the tail, so the label stays on a real token; models with a pre-training cache from 0.2.5 need one run with `use_cache=False`.

- `TrainingArguments.label_token_protocol` (`"first_differing_token"` by default, or `"no_leading_space"`): which token a decoder-only (`clm_next_token`) training row supervises. `first_differing_token` labels the first token where the factual and the alternative target differ and uses the tokens they share as context; a row without an alternative gets the same token variant as a contrast row with the same prefix: the leading-space variant (`▁she`, `Ġshe`) after a prefix that ends in whitespace, the literal token after a glued or empty prefix (`(`, `
`). Row-wise decoder evaluation scores rows with a shared token prefix at the first differing token (rows without one keep the scoring of 0.2.5); the pre-training alignment check and `create_inputs(..., alternative=...)` use the same rule. `no_leading_space` is the rule of 0.2.5 and earlier (`vocab[label]`, no alignment) and exists to reproduce models with a plain tokenizer trained with those releases; an Instruct model supports only `first_differing_token` (see Changed).
- Divergence-point supervision for decoder-only causal language models: when the factual and the alternative target are multi-token and share a token prefix of length $k$, the shared tokens become context and the first differing token is supervised. Row-wise decoder evaluation scores such rows at the same token.
- `audit_clm_alignment(tokenizer, pairs_or_df, context_prefix="")` and `TrainingArguments.clm_unalignable_rows` (`"error"` by default, or `"drop"`). Before training, pruning or a learning-rate search, every decoder-only row is checked. A row whose two targets are identical in tokens, or where one is a token prefix of the other (`bride` and `bridegroom`), cannot be supervised: `"error"` raises one `ClmAlignmentError` that lists all such pairs, `"drop"` removes the rows and logs the counts per class. The check also runs when the model is given as a Hub id.
- `ClmDivergence`, `ClmAlignmentError`, `align_clm_divergence` and `audit_clm_alignment` are exported from `gradiend`.
- `TextPredictionModelWithGradiend.create_inputs(masked_text, label, alternative=None)` and `create_gradients(..., alternative=None)`: with an alternative, the row is built like a training row (shared tokens as context, label at the first differing token). An alternative equal to the label is a plain row.
- `TrainingArguments.instruction_model` (`None` by default): whether the model is an Instruct model whose input is chat-formatted. `None` decides by the model name (see Changed), `True` chat-formats a model whose name does not match, `False` reads plain text even if the name matches. `AutoTokenizerForLM.from_pretrained(..., instruction_model=...)` takes the same value. The decision in force (`True` or `False`) is stored with the trained model in `training.json`, and a loaded model keeps it, whatever the name rule says later. A model saved without it (0.2.5 and earlier) is decided by the old rule, `instruct` anywhere in the name, so an old plain-trained `gemma-2-2b-it` still loads as plain text.
- `TrainingArguments.chat_template_date` (ISO date `YYYY-MM-DD`, `None` by default) and `DEFAULT_CHAT_TEMPLATE_DATE` (`"2024-07-26"`, in `gradiend.trainer.text.common.chat_format`): chat templates that write the current date into the prompt (Llama 3.1 and later, gpt-oss) get a fixed date instead of the system clock, so the rows do not change with the day they are built. These models work out of the box; the first time a template asks for the date, one INFO line names the date used. The date a trained model used is stored with it, and a loaded model keeps it.
- `register_chat_format`, `ChatFormat`, `TemplateChatFormat`, `chat_format_for_tokenizer`, `uses_chat_format` and `instruction_name_marker` (`gradiend.trainer.text.common`): the chat format of a model family. `register_chat_format` plugs in a family whose tokenizer has no `chat_template`, without editing GRADIEND ([guide](docs/guides/decoder-only.md#instruct-models)).

### Changed

- Behaviour change for decoder-only models (`clm_next_token`): the default `label_token_protocol="first_differing_token"` replaces the label rule of 0.2.5 and earlier. Multi-token targets with a shared token prefix (GPT-2: `anglicised` and `anglicized` share ` ang` and `lic`) were supervised at their first token, where both sides are equal, so the contrast was zero; now the first differing token is supervised. For all other rows the training label is the leading-space token (`▁she`, `Ġshe`) instead of `vocab[label]` (`she`), which is the token `create_inputs` already used, so training and single-row scoring now agree. Retrain to get the new behaviour, or pass `label_token_protocol="no_leading_space"` to reproduce 0.2.5 results.
- A trained model keeps the label rule it was trained with. When a trainer loads a saved model, the rule recorded in its `training.json` is used for the trainer's datasets and evaluation (a warning names the difference to the trainer's own `label_token_protocol`). A saved setting without the key, as written by 0.2.5 and earlier, loads as `no_leading_space`. The names `canonical` and `legacy` are accepted when loading saved settings and mean `first_differing_token` and `no_leading_space`.
- Instruct models are chat-formatted in the chat format of their own family. 0.2.5 wrote the Llama-3 layout by hand for every Instruct model, which is wrong for Qwen, Mistral, Gemma and others; now the format is the tokenizer's own `chat_template`, and Llama-3 is no longer special (its prompt is the one of its `chat_template`: a blank line after each role header, and from Llama 3.1 on a `Today Date` block with a fixed date, see `chat_template_date`). A tokenizer without a `chat_template` raises a `ValueError` that names the model and points to `register_chat_format`. Every row is a chat prompt with a system prompt: training rows (also rows with shared tokens, and identity rows), neutral rows, `create_inputs`, row-wise decoder evaluation and the language-model score. A training row is the chat prompt of the text before `[MASK]` (without its trailing whitespace), then the shared answer tokens, then a slot token that carries the label; the label is predicted from the end of the generation prompt, where the model starts its answer. The answer words are tokenized without a leading space, because the answer opens the assistant turn; the alignment audit and the row-wise decoder evaluation use the same tokens. A context that does not fit into `max_length` is shortened from the left, inside the user message; the chat frame is kept.
- Breaking: which models are Instruct models. A model is an Instruct model when the whole word `instruct`, `chat` or `it` is in the base name of its id or path (the part after the last `/`, split at every character that is not a letter or digit): `Qwen/Qwen2.5-1.5B-Instruct`, `google/gemma-2-2b-it` and `TinyLlama/TinyLlama-1.1B-Chat-v1.0` match, `fitness` and a local path whose parent directory says `instruct` do not. 0.2.5 looked for the text `instruct` anywhere in the name, also in the organisation or a parent directory. `AutoTokenizerForLM.from_pretrained` logs one INFO line when it chat-wraps a tokenizer. A name that matches on a model without a chat format (an encoder or seq2seq model such as `opus-mt-en-it`) falls back to plain text with one warning; Hugging Face snapshot paths are judged by the model name in the path. `InstructTokenizerWrapper` no longer takes `user_prompt_header` and `assistant_prompt_header` and has no such attributes: the role headers come from the chat template.
- Breaking: an Instruct model supports only `label_token_protocol="first_differing_token"`. A chat tokenizer combined with `no_leading_space` raises a `ValueError` when a dataset is built, when the trainer sets up a model (also a loaded model whose record says `no_leading_space`), in `create_inputs` and in row-wise decoder evaluation. The Instruct rule of 0.2.5 (the whole prompt in Llama-3 tokens for every family, single-token labels, the label on the last prompt token) cannot be reproduced: a trainer that loads a model trained with it raises that error, and the model has to be retrained.

### Fixed

- Learning-rate search (`tune_learning_rate`, `learning_rate="tune"`): a probe is judged in the correlation sign it learned. The convergence check uses the absolute correlation, but the probe classifier read the signed score, so an encoder that separates the classes with a negative sign looked as if it was degrading while its absolute correlation was rising, and the search was sent to lower learning rates. Positive probes are unchanged, the trace records the `orientation`.
- The pre-training alignment check ran no data check when the model was given as a Hub id, because the objective could not be resolved from a model name: unalignable rows then crashed `pre_prune` even with `clm_unalignable_rows="drop"`. The objective is resolved from the tokenizer while the model is not loaded.
- `create_inputs(..., alternative=...)` with an alternative identical to the label builds a plain row, like the training dataset, instead of failing on a divergence that does not exist.
- Neutral rows of an Instruct model: `create_masked_pair_from_text` tokenized the chat prompt instead of the sentence, so the masked word could be taken from the system prompt or the chat headers. It splits the raw sentence now. The offset mappings of a chat tokenizer also pointed into the chat prompt; they point into the sentence now, and a token of the chat frame gets an empty span.
- The row-wise language-model score of an Instruct model located the user text with the Llama-3 header token; it takes the length of the model's own chat frame, so it is right for every family and for either padding side.
- A `torch_dtype` set in the training arguments (for example `torch.bfloat16`) was silently reset to `float32` whenever the arguments were copied or restored, which `train()` does, so training ran in `float32`. Training with an explicit `torch_dtype` now really uses it, and a dtype name that does not exist raises instead of becoming `float32`. Models trained before keep their numbers; to reproduce a run trained under 0.2.5 with an explicit `torch_dtype`, set `torch_dtype=torch.float32`. `torch_dtype="auto"` and unknown dtype names now raise instead of becoming `float32`.

## [0.2.5] - 2026-11-02

Patch release. Training defaults are unchanged from 0.2.4.

### Added

- `max_size_per_target` parameter in `TextPredictionDataCreator` and `TextPredictionDataCreator.generate_training_data()`: caps the number of rows per target token within each class to prevent dominant target tokens from overpowering rarer targets, without oversampling rare targets.

### Fixed

- Counterfactual token sampling in `per_class_dict_to_unified()` now samples only from the same split of the counterpart class when the dataset contains split metadata. Previously, counterfactuals were sampled globally across all splits, which leaked held-out validation and test target vocabulary into training counterfactuals when using `split_by_target=True`.

## [0.2.4] - 2026-10-30

Patch release. Training defaults are unchanged from 0.2.3.

### Fixed

- Decoder evaluation with a token that belongs to more than one class (for example German `sie`, singular and plural). The evaluation then scores every row on its own, and its summary was assembled wrongly: `evaluate_decoder(target_class=...)` failed with `No decoder grid candidates for strengthen ...`, and `evaluate_decoder()` for all classes reported each class under the other class's name and feature factor. Strengthening a class now reads its probability from the other class's rows, as it does for classes with disjoint tokens. Cached decoder results of such runs are not valid: run `evaluate_decoder(..., use_cache=False)` once so that they are recomputed.
- Decoder-only training (Qwen, Gemma, GPT-2, ...) rebuilt the tokenizer vocabulary several times for every training item, about 0.1 s of CPU time per item for a 150k vocabulary, which made training CPU-bound. The vocabulary is now read once per dataset. The supervised label ids and all results are unchanged.

## [0.2.3] - 2026-10-23

Patch release. Training defaults are unchanged from 0.2.2.

### Added

- `LRSearchConfig.probe_seeds` (default `1`): with `probe_seeds > 1` a probe run that ends `collapsed` or `degrading` is repeated with the next seeds before the learning rate is judged, so one unlucky seed does not decide the search ([guide](docs/guides/learning-rate-search.md)).

### Fixed

- Masked-language-model training: a row whose mask lay beyond `max_length` lost its mask in the truncation and was trained on without any label (loss over zero positions, no error). The text is now cut to a window around the mask; if the mask cannot be kept, a `ValueError` names the row.
- The quick-start example in the README (`gradiend/examples/readme.py`) failed in about 30% of the runs with `fail_on_non_convergence`: only 3 validation rows were left, so an untrained model could reach a perfect correlation at step 0 by chance and no later step could beat it. It now uses a 60/20/20 split.
- `gradiend/examples/create_german_article_data.py` produced no rows (the default of ten context words before the article is longer than its example sentences) and failed on its tiny splits.
- The neutral data of the encoder evaluation no longer excludes the words of `decoder_eval_targets`. Those only score the decoder and can be a large subset of the vocabulary, which removed thousands of unrelated words from the neutral data.

## [0.2.2] - 2026-10-22

Patch release. Training defaults and numerics are unchanged from 0.2.1.

### Added

- Automatic learning-rate search with short probe runs: `learning_rate="tune"` or `trainer.tune_learning_rate()`, options in `LRSearchConfig`, and `trainer.lr_search_result.plot()` shows what the search did ([guide](docs/guides/learning-rate-search.md)).
- `PairedTextFilterConfig` for `TextPredictionDataCreator`: list 1:1 counterpart words once (e.g. British/American spelling) and every row carries its explicit alternative. `split_by_target=True` holds out whole target words (or pairs) for validation and test; the trainer's `split_col="heldout"` keeps such pairs and their alternatives intact.
- Callbacks can stop training after any step by setting `control["should_stop"]`.
- `plot_topk_overlap`, `plot_topk_overlap_venn` and the top-k heatmaps accept a list of models, labelled after their checkpoint folders.
- `mask_placeholder` (`TextPredictionConfig` / `TrainingArguments`, default `"[MASK]"`) for data that marks the prediction slot differently, e.g. `"[PRONOUN]"`. Templates without the placeholder now fail early. Decoder evaluation still expects `[MASK]`.
- `gradiend.comparison.load_saved_cross_encoding_inputs`: the class pairs, sources and cached encodings of saved run folders, so the oriented cross-encoding matrix can be computed without trainers or data (e.g. for the published multilingual models).

### Changed

- `plot_encoder_by_target` shows targets that differ only in casing as one target, in their most frequent spelling (`ignore_case=True`, the default).
- The interactive by-target plot (`interactive=True`) shows the masked sentence on hover, together with the factual, alternative and encoder input token. Before, it showed the original sentence with the factual word, which did not contain the plotted token (the alternative is the encoder input by default). For this, the encoder rows of text training data keep their `template` column; encoder caches written by earlier versions lack it and fall back to the original sentence.
- `multilingual_demo_labels` (labels and styles of the multilingual demo figures) moved from `gradiend.visualizer` to `gradiend.examples` and is no longer part of the installed package.

### Fixed

- `split_by_target=True` with plain `TextFilterConfig` targets treated `Bad` and `bad` as different targets, so a word could end up in several splits. Casing and surrounding whitespace are now ignored (override with `split_group_key`). Paired targets were not affected.
- Tokenizers that pad on the left (Gemma-2/3): the supervised label landed in the padding, so all inputs of a class were identical and training "converged" without learning anything. Models trained with 0.2.1 or earlier on such tokenizers should be retrained.
- Language-model score (LMS) of decoder-only models: the causal-LM perplexity compared each position's logits with the token at the same position instead of the next token, so it measured how well the model repeats the current token, not its language-model loss. It now uses the next-token shift and ignores targets predicted from padding. LMS values, and the update strength `evaluate_decoder` selects with them, can differ from 0.2.1 for decoder-only models; re-run their decoder evaluations. Masked language models are not affected.
- Decoder-only detection reads the model config instead of the tokenizer's mask token. Gemma-3 is causal but ships an unused mask token and was trained as a masked LM.
- Loading a base model fails loudly when nearly all weights come back uninitialised (seen with Gemma 3) instead of silently training on random weights. Incomplete local checkpoints get a clearer error.
- CUDA out-of-memory (or other CUDA errors) during evaluation now abort training. Before, evaluation was skipped silently and no checkpoint was selected.
- A stale decoder-grid cache (other direction or target, or missing metrics) is recomputed instead of reused.
- Decoder evaluation no longer fails with `KeyError` when no candidate reaches the language-model score (LMS) threshold (seen in multi-seed runs). In that case the candidate with the smallest non-zero learning rate is selected, as before; the fallback only broke on the dict-style candidate ids of multi-seed grids.
- `GradientTrainingDataset` raises `IndexError` for an out-of-range index (before: an empty batch and a confusing `KeyError` later).

## [0.2.1] - 2026-07-16

### Changed

- Multi-seed dispersion heatmaps now use dispersion-specific autoscaling and one-decimal percentage annotations, making small seed-to-seed variation readable.
- Documentation now explains dispersion heatmap values as standard deviations of percentage overlap across seeds.
- Decoder evaluation cache keys now include split and size caps so cached grids are reused only for matching evaluation settings.

### Fixed

- `evaluate_decoder()` consistently accepts `split` and shared `max_size` arguments; `max_size` caps both training-like and neutral decoder evaluation rows unless explicit caps are supplied.
- Text-prediction and classification decoder evaluation now honor requested splits.
- Decoder plotting analysis honors `TrainingArguments.decoder_eval_max_size_training_like`, `decoder_eval_max_size_neutral`, and `eval_batch_size`, preventing uncapped neutral/LMS evaluation during probability-shift plotting.
- Encoder metrics caches are reused when `Trainer.evaluate_encoder()` supplies a trainer-owned cached encoder DataFrame.
- Multi-seed per-call `return_per_seed=True` is handled by `MultiSeedTrainerView` instead of leaking into single-seed trainer calls.
- Multi-seed encoder distribution, scatter, and strip-by-split plots now build per-seed encoder DataFrames before plotting.
- API autoref links in docs were refreshed for CI.

## [0.2.0] - 2026-07-13

### Added

- **Trainer suites** — [`TrainerSuite`][gradiend.trainer.suite.base.TrainerSuite], [`PositiveTrainerSuite`][gradiend.trainer.suite.positive.PositiveTrainerSuite], [`SymmetricTrainerSuite`][gradiend.trainer.suite.symmetric.SymmetricTrainerSuite], and [`TrainerCollection`][gradiend.trainer.suite.collection.TrainerCollection] for training and comparing multiple GRADIEND models from declarative pair/feature definitions.
- **Multi-seed analysis** — [`Trainer.multi_seed()`][gradiend.trainer.trainer.Trainer.multi_seed] and [`MultiSeedTrainerView`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView] for seed aggregation, selection, and comparison plots.
- **Cross-model comparison** — similarity, cross-encoding, anchor-aligned, and GRADIEND-feature cross-encoding matrix helpers exported from `gradiend` (`compute_similarity_matrix`, `compute_grouped_similarity_matrices`, `compute_cross_encoding_matrix`, `compute_anchor_aligned_encoding_matrix`, `compute_gradiend_feature_cross_encoding_matrix`, `compute_gradiend_transition_cross_encoding_matrix`).
- **Plot styling API** — [`PlotStyleConfig`][gradiend.visualizer.plot_style_config.PlotStyleConfig], [`configure_plot_style()`][gradiend.visualizer.plot_style.configure_plot_style], transition label helpers ([`format_transition_label`][gradiend.visualizer.labels.format_transition_label]), `GRADIEND_PLOT_*` environment variables, and automatic `amsmath`/`amssymb` preamble when LaTeX is enabled. Guide: [Plot styling & LaTeX](guides/plot-styling-latex.md).
- **Comparison visualizations** — heatmap helpers (`plot_similarity_heatmap`, `plot_cross_encoding_heatmap`, `plot_gradiend_feature_cross_encoding_heatmap`, `plot_gradiend_transition_cross_encoding_heatmap`, `plot_comparison_heatmap`, `plot_topk_overlap_heatmap`, `plot_topk_overlap_venn`) and [`check_plot_environment`][gradiend.visualizer.plot_style.check_plot_environment].
- **Training cache policy** — `use_cache="only_convergent"` and training checkpoint fingerprinting (`cache_fingerprint` in `training.json`) for safer checkpoint reuse.
- **Data splits** — unified split columns, vocabulary-held-out splits, split-policy validation, and balancing helpers.
- **Transition selection** — [`TransitionSpec`][gradiend.trainer.core.transition_selection.TransitionSpec], [`pair()`][gradiend.trainer.core.transition_selection.pair], and [`identity()`][gradiend.trainer.core.transition_selection.identity] for explicit encoder-eval transition lists.
- **Decoder evaluation** — row-wise targets, explicit target validation, and richer probability-shift plotting.
- **Decoder-only MLM head** — auxiliary head training/saving and pooling-length ablation support.
- **Seq2seq objectives** — `seq2seq_decoder`, `seq2seq_encoder_mlm`, and `seq2seq_decoder_sequence_cloze` prediction objectives (experimental).
- **Runtime monitor** — optional heartbeat and CUDA OOM logging during training.
- **Documentation** — guides for trainer suites, multi-seed analysis, cross-model comparison, oriented cross-encoding matrices, data splits, decoder eval targets, token prediction methods, and plot styling/LaTeX.

### Changed

- Top-level `gradiend` exports expanded (suites, comparison, visualization helpers, plot styling); optional-import failures surface a clearer lazy error via `__getattr__`.
- Default `TextPredictionDataCreator.min_left_context_words` is now `10` (use `0` when sentence-initial tokens must match).
- Split-policy validation runs at training preparation time rather than on raw data load.
- `TextClassificationDataCreator` removed from top-level exports (still available under `gradiend.data.text.classification`).

### Deprecated

- Heatmap **`fmt`** argument — use **`annot_fmt`** instead (`plot_comparison_heatmap`, `plot_similarity_heatmap`, `plot_topk_overlap_heatmap`, and related wrappers).
- Method argument **`use_all_transitions`** — use **`include_other_classes`** instead (same behavior: include all class transitions in encoder evaluation when `len(all_classes) > 2`). Configure the default via [`TrainingArguments.include_other_classes`][gradiend.trainer.core.arguments.TrainingArguments].
- **`configure_matplotlib_style()`** — use **`configure_plot_style()`** instead.

### Fixed

- Trainer suite construction with explicit pair definitions no longer loads unresolved Hugging Face dataset ids during validation.
- Decoder probability selection recognizes unified `alternative_class` as well as legacy `alternative_id`.
- `ModelWithGradiend` deepcopy/pickle restores gradient locks correctly.
- Python 3.9 compatibility across trainer, comparison, and visualization code paths.
- CI unit tests no longer import from gitignored `experiments` trees at module level.
- LaTeX transition arrows (`\rightleftarrows`) include `amsmath` and `amssymb` in the matplotlib preamble; checkpoint `model.source` is no longer overwritten from `TrainingArguments` when loading finished checkpoints.

## [0.1.0] - 2025-02-04

Initial public release of the GRADIEND Python package.

[Unreleased]: https://github.com/aieng-lab/gradiend/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/aieng-lab/gradiend/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/aieng-lab/gradiend/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/aieng-lab/gradiend/releases/tag/v0.1.0
