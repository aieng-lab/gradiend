# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
