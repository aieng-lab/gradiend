# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Trainer suites** — [`TrainerSuite`][gradiend.trainer.suite.base.TrainerSuite], [`PositiveTrainerSuite`][gradiend.trainer.suite.positive.PositiveTrainerSuite], [`SymmetricTrainerSuite`][gradiend.trainer.suite.symmetric.SymmetricTrainerSuite], and [`TrainerCollection`][gradiend.trainer.suite.collection.TrainerCollection] for training and comparing multiple GRADIEND models from declarative pair/feature definitions.
- **Multi-seed analysis** — [`Trainer.multi_seed()`][gradiend.trainer.trainer.Trainer.multi_seed] and [`MultiSeedTrainerView`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView] for seed aggregation, selection, and comparison plots.
- **Cross-model comparison** — similarity, cross-encoding, anchor-aligned, and GRADIEND-feature cross-encoding matrix helpers exported from `gradiend` (`compute_similarity_matrix`, `compute_grouped_similarity_matrices`, `compute_cross_encoding_matrix`, `compute_anchor_aligned_encoding_matrix`, `compute_gradiend_feature_cross_encoding_matrix`, `compute_gradiend_transition_cross_encoding_matrix`).
- **Comparison visualizations** — heatmap helpers (`plot_similarity_heatmap`, `plot_cross_encoding_heatmap`, `plot_gradiend_feature_cross_encoding_heatmap`, `plot_gradiend_transition_cross_encoding_heatmap`, `plot_comparison_heatmap`, `plot_topk_overlap_heatmap`, `plot_topk_overlap_venn`) and [`check_plot_environment`][gradiend.visualizer.plot_style.check_plot_environment].
- **Training cache policy** — `use_cache="only_convergent"` and training checkpoint fingerprinting (`cache_fingerprint` in `training.json`) for safer checkpoint reuse.
- **Data splits** — unified split columns, vocabulary-held-out splits, split-policy validation, and balancing helpers.
- **Transition selection** — [`TransitionSpec`][gradiend.trainer.core.transition_selection.TransitionSpec], [`pair()`][gradiend.trainer.core.transition_selection.pair], and [`identity()`][gradiend.trainer.core.transition_selection.identity] for explicit encoder-eval transition lists.
- **Decoder evaluation** — row-wise targets, explicit target validation, and richer probability-shift plotting.
- **Decoder-only MLM head** — auxiliary head training/saving and pooling-length ablation support.
- **Seq2seq objectives** — `seq2seq_decoder`, `seq2seq_encoder_mlm`, and `seq2seq_decoder_sequence_cloze` prediction objectives (experimental).
- **Runtime monitor** — optional heartbeat and CUDA OOM logging during training.
- **Documentation** — guides for trainer suites, multi-seed analysis, cross-model comparison, oriented cross-encoding matrices, data splits, decoder eval targets, and token prediction methods.

### Changed

- Top-level `gradiend` exports expanded (suites, comparison, visualization helpers); optional-import failures surface a clearer lazy error via `__getattr__`.
- Default `TextPredictionDataCreator.min_left_context_words` is now `10` (use `0` when sentence-initial tokens must match).
- Split-policy validation runs at training preparation time rather than on raw data load.
- `TextClassificationDataCreator` removed from top-level exports (still available under `gradiend.data.text.classification`).

### Deprecated

- Heatmap **`fmt`** argument — use **`annot_fmt`** instead (`plot_comparison_heatmap`, `plot_similarity_heatmap`, `plot_topk_overlap_heatmap`, and related wrappers).
- Method argument **`use_all_transitions`** — use **`include_other_classes`** instead (same behavior: include all class transitions in encoder evaluation when `len(all_classes) > 2`). Configure the default via [`TrainingArguments.include_other_classes`][gradiend.trainer.core.arguments.TrainingArguments].

### Fixed

- Trainer suite construction with explicit pair definitions no longer loads unresolved Hugging Face dataset ids during validation.
- Decoder probability selection recognizes unified `alternative_class` as well as legacy `alternative_id`.
- `ModelWithGradiend` deepcopy/pickle restores gradient locks correctly.
- Python 3.9 compatibility across trainer, comparison, and visualization code paths.
- CI unit tests no longer import from gitignored `experiments` trees at module level.

## [0.1.0] - 2025-02-04

Initial public release of the GRADIEND Python package.

[Unreleased]: https://github.com/aieng-lab/gradiend/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/aieng-lab/gradiend/releases/tag/v0.1.0
