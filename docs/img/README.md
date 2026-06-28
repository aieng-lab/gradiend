# Doc images

Images here are used by the built documentation. PNG/PDF in this folder are **not** ignored by git so they can be committed.

Plot regeneration instructions also appear as HTML comments (`<!-- DOC_PLOT: ... -->`) in the markdown files that reference each figure. MkDocs does not render those comments.

## Plot screenshots

| File | Where it's used | How to generate |
|------|-----------------|------------------|
| `workflow-diagram.png` | [index.md](../index.md), [detailed-workflow.md](../tutorials/detailed-workflow.md) | Render `workflow-diagram.tex` |
| `start_workflow_training_convergence.png` | [start.md](../start.md), [evaluation-visualization.md](../guides/evaluation-visualization.md) | [start_workflow.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/start_workflow.py) |
| `start_workflow_encoder_analysis_split_test.png` | [start.md](../start.md), [evaluation-visualization.md](../guides/evaluation-visualization.md) | Same |
| `start_workflow_decoder_probability_shifts_3SG.png` | [start.md](../start.md) | Same; decoder eval for `3SG` |
| `data_splits_encoder_by_target_test.png` | [data-splits.md](../guides/data-splits.md) | [train_sentiment.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_sentiment.py) |
| `suite_similarity_heatmap.png` | [trainer-suites.md](../guides/trainer-suites.md) | [train_sentiment_positive_suite.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_sentiment_positive_suite.py) (`--write-docs-images`) |
| `suite_cross_encoding_heatmap.png` | [trainer-suites.md](../guides/trainer-suites.md), [cross-model-comparison.md](../guides/cross-model-comparison.md) | Same (`--write-docs-images`) |
| `symmetric_suite_topk_overlap.png` | [trainer-suites.md](../guides/trainer-suites.md) | [train_race_symmetric_suite.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_race_symmetric_suite.py) (`--write-docs-images`) |
| `symmetric_suite_cross_encoding.png` | [trainer-suites.md](../guides/trainer-suites.md) | Same |
| `multi_seed_layerwise_similarity.png` | [cross-model-comparison.md](../guides/cross-model-comparison.md) | `python -m gradiend.examples.train_multi_seed_stability --write-docs-images` |
| `multi_seed_encoder_by_target_seeds.png` | [multi-seed.md](../guides/multi-seed.md) | Same |
| `multi_seed_encoder_by_target_combined.png` | [multi-seed.md](../guides/multi-seed.md) | Same |
| `multi_seed_encoder_by_target_errorbar.png` | [multi-seed.md](../guides/multi-seed.md) | Same |
| `multi_seed_encoder_distributions.png` | [multi-seed.md](../guides/multi-seed.md) | Same |
| `multi_seed_encoder_scatter.png` | [multi-seed.md](../guides/multi-seed.md) | Same |
| `multi_seed_training_convergence.png` | [multi-seed.md](../guides/multi-seed.md) | Same |
| `multi_seed_probability_shifts.png` | [multi-seed.md](../guides/multi-seed.md) | Same |
| `seed_comparison_topk_overlap.png` | [cross-model-comparison.md](../guides/cross-model-comparison.md) | Same |
| `seed_comparison_decoder_cosine.png` | [cross-model-comparison.md](../guides/cross-model-comparison.md) | Same |
| `multi_seed_component_similarity.png` | [cross-model-comparison.md](../guides/cross-model-comparison.md) | Same |
| `topk_overlap_heatmap.png` | [evaluation-visualization.md](../guides/evaluation-visualization.md) | [train_gender_de_detailed.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de_detailed.py) |
| `topk_overlap_venn.png` | [evaluation-visualization.md](../guides/evaluation-visualization.md) | Same |
| `cross_encoding_gradiend_by_transition.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | [multilingual_gradiend_demo_small.py](https://github.com/aieng-lab/gradiend/blob/main/experiments/multilingual_gradiend_demo_small.py) (`--plot-only`) |
| `cross_encoding_oriented_counterfactual.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_synthetic_preanchor_overview.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | `python scripts/generate_cross_encoding_matrix_doc_figures.py` |
| `cross_encoding_synthetic_oriented_overview.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_synthetic_preanchor_diagonal_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_synthetic_oriented_diagonal_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_synthetic_aggregation_diagonal.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_synthetic_preanchor_offdiag_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_synthetic_oriented_offdiag_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_synthetic_aggregation_offdiag.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `decoder_eval_commutative_probability_shifts.png` | [decoder-eval-targets.md](../guides/decoder-eval-targets.md) | Inline snippet in guide (no dedicated script) |

After adding or updating these files, commit them so deployed docs show the plots.
