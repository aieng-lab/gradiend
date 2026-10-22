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
| `multi_seed_suite_dispersion_heatmap.png` | [trainer-suites.md](../guides/trainer-suites.md) | Multilingual demo multi-seed suite output, converted from the generated PDF |
| `topk_overlap_heatmap.png` | [evaluation-visualization.md](../guides/evaluation-visualization.md) | [train_gender_de_detailed.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de_detailed.py) |
| `topk_overlap_venn.png` | [evaluation-visualization.md](../guides/evaluation-visualization.md) | Same |
| `pruning_analysis_metric_grid.png` | [pruning-guide.md](../guides/pruning-guide.md) | `runs/paper/pruning_analysis_metric_grid.pdf` |
| `cross_encoding_example_race_gender_oriented_counterfactual.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Race + English gender subset from the multilingual demo BERT cross-encoding outputs |
| `cross_encoding_example_race_gender_preanchor.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_example_race_gender_preanchor_black_black_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_example_race_gender_oriented_black_black_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_example_race_gender_preanchor_black_asian_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_example_race_gender_oriented_black_asian_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_example_race_gender_preanchor_asian_asian_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_example_race_gender_oriented_asian_asian_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `cross_encoding_example_race_gender_oriented_white_asian_highlight.png` | [cross-encoding-matrix.md](../guides/cross-encoding-matrix.md) | Same |
| `lr_search_example.png` | [learning-rate-search.md](../guides/learning-rate-search.md) | `python scripts/plot_lr_search_example.py docs/img/lr_search_example.png` (simulated training response) |

After adding or updating these files, commit them so deployed docs show the plots.
