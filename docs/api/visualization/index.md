# Visualization

Standalone plot functions (module-level). Trainer convenience wrappers such as
`trainer.plot_training_convergence()` are documented on
[`Trainer`][gradiend.trainer.trainer.Trainer] and
[`TextPredictionTrainer`][gradiend.trainer.text.prediction.trainer.TextPredictionTrainer].

## Training and encoder plots

- **[`plot_training_convergence`][gradiend.visualizer.convergence.plot_training_convergence]** — Convergence over training steps
- **[`plot_encoder_distributions`][gradiend.visualizer.encoder_distributions.plot_encoder_distributions]** — Split violins of encoded values
- **[`plot_encoder_scatter`][gradiend.visualizer.encoder_scatter.plot_encoder_scatter]** — Interactive encoder scatter (Plotly)
- **[`plot_encoder_by_target`][gradiend.visualizer.encoder_by_target.plot_encoder_by_target]** — Encoded values per masked target token

## Multi-model comparison

- **[`plot_topk_overlap_heatmap`][gradiend.visualizer.topk.pairwise_heatmap.plot_topk_overlap_heatmap]** — Pairwise top-k weight overlap
- **[`plot_topk_overlap_venn`][gradiend.visualizer.topk.venn_.plot_topk_overlap_venn]** — Top-k overlap Venn diagram
- **[`plot_similarity_heatmap`][gradiend.visualizer.heatmaps.similarity.plot_similarity_heatmap]** — Similarity matrix heatmap
- **[`plot_cross_encoding_heatmap`][gradiend.visualizer.heatmaps.encoding.plot_cross_encoding_heatmap]** — Oriented cross-encoding heatmap
- **[`plot_gradiend_transition_cross_encoding_heatmap`][gradiend.visualizer.heatmaps.encoding.plot_gradiend_transition_cross_encoding_heatmap]** — GRADIEND × transition heatmap
- **[`plot_comparison_heatmap`][gradiend.visualizer.heatmaps.base.plot_comparison_heatmap]** — Generic comparison heatmap (e.g. seed comparison)

## Related

- [`compute_similarity_matrix`][gradiend.comparison.similarity.compute_similarity_matrix] — Build similarity data for heatmaps
- [Evaluation visualization guide](../../guides/evaluation-visualization.md) — Plot customization and examples
