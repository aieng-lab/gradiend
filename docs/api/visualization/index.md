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

- **[`plot_topk_overlap`][gradiend.visualizer.topk.plot_topk_overlap]** — Top-k weight overlap for a dict or list of models
- **[`plot_topk_overlap_heatmap`][gradiend.visualizer.topk.pairwise_heatmap.plot_topk_overlap_heatmap]** — Pairwise top-k weight overlap
- **[`plot_topk_overlap_venn`][gradiend.visualizer.topk.venn_.plot_topk_overlap_venn]** — Top-k overlap Venn diagram
- **[`plot_similarity_heatmap`][gradiend.visualizer.heatmaps.similarity.plot_similarity_heatmap]** — Similarity matrix heatmap
- **[`plot_cross_encoding_heatmap`][gradiend.visualizer.heatmaps.encoding.plot_cross_encoding_heatmap]** — Oriented cross-encoding heatmap
- **[`plot_gradiend_transition_cross_encoding_heatmap`][gradiend.visualizer.heatmaps.encoding.plot_gradiend_transition_cross_encoding_heatmap]** — GRADIEND × transition heatmap
- **[`plot_gradiend_feature_cross_encoding_heatmap`][gradiend.visualizer.heatmaps.encoding.plot_gradiend_feature_cross_encoding_heatmap]** — GRADIEND × feature-class heatmap
- **[`plot_comparison_heatmap`][gradiend.visualizer.heatmaps.base.plot_comparison_heatmap]** — Generic comparison heatmap (e.g. seed comparison)

## Environment

- **[`check_plot_environment`][gradiend.visualizer.plot_style.check_plot_environment]** — Verify matplotlib/LaTeX/font setup
- **[`configure_plot_style`][gradiend.visualizer.plot_style.configure_plot_style]** — Apply GRADIEND matplotlib defaults
- **[`PlotStyleConfig`][gradiend.visualizer.plot_style_config.PlotStyleConfig]** — LaTeX, font, preamble, and transition-arrow options
- **[`format_transition_label`][gradiend.visualizer.labels.format_transition_label]** — Render `A -> B` labels for heatmaps

See also [Plot styling & LaTeX guide](../../guides/plot-styling-latex.md).

## Related

- [`compute_similarity_matrix`][gradiend.comparison.similarity.compute_similarity_matrix] — Build similarity data for heatmaps
- [Evaluation visualization guide](../../guides/evaluation-visualization.md) — Plot customization and examples
