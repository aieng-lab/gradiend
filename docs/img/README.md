# Doc images

Images here are used by the built documentation. PNG/PDF in this folder are **not** ignored by git so they can be committed.

## Plot screenshots (generate and add)

Generate these by running the relevant code, then save outputs into this directory with the names below so the markdown links resolve.

| File | Where it's used | How to generate |
|------|-----------------|------------------|
| `system-overview.png` | [index.md](../index.md) | Schematic of GRADIEND components (replace AI placeholder when you have a final figure). |
| `start_workflow_training_convergence.png` | [start.md](../start.md) | Run start workflow, then `trainer.plot_training_convergence(output="docs/img/start_workflow_training_convergence.png", img_format="png")`. |
| `start_workflow_encoder_analysis_split_test.png` | [start.md](../start.md) | Run start workflow, then `trainer.evaluate_encoder(..., plot=True)` and save from the plot or use the trainer’s plot output path. |
| `training_convergence.png` | [evaluation-visualization.md](../guides/evaluation-visualization.md) | `plot_training_convergence(..., output="docs/img/training_convergence.png", img_format="png")`. |
| `encoder_analysis_max_size_100_split_test.png` | [evaluation-visualization.md](../guides/evaluation-visualization.md) | `evaluate_encoder(..., plot=True)` and save the figure to this path (or copy from experiment_dir). |

After adding or updating these files, commit them so the deployed docs show the plots.
