# GRADIEND Experiments

This folder contains larger committed experiments, benchmark scripts, and paper-analysis
workflows. They are reproducibility material rather than public package API.

Scripts here may assume local datasets, trained model artifacts, GPUs, SLURM settings, or
paper-specific output folders. Keep reusable library behavior in `gradiend/`, small public
examples in `gradiend/examples/`, and project-specific but committed studies in `projects/`.

When promoting an experiment into documentation, first add focused tests and move the
minimal user-facing workflow into `gradiend/examples/` or the docs.
