# Saving & loading (modality-agnostic)

This guide explains where artifacts are stored and how to reload trained models. It is intended to be modality‑agnostic.

## Experiment directories

`TrainingArguments.experiment_dir` defines the root output directory for runs. When `run_id` is set, outputs are stored under `experiment_dir/run_id`.

Artifacts typically include:

- Model weights and config (`model.safetensors` or `pytorch_model.bin`, `config.json`)
- Optional `training.json`
- Encoder/decoder evaluation caches and plots

## Load a trained model

```python
model = trainer.get_model()
```

`trainer.get_model()` loads a `ModelWithGradiend` instance from the current `model_path`. Use `load_directory` to load a specific checkpoint.

## Save a trained model

```python
model = trainer.get_model()
model.save_pretrained("path/to/output")
```

This saves the GRADIEND checkpoint plus adapter configuration required for reload.
