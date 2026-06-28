# Training

Trainer, arguments and pruning:

- **[`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments]** — Training hyperparameters (HF-like)
- **[`Trainer`][gradiend.trainer.trainer.Trainer]** — Generic trainer
- **[`TextPredictionConfig`][gradiend.trainer.text.prediction.trainer.TextPredictionConfig]** — Config for text prediction
- **[`TextPredictionTrainer`][gradiend.trainer.text.prediction.trainer.TextPredictionTrainer]** — Trainer for text prediction
- **[`PrePruneConfig`][gradiend.trainer.core.pruning.PrePruneConfig]** — Pre-pruning
- **[`PostPruneConfig`][gradiend.trainer.core.pruning.PostPruneConfig]** — Post-pruning

Additional symbols (not in the navigation): [`load_training_stats`][gradiend.trainer.core.stats.load_training_stats], [`create_model_with_gradiend`][gradiend.trainer.factory.create_model_with_gradiend], [`GradientTrainingDataset`][gradiend.trainer.core.dataset.GradientTrainingDataset], [`TextGradientTrainingDataset`][gradiend.trainer.text.common.dataset.TextGradientTrainingDataset].
