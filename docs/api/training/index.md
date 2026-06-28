# Training

Trainer, arguments, suites, and pruning:

- **[`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments]** — Training hyperparameters (HF-like)
- **[`Trainer`][gradiend.trainer.trainer.Trainer]** — Generic trainer
- **[`TextPredictionConfig`][gradiend.trainer.text.prediction.trainer.TextPredictionConfig]** — Config for text prediction
- **[`TextPredictionTrainer`][gradiend.trainer.text.prediction.trainer.TextPredictionTrainer]** — Trainer for text prediction
- **[`TrainerConfig`][gradiend.trainer.config.TrainerConfig]** — Shared trainer configuration base
- **[`PrePruneConfig`][gradiend.trainer.core.pruning.PrePruneConfig]** — Pre-pruning
- **[`PostPruneConfig`][gradiend.trainer.core.pruning.PostPruneConfig]** — Post-pruning

## Trainer suites

- **[`TrainerSuite`][gradiend.trainer.suite.base.TrainerSuite]** — Base suite for paired trainers
- **[`PositiveTrainerSuite`][gradiend.trainer.suite.positive.PositiveTrainerSuite]** — One-vs-rest positive feature suites
- **[`SymmetricTrainerSuite`][gradiend.trainer.suite.symmetric.SymmetricTrainerSuite]** — Symmetric pair suites
- **[`TrainerCollection`][gradiend.trainer.suite.collection.TrainerCollection]** — Group suites for joint analysis
- **[`SuitePairDefinition`][gradiend.trainer.suite.definitions.SuitePairDefinition]** — Declarative pair spec
- **[`PositiveFeatureDefinition`][gradiend.trainer.suite.definitions.PositiveFeatureDefinition]** — Declarative positive-feature spec
- **[`MultiSeedTrainerView`][gradiend.trainer.core.multi_seed.MultiSeedTrainerView]** — Multi-seed analysis view

## Transitions and utilities

- **[`TransitionSpec`][gradiend.trainer.core.transition_selection.TransitionSpec]** — Explicit encoder-eval transition (`pair()`, `identity()`)
- **[`set_seed`][gradiend.trainer.trainer.set_seed]** — Reproducible RNG seeding

Additional symbols (not in the navigation): [`load_training_stats`][gradiend.trainer.core.stats.load_training_stats], [`create_model_with_gradiend`][gradiend.trainer.factory.create_model_with_gradiend], [`GradientTrainingDataset`][gradiend.trainer.core.dataset.GradientTrainingDataset], [`TextGradientTrainingDataset`][gradiend.trainer.text.common.dataset.TextGradientTrainingDataset].
