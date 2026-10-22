"""
GRADIEND Training Module

This package exposes trainer APIs lazily so lightweight submodules such as
``gradiend.trainer.core.unified_schema`` can be imported without pulling in the
full text trainer, evaluator, and visualization stack.
"""

__all__ = [
    "Trainer",
    "MultiSeedTrainerView",
    "is_multi_seed_view",
    "SeedModelGroup",
    "TrainerSuite",
    "TrainerCollection",
    "PositiveTrainerSuite",
    "SymmetricTrainerSuite",
    "SuitePairDefinition",
    "PositiveFeatureDefinition",
    "set_seed",
    "TrainerConfig",
    "load_training_stats",
    "train_core",
    "TrainingArguments",
    "TransitionSpec",
    "pair",
    "identity",
    "expand_transition_selection",
    "PostPruneConfig",
    "PrePruneConfig",
    "LRSearchConfig",
    "post_prune",
    "pre_prune",
    "GradientTrainingDataset",
    "PreComputedTrainingDataset",
    "TextGradientTrainingDataset",
    "create_model_with_gradiend",
    "TrainingCallback",
    "EvaluationCallback",
    "CheckpointCallback",
    "NormalizationCallback",
    "LoggingCallback",
    "get_default_callbacks",
]

_LAZY_IMPORTS = {
    "Trainer": ("gradiend.trainer.trainer", "Trainer"),
    "set_seed": ("gradiend.trainer.trainer", "set_seed"),
    "MultiSeedTrainerView": ("gradiend.trainer.core.multi_seed", "MultiSeedTrainerView"),
    "is_multi_seed_view": ("gradiend.trainer.core.multi_seed", "is_multi_seed_view"),
    "SeedModelGroup": ("gradiend.trainer.core.seed_models", "SeedModelGroup"),
    "load_training_stats": ("gradiend.trainer.core.stats", "load_training_stats"),
    "train_core": ("gradiend.trainer.core.training", "train"),
    "TrainingCallback": ("gradiend.trainer.core.callbacks", "TrainingCallback"),
    "EvaluationCallback": ("gradiend.trainer.core.callbacks", "EvaluationCallback"),
    "CheckpointCallback": ("gradiend.trainer.core.callbacks", "CheckpointCallback"),
    "NormalizationCallback": ("gradiend.trainer.core.callbacks", "NormalizationCallback"),
    "LoggingCallback": ("gradiend.trainer.core.callbacks", "LoggingCallback"),
    "get_default_callbacks": ("gradiend.trainer.core.callbacks", "get_default_callbacks"),
    "TrainingArguments": ("gradiend.trainer.core.arguments", "TrainingArguments"),
    "TransitionSpec": ("gradiend.trainer.core.transition_selection", "TransitionSpec"),
    "pair": ("gradiend.trainer.core.transition_selection", "pair"),
    "identity": ("gradiend.trainer.core.transition_selection", "identity"),
    "expand_transition_selection": ("gradiend.trainer.core.transition_selection", "expand_transition_selection"),
    "TrainerConfig": ("gradiend.trainer.config", "TrainerConfig"),
    "PostPruneConfig": ("gradiend.trainer.core.pruning", "PostPruneConfig"),
    "PrePruneConfig": ("gradiend.trainer.core.pruning", "PrePruneConfig"),
    "LRSearchConfig": ("gradiend.trainer.core.lr_search", "LRSearchConfig"),
    "post_prune": ("gradiend.trainer.core.pruning", "post_prune"),
    "pre_prune": ("gradiend.trainer.core.pruning", "pre_prune"),
    "GradientTrainingDataset": ("gradiend.trainer.core.dataset", "GradientTrainingDataset"),
    "PreComputedTrainingDataset": ("gradiend.trainer.core.dataset", "PreComputedTrainingDataset"),
    "TextGradientTrainingDataset": ("gradiend.trainer.text.common.dataset", "TextGradientTrainingDataset"),
    "create_model_with_gradiend": ("gradiend.trainer.factory", "create_model_with_gradiend"),
    "TrainerSuite": ("gradiend.trainer.suite", "TrainerSuite"),
    "TrainerCollection": ("gradiend.trainer.suite", "TrainerCollection"),
    "PositiveTrainerSuite": ("gradiend.trainer.suite", "PositiveTrainerSuite"),
    "SymmetricTrainerSuite": ("gradiend.trainer.suite", "SymmetricTrainerSuite"),
    "SuitePairDefinition": ("gradiend.trainer.suite", "SuitePairDefinition"),
    "PositiveFeatureDefinition": ("gradiend.trainer.suite", "PositiveFeatureDefinition"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib

        module_name, attr_name = _LAZY_IMPORTS[name]
        value = getattr(importlib.import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
