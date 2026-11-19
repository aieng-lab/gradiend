"""Every model the generic trainer builds is handed to ``_configure_model`` right after the factory.

``Trainer`` builds its training models with ``gradiend.trainer.factory.create_model_with_gradiend``, not
with ``FeatureLearningDefinition.create_model_with_gradiend`` (which calls the hook itself). The hook is
where ``TextPredictionTrainer`` gives a model the label-token rule it must be scored with, so each factory
call in the trainer has to be followed by an explicit ``self._configure_model(model)``.
"""

from unittest.mock import MagicMock, patch

import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.feature_definition import FeatureLearningDefinition
from gradiend.trainer.core.pruning import PrePruneConfig
from tests.test_trainer_model import MockModelWithGradiendForTest, MockTrainerForTest


class _Stop(BaseException):
    """Ends the run right after the hook was called (a BaseException, so no ``except Exception`` swallows it)."""


class _RecordingTrainer(MockTrainerForTest):
    """Records every model passed to the hook; optionally stops the run there."""

    stop_after_hook = False

    def _configure_model(self, model):
        self.configured.append(model)
        if self.stop_after_hook:
            raise _Stop


def _trainer(temp_dir, *, stop_after_hook, **args):
    trainer = _RecordingTrainer(model="mock-base", args=TrainingArguments(experiment_dir=temp_dir, **args))
    trainer.configured = []
    trainer.stop_after_hook = stop_after_hook
    return trainer


def test_the_default_hook_changes_nothing():
    model = MagicMock()
    before = dict(vars(model))
    assert FeatureLearningDefinition._configure_model(object.__new__(MockTrainerForTest), model) is None
    assert dict(vars(model)) == before


def test_single_run_training_configures_the_model_it_creates(temp_dir):
    trainer = _trainer(temp_dir, stop_after_hook=True)
    created = MagicMock(name="created")
    with patch("gradiend.trainer.trainer.create_model_with_gradiend", return_value=created) as factory:
        with pytest.raises(_Stop):
            trainer._train(
                output_dir=temp_dir,
                args=trainer._training_args,
                model="mock-base",
                model_with_gradiend_cls=MockModelWithGradiendForTest,
                callbacks=None,
            )
    factory.assert_called_once()
    assert trainer.configured == [created]


def test_pre_prune_model_recreation_configures_the_new_model(temp_dir):
    trainer = _trainer(temp_dir, stop_after_hook=False, pre_prune_config=PrePruneConfig(n_samples=2, topk=0.5))
    eager = MagicMock()
    eager.gradiend._lazy_init = False
    lazy = MagicMock()
    lazy.gradiend._lazy_init = True
    with patch("gradiend.trainer.trainer.create_model_with_gradiend", return_value=lazy):
        result = trainer._prepare_model_for_pre_prune_if_needed(eager, trainer._training_args)
    assert result is lazy
    assert trainer.configured == [lazy]


def test_a_model_that_needs_no_recreation_is_not_configured_again(temp_dir):
    trainer = _trainer(temp_dir, stop_after_hook=False, pre_prune_config=PrePruneConfig(n_samples=2, topk=0.5))
    lazy = MagicMock()
    lazy.gradiend._lazy_init = True
    with patch("gradiend.trainer.trainer.create_model_with_gradiend") as factory:
        assert trainer._prepare_model_for_pre_prune_if_needed(lazy, trainer._training_args) is lazy
    factory.assert_not_called()
    assert trainer.configured == []


def test_multi_seed_pre_prune_configures_the_model_it_creates(temp_dir):
    trainer = _trainer(
        temp_dir,
        stop_after_hook=True,
        max_seeds=2,
        use_cache=False,
        pre_prune_config=PrePruneConfig(n_samples=2, topk=0.5),
    )
    created = MagicMock(name="created")
    with patch("gradiend.trainer.trainer.create_model_with_gradiend", return_value=created) as factory:
        with pytest.raises(_Stop):
            trainer.train(model="mock-base", model_with_gradiend_cls=MockModelWithGradiendForTest)
    factory.assert_called_once()
    assert trainer.configured == [created]
