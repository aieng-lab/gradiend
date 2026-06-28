"""Per-problem learning rates for multilingual_gradiend_demo."""

import argparse
import sys

import pytest

from gradiend import TrainerCollection


@pytest.fixture(scope="module")
def demo():
    import experiments.multilingual_gradiend_demo as mod

    return mod


def _config(demo, **overrides):
    monkeypatch_argv = overrides.pop("_argv", ["multilingual_gradiend_demo.py"])
    import sys as _sys

    old_argv = _sys.argv
    _sys.argv = monkeypatch_argv
    try:
        cli = demo.parse_args()
        ns = {**vars(cli), **overrides}
        return demo.build_experiment_config(argparse.Namespace(**ns))
    finally:
        _sys.argv = old_argv


def test_learning_rate_for_problem_uses_decoder_table(demo):
    config = _config(demo, decoder_eval_mode="mlm_head", problem_learning_rate=[])
    assert config.decoder_eval_mode is demo.DecoderEvalMode.MLM_HEAD
    assert demo.learning_rate_for_problem(config, "pronoun") == demo.DECODER_LEARNING_RATE_BY_PROBLEM["pronoun"]
    assert demo.learning_rate_for_problem(config, "gender_en") == demo.DECODER_LEARNING_RATE_BY_PROBLEM["gender_en"]


def test_learning_rate_for_problem_uses_encoder_table(demo):
    config = _config(demo, decoder_eval_mode="none", problem_learning_rate=[])
    assert demo.learning_rate_for_problem(config, "pronoun") == demo.ENCODER_LEARNING_RATE_BY_PROBLEM["pronoun"]


def test_cli_problem_learning_rate_override(demo):
    config = _config(
        demo,
        _argv=["multilingual_gradiend_demo.py", "--problem-learning-rate", "pronoun=3e-5"],
        decoder_eval_mode="mlm_head",
    )
    assert demo.learning_rate_for_problem(config, "pronoun") == pytest.approx(3e-5)
    assert demo.learning_rate_for_problem(config, "sentiment") == demo.DECODER_LEARNING_RATE_BY_PROBLEM["sentiment"]


def test_parse_problem_learning_rate_rejects_unknown_problem(demo):
    with pytest.raises(ValueError, match="Unknown problem"):
        demo._parse_problem_learning_rate_overrides(["unknown=1e-4"])


def test_trainer_collection_unloads_after_uncached_train_when_not_retaining():
    class DummyTrainer:
        def __init__(self):
            self.run_id = "child"
            self.trained_with_use_cache = None
            self._last_train_used_cache = False
            self.unloaded = False

        def train(self, *, use_cache=True):
            self.trained_with_use_cache = use_cache

        def unload_model(self):
            self.unloaded = True

    trainer = DummyTrainer()
    suite = TrainerCollection(trainer, retain_models_in_memory=False)

    suite.train(use_cache=False)

    assert trainer.trained_with_use_cache is False
    assert trainer.unloaded is True
