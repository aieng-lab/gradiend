from types import SimpleNamespace

import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.suite.collection import TrainerCollection
from tests.test_trainer_model import MockTrainerForTest


@pytest.fixture(scope="module")
def demo():
    import experiments.multilingual_gradiend_demo as mod

    return mod


def test_sentiment_suite_contains_only_full_sentiment_and_good_bad(demo, monkeypatch):
    full_trainer = MockTrainerForTest(
        model="mock-base",
        run_id="sentiment_positive_negative",
        args=TrainingArguments(experiment_dir=None),
    )
    good_bad_trainer = MockTrainerForTest(
        model="mock-base",
        run_id="sentiment_good_bad",
        args=TrainingArguments(experiment_dir=None),
    )
    full_suite = TrainerCollection(full_trainer, retain_models_in_memory=False)

    monkeypatch.setattr(
        demo,
        "_build_sentiment_full_lexicon_suite",
        lambda config, *, retain_models_in_memory: full_suite,
    )
    monkeypatch.setattr(
        demo,
        "_build_sentiment_good_bad_trainer",
        lambda config: good_bad_trainer,
    )

    suite = demo.build_sentiment_suite(
        SimpleNamespace(),
        retain_models_in_memory=False,
    )

    assert demo.SENTIMENT_GOOD_BAD_PAIR == ("good", "bad")
    assert demo.SENTIMENT_PRE_PRUNE_TOPK == 0.1
    assert demo.SENTIMENT_POST_PRUNE_TOPK == 0.01
    assert list(suite.trainers) == [
        "sentiment_positive_negative",
        "sentiment_good_bad",
    ]
