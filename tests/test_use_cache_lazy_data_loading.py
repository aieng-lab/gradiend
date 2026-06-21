import json
import os
from types import SimpleNamespace

import pandas as pd

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.trainer.core.arguments import TrainingArguments as CoreTrainingArguments
from gradiend.trainer.text.classification.config import TextClassificationConfig
from gradiend.trainer.text.classification.trainer import TextClassificationTrainer
import gradiend.trainer.trainer as trainer_module


def _patch_training_cache_hit(monkeypatch) -> None:
    """Train() checks should_reuse_training_cache, not trainer.has_saved_model directly."""
    monkeypatch.setattr(
        trainer_module,
        "should_reuse_training_cache",
        lambda *args, **kwargs: True,
    )


class _DummyPredictionTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 1
    pad_token = "[PAD]"
    pad_token_id = 0
    eos_token = "[EOS]"


class _DummyClassificationTokenizer:
    pad_token_id = 0


def _prediction_data():
    return {
        "3SG": pd.DataFrame(
            {
                "masked": ["[MASK] went home", "[MASK] is home", "[MASK] stayed home"],
                "split": ["train", "validation", "test"],
                "3SG": ["he", "he", "he"],
            }
        ),
        "3PL": pd.DataFrame(
            {
                "masked": ["[MASK] went home", "[MASK] are home", "[MASK] stayed home"],
                "split": ["train", "validation", "test"],
                "3PL": ["they", "they", "they"],
            }
        ),
    }


def _classification_data():
    return pd.DataFrame(
        [
            {"text": "he went home", "label": "3SG", "split": "train"},
            {"text": "they went home", "label": "3PL", "split": "train"},
            {"text": "he stayed home", "label": "3SG", "split": "test"},
            {"text": "they stayed home", "label": "3PL", "split": "test"},
        ]
    )


def _make_prediction_trainer(experiment_dir: str) -> TextPredictionTrainer:
    args = TrainingArguments(
        experiment_dir=experiment_dir,
        use_cache=True,
        train_batch_size=2,
        max_steps=2,
    )
    return TextPredictionTrainer(
        model="bert-base-uncased",
        data=_prediction_data(),
        target_classes=["3SG", "3PL"],
        args=args,
        use_class_names_as_columns=True,
    )


def _make_classification_trainer(experiment_dir: str) -> TextClassificationTrainer:
    args = CoreTrainingArguments(
        experiment_dir=experiment_dir,
        use_cache=True,
        train_batch_size=2,
        max_steps=2,
    )
    config = TextClassificationConfig(
        data=_classification_data(),
        target_classes=["3SG", "3PL"],
    )
    return TextClassificationTrainer(
        model="bert-base-uncased",
        args=args,
        config=config,
    )


def test_prediction_cached_train_defers_data_loading(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")

    trainer.train()

    assert trainer._last_train_used_cache is True
    assert trainer._data_loaded is False


def test_prediction_evaluate_encoder_keeps_data_unloaded_when_encoder_df_is_provided(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer.train()
    trainer._evaluator = SimpleNamespace(
        evaluate_encoder=lambda **kwargs: {"n_samples": len(kwargs["encoder_df"])}
    )

    result = trainer.evaluate_encoder(
        encoder_df=pd.DataFrame({"encoded": [0.1], "label": [1.0], "type": ["training"]}),
        use_cache=True,
    )

    assert result["n_samples"] == 1
    assert trainer._data_loaded is False


def test_prediction_evaluate_encoder_lazy_loads_data_when_needed(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer.train()
    trainer._evaluator = SimpleNamespace(evaluate_encoder=lambda **kwargs: {"ok": True})
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyPredictionTokenizer())

    def _fake_analyze_encoder(model_with_gradiend, **kwargs):
        assert trainer._data_loaded is False
        trainer.create_training_data(
            model_with_gradiend.tokenizer,
            split="test",
            batch_size=1,
            max_size=1,
            balance_column=None,
        )
        assert trainer._data_loaded is True
        return pd.DataFrame({"encoded": [0.1], "label": [1.0], "type": ["training"]})

    trainer._analyze_encoder = _fake_analyze_encoder

    trainer.evaluate_encoder(use_cache=False)

    assert trainer._data_loaded is True


def test_prediction_evaluate_decoder_keeps_data_unloaded_when_frames_are_supplied(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer.train()
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyPredictionTokenizer())
    trainer._evaluator = SimpleNamespace(
        evaluate_decoder=lambda **kwargs: {
            "training_rows": len(kwargs["training_like_df"]),
            "neutral_rows": len(kwargs["neutral_df"]),
        }
    )

    result = trainer.evaluate_decoder(
        training_like_df=pd.DataFrame({"masked": ["[MASK] went home"], "factual": ["he"], "alternative": ["they"]}),
        neutral_df=pd.DataFrame({"text": ["he went home"]}),
        use_cache=False,
    )

    assert result["training_rows"] == 1
    assert result["neutral_rows"] == 1
    assert trainer._data_loaded is False


def test_prediction_evaluate_decoder_lazy_loads_data_when_needed(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_prediction_trainer("prediction-cache-exp")
    trainer.train()
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyPredictionTokenizer())

    def _fake_evaluate_decoder(**kwargs):
        assert trainer._data_loaded is False
        training_like_df, neutral_df = trainer._get_decoder_eval_dataframe(
            _DummyPredictionTokenizer(),
            cached_training_like_df=kwargs.get("training_like_df"),
            cached_neutral_df=kwargs.get("neutral_df"),
        )
        assert trainer._data_loaded is True
        return {
            "training_rows": len(training_like_df),
            "neutral_rows": len(neutral_df),
        }

    trainer._evaluator = SimpleNamespace(evaluate_decoder=_fake_evaluate_decoder)

    result = trainer.evaluate_decoder(use_cache=False)

    assert result["training_rows"] >= 1
    assert result["neutral_rows"] >= 1
    assert trainer._data_loaded is True


def test_classification_cached_train_defers_data_loading(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")

    trainer.train()

    assert trainer._last_train_used_cache is True
    assert trainer._combined_data is None


def test_classification_evaluate_encoder_keeps_data_unloaded_when_encoder_df_is_provided(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")
    trainer.train()
    trainer._evaluator = SimpleNamespace(
        evaluate_encoder=lambda **kwargs: {"n_samples": len(kwargs["encoder_df"])}
    )

    result = trainer.evaluate_encoder(
        encoder_df=pd.DataFrame({"encoded": [0.1], "label": [1.0], "type": ["training"]}),
        use_cache=True,
    )

    assert result["n_samples"] == 1
    assert trainer._combined_data is None


def test_classification_evaluate_encoder_lazy_loads_data_when_needed(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")
    trainer.train()
    trainer._evaluator = SimpleNamespace(evaluate_encoder=lambda **kwargs: {"ok": True})
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyClassificationTokenizer())

    def _fake_analyze_encoder(model_with_gradiend, **kwargs):
        assert trainer._combined_data is None
        trainer.create_training_data(model_with_gradiend.tokenizer, split="test", batch_size=1, max_size=1)
        assert trainer._combined_data is not None
        return pd.DataFrame({"encoded": [0.1], "label": [1.0], "type": ["training"]})

    trainer._analyze_encoder = _fake_analyze_encoder

    trainer.evaluate_encoder(use_cache=False)

    assert trainer._combined_data is not None


def test_classification_evaluate_decoder_keeps_data_unloaded_when_frames_are_supplied(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")
    trainer.train()
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyClassificationTokenizer())
    trainer._evaluator = SimpleNamespace(
        evaluate_decoder=lambda **kwargs: {
            "training_rows": len(kwargs["training_like_df"]),
            "neutral_rows": len(kwargs["neutral_df"]),
        }
    )

    result = trainer.evaluate_decoder(
        training_like_df=pd.DataFrame({"text": ["he stayed home"], "label_class": ["3SG"]}),
        neutral_df=pd.DataFrame({"text": ["they stayed home"]}),
        use_cache=False,
    )

    assert result["training_rows"] == 1
    assert result["neutral_rows"] == 1
    assert trainer._combined_data is None


def test_classification_evaluate_decoder_lazy_loads_data_when_needed(monkeypatch):
    _patch_training_cache_hit(monkeypatch)
    trainer = _make_classification_trainer("classification-cache-exp")
    trainer.train()
    trainer.get_model = lambda: SimpleNamespace(tokenizer=_DummyClassificationTokenizer())

    def _fake_evaluate_decoder(**kwargs):
        assert trainer._combined_data is None
        training_like_df, neutral_df = trainer._get_decoder_eval_dataframe(
            _DummyClassificationTokenizer(),
            cached_training_like_df=kwargs.get("training_like_df"),
            cached_neutral_df=kwargs.get("neutral_df"),
        )
        assert trainer._combined_data is not None
        return {
            "training_rows": len(training_like_df),
            "neutral_rows": len(neutral_df),
        }

    trainer._evaluator = SimpleNamespace(evaluate_decoder=_fake_evaluate_decoder)

    result = trainer.evaluate_decoder(use_cache=False)

    assert result["training_rows"] >= 1
    assert result["neutral_rows"] >= 1
    assert trainer._combined_data is not None
