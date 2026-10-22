"""The encoder rows of text training data keep the masked template (shown in interactive plots)."""

from types import SimpleNamespace

from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer


def test_training_encoder_rows_keep_masked_template(monkeypatch):
    entry = {
        "encoded": 0.9,
        "label": 1.0,
        "source_id": "BE",
        "target_id": "BE",
        "feature_class_id": 1,
        "text": "We met at the center today.",
        "template": "We met at the [MASK] today.",
        "factual_token": "center",
        "alternative_token": "centre",
        "source_token": "centre",
        "data_split": "train",
    }
    evaluator = SimpleNamespace(evaluate_encoder=lambda **kwargs: {"training_rows": [entry]})
    monkeypatch.setattr(TextPredictionTrainer, "evaluator", property(lambda self: evaluator))
    trainer = object.__new__(TextPredictionTrainer)
    monkeypatch.setattr(TextPredictionTrainer, "eval_group", property(lambda self: "feature_class_id"))
    monkeypatch.setattr(TextPredictionTrainer, "_default_from_training_args", lambda self, *args, **kwargs: False)

    rows = trainer._encode_training_rows(None, [entry], "alternative", None, {"split": "test"})

    assert rows[0]["template"] == "We met at the [MASK] today."
    assert rows[0]["text"] == "We met at the center today."
    assert rows[0]["source_token"] == "centre"
