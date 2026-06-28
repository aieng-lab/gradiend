import json
import shutil
import uuid
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.unified_data import UNIFIED_MASKED, UNIFIED_FACTUAL, UNIFIED_ALTERNATIVE
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer


class _DummyDecoderTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 103
    pad_token = ""
    eos_token = "[EOS]"

    def __call__(self, *args, **kwargs):
        return {
            "input_ids": [[101, 102, 0, 0]],
            "attention_mask": [[1, 1, 0, 0]],
        }


def test_create_training_data_loads_mlm_head_labels_from_run_experiment_dir():
    mock_objective = Mock()
    mock_objective.name = "clm_mlm_head"
    root = Path("codex_tmp_decoder_mlm_head_dataset") / uuid.uuid4().hex
    base_exp = root / "experiment"
    run_exp = base_exp / "sentiment_positive_negative"
    head_dir = run_exp / "decoder_mlm_head"
    head_dir.mkdir(parents=True)
    (head_dir / "config.json").write_text("{}", encoding="utf-8")
    (head_dir / "config_mlm_head.json").write_text(
        json.dumps({"target_labels": ["happy", "sad"]}),
        encoding="utf-8",
    )

    try:
        data = pd.DataFrame(
            [
                {
                    "masked": "I feel [MASK]",
                    "split": "train",
                    "label_class": "positive",
                    "label": "happy",
                    "alternative_class": "negative",
                    "alternative": "sad",
                },
                {
                    "masked": "I feel [MASK]",
                    "split": "train",
                    "label_class": "negative",
                    "label": "sad",
                    "alternative_class": "positive",
                    "alternative": "happy",
                },
            ]
        )
        config = TextPredictionConfig(
            data=data,
            target_classes=["positive", "negative"],
            masked_col="masked",
            split_col="split",
            run_id="sentiment_positive_negative",
        )
        trainer = TextPredictionTrainer(
            model="gpt2",
            config=config,
            args=TrainingArguments(
                experiment_dir=str(base_exp),
                prediction_objective="clm_mlm_head",
            ),
        )
        trainer._prediction_objective = lambda _tokenizer=None: mock_objective

        dataset = trainer.create_training_data(_DummyDecoderTokenizer(), split="train", batch_size=1)

        assert dataset.mlm_head_target_labels == ["happy", "sad"]
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_neutral_encoder_prediction_objective_falls_back_to_clm_next_token():
    assert TextPredictionTrainer._neutral_encoder_prediction_objective("clm_mlm_head") == "clm_next_token"
    assert TextPredictionTrainer._neutral_encoder_prediction_objective("clm_next_token") == "clm_next_token"
    assert TextPredictionTrainer._neutral_encoder_prediction_objective("mlm_mask_token") == "mlm_mask_token"


def test_neutral_training_masked_dataset_uses_clm_next_token_without_mlm_head_labels():
    df = pd.DataFrame(
        [
            {
                UNIFIED_MASKED: "Der [MASK] Mann",
                UNIFIED_FACTUAL: "alte",
                UNIFIED_ALTERNATIVE: "alte",
                "factual_id": "neutral",
                "alternative_id": "neutral",
                "label": 0,
                "feature_class_id": 0,
            }
        ]
    )
    tokenizer = _DummyDecoderTokenizer()
    dataset = TextTrainingDataset(
        df,
        tokenizer,
        batch_size=1,
        is_decoder_only_model=True,
        prediction_objective=TextPredictionTrainer._neutral_encoder_prediction_objective("clm_mlm_head"),
        target_key="label",
        balance_column="feature_class_id",
    )
    assert dataset.prediction_objective == "clm_next_token"
    assert dataset.mlm_head_target_labels is None


def test_decoder_mlm_training_data_includes_labels_from_other_splits():
    """MLM-head labels must cover tokens only seen in val/test (encoder eval uses validation)."""
    data = pd.DataFrame(
        [
            {
                "masked": "I feel [MASK]",
                "split": "train",
                "label_class": "positive",
                "label": "happy",
                "alternative_class": "negative",
                "alternative": "sad",
            },
            {
                "masked": "That is [MASK]",
                "split": "validation",
                "label_class": "positive",
                "label": "true",
                "alternative_class": "negative",
                "alternative": "fake",
            },
        ]
    )
    config = TextPredictionConfig(
        data=data,
        target_classes=["positive", "negative"],
        masked_col="masked",
        split_col="split",
    )
    trainer = TextPredictionTrainer(
        model="gpt2",
        config=config,
        args=TrainingArguments(experiment_dir="runs/test_mlm_label_coverage"),
    )
    train_df = trainer.get_decoder_mlm_training_data(split="train")
    labels = set(train_df["label"].astype(str).str.strip())
    assert "happy" in labels
    assert "true" in labels
    assert "sad" in labels
    assert "fake" in labels
