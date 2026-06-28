"""Regression tests: neutral encoder rows under clm_mlm_head must not require MLM-head labels."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import torch

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from gradiend.trainer.text.prediction.dataset import create_masked_pair_from_text
from tests.conftest import MockTokenizer


class _Gpt2StyleTokenizer(MockTokenizer):
    """Decoder-only tokenizer stub (no native [MASK] token)."""

    mask_token = None
    mask_token_id = None


def _make_clm_mlm_head_trainer() -> TextPredictionTrainer:
    trainer = object.__new__(TextPredictionTrainer)
    trainer._training_args = TrainingArguments(
        experiment_dir="runs/test_decoder_mlm_head_neutral",
        prediction_objective="clm_mlm_head",
        seed=0,
    )
    mock_objective = Mock()
    mock_objective.name = "clm_mlm_head"
    trainer._prediction_objective = lambda _tokenizer=None: mock_objective
    trainer.config = TextPredictionConfig(
        data=pd.DataFrame(),
        target_classes=["masc_nom", "fem_nom"],
        masked_col="masked",
        split_col="split",
    )
    return trainer


def _make_model_with_gradiend(tokenizer: MockTokenizer) -> Mock:
    gradiend = Mock()
    gradiend.torch_dtype = torch.float32
    gradiend.device_encoder = torch.device("cpu")

    model = Mock()
    model.tokenizer = tokenizer
    model.base_model = tokenizer
    model.is_seq2seq_model = False
    model.gradiend = gradiend
    model.forward_clm_gradients = Mock(return_value=torch.randn(16, dtype=torch.float32))
    model.gradient_creator = Mock(side_effect=AssertionError("MLM-head gradient_creator must not be used"))
    model.encode = lambda _grad, **kwargs: 0.42 if kwargs.get("return_float") else torch.tensor(0.42)
    return model


def test_neutral_encoder_gradient_creator_uses_forward_clm_gradients():
    trainer = _make_clm_mlm_head_trainer()
    model = Mock()
    model.forward_clm_gradients = Mock()
    model.gradient_creator = Mock()

    assert trainer._neutral_encoder_gradient_creator(model) is model.forward_clm_gradients


def test_neutral_encoder_gradient_creator_keeps_default_for_other_objectives():
    trainer = _make_clm_mlm_head_trainer()
    mock_objective = Mock()
    mock_objective.name = "clm_next_token"
    trainer._prediction_objective = lambda _tokenizer=None: mock_objective
    model = Mock()
    model.forward_clm_gradients = Mock()
    model.gradient_creator = Mock()

    assert trainer._neutral_encoder_gradient_creator(model) is model.gradient_creator


@pytest.fixture
def train_eval_entries():
    return [
        {
            "template": "Der [MASK] Mann läuft schnell",
            "factual_token": "große",
            "alternative_token": "kleine",
        },
        {
            "template": "Die [MASK] Frau sitzt ruhig",
            "factual_token": "alte",
            "alternative_token": "junge",
        },
    ]


def test_encode_neutral_training_masked_rows_under_clm_mlm_head(train_eval_entries):
    trainer = _make_clm_mlm_head_trainer()
    tokenizer = _Gpt2StyleTokenizer()
    model = _make_model_with_gradiend(tokenizer)

    with patch(
        "gradiend.trainer.text.prediction.trainer.create_masked_pair_from_text",
        wraps=create_masked_pair_from_text,
    ) as masked_pair:
        rows = trainer._encode_neutral_training_masked_rows(
            model,
            train_eval_entries,
            excluded_tokens=["der", "die"],
            factual_token_key="factual_token",
            alternative_token_key="alternative_token",
            max_size=2,
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
        )

    assert masked_pair.call_count == 2
    assert len(rows) == 2
    assert all(row["type"] == "neutral_training_masked" for row in rows)
    assert all(isinstance(row["encoded"], float) for row in rows)
    assert model.forward_clm_gradients.call_count == 2
    model.gradient_creator.assert_not_called()


def test_encode_neutral_dataset_rows_under_clm_mlm_head():
    trainer = _make_clm_mlm_head_trainer()
    tokenizer = _Gpt2StyleTokenizer()
    model = _make_model_with_gradiend(tokenizer)
    neutral_df = pd.DataFrame(
        {
            "text": [
                "Der Hund schläft auf dem Sofa heute Abend.",
                "Die Katze springt über den Zaun im Garten.",
            ],
        }
    )

    rows = trainer._encode_neutral_dataset_rows(
        model,
        neutral_df,
        encoder_kwargs={"text_col": "text"},
        masked_col_name="masked",
        excluded_tokens=["der", "die"],
        max_size=2,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert len(rows) == 2
    assert all(row["type"] == "neutral_dataset" for row in rows)
    assert all(isinstance(row["encoded"], float) for row in rows)
    assert model.forward_clm_gradients.call_count == 2
    model.gradient_creator.assert_not_called()
