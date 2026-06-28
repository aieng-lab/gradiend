"""Tests for neutral encoder mask-target exclusion (training masked + dataset)."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest
import torch

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.dataset import create_masked_pair_from_text
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from tests.conftest import MockTokenizer


def _make_trainer(**config_overrides) -> TextPredictionTrainer:
    trainer = object.__new__(TextPredictionTrainer)
    trainer._training_args = TrainingArguments(experiment_dir="runs/test_neutral_exclusions", seed=0)
    mock_objective = Mock()
    mock_objective.name = "mlm"
    trainer._prediction_objective = lambda _tokenizer=None: mock_objective
    cfg = TextPredictionConfig(
        data=pd.DataFrame(
            {
                "masked": ["The chef [MASK] added pepper"],
                "factual_class": ["3SG"],
                "alternative_class": ["3PL"],
                "factual": ["he"],
                "alternative": ["they"],
                "transition": ["3SG->3PL"],
                "split": ["test"],
            }
        ),
        target_classes=["3SG", "3PL"],
        masked_col="masked",
        split_col="split",
        eval_neutral_additional_excluded_words=["you", "we"],
    )
    for key, value in config_overrides.items():
        setattr(cfg, key, value)
    trainer.config = cfg
    trainer.run_id = "test"
    trainer._ensure_data = lambda: None
    trainer._infer_decoder_eval_targets = lambda: (
        {"3SG": ["he", "she", "it"], "3PL": ["they"]},
        False,
    )
    return trainer


def test_resolve_encoder_neutral_excluded_tokens_includes_targets_and_additional():
    trainer = _make_trainer()
    excluded = trainer._resolve_encoder_neutral_excluded_tokens()
    lowered = {w.lower() for w in excluded}
    assert "he" in lowered
    assert "they" in lowered
    assert "you" in lowered
    assert "we" in lowered


def test_create_masked_pair_from_text_skips_excluded_tokens():
    tokenizer = MockTokenizer()
    pair = create_masked_pair_from_text(
        "The chef he added pepper",
        tokenizer,
        is_decoder_only_model=False,
        excluded_tokens=["he", "they", "pepper"],
        mask_token="[MASK]",
    )
    assert pair is not None
    _masked_text, target_token = pair
    assert target_token.lower() not in {"he", "they", "pepper"}


def test_neutral_training_masked_decoder_only_remasks_instead_of_using_factual_token():
    trainer = _make_trainer()
    trainer._training_args = TrainingArguments(
        experiment_dir="runs/test_neutral_exclusions",
        prediction_objective="clm_mlm_head",
        seed=0,
    )
    mock_objective = Mock()
    mock_objective.name = "clm_mlm_head"
    trainer._prediction_objective = lambda _tokenizer=None: mock_objective

    class _Gpt2StyleTokenizer(MockTokenizer):
        mask_token = None
        mask_token_id = None

    tokenizer = _Gpt2StyleTokenizer()
    model = Mock()
    model.tokenizer = tokenizer
    model.base_model = tokenizer
    model.is_seq2seq_model = False
    model.gradiend = Mock(torch_dtype=torch.float32, device_encoder=torch.device("cpu"))
    model.forward_clm_gradients = Mock(return_value=torch.randn(16, dtype=torch.float32))
    model.gradient_creator = Mock()
    model.encode = lambda _grad, **kwargs: 0.1

    with patch(
        "gradiend.trainer.text.prediction.trainer.create_masked_pair_from_text",
        wraps=create_masked_pair_from_text,
    ) as masked_pair:
        rows = trainer._encode_neutral_training_masked_rows(
            model,
            [
                {
                    "template": "The chef [MASK] added pepper quickly",
                    "factual_token": "he",
                    "alternative_token": "they",
                }
            ],
            excluded_tokens=["he", "they", "you", "we"],
            factual_token_key="factual_token",
            alternative_token_key="alternative_token",
            max_size=1,
            torch_dtype=torch.float32,
            device=torch.device("cpu"),
        )
    assert len(rows) == 1
    assert masked_pair.call_count == 1
    assert rows[0]["masked"] is not None
    assert "[MASK]" in str(rows[0]["masked"])
    assert rows[0]["factual_token"] is not None
    assert rows[0]["source_token"] == rows[0]["factual_token"]


def test_neutral_training_masked_skips_when_only_excluded_tokens_remain():
    trainer = _make_trainer()
    tokenizer = MockTokenizer()
    model = Mock()
    model.tokenizer = tokenizer
    model.base_model = tokenizer
    model.is_seq2seq_model = False
    model.gradiend = Mock(torch_dtype=torch.float32, device_encoder=torch.device("cpu"))
    model.gradient_creator = Mock(return_value=torch.randn(16, dtype=torch.float32))
    model.encode = lambda _grad, **kwargs: 0.0

    rows = trainer._encode_neutral_training_masked_rows(
        model,
        [
            {
                "template": "[MASK] they",
                "factual_token": "he",
                "alternative_token": "they",
            }
        ],
        excluded_tokens=["he", "they"],
        factual_token_key="factual_token",
        alternative_token_key="alternative_token",
        max_size=1,
        torch_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert rows == []
