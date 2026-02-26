"""
Test that the standard workflow (train, evaluate_encoder, evaluate_decoder) works
for an MLM (encoder) model, including when using pre_prune_config (lazy_init).

Uses mocked model/tokenizer to avoid loading BERT and keep tests fast and light on RAM.
"""

import os
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from gradiend import (
    TextFilterConfig,
    TextPredictionDataCreator,
    TrainingArguments,
    TextPredictionTrainer,
    PrePruneConfig,
)
from tests.conftest import SimpleMockModel, MockTokenizer

MINI_TEXTS = [
    "The chef tasted the soup, then he added a pinch of pepper.",
    "She opened the window and watched the leaves fall.",
    "The dog ran to the door; it wanted to go outside.",
    "The mechanic wiped his hands and said the car would be ready.",
    "The gardener pruned the roses and he left the cuttings by the gate.",
    "The players huddled on the pitch before they ran back.",
    "The committee members met on Tuesday and they voted.",
    "The volunteers packed the boxes and said they would load the van.",
    "The staff members finished the inventory and they reported the count.",
    "The panel members discussed the proposal and they reached a consensus.",
    "The report will be ready for review by the end of the week.",
    "The museum opens at nine and closes at six on weekdays.",
]


def _make_mock_load_model(mock_model, mock_tokenizer):
    """Return a _load_model classmethod that returns (mock_model, mock_tokenizer) to avoid HF loading."""

    def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
        return (mock_model, mock_tokenizer)

    return classmethod(mock_load_model)


@pytest.mark.parametrize("use_pre_prune", [True, False], ids=["with_pre_prune", "without_pre_prune"])
def test_standard_workflow_mlm_train_encoder_decoder(use_pre_prune):
    """
    Full workflow: train -> evaluate_encoder -> evaluate_decoder.

    With use_pre_prune: exercises lazy_init path (pre_prune_config).
    Without: exercises non-lazy path. Uses mocked model so no BERT load or heavy RAM.
    """
    mock_model = SimpleMockModel(name_or_path="mock-model", dtype=torch.float32)
    mock_tokenizer = MockTokenizer(vocab_size=1000)

    creator = TextPredictionDataCreator(
        base_data=MINI_TEXTS,
        feature_targets=[
            TextFilterConfig(targets=["he", "she", "it"], id="3SG"),
            TextFilterConfig(targets=["they"], id="3PL"),
        ],
    )
    training = creator.generate_training_data(
        max_size_per_class=15,
        min_rows_per_class_for_split=0,
    )
    neutral = creator.generate_neutral_data(
        additional_excluded_words=["i", "we", "you", "he", "she", "it", "they"],
        max_size=10,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        args = TrainingArguments(
            train_batch_size=1,
            eval_steps=2,
            num_train_epochs=1,
            max_steps=4,
            learning_rate=1e-4,
            experiment_dir=os.path.join(tmpdir, "workflow_test"),
            use_cache=False,
            do_eval=False,
            pre_prune_config=PrePruneConfig(n_samples=4, topk=0.5) if use_pre_prune else None,
        )
        with patch(
            "gradiend.trainer.text.common.model_base.TextModelWithGradiend._load_model",
            _make_mock_load_model(mock_model, mock_tokenizer),
        ):
            trainer = TextPredictionTrainer(
                model="bert-base-uncased",
                data=training,
                eval_neutral_data=neutral,
                max_counterfactuals_per_sentence=1,
                args=args,
            )
            trainer.train()
            enc_result = trainer.evaluate_encoder(plot=False)
            assert "correlation" in enc_result
            dec = trainer.evaluate_decoder(plot=False, target_class="3SG")
            assert "3SG" in dec
            assert "value" in dec["3SG"]
            if use_pre_prune:
                changed = trainer.rewrite_base_model(decoder_results=dec, target_class="3SG")
                assert changed is not None
