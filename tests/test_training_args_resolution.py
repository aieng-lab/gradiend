"""Arguments built from keyword overrides start from a complete ``TrainingArguments``.

``TrainingArguments.from_dict`` reads serialized records: a record without ``label_token_protocol`` was written
before the protocol existed and means ``no_leading_space``. User overrides are not a record. ``trainer.train(max_steps=100)``
(and the learning-rate probes, and ``gradiend.trainer.core.train(model, data, **kwargs)``) used to build the arguments
with ``from_dict(overrides)`` when no arguments were stored, so a run that only overrode a number trained with the
pre-0.2.6 label rule, while ``trainer.train()`` and ``TrainingArguments(...)`` used the new default.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.training import train as core_train
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from tests.test_trainer_model import MockTrainerForTest

NEW = "first_differing_token"
OLD = "no_leading_space"


class _Stop(BaseException):
    """Ends the run at the first consumer of the merged arguments (a BaseException, so no ``except Exception`` swallows it)."""


def _trainer(args=None):
    return MockTrainerForTest(model="mock-base", args=args)


# ----------------------------------------------------------------------------- the helper

def test_overrides_without_stored_arguments_use_the_current_default():
    args = _trainer(None)._resolve_training_args({"max_steps": 100})
    assert args.max_steps == 100
    assert args.label_token_protocol == NEW


def test_no_overrides_and_no_stored_arguments_are_the_defaults():
    assert _trainer(None)._resolve_training_args({}).to_dict() == TrainingArguments().to_dict()


def test_stored_arguments_keep_their_protocol_under_overrides():
    stored = TrainingArguments(label_token_protocol=OLD, max_steps=7)
    args = _trainer(stored)._resolve_training_args({"max_steps": 100})
    assert args.label_token_protocol == OLD
    assert args.max_steps == 100
    assert stored.max_steps == 7  # the stored instance is not modified


def test_an_explicit_override_of_the_protocol_wins():
    assert _trainer(None)._resolve_training_args({"label_token_protocol": OLD}).label_token_protocol == OLD
    stored = TrainingArguments(label_token_protocol=OLD)
    assert _trainer(stored)._resolve_training_args({"label_token_protocol": NEW}).label_token_protocol == NEW


def test_keys_that_are_not_arguments_are_ignored_as_before():
    args = _trainer(None)._resolve_training_args({"max_steps": 100, "not_an_argument": 1})
    assert args.max_steps == 100 and not hasattr(args, "not_an_argument")


def test_serialized_records_without_the_key_still_load_as_no_leading_space():
    record = TrainingArguments().to_dict()
    del record["label_token_protocol"]
    assert TrainingArguments.from_dict(record).label_token_protocol == OLD


# ----------------------------------------------------------------------------- through the real entry points

def test_generic_train_with_only_overrides_trains_under_the_new_default(tmp_path):
    trainer = _trainer(None)
    seen = []

    def stop(use_cache, output_dir, *, training_args=None, **kwargs):
        seen.append(training_args)
        raise _Stop

    with patch("gradiend.trainer.trainer.should_reuse_training_cache", side_effect=stop):
        with pytest.raises(_Stop):
            trainer.train(output_dir=str(tmp_path / 'model'), max_steps=100)
    assert [a.label_token_protocol for a in seen] == [NEW]
    assert seen[0].max_steps == 100


def test_generic_train_keeps_the_stored_protocol(tmp_path):
    trainer = _trainer(TrainingArguments(label_token_protocol=OLD))
    seen = []

    def stop(use_cache, output_dir, *, training_args=None, **kwargs):
        seen.append(training_args)
        raise _Stop

    with patch("gradiend.trainer.trainer.should_reuse_training_cache", side_effect=stop):
        with pytest.raises(_Stop):
            trainer.train(output_dir=str(tmp_path / 'model'), max_steps=100)
    assert seen[0].label_token_protocol == OLD


def test_core_train_without_arguments_uses_the_new_default_under_kwargs():
    seen = []

    def stop(training_args):
        seen.append(training_args)
        raise _Stop

    with patch("gradiend.trainer.core.training.get_default_callbacks", side_effect=stop):
        with pytest.raises(_Stop):
            core_train(MagicMock(), [object()], max_steps=100)
    assert seen[0].label_token_protocol == NEW and seen[0].max_steps == 100


@pytest.fixture(scope="module")
def text_trainer_parts():
    tok = AutoTokenizer.from_pretrained("gpt2")
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer, config.n_head, config.n_embd, config.vocab_size = 2, 2, 64, len(tok)
    model = AutoModelForCausalLM.from_config(config)
    model.tokenizer = tok
    data = pd.DataFrame([
        {"masked": "The word was [MASK]", "factual": "good", "alternative": "bad", "split": "train",
         "label_class": "POS", "alternative_class": "NEG"},
    ])
    cfg = TextPredictionConfig(
        target_classes=["POS", "NEG"], data=data, masked_col="masked", label_col="factual",
        alternative_col="alternative", label_class_col="label_class", alternative_class_col="alternative_class",
        split_col="split",
    )
    return model, cfg


def test_text_prediction_train_with_only_overrides_audits_and_trains_under_the_new_default(text_trainer_parts, tmp_path):
    model, cfg = text_trainer_parts
    trainer = TextPredictionTrainer(model=model, config=cfg)
    audited, trained = [], []

    def stop(use_cache, output_dir, *, training_args=None, **kwargs):
        trained.append(training_args)
        raise _Stop

    with patch.object(TextPredictionTrainer, "_check_and_filter_clm_alignment", lambda self, args=None, model=None: audited.append(args)):
        with patch("gradiend.trainer.trainer.should_reuse_training_cache", side_effect=stop):
            with pytest.raises(_Stop):
                trainer.train(output_dir=str(tmp_path / 'model'), max_steps=100)
    assert [a.label_token_protocol for a in audited] == [NEW]
    assert [a.label_token_protocol for a in trained] == [NEW]


def test_learning_rate_probes_of_a_trainer_without_stored_arguments_use_the_new_default(text_trainer_parts, tmp_path):
    """The probes call ``trainer.train(**overrides)`` with overrides only."""
    model, cfg = text_trainer_parts
    trainer = TextPredictionTrainer(model=model, config=cfg)
    trained = []

    def stop(use_cache, output_dir, *, training_args=None, **kwargs):
        trained.append(training_args)
        raise _Stop

    overrides = dict(learning_rate=1e-5, experiment_dir=str(tmp_path / 'probe'), max_seeds=1, use_cache=False, max_steps=150)
    with patch.object(TextPredictionTrainer, "_check_and_filter_clm_alignment", lambda self, args=None, model=None: None):
        with patch("gradiend.trainer.trainer.should_reuse_training_cache", side_effect=stop):
            with pytest.raises(_Stop):
                trainer.train(**overrides)
    assert trained[0].label_token_protocol == NEW


# ----------------------------------------------------------------------------- the dtype survives the merge

def test_a_configured_torch_dtype_survives_the_merge_with_overrides():
    """The merge copies the arguments through ``to_dict`` and ``from_dict``; a bfloat16 run must not become float32."""
    stored = TrainingArguments(torch_dtype=torch.bfloat16)
    assert _trainer(stored)._resolve_training_args({}).torch_dtype == torch.bfloat16
    assert _trainer(stored)._resolve_training_args({"max_steps": 100}).torch_dtype == torch.bfloat16
    # an override of the dtype itself is applied, also on top of the default arguments
    assert _trainer(None)._resolve_training_args({"torch_dtype": torch.float16}).torch_dtype == torch.float16
    assert TrainingArguments()._with_overrides({"torch_dtype": torch.bfloat16}).torch_dtype == torch.bfloat16
