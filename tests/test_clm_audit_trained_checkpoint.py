"""The CLM alignment audit follows the base model of a trained checkpoint.

After ``Trainer.train()`` the trainer's ``_model_arg`` is the saved GRADIEND checkpoint (and so it is for every seed
while the seeds are evaluated). A checkpoint directory is not a Hugging Face model directory: it has no ``model_type``
and no tokenizer, only the id of the base model it was trained on. The audit must therefore take its objective and its
tokenizer from that base model:

- a masked LM (BERT) needs no audit at all and must not log that an audit was skipped;
- a decoder-only model (GPT-2) under ``clm_next_token`` is audited before training, is not audited again for the
  checkpoint it produced, and is audited for real (not skipped with a warning) when a trainer starts from a checkpoint.
"""

import contextlib
import json
import logging
import os
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from transformers import AutoTokenizer, BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel

from gradiend import PostPruneConfig, PrePruneConfig, TextPredictionConfig, TextPredictionTrainer, TrainingArguments
from gradiend.model.model_with_gradiend import _is_gradiend_checkpoint
from gradiend.trainer.text.prediction import clm_alignment
from gradiend.trainer.text.prediction.clm_alignment import ClmAlignmentError

FDT = "first_differing_token"
SKIPPED = "alignment audit skipped"


# ----------------------------------------------------------------------------- tiny models, data, helpers

def _save_tiny_model(directory, kind):
    torch.manual_seed(0)
    if kind == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = BertForMaskedLM(BertConfig(
            vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
        ))
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_embd=32, n_layer=2, n_head=2))
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return str(directory)


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    return _save_tiny_model(tmp_path_factory.mktemp("tiny_bert"), "bert")


@pytest.fixture(scope="module")
def tiny_gpt2(tmp_path_factory):
    return _save_tiny_model(tmp_path_factory.mktemp("tiny_gpt2"), "gpt2")


def _pronoun_data():
    templates = [
        "[MASK] was very happy today", "[MASK] likes to read books", "yesterday [MASK] went to the market",
        "[MASK] has a small garden", "I think [MASK] will come later", "[MASK] always arrives on time",
        "after lunch [MASK] fell asleep", "[MASK] is looking for a new job",
    ]
    singular = ["he", "she", "it"]
    rows = []
    for i in range(300):
        split = "train" if i < 240 else ("validation" if i < 270 else "test")
        label_class = "3SG" if i % 2 == 0 else "3PL"
        label = singular[(i // 2) % 3] if label_class == "3SG" else "they"
        rows.append({"masked": templates[i % len(templates)], "label": label, "label_class": label_class, "split": split})
    neutral = pd.DataFrame({"text": ["The weather is nice", "Books are on the table", "A garden needs water"] * 10})
    return pd.DataFrame(rows), neutral


class _WarningCollector(logging.Handler):
    """Collects the text of every warning logged while it is attached to the root logger."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _watching():
    """Yields ``(warnings, audited_row_counts)``: what is logged, and the size of every audit that actually runs."""
    collector = _WarningCollector()
    audited = []
    original = clm_alignment.audit_clm_alignment

    def counting(tokenizer, data, *args, **kwargs):
        audited.append(len(data))
        return original(tokenizer, data, *args, **kwargs)

    root = logging.getLogger()
    root.addHandler(collector)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(clm_alignment, "audit_clm_alignment", counting)
            yield collector.messages, audited
    finally:
        root.removeHandler(collector)


def _train_and_evaluate(model_dir, experiment_dir):
    """The paper flow in miniature: two seeds, pre- and post-pruning, then the encoder analysis of the trained model."""
    data, neutral = _pronoun_data()
    config = TextPredictionConfig(run_id="pronouns", data=data, target_classes=["3SG", "3PL"], eval_neutral_data=neutral)
    args = TrainingArguments(
        experiment_dir=str(experiment_dir), train_batch_size=4, eval_steps=5, max_steps=10, eval_batch_size=4,
        learning_rate=1e-3, source="alternative", target="diff", max_seeds=2, fail_on_non_convergence=False,
        use_cache=False,
        pre_prune_config=PrePruneConfig(n_samples=8, topk=0.2), post_prune_config=PostPruneConfig(topk=0.2),
    )
    trainer = TextPredictionTrainer(model=model_dir, config=config, args=args)
    with _watching() as (warnings, audited):
        trainer.train()
        trainer.evaluate_encoder(max_size=30, use_cache=False)
    return SimpleNamespace(trainer=trainer, model_dir=model_dir, warnings=warnings, audited=audited)


@pytest.fixture(scope="module")
def bert_run(tiny_bert, tmp_path_factory):
    experiment_dir = tmp_path_factory.mktemp("bert_experiment")
    yield _train_and_evaluate(tiny_bert, experiment_dir)
    shutil.rmtree(experiment_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def gpt2_run(tiny_gpt2, tmp_path_factory):
    experiment_dir = tmp_path_factory.mktemp("gpt2_experiment")
    yield _train_and_evaluate(tiny_gpt2, experiment_dir)
    shutil.rmtree(experiment_dir, ignore_errors=True)


def _recorded_base_model(checkpoint):
    with open(os.path.join(checkpoint, "config.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)["metadata"]["base_model"]


def _skipped(messages):
    return [message for message in messages if SKIPPED in message]


def _fake_checkpoint(directory, base_model, **metadata):
    """The files of a trained checkpoint that the audit reads: ``config.json`` and ``training.json``."""
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "format_version": 0,
        "architecture": {"input_dim": 4, "latent_dim": 1},
        "metadata": {"base_model": base_model, **metadata},
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "training.json").write_text(
        json.dumps({"training_args": {"label_token_protocol": FDT}}), encoding="utf-8"
    )
    return str(directory)


def _unalignable_trainer(model, **args):
    """A trainer whose second row cannot be supervised under clm_next_token ('bride' is a prefix of 'bridegroom')."""
    data = pd.DataFrame([
        {"masked": "The word was [MASK]", "factual": "anglicised", "alternative": "anglicized",
         "split": "train", "label_class": "UK", "alternative_class": "US"},
        {"masked": "She saw the [MASK]", "factual": "bride", "alternative": "bridegroom",
         "split": "train", "label_class": "UK", "alternative_class": "US"},
    ])
    config = TextPredictionConfig(
        target_classes=["UK", "US"], data=data, masked_col="masked", label_col="factual",
        alternative_col="alternative", label_class_col="label_class",
        alternative_class_col="alternative_class", split_col="split",
    )
    return TextPredictionTrainer(
        model=model, config=config, training_args=TrainingArguments(label_token_protocol=FDT, **args)
    )


# ----------------------------------------------------------------------------- the paper flow, end to end

def test_masked_lm_flow_logs_no_audit_skipped_and_audits_nothing(bert_run):
    trainer = bert_run.trainer
    # the flow really ended on the saved checkpoint, the path that used to be read as a Hugging Face directory
    assert _is_gradiend_checkpoint(trainer._model_arg)
    assert _recorded_base_model(trainer._model_arg) == bert_run.model_dir
    assert _skipped(bert_run.warnings) == []
    assert bert_run.audited == []  # a masked LM never needs the audit


def test_decoder_only_flow_audits_once_before_training_and_says_nothing_for_the_checkpoint(gpt2_run):
    trainer = gpt2_run.trainer
    assert _is_gradiend_checkpoint(trainer._model_arg)
    assert _recorded_base_model(trainer._model_arg) == gpt2_run.model_dir
    assert _skipped(gpt2_run.warnings) == []
    assert gpt2_run.audited == [300]  # every row, once; the checkpoints it produced are not audited again


# ----------------------------------------------------------------------------- the audit of a checkpoint path

def test_a_checkpoint_path_is_audited_as_the_decoder_only_model_it_was_trained_from(tiny_gpt2, tmp_path):
    checkpoint = _fake_checkpoint(tmp_path / "model", base_model=tiny_gpt2)
    trainer = _unalignable_trainer(checkpoint, clm_unalignable_rows="error")
    with _watching() as (warnings, audited):
        with pytest.raises(ClmAlignmentError, match="bridegroom"):
            trainer._check_and_filter_clm_alignment()
    assert _skipped(warnings) == []
    assert audited == [2]


def test_a_checkpoint_path_drops_unalignable_rows_like_its_base_model(tiny_gpt2, tmp_path):
    checkpoint = _fake_checkpoint(tmp_path / "model", base_model=tiny_gpt2)
    trainer = _unalignable_trainer(checkpoint, clm_unalignable_rows="drop")
    trainer._check_and_filter_clm_alignment()
    assert trainer._combined_data["factual"].tolist() == ["anglicised"]


def test_a_masked_lm_checkpoint_path_is_decided_from_cheap_facts_without_a_warning(tiny_bert, tmp_path):
    checkpoint = _fake_checkpoint(tmp_path / "model", base_model=tiny_bert)
    trainer = _unalignable_trainer(checkpoint, clm_unalignable_rows="error")
    with _watching() as (warnings, audited):
        trainer._check_and_filter_clm_alignment()
    assert _skipped(warnings) == []
    assert audited == []
    assert trainer._data_loaded is False  # the data was not even touched


def test_an_audit_done_before_training_is_remembered_for_the_checkpoint_it_produces(tiny_gpt2, tmp_path):
    trainer = _unalignable_trainer(tiny_gpt2, clm_unalignable_rows="drop")
    with _watching() as (warnings, audited):
        trainer._check_and_filter_clm_alignment()
        assert audited == [2]
        audited_data = trainer._combined_data
        # what Trainer.train() does when it is done: the model in force is now the saved checkpoint
        trainer._model_arg = _fake_checkpoint(tmp_path / "model", base_model=tiny_gpt2)
        trainer._check_and_filter_clm_alignment()
    assert audited == [2]  # not audited again
    assert trainer._combined_data is audited_data
    assert _skipped(warnings) == []


def test_a_checkpoint_of_another_model_is_not_covered_by_the_remembered_audit(tiny_gpt2, tiny_bert, tmp_path):
    trainer = _unalignable_trainer(tiny_gpt2, clm_unalignable_rows="drop")
    trainer._check_and_filter_clm_alignment()
    trainer._model_arg = _fake_checkpoint(tmp_path / "model", base_model=tiny_bert)
    with _watching() as (warnings, audited):
        trainer._check_and_filter_clm_alignment()
    assert audited == []  # decided again, now for a masked LM: nothing to audit
    assert _skipped(warnings) == []


# ----------------------------------------------------------------------------- the checkpoint reader

def test_checkpoint_base_model_id_reads_only_the_recorded_base_model(tmp_path):
    from gradiend.model.model_with_gradiend import checkpoint_base_model_id

    assert checkpoint_base_model_id(_fake_checkpoint(tmp_path / "ckpt", base_model="some/base")) == "some/base"


def test_checkpoint_base_model_id_is_none_for_everything_that_is_no_checkpoint(tmp_path):
    from gradiend.model.model_with_gradiend import checkpoint_base_model_id

    hf_directory = tmp_path / "hf"
    hf_directory.mkdir()
    (hf_directory / "config.json").write_text(json.dumps({"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}))
    assert checkpoint_base_model_id(str(hf_directory)) is None
    assert checkpoint_base_model_id(str(tmp_path / "missing")) is None
    assert checkpoint_base_model_id("gpt2") is None


def test_checkpoint_without_a_recorded_base_model_fails_loudly(tmp_path):
    from gradiend.model.model_with_gradiend import checkpoint_base_model_id

    directory = tmp_path / "ckpt"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"architecture": {}, "metadata": {}}))
    with pytest.raises(ValueError, match="records no base model"):
        checkpoint_base_model_id(str(directory))
