"""Regression tests for the release-review findings on the CLM audit, the label rule and the row-wise evaluation.

S1  the alignment audit is tied to the data it audited (reloaded data is audited again)
S3  row-wise divergence scoring takes the softmax of the read position only
S5  the audit decides from cheap facts first and never loads a tokenizer for a non-CLM trainer
S6  the audit uses the label rule that is in force (a saved model's recorded one)
S9  a row without an alternative gets the same token variant as a contrast row with the same prefix
"""

import json

import pandas as pd
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gradiend import TrainingArguments
from gradiend.trainer.text.prediction.clm_alignment import (
    ClmAlignmentError,
    label_token_ids,
    resolve_clm_labels,
)
from gradiend.trainer.text.prediction.decoder_eval_utils import compute_probability_shift_score_row_wise

FDT = "first_differing_token"
NLS = "no_leading_space"


@pytest.fixture(scope="module")
def tok():
    t = AutoTokenizer.from_pretrained("gpt2")
    t.pad_token = t.eos_token
    return t


@pytest.fixture(scope="module")
def tiny_model(tok):
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer, config.n_head, config.n_embd, config.vocab_size = 2, 2, 64, len(tok)
    return AutoModelForCausalLM.from_config(config).eval()


def _data():
    return pd.DataFrame([
        {"masked": "The word was [MASK]", "factual": "anglicised", "alternative": "anglicized",
         "split": "train", "label_class": "UK", "alternative_class": "US"},
        {"masked": "She saw the [MASK]", "factual": "bride", "alternative": "bridegroom",
         "split": "train", "label_class": "UK", "alternative_class": "US"},
    ])


def _trainer(model, **args):
    from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer

    cfg = TextPredictionConfig(
        target_classes=["UK", "US"], data=_data(), masked_col="masked", label_col="factual",
        alternative_col="alternative", label_class_col="label_class",
        alternative_class_col="alternative_class", split_col="split",
    )
    return TextPredictionTrainer(model=model, config=cfg, training_args=TrainingArguments(**args))


@pytest.fixture
def model_with_tokenizer(tok, tiny_model):
    tiny_model.tokenizer = tok
    return tiny_model


# ----------------------------------------------------------------------------- S1

def test_reloaded_data_is_audited_again_and_dropped_rows_stay_dropped(model_with_tokenizer):
    args = TrainingArguments(label_token_protocol=FDT, clm_unalignable_rows="drop")
    trainer = _trainer(model_with_tokenizer, label_token_protocol=FDT, clm_unalignable_rows="drop")
    trainer._check_and_filter_clm_alignment(args)
    assert len(trainer._combined_data) == 1
    # materialising all class transitions wipes and reloads the data (all rows are back)
    trainer._ensure_data(materialize_all_class_transitions=True)
    trainer._check_and_filter_clm_alignment(args)
    assert len(trainer._combined_data) == 1
    # the same data is not audited twice
    audited = trainer._combined_data
    trainer._check_and_filter_clm_alignment(args)
    assert trainer._combined_data is audited


def test_drop_filter_is_positional_for_a_non_default_index(model_with_tokenizer):
    args = TrainingArguments(label_token_protocol=FDT, clm_unalignable_rows="drop")
    trainer = _trainer(model_with_tokenizer, label_token_protocol=FDT, clm_unalignable_rows="drop")
    trainer._ensure_data()
    trainer._combined_data.index = [10, 20]
    trainer._check_and_filter_clm_alignment(args)
    assert trainer._combined_data["factual"].tolist() == ["anglicised"]


# ----------------------------------------------------------------------------- S5

def test_non_clm_trainer_never_loads_a_tokenizer_and_decides_once(monkeypatch):
    from gradiend.trainer.text.common import loading

    loads = []
    monkeypatch.setattr(loading.AutoTokenizerForLM, "from_pretrained", lambda *a, **k: loads.append(a) or None)
    trainer = _trainer("bert-base-uncased", label_token_protocol=FDT)
    calls = []
    original = trainer._model_for_clm_objective
    monkeypatch.setattr(trainer, "_model_for_clm_objective", lambda *a, **k: calls.append(1) or original(*a, **k))
    trainer._check_and_filter_clm_alignment(trainer._training_args)
    trainer._check_and_filter_clm_alignment(trainer._training_args)
    assert loads == []
    assert len(calls) == 1
    assert trainer._data_loaded is False  # the data was not even touched


def test_audit_failures_are_logged_not_swallowed(monkeypatch, caplog):
    from gradiend.trainer.text.common import loading

    def broken(*args, **kwargs):
        raise OSError("hub is down")

    monkeypatch.setattr(loading.AutoTokenizerForLM, "from_pretrained", broken)
    trainer = _trainer("gpt2", label_token_protocol=FDT)
    with caplog.at_level("WARNING"):
        trainer._check_and_filter_clm_alignment(trainer._training_args)
    assert any("hub is down" in record.getMessage() for record in caplog.records)


# ----------------------------------------------------------------------------- S6

def _saved_model(model_with_tokenizer, record):
    from types import SimpleNamespace

    kwargs = {"training": {"training_args": record}}
    return SimpleNamespace(
        tokenizer=model_with_tokenizer.tokenizer, config=model_with_tokenizer.config,
        gradiend=SimpleNamespace(kwargs=kwargs),
    )


def test_recorded_rule_is_read_from_a_saved_model_directory(tmp_path):
    from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer

    assert TextPredictionTrainer._recorded_label_token_protocol(str(tmp_path)) is None
    (tmp_path / "training.json").write_text(json.dumps({"training_args": {"max_steps": 5}}), encoding="utf-8")
    assert TextPredictionTrainer._recorded_label_token_protocol(str(tmp_path)) == NLS
    (tmp_path / "training.json").write_text(
        json.dumps({"training_args": {"label_token_protocol": FDT}}), encoding="utf-8"
    )
    assert TextPredictionTrainer._recorded_label_token_protocol(str(tmp_path)) == FDT


def test_audit_uses_the_rule_recorded_with_a_saved_model(model_with_tokenizer):
    # a 0.2.5 record carries no label_token_protocol: the model was trained under no_leading_space
    saved = _saved_model(model_with_tokenizer, {"max_steps": 5})
    trainer = _trainer(model_with_tokenizer, label_token_protocol=FDT, clm_unalignable_rows="error")
    trainer._check_and_filter_clm_alignment(trainer._training_args, model=saved)
    # without a record the new default applies and the unalignable row is found
    other = _trainer(model_with_tokenizer, label_token_protocol=FDT, clm_unalignable_rows="error")
    with pytest.raises(ClmAlignmentError, match="bridegroom"):
        other._check_and_filter_clm_alignment(other._training_args)


def test_train_passes_the_model_it_trains_to_the_audit(model_with_tokenizer, monkeypatch):
    from gradiend.trainer.trainer import Trainer

    saved = _saved_model(model_with_tokenizer, {"max_steps": 5})
    monkeypatch.setattr(Trainer, "train", lambda self, *a, **k: "trained")
    trainer = _trainer(model_with_tokenizer, label_token_protocol=FDT, clm_unalignable_rows="error")
    assert trainer.train(model=saved) == "trained"


# ----------------------------------------------------------------------------- S9

@pytest.mark.parametrize("prefix", ["We saw the ", "(", "", "We saw the"])
@pytest.mark.parametrize("word", ["colour", "good"])
def test_row_without_alternative_has_the_variant_of_a_contrast_row(tok, prefix, word):
    alone = resolve_clm_labels(tok, prefix, word)
    contrast = resolve_clm_labels(tok, prefix, word, "zzzqx")
    assert alone[1] == contrast[1] or contrast[0] != ()  # same first token unless the pair shares a prefix
    assert alone[1] == label_token_ids(tok, prefix, word)[0]


def test_glued_or_empty_prefix_gets_the_literal_token_and_spaced_prefix_the_leading_space_token(tok):
    plain = tok.encode("good", add_special_tokens=False)
    spaced = tok.encode(" good", add_special_tokens=False)
    assert plain != spaced
    assert label_token_ids(tok, "(", "good") == plain
    assert label_token_ids(tok, "", "good") == plain
    assert label_token_ids(tok, "The review was ", "good") == spaced
    assert resolve_clm_labels(tok, "The review was ", "good")[1] == spaced[0]
    assert resolve_clm_labels(tok, "(", "good")[1] == plain[0]


def test_no_leading_space_rule_ignores_the_prefix(tok):
    expected = [tok.vocab.get("good", tok.encode("good", add_special_tokens=False)[0])]
    assert label_token_ids(tok, "(", "good", NLS) == expected
    assert label_token_ids(tok, "The review was ", "good", NLS) == expected


# ----------------------------------------------------------------------------- S3

@pytest.mark.parametrize("side", ["left", "right"])
def test_divergence_scoring_softmaxes_one_position_and_right_padding_is_invariant(tok, tiny_model, monkeypatch, side):
    tok_side = AutoTokenizer.from_pretrained("gpt2")
    tok_side.pad_token = tok_side.eos_token
    tok_side.padding_side = side
    rows = pd.DataFrame([
        {"factual_id": "A", "alternative_id": "B", "masked": "The word is [MASK]",
         "factual": "anglicised", "alternative": "anglicized"},
        {"factual_id": "A", "alternative_id": "B", "masked": "In a long sentence the word was [MASK]",
         "factual": "anglicised", "alternative": "anglicized"},
    ])
    seen = []
    real_softmax = torch.softmax

    def spy(x, *args, **kwargs):
        seen.append(x.dim())
        return real_softmax(x, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", spy)
    _, batched = compute_probability_shift_score_row_wise(
        tiny_model, tok_side, rows, return_per_row_df=True, batch_size=2
    )
    assert seen and max(seen) <= 2, "softmax must run on B x V, not on B x L x V"
    monkeypatch.undo()
    if side == "left":
        # GPT-2 derives position ids from the sequence index, not the mask, so left padding shifts its positions:
        # a property of the model, not of the position that is read. Only the position and softmax shape are checked.
        return
    for i in range(2):
        _, single = compute_probability_shift_score_row_wise(
            tiny_model, tok_side, rows.iloc[[i]], return_per_row_df=True, batch_size=1
        )
        for column in ("P_factual", "P_alternative"):
            assert batched[column].iloc[i] == pytest.approx(single[column].iloc[0], rel=1e-5)


def _training_label(tok, text, factual, alternative):
    from gradiend.trainer.text.prediction.dataset import TextTrainingDataset

    df = pd.DataFrame([{"masked": text, "factual": factual, "alternative": alternative or factual, "label": 1,
                        "feature_class_id": 0, "factual_id": "A", "alternative_id": "B"}])
    ds = TextTrainingDataset(df, tok, batch_size=1, is_decoder_only_model=True, max_length=16,
                             balance_column=None, label_token_protocol=FDT)
    item = ds._create_item(text, factual, alternative_target=alternative)
    return int(item["labels"][item["labels"] != -100].item())


@pytest.mark.parametrize("text,leading_space", [("We saw the [MASK]", True), ("We saw the[MASK]", False)])
def test_training_row_without_alternative_follows_the_prefix_like_a_contrast_row(tok, text, leading_space):
    word = "colour"
    expected = tok.encode((" " if leading_space else "") + word, add_special_tokens=False)[0]
    assert _training_label(tok, text, word, None) == expected
    assert _training_label(tok, text, word, "color") == expected
