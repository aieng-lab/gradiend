"""Behaviour of ``label_token_protocol="first_differing_token"`` that differs from gradiend 0.2.5 by design.

The rows that did not change are pinned in ``test_label_protocol_characterization.py``.
"""

import pandas as pd
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gradiend import TrainingArguments, create_model_with_gradiend
from gradiend.trainer.text.prediction.clm_alignment import (
    ClmAlignmentError,
    align_clm_divergence,
    build_shared_prefix_item,
)
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset
from gradiend.trainer.text.prediction.decoder_eval_utils import compute_probability_shift_score_row_wise
from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend

PAD = 50256
MAX_LEN = 12
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


def _items(tok, protocol, text, factual, alternative):
    df = pd.DataFrame([{
        "masked": text, "factual": factual, "alternative": alternative, "label": 1,
        "feature_class_id": 0, "factual_id": "A", "alternative_id": "B",
    }])
    ds = TextTrainingDataset(
        df, tok, batch_size=1, is_decoder_only_model=True, max_length=MAX_LEN,
        balance_column=None, label_token_protocol=protocol,
    )
    return ds[0]


# row -> (text, factual, alternative, real tokens, label position, factual id, alternative id,
#         label ids of a plain leading-space rule without alignment; documentation only, not asserted)
BY_DESIGN = {
    # shared leading token "ang": a plain leading-space rule labels that shared token for both sides
    "k_gt0": ("The word is [MASK]", "anglicised", "anglicized",
              [464, 1573, 318, 3550, 677, PAD], 5, 1417, 1143, (3550, 3550)),
    # target glued to the previous word: a plain leading-space rule puts the label ON the last context token ("the")
    "k0_no_trailing_space": ("We saw the[MASK]", "colour", "color",
                             [1135, 2497, 262, PAD], 3, 49903, 8043, (9568, 3124)),
    # target is the first word: a plain leading-space rule labels " good" / " bad"
    "empty_prefix": ("[MASK] is nice", "good", "bad", [], 11, 11274, 14774, (922, 2089)),
}


@pytest.mark.parametrize("row", sorted(BY_DESIGN))
def test_items_that_change_by_design(tok, row):
    text, f, a, real, pos, f_label, a_label, _old = BY_DESIGN[row]
    item = _items(tok, FDT, text, f, a)
    for side, label in (("factual", f_label), ("alternative", a_label)):
        x = item[side]
        assert x["input_ids"].tolist() == real + [PAD] * (MAX_LEN - len(real))
        assert x["attention_mask"].tolist() == [1] * len(real) + [0] * (MAX_LEN - len(real))
        expected = [-100] * MAX_LEN
        expected[pos] = label
        assert x["labels"].tolist() == expected


def test_no_leading_space_does_not_align_shared_prefix_rows(tok):
    item = _items(tok, NLS, "The word is [MASK]", "anglicised", "anglicized")
    assert item["factual"]["input_ids"].tolist() == item["alternative"]["input_ids"].tolist()
    assert item["factual"]["labels"].tolist() == item["alternative"]["labels"].tolist()  # both 648, as in 0.2.5


def test_prefix_overlap_row_is_unalignable(tok):
    with pytest.raises(ClmAlignmentError):
        _items(tok, FDT, "She saw the [MASK]", "bride", "bridegroom")
    # a plain leading-space rule labels the shared first token 26619 for both sides; no_leading_space (0.2.5) labels 1671
    legacy = _items(tok, NLS, "She saw the [MASK]", "bride", "bridegroom")["factual"]["labels"]
    assert legacy.max().item() == 1671


def test_identical_strings_are_a_neutral_identity_row_with_the_plain_label(tok):
    item = _items(tok, FDT, "The word is [MASK]", "medium", "medium")
    assert item["factual"]["labels"][3].item() == 7090  # the plain leading-space label, " medium"


def test_k0_label_is_the_leading_space_token_when_the_target_follows_whitespace(tok):
    text, f, a = "We saw the [MASK]", "colour", "color"
    for word, side in ((f, "factual"), (a, "alternative")):
        old = tok(f" {word}", add_special_tokens=False)["input_ids"][0]
        got = _items(tok, FDT, text, f, a)[side]["labels"]
        assert got[got != -100].item() == old


def test_colour_color_share_no_first_token_but_anglicised_pair_does(tok):
    c = [tok(f" {w}", add_special_tokens=False)["input_ids"] for w in ("colour", "color")]
    assert c[0][0] != c[1][0]
    d = [tok(f" {w}", add_special_tokens=False)["input_ids"] for w in ("anglicised", "anglicized")]
    assert d[0][0] == d[1][0] and d[0][2] != d[1][2]


# ----------------------------------------------------------------------------- name handling

@pytest.mark.parametrize("old,new", [("canonical", FDT), ("legacy", NLS)])
def test_old_names_load_via_from_dict_only(old, new):
    assert TrainingArguments.from_dict({"label_token_protocol": old}).label_token_protocol == new
    with pytest.raises(ValueError):
        TrainingArguments(label_token_protocol=old)


def test_saved_settings_without_the_key_load_as_no_leading_space():
    d = TrainingArguments().to_dict()
    assert d["label_token_protocol"] == FDT
    d.pop("label_token_protocol")
    assert TrainingArguments.from_dict(d).label_token_protocol == NLS


# ----------------------------------------------------------------------------- builder

class _Tok:
    """Minimal tokenizer: ids are word lengths; eos id 0 is valid and must not be replaced."""

    eos_token_id = 0
    pad_token_id = None
    padding_side = "right"
    truncation_side = "right"

    def __call__(self, text, add_special_tokens=True, truncation=False, **kw):
        return {"input_ids": [len(w) for w in text.split()]}


def test_builder_keeps_the_tail_and_the_label_when_truncating():
    t = _Tok()
    for side in ("left", "right"):
        t.truncation_side = side
        item = build_shared_prefix_item(t, "a bb ccc dddd eeeee ", [7, 8], 99, max_length=4, pad_to_max_length=True)
        assert item["input_ids"][0].tolist() == [5, 7, 8, 0]
        assert item["labels"][0].tolist() == [-100, -100, -100, 99]


def test_builder_id_zero_is_a_valid_slot_and_pad_id():
    item = build_shared_prefix_item(_Tok(), "a bb", [], 5, max_length=5, pad_to_max_length=True)
    assert item["input_ids"][0].tolist() == [1, 2, 0, 0, 0]  # eos 0 is slot AND pad (pad_token_id None)
    assert item["attention_mask"][0].tolist() == [1, 1, 1, 0, 0]
    assert item["labels"][0].tolist() == [-100, -100, 5, -100, -100]


def test_builder_left_padding_and_unpadded():
    t = _Tok()
    t.padding_side = "left"
    item = build_shared_prefix_item(t, "a bb", [3], 5, max_length=6, pad_to_max_length=True)
    assert item["attention_mask"][0].tolist() == [0, 0, 1, 1, 1, 1]
    assert item["labels"][0].tolist() == [-100] * 5 + [5]
    plain = build_shared_prefix_item(t, "a bb", [3], 5, max_length=6, pad_to_max_length=False)
    assert plain["input_ids"][0].tolist() == [1, 2, 3, 0]
    assert plain["labels"][0].tolist() == [-100, -100, -100, 5]


# ----------------------------------------------------------------------------- create_inputs

@pytest.fixture
def wrapper(tok, tiny_model):
    return create_model_with_gradiend(tiny_model, TextPredictionModelWithGradiend, tokenizer=tok)


def test_create_inputs_with_alternative_builds_the_divergence_item(tok, wrapper):
    wrapper.label_token_protocol = FDT
    item = wrapper.create_inputs("The word is [MASK]", "anglicised", alternative="anglicized")
    assert item["input_ids"].tolist() == [[464, 1573, 318, 3550, 677, PAD]]
    assert item["labels"].tolist() == [[-100] * 5 + [1417]]
    item = wrapper.create_inputs("We saw the [MASK]", "colour", alternative="color")
    assert item["input_ids"].tolist() == [[1135, 2497, 262, PAD]]
    assert item["labels"].tolist() == [[-100, -100, -100, 9568]]


def test_create_inputs_with_alternative_matches_the_training_item(tok, wrapper):
    wrapper.label_token_protocol = FDT
    train = _items(tok, FDT, "The word is [MASK]", "anglicised", "anglicized")["factual"]
    n = int(train["attention_mask"].sum())
    item = wrapper.create_inputs("The word is [MASK]", "anglicised", alternative="anglicized")
    assert item["input_ids"][0].tolist() == train["input_ids"][:n].tolist()
    assert item["labels"][0].tolist() == train["labels"][:n].tolist()


def test_create_inputs_with_alternative_under_no_leading_space_raises(wrapper):
    wrapper.label_token_protocol = NLS
    with pytest.raises(ValueError, match="first_differing_token"):
        wrapper.create_inputs("The word is [MASK]", "good", alternative="bad")


def test_create_inputs_with_identical_alternative_is_the_plain_row(wrapper):
    """A neutral identity row (alternative == label) has no contrast: it gets the plain leading-space label, under both rules."""
    for protocol in (FDT, NLS):
        wrapper.label_token_protocol = protocol
        item = wrapper.create_inputs("The word is [MASK]", "good", alternative="good")
        plain = wrapper.create_inputs("The word is [MASK]", "good")
        assert item["input_ids"].tolist() == plain["input_ids"].tolist()
        assert item["labels"].tolist() == plain["labels"].tolist()


def test_create_inputs_without_alternative_ignores_the_protocol(wrapper):
    """Without an alternative create_inputs always uses the leading-space rule (as in 0.2.5), also under no_leading_space."""
    wrapper.label_token_protocol = NLS
    item = wrapper.create_inputs("The word is [MASK]", "good")
    assert item["labels"][0, -1].item() == 922


# ----------------------------------------------------------------------------- row-wise eval

def _eval_frame(rows):
    return pd.DataFrame([
        {"factual_id": "A", "alternative_id": "B", "masked": m, "factual": f, "alternative": a} for m, f, a in rows
    ])


def test_row_wise_eval_scores_shared_prefix_rows_at_the_divergence_token(tok, tiny_model):
    text, f, a = "The word is [MASK]", "anglicised", "anglicized"
    frame = _eval_frame([(text, f, a)])
    _, per_row = compute_probability_shift_score_row_wise(
        tiny_model, tok, frame, return_per_row_df=True, label_token_protocol=FDT
    )
    div = align_clm_divergence(tok, "The word is ", f, a)
    ids = torch.tensor([tok("The word is", add_special_tokens=False)["input_ids"] + list(div.shared_ids)], device=tiny_model.device)
    with torch.no_grad():
        probs = torch.softmax(tiny_model(input_ids=ids).logits[0, -1], dim=-1)
    assert per_row["P_factual"][0] == pytest.approx(float(probs[div.factual_id]), rel=1e-4)
    assert per_row["P_alternative"][0] == pytest.approx(float(probs[div.alternative_id]), rel=1e-4)
    assert per_row["P_factual"][0] != pytest.approx(per_row["P_alternative"][0], rel=1e-3)
    # no_leading_space scores both sides at the shared first token, i.e. identically (0.2.5 resolves the shared
    # token through get_vocab(), whose order is random, so the common value itself is not pinned)
    _, legacy = compute_probability_shift_score_row_wise(
        tiny_model, tok, frame, return_per_row_df=True, label_token_protocol=NLS
    )
    assert legacy["P_factual"][0] == pytest.approx(legacy["P_alternative"][0], rel=1e-6)


@pytest.mark.parametrize("bad", [
    ("She saw the [MASK]", "bride", "bridegroom"),
    ("The word is [MASK]", "medium", "medium"),
])
def test_row_wise_eval_unalignable_rows_follow_the_policy(tok, tiny_model, bad):
    df = _eval_frame([("We saw the [MASK]", "colour", "color"), bad])
    with pytest.raises(ClmAlignmentError):
        compute_probability_shift_score_row_wise(tiny_model, tok, df, label_token_protocol=FDT)
    _, per_row = compute_probability_shift_score_row_wise(
        tiny_model, tok, df, return_per_row_df=True, label_token_protocol=FDT, clm_unalignable_rows="drop"
    )
    assert len(per_row) == 1
    _, kept = compute_probability_shift_score_row_wise(
        tiny_model, tok, df, return_per_row_df=True, label_token_protocol=NLS
    )
    assert len(kept) == 2


# ----------------------------------------------------------------------------- trainer pre-training check

def _trainer(tok, tiny_model, **args):
    from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer

    data = pd.DataFrame([
        {"masked": "The word was [MASK]", "factual": "anglicised", "alternative": "anglicized",
         "split": "train", "label_class": "UK", "alternative_class": "US"},
        {"masked": "She saw the [MASK]", "factual": "bride", "alternative": "bridegroom",
         "split": "train", "label_class": "UK", "alternative_class": "US"},
    ])
    cfg = TextPredictionConfig(
        target_classes=["UK", "US"], data=data, masked_col="masked", label_col="factual",
        alternative_col="alternative", label_class_col="label_class",
        alternative_class_col="alternative_class", split_col="split",
    )
    tiny_model.tokenizer = tok
    return TextPredictionTrainer(model=tiny_model, config=cfg, training_args=TrainingArguments(**args))


def test_trainer_checks_alignment_before_training(tok, tiny_model):
    with pytest.raises(ClmAlignmentError, match="bridegroom"):
        _trainer(tok, tiny_model, label_token_protocol=FDT).create_training_data(tok)
    dropping = _trainer(tok, tiny_model, label_token_protocol=FDT, clm_unalignable_rows="drop")
    assert len(dropping.create_training_data(tok)) == 1


def test_pre_training_audit_runs_when_the_model_is_given_as_a_hub_id(tok):
    """A model passed as a string id is not loaded yet at audit time: the audit must still see a CLM.

    Regression: ``is_decoder_only_model("gpt2")`` is False for a bare string, so the audit resolved an
    MLM objective, skipped itself, and the bridegroom/bride row reached ``pre_prune`` and crashed there.
    """
    from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer

    data = pd.DataFrame([
        {"masked": "The word was [MASK]", "factual": "anglicised", "alternative": "anglicized",
         "split": "train", "label_class": "UK", "alternative_class": "US"},
        {"masked": "She saw the [MASK]", "factual": "bride", "alternative": "bridegroom",
         "split": "train", "label_class": "UK", "alternative_class": "US"},
    ])
    cfg = TextPredictionConfig(
        target_classes=["UK", "US"], data=data, masked_col="masked", label_col="factual",
        alternative_col="alternative", label_class_col="label_class",
        alternative_class_col="alternative_class", split_col="split",
    )
    args = TrainingArguments(label_token_protocol=FDT, clm_unalignable_rows="drop")
    trainer = TextPredictionTrainer(model="gpt2", config=cfg, training_args=args)
    trainer._check_and_filter_clm_alignment(args)
    assert len(trainer._combined_data) == 1

    failing = TextPredictionTrainer(
        model="gpt2", config=cfg,
        training_args=TrainingArguments(label_token_protocol=FDT, clm_unalignable_rows="error"),
    )
    with pytest.raises(ClmAlignmentError, match="bridegroom"):
        failing._check_and_filter_clm_alignment(failing._training_args)


# ----------------------------------------------------------------------------- MLM unaffected

def test_mlm_items_ignore_the_protocol():
    """MLM items are the same under either value (the default argument is first_differing_token)."""
    mlm_tok = AutoTokenizer.from_pretrained("bert-base-cased")
    df = pd.DataFrame([{
        "masked": "The word is [MASK]", "factual": "good", "alternative": "bad", "label": 1,
        "feature_class_id": 0, "factual_id": "A", "alternative_id": "B",
    }])
    built = {}
    for protocol in (FDT, NLS):
        ds = TextTrainingDataset(df, mlm_tok, batch_size=1, is_decoder_only_model=False, max_length=12,
                                 balance_column=None, label_token_protocol=protocol)
        built[protocol] = ds[0]
    for side in ("factual", "alternative"):
        for key in ("input_ids", "attention_mask", "labels"):
            assert built[FDT][side][key].tolist() == built[NLS][side][key].tolist()
    row = built[FDT]["factual"]
    is_mask = row["input_ids"] == mlm_tok.mask_token_id
    assert int(is_mask.sum()) == 1
    assert row["labels"][is_mask].item() == mlm_tok.convert_tokens_to_ids("good")
