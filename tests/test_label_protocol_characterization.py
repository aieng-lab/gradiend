"""Characterization tests for ``label_token_protocol`` on decoder-only models.

Expected values are the ones of gradiend 0.2.5 (before CLM divergence supervision existed) and are hard-coded.

- ``"no_leading_space"`` must reproduce 0.2.5 byte-for-byte: training items, ``create_inputs``, and no
  alignment check.
- ``"first_differing_token"`` labels the first token where factual and alternative differ. For rows whose first
  tokens already differ and whose target follows whitespace this is the leading-space label; the items differ
  from 0.2.5 only in the input id AT the label position, which no prediction reads. Shared-prefix rows and
  rows whose target does not follow whitespace change by design; they are pinned in
  ``test_first_differing_token_changes.py``.

Row-wise evaluation of rows WITHOUT a shared prefix is not pinned to numbers: 0.2.5 resolves the target word to
a token id by iterating ``tokenizer.get_vocab()``, whose order is random for a fast tokenizer, so the same row
scores differently from call to call. These rows keep that scorer under both protocols, which the
row-wise test below checks with a fixed vocabulary order.
"""

import pandas as pd
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gradiend import create_model_with_gradiend
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset
from gradiend.trainer.text.prediction.decoder_eval_utils import compute_probability_shift_score_row_wise
from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend

PAD = 50256
MAX_LEN = 12

# name -> (masked text, factual, alternative)
ROWS = {
    "k0": ("We saw the [MASK]", "colour", "color"),
    "k0_no_trailing_space": ("We saw the[MASK]", "colour", "color"),
    "k_gt0": ("The word is [MASK]", "anglicised", "anglicized"),
    "prefix_overlap": ("She saw the [MASK]", "bride", "bridegroom"),
    "identical": ("The word is [MASK]", "medium", "medium"),
    "empty_prefix": ("[MASK] is nice", "good", "bad"),
}

# (protocol, row) -> (real tokens, label position, factual label id, alternative label id), as in 0.2.5
# (for first_differing_token: the leading-space label)
MAIN_ITEMS = {
    ("no_leading_space", "k0"): ([1135, 2497, 262, 220], 3, 49903, 8043),
    ("no_leading_space", "k0_no_trailing_space"): ([1135, 2497, 262], 2, 49903, 8043),
    ("no_leading_space", "k_gt0"): ([464, 1573, 318, 220], 3, 648, 648),
    ("no_leading_space", "prefix_overlap"): ([3347, 2497, 262, 220], 3, 1671, 1671),
    ("no_leading_space", "identical"): ([464, 1573, 318, 220], 3, 24132, 24132),
    ("no_leading_space", "empty_prefix"): ([], 11, 11274, 14774),
    ("first_differing_token", "k0"): ([1135, 2497, 262, 220], 3, 9568, 3124),
    ("first_differing_token", "identical"): ([464, 1573, 318, 220], 3, 7090, 7090),
}


@pytest.fixture(scope="module")
def tok():
    t = AutoTokenizer.from_pretrained("gpt2")
    t.pad_token = t.eos_token
    return t


@pytest.fixture(scope="module")
def tiny_model(tok):
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 2
    config.n_head = 2
    config.n_embd = 64
    config.vocab_size = len(tok)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def _items(tok, protocol, row):
    text, factual, alternative = ROWS[row]
    df = pd.DataFrame([{
        "masked": text, "factual": factual, "alternative": alternative, "label": 1,
        "feature_class_id": 0, "factual_id": "A", "alternative_id": "B",
    }])
    ds = TextTrainingDataset(
        df, tok, batch_size=1, is_decoder_only_model=True, max_length=MAX_LEN,
        balance_column=None, label_token_protocol=protocol,
    )
    return ds[0]


def _expected_items(protocol, row, side):
    real, pos, f_label, a_label = MAIN_ITEMS[(protocol, row)]
    ids = real + [PAD] * (MAX_LEN - len(real))
    mask = [1] * len(real) + [0] * (MAX_LEN - len(real))
    labels = [-100] * MAX_LEN
    labels[pos] = f_label if side == "factual" else a_label
    return ids, mask, labels, pos


# ----------------------------------------------------------------------------- training items

@pytest.mark.parametrize("row", sorted(ROWS))
def test_no_leading_space_items_are_byte_identical_to_0_2_5(tok, row):
    item = _items(tok, "no_leading_space", row)
    for side in ("factual", "alternative"):
        ids, mask, labels, _ = _expected_items("no_leading_space", row, side)
        assert item[side]["input_ids"].tolist() == ids
        assert item[side]["attention_mask"].tolist() == mask
        assert item[side]["labels"].tolist() == labels


@pytest.mark.parametrize("row", ["k0", "identical"])
def test_first_differing_token_items_equal_the_leading_space_items_for_first_token_differing_rows(tok, row):
    """Same label, same mask; the input id AT the label position (never read by any prediction) is the
    slot token instead of the trailing-space token of 0.2.5."""
    item = _items(tok, "first_differing_token", row)
    for side in ("factual", "alternative"):
        ids, mask, labels, pos = _expected_items("first_differing_token", row, side)
        got = item[side]
        assert got["labels"].tolist() == labels
        assert got["attention_mask"].tolist() == mask
        got_ids = got["input_ids"].tolist()
        assert got_ids[:pos] + got_ids[pos + 1:] == ids[:pos] + ids[pos + 1:]


def test_first_differing_token_k0_loss_and_gradient_equal_the_0_2_5_input(tok, tiny_model):
    """The changed input id at the label position does not change the loss or the gradient."""
    item = _items(tok, "first_differing_token", "k0")["factual"]
    old_ids = item["input_ids"].clone()
    old_ids[3] = 220  # the trailing-space token 0.2.5 had at the label position

    def loss(ids):
        tiny_model.zero_grad(set_to_none=True)
        out = tiny_model(
            input_ids=ids[None], attention_mask=item["attention_mask"][None], labels=item["labels"][None]
        )
        out.loss.backward()
        return out.loss.item(), tiny_model.lm_head.weight.grad.clone()

    l_new, g_new = loss(item["input_ids"])
    l_old, g_old = loss(old_ids)
    assert l_new == l_old
    assert torch.equal(g_new, g_old)


# ----------------------------------------------------------------------------- create_inputs

# (text, label) -> (input_ids, label id); the label sits on the last token, "[MASK]" stays literal text
EXPECTED_CREATE_INPUTS = {
    ("The word is [MASK]", "good"): ([464, 1573, 318, 685, 31180, 42, 60], 922),
    ("We saw the [MASK]", "colour"): ([1135, 2497, 262, 685, 31180, 42, 60], 9568),
    ("The word is[MASK]", "good"): ([464, 1573, 318, 58, 31180, 42, 60], 922),
}


@pytest.fixture
def wrapper(tok, tiny_model):
    return create_model_with_gradiend(tiny_model, TextPredictionModelWithGradiend, tokenizer=tok)


@pytest.mark.parametrize("protocol", [None, "no_leading_space", "first_differing_token"])
@pytest.mark.parametrize("text,label", sorted(EXPECTED_CREATE_INPUTS))
def test_create_inputs_without_alternative_is_unchanged(wrapper, protocol, text, label):
    if protocol is not None:
        wrapper.label_token_protocol = protocol
    item = wrapper.create_inputs(text, label)
    ids, label_id = EXPECTED_CREATE_INPUTS[(text, label)]
    if text.endswith("is[MASK]") and protocol != "no_leading_space":
        # Deliberate change of first_differing_token: the mask is glued to the previous word ("The word is[MASK]"),
        # so the word really has no leading space. 0.2.5 create_inputs used the leading-space token (922) here while
        # its training used the literal one; both paths now agree on the literal token (11274, "good").
        label_id = 11274
    assert item["input_ids"].tolist() == [ids]
    assert item["attention_mask"].tolist() == [[1] * len(ids)]
    assert item["labels"].tolist() == [[-100] * (len(ids) - 1) + [label_id]]


@pytest.mark.parametrize("protocol", [None, "no_leading_space", "first_differing_token"])
def test_create_inputs_rejects_multi_token_label(wrapper, protocol):
    if protocol is not None:
        wrapper.label_token_protocol = protocol
    with pytest.raises(ValueError, match="Only a single label token is supported"):
        wrapper.create_inputs("The word is [MASK]", "anglicised")


# ----------------------------------------------------------------------------- row-wise evaluation

def _eval_frame(rows):
    return pd.DataFrame([
        {"factual_id": "A", "alternative_id": "B", "masked": m, "factual": f, "alternative": a}
        for m, f, a in rows
    ])


@pytest.fixture
def fixed_vocab_order(monkeypatch):
    """0.2.5 resolves a target word through ``tokenizer.get_vocab()``, whose order is random for a fast tokenizer.

    A fixed order makes that scorer repeatable, so the two protocols can be compared on the rows that use it.
    """
    from collections import defaultdict

    from gradiend.trainer.text.prediction import decoder_eval_utils

    def ordered(tokenizer):
        norm_map = defaultdict(list)
        for token, _ in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1]):
            norm_map[decoder_eval_utils._normalize_token_string(token)].append(token)
        return norm_map

    monkeypatch.setattr(decoder_eval_utils, "_build_vocab_norm_map", ordered)


def test_row_wise_eval_of_rows_without_a_shared_prefix_is_the_same_under_both_protocols(tok, tiny_model, fixed_vocab_order):
    """Only shared-prefix and unalignable rows are scored differently by first_differing_token; the rest keeps the 0.2.5 scorer."""
    df = _eval_frame([ROWS["k0"], ("The review was [MASK]", "good", "bad"), ROWS["empty_prefix"]])
    results = {
        protocol: compute_probability_shift_score_row_wise(
            tiny_model, tok, df, return_per_row_df=True, label_token_protocol=protocol
        )[1]
        for protocol in ("no_leading_space", "first_differing_token")
    }
    assert len(results["no_leading_space"]) == len(results["first_differing_token"]) == 2  # empty prefix: no row
    for column in ("P_factual", "P_alternative"):
        assert results["first_differing_token"][column].tolist() == results["no_leading_space"][column].tolist()


# ----------------------------------------------------------------------------- trainer pre-training check

def _trainer(tok, tiny_model, **args):
    from gradiend import TrainingArguments
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


def test_no_leading_space_trainer_keeps_unalignable_rows_and_runs_no_alignment_check(tok, tiny_model):
    assert len(_trainer(tok, tiny_model, label_token_protocol="no_leading_space").create_training_data(tok)) == 2




# ----------------------------------------------------------------------------- Qwen2.5 tokenizer (0.2.5 values)

QWEN_PAD = 151643
# (protocol, row) -> (real tokens, label position, factual label id, alternative label id), as in 0.2.5
QWEN_MAIN_ITEMS = {
    ("no_leading_space", "k0"): ([1654, 5485, 279, 220], 3, 47339, 3423),
    ("no_leading_space", "k0_no_trailing_space"): ([1654, 5485, 279], 2, 47339, 3423),
    ("no_leading_space", "k_gt0"): ([785, 3409, 374, 220], 3, 524, 524),
    ("no_leading_space", "prefix_overlap"): ([7941, 5485, 279, 220], 3, 1323, 1323),
    ("no_leading_space", "identical"): ([785, 3409, 374, 220], 3, 26086, 26086),
    ("no_leading_space", "empty_prefix"): ([], 11, 18536, 13855),
    ("first_differing_token", "k0"): ([1654, 5485, 279, 220], 3, 12463, 1894),
    ("first_differing_token", "identical"): ([785, 3409, 374, 220], 3, 11051, 11051),
}


@pytest.fixture(scope="module")
def qwen_tok():
    pytest.importorskip("transformers")
    try:
        t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    except Exception:
        pytest.skip("Qwen2.5 tokenizer not cached")
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


@pytest.mark.parametrize("key", sorted(QWEN_MAIN_ITEMS))
def test_qwen_items_match_0_2_5(qwen_tok, key):
    protocol, row = key
    item = _items(qwen_tok, protocol, row)
    real, pos, f_label, a_label = QWEN_MAIN_ITEMS[key]
    for side, label in (("factual", f_label), ("alternative", a_label)):
        got = item[side]
        assert got["attention_mask"].tolist() == [1] * len(real) + [0] * (MAX_LEN - len(real))
        expected = [-100] * MAX_LEN
        expected[pos] = label
        assert got["labels"].tolist() == expected
        ids = got["input_ids"].tolist()
        want = real + [QWEN_PAD] * (MAX_LEN - len(real))
        # first_differing_token: the input id AT the label position is never read, so it may differ (slot token)
        assert ids == want if protocol == "no_leading_space" else ids[:pos] + ids[pos + 1:] == want[:pos] + want[pos + 1:]


# ----------------------------------------------------------------------------- old names via saved settings

@pytest.mark.parametrize("row", sorted(ROWS))
def test_settings_saved_with_the_legacy_name_reproduce_0_2_5(tok, row):
    from gradiend import TrainingArguments

    protocol = TrainingArguments.from_dict({"label_token_protocol": "legacy"}).label_token_protocol
    assert protocol == "no_leading_space"
    item = _items(tok, protocol, row)
    for side in ("factual", "alternative"):
        ids, mask, labels, _ = _expected_items("no_leading_space", row, side)
        assert item[side]["input_ids"].tolist() == ids
        assert item[side]["labels"].tolist() == labels
