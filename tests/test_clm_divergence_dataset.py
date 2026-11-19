"""Unit and integration tests for CLM divergence dataset creation, training loss equivalence, and trainer checks.

Covers:
- Dataset item construction with ctx + shared + [slot] and labels at slot position.
- Bit-identical regression against legacy single-token and k=0 items.
- Numerical equivalence: training HF loss == -log_softmax(eval_logits_at_k)[F[k]].
- Padding-side handling (left padding and right padding).
- Row-wise eval P_factual and P_alternative computed at divergence position.
- Pre-training check in TextPredictionTrainer (aggregated error vs drop mode).
"""

import pytest
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from gradiend import TrainingArguments
from gradiend.trainer.text.prediction.clm_alignment import ClmAlignmentError, align_clm_divergence
from gradiend.trainer.text.prediction.dataset import TextTrainingDataset
from gradiend.trainer.text.prediction.decoder_eval_utils import compute_probability_shift_score_row_wise
from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer, TextPredictionConfig
from gradiend.util.positions import last_real_token_positions


@pytest.fixture(scope="module")
def gpt2_tok():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _tiny_gpt2_config(tokenizer):
    """Config of the tiny GPT-2 used by the tests of this module."""
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 2
    config.n_head = 2
    config.n_embd = 64
    config.vocab_size = len(tokenizer)
    return config


@pytest.fixture(scope="module")
def tiny_causal_model(gpt2_tok):
    """Tiny randomly-initialized CausalLM (no download) for mathematical equivalence tests."""
    model = AutoModelForCausalLM.from_config(_tiny_gpt2_config(gpt2_tok))
    model.eval()
    return model


def _float64_tiny_causal_model(tokenizer, seed):
    """Seeded float64 tiny CausalLM for the exact padded-vs-unpadded equivalence tests.

    Why float64: the padded training item (32 tokens) and the unpadded evaluation sequence are the same maths, but
    they go through different tensor shapes, and in float32 the BLAS kernels (which depend on the machine) round
    them differently: gradient differences of 1e-6 to 2e-5 on gradients of size ~3, so a float32 comparison passes
    on one machine and fails on another. In float64 the rounding is ~1e-15, so a real mismatch (a label one
    position off, or on the wrong token) cannot hide in rounding noise and the result does not depend on the machine.
    A local copy, not the module fixture, so no other test sees a double model; the inputs are integer ids, so only
    the model changes. ``seed`` makes the weights reproducible without touching the global random state.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AutoModelForCausalLM.from_config(_tiny_gpt2_config(tokenizer))
    return model.double().eval()


def _next_token_cross_entropy(logits, labels):
    """Mean cross-entropy of ``logits[t]`` against ``labels[t + 1]`` over the supervised positions (``-100`` ignored).

    This is the definition of the HF causal-LM loss, computed in the dtype of the logits. HF casts the logits to
    float32 first, which puts a float32 floor under a float64 comparison (about 1e-7 on the gradients here, 1e-6 on
    a loss of ~11).
    """
    return torch.nn.functional.cross_entropy(
        logits[..., :-1, :].reshape(-1, logits.size(-1)), labels[..., 1:].reshape(-1), ignore_index=-100
    )


def test_divergence_item_construction(gpt2_tok):
    """Multi-token divergence rows must share input_ids and place divergence tokens on the slot."""
    text = "The word is [MASK]"
    f = "anglicised"
    a = "anglicized"
    df = pd.DataFrame([{
        "masked": text,
        "factual": f,
        "alternative": a,
        "label": 1,
        "feature_class_id": 0,
        "factual_id": "UK",
        "alternative_id": "US",
    }])

    ds = TextTrainingDataset(df, gpt2_tok, batch_size=1, is_decoder_only_model=True, max_length=32, balance_column=None, label_token_protocol="first_differing_token")
    item_f = ds._create_item(text, f, alternative_target=a)
    item_a = ds._create_item(text, a, alternative_target=f)

    # Both factual and alternative items must have identical input_ids and attention_mask
    assert torch.equal(item_f["input_ids"], item_a["input_ids"])
    assert torch.equal(item_f["attention_mask"], item_a["attention_mask"])

    div = align_clm_divergence(gpt2_tok, "The word is ", f, a)
    pos = int(last_real_token_positions(item_f["attention_mask"].unsqueeze(0))[0].item())

    # Label sitting at slot position
    assert item_f["labels"][pos].item() == div.factual_id
    assert item_a["labels"][pos].item() == div.alternative_id

    # Everywhere else labels are -100
    assert (item_f["labels"] != -100).sum().item() == 1
    assert (item_a["labels"] != -100).sum().item() == 1


def test_single_token_bit_identity(gpt2_tok):
    """Single-token rows must produce bit-identical items whether alternative_target is given or omitted."""
    text = "The review was [MASK]"
    f = "good"
    a = "bad"
    df = pd.DataFrame([{
        "masked": text,
        "factual": f,
        "alternative": a,
        "label": 1,
        "feature_class_id": 0,
        "factual_id": "pos",
        "alternative_id": "neg",
    }])

    ds = TextTrainingDataset(df, gpt2_tok, batch_size=1, is_decoder_only_model=True, max_length=32, balance_column=None, label_token_protocol="first_differing_token")
    item1 = ds._create_item(text, f, alternative_target=a)
    item2 = ds._create_item(text, f, alternative_target=a)

    assert torch.equal(item1["input_ids"], item2["input_ids"])
    assert torch.equal(item1["attention_mask"], item2["attention_mask"])
    assert torch.equal(item1["labels"], item2["labels"])
    # Label is leading-space token
    lead_id = gpt2_tok.encode(" good", add_special_tokens=False)[0]
    last_pos = last_real_token_positions(item1["attention_mask"].unsqueeze(0))[0].item()
    assert item1["labels"][last_pos].item() == lead_id


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mathematical_loss_equivalence(gpt2_tok, seed):
    """The training item's loss with the slot token equals -log_softmax(eval_logits)[target] of the unpadded sequence.

    Run in float64 (see :func:`_float64_tiny_causal_model`), so the padded and the unpadded forward pass agree to
    rounding at 1e-15 whatever the machine's BLAS does.
    """
    text = "The spelling used in the paper was [MASK]"
    f = "anglicised"
    a = "anglicized"
    df = pd.DataFrame([{
        "masked": text,
        "factual": f,
        "alternative": a,
        "label": 1,
        "feature_class_id": 0,
        "factual_id": "UK",
        "alternative_id": "US",
    }])

    ds = TextTrainingDataset(df, gpt2_tok, batch_size=1, is_decoder_only_model=True, max_length=32, balance_column=None, label_token_protocol="first_differing_token")
    item_f = ds._create_item(text, f, alternative_target=a)

    # 1. Forward training item through CausalLM
    model = _float64_tiny_causal_model(gpt2_tok, seed)
    device = next(model.parameters()).device
    inputs = {
        "input_ids": item_f["input_ids"].unsqueeze(0).to(device),
        "attention_mask": item_f["attention_mask"].unsqueeze(0).to(device),
        "labels": item_f["labels"].unsqueeze(0).to(device),
    }
    with torch.no_grad():
        out_train = model(**inputs)
    train_loss = float(out_train.loss.item())
    train_loss_float64 = float(_next_token_cross_entropy(out_train.logits, inputs["labels"]))

    # 2. Forward eval item (prefix.rstrip() + shared, no slot token)
    prefix = "The spelling used in the paper was "
    div = align_clm_divergence(gpt2_tok, prefix, f, a)
    prefix_ids = gpt2_tok.encode(prefix.rstrip(), add_special_tokens=True)
    eval_ids = prefix_ids + list(div.shared_ids)
    eval_inputs = {"input_ids": torch.tensor([eval_ids], dtype=torch.long, device=device)}
    with torch.no_grad():
        out_eval = model(**eval_inputs)
    eval_last_logits = out_eval.logits[0, -1, :]
    manual_loss = float((-torch.log_softmax(eval_last_logits, dim=-1)[div.factual_id]).item())

    # the loss of the item labels, computed in float64: equal up to rounding at 1e-15
    assert abs(train_loss_float64 - manual_loss) < 1e-9, f"Train loss {train_loss_float64} != eval loss {manual_loss}"
    # the HF loss casts the logits to float32 and returns a float32 loss, so it agrees only to about one float32 ulp
    # of the loss (~1e-6 at a loss of ~11)
    assert abs(train_loss - manual_loss) < 1e-5, f"HF train loss {train_loss} != eval loss {manual_loss}"


def test_row_wise_eval_reads_divergence(gpt2_tok, tiny_causal_model):
    """compute_probability_shift_score_row_wise computes P(factual) and P(alternative) at the divergence position."""
    df = pd.DataFrame([
        {
            "masked": "The word is [MASK]",
            "factual": "anglicised",
            "alternative": "anglicized",
            "factual_id": "UK",
            "alternative_id": "US",
        },
        {
            "masked": "The weather was [MASK]",
            "factual": "good",
            "alternative": "bad",
            "factual_id": "POS",
            "alternative_id": "NEG",
        },
    ])

    scores, per_row_df = compute_probability_shift_score_row_wise(
        tiny_causal_model,
        gpt2_tok,
        df,
        return_per_row_df=True,
    )

    assert "UK" in scores
    assert "POS" in scores
    assert len(per_row_df) == 2
    assert "P_factual" in per_row_df.columns
    assert "P_alternative" in per_row_df.columns
    for _, row in per_row_df.iterrows():
        assert 0.0 <= row["P_factual"] <= 1.0
        assert 0.0 <= row["P_alternative"] <= 1.0


def test_trainer_pre_training_check_error_mode(gpt2_tok, tiny_causal_model):
    """Pre-training check raises aggregated ClmAlignmentError when unalignable rows exist under 'error'."""
    data = pd.DataFrame([
        {"masked": "The word was [MASK]", "factual": "anglicised", "alternative": "anglicized", "split": "train", "label_class": "UK", "alternative_class": "US"},
        {"masked": "She saw the [MASK]", "factual": "bride", "alternative": "bridegroom", "split": "train", "label_class": "UK", "alternative_class": "US"},
    ])
    cfg = TextPredictionConfig(
        target_classes=["UK", "US"],
        data=data,
        masked_col="masked",
        label_col="factual",
        alternative_col="alternative",
        label_class_col="label_class",
        alternative_class_col="alternative_class",
        split_col="split",
    )
    trainer = TextPredictionTrainer(
        model=tiny_causal_model,
        config=cfg,
        training_args=TrainingArguments(clm_unalignable_rows="error"),
    )
    tiny_causal_model.tokenizer = gpt2_tok  # the audit reads the tokenizer from the model object

    with pytest.raises(ClmAlignmentError) as exc_info:
        trainer.create_training_data(gpt2_tok)

    msg = str(exc_info.value)
    assert "unalignable row(s)" in msg
    assert "bride" in msg
    assert "bridegroom" in msg
    assert "clm_unalignable_rows='drop'" in msg


def test_trainer_pre_training_check_drop_mode(gpt2_tok, tiny_causal_model):
    """Pre-training check drops only unalignable rows and logs counts under 'drop'."""
    data = pd.DataFrame([
        {"masked": "The word was [MASK]", "factual": "anglicised", "alternative": "anglicized", "split": "train", "label_class": "UK", "alternative_class": "US"},
        {"masked": "She saw the [MASK]", "factual": "bride", "alternative": "bridegroom", "split": "train", "label_class": "UK", "alternative_class": "US"},
    ])
    cfg = TextPredictionConfig(
        target_classes=["UK", "US"],
        data=data,
        masked_col="masked",
        label_col="factual",
        alternative_col="alternative",
        label_class_col="label_class",
        alternative_class_col="alternative_class",
        split_col="split",
    )
    trainer = TextPredictionTrainer(
        model=tiny_causal_model,
        config=cfg,
        training_args=TrainingArguments(clm_unalignable_rows="drop"),
    )
    tiny_causal_model.tokenizer = gpt2_tok  # the audit reads the tokenizer from the model object

    train_data = trainer.create_training_data(gpt2_tok)
    # The invalid bride/bridegroom row was dropped, only anglicised/anglicized remains
    assert len(train_data) == 1


def test_left_padding_divergence_item(gpt2_tok):
    """Left-padded tokenizers must place the pad block on the left and label on the rightmost real slot."""
    text = "The word was [MASK]"
    f = "anglicised"
    a = "anglicized"
    df = pd.DataFrame([{
        "masked": text,
        "factual": f,
        "alternative": a,
        "label": 1,
        "feature_class_id": 0,
        "factual_id": "UK",
        "alternative_id": "US",
    }])

    orig_pad_side = gpt2_tok.padding_side
    try:
        gpt2_tok.padding_side = "left"
        ds = TextTrainingDataset(df, gpt2_tok, batch_size=1, is_decoder_only_model=True, max_length=32, balance_column=None, label_token_protocol="first_differing_token")
        item_f = ds._create_item(text, f, alternative_target=a)

        # First tokens must be pad tokens
        assert item_f["attention_mask"][0].item() == 0
        # Last token must be real slot token
        assert item_f["attention_mask"][-1].item() == 1

        pos = int(last_real_token_positions(item_f["attention_mask"].unsqueeze(0))[0].item())
        assert pos == 31  # rightmost token

        div = align_clm_divergence(gpt2_tok, "The word was ", f, a)
        assert item_f["labels"][pos].item() == div.factual_id
        assert (item_f["labels"] != -100).sum().item() == 1
    finally:
        gpt2_tok.padding_side = orig_pad_side


def test_model_with_gradiend_create_inputs(gpt2_tok, tiny_causal_model):
    """TextPredictionModelWithGradiend.create_inputs routes through alignment when alternative is given."""
    from gradiend import create_model_with_gradiend
    from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
    wrapper = create_model_with_gradiend(tiny_causal_model, TextPredictionModelWithGradiend, tokenizer=gpt2_tok)

    # 1. Multi-token without alternative -> raises ValueError("Only a single label token is supported")
    with pytest.raises(ValueError) as exc_info:
        wrapper.create_inputs("The word is [MASK]", "anglicised")
    assert "Only a single label token is supported" in str(exc_info.value)

    # 2. Multi-token with alternative -> divergence aligned
    item = wrapper.create_inputs("The word is [MASK]", "anglicised", alternative="anglicized")
    div = align_clm_divergence(gpt2_tok, "The word is ", "anglicised", "anglicized")
    assert item["labels"][0, -1].item() == div.factual_id


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_divergence_equivalence(gpt2_tok, seed):
    """Base model gradient from training item matches gradient from next-token cross-entropy loss.

    Run in float64 (see :func:`_float64_tiny_causal_model`): the padded 32-token item and the unpadded evaluation
    sequence are the same maths, only float32 rounding of different kernel shapes made this comparison depend on
    the machine. The loss of the item is the next-token cross-entropy of its labels in float64
    (:func:`_next_token_cross_entropy`), so the float32 cast of the logits inside the HF loss adds no noise floor
    either.
    """
    model = _float64_tiny_causal_model(gpt2_tok, seed)
    text = "The spelling used was [MASK]"
    f = "anglicised"
    a = "anglicized"
    df = pd.DataFrame([{
        "masked": text,
        "factual": f,
        "alternative": a,
        "label": 1,
        "feature_class_id": 0,
        "factual_id": "UK",
        "alternative_id": "US",
    }])

    ds = TextTrainingDataset(df, gpt2_tok, batch_size=1, is_decoder_only_model=True, max_length=32, balance_column=None, label_token_protocol="first_differing_token")
    item_f = ds._create_item(text, f, alternative_target=a)

    device = next(model.parameters()).device
    inputs = {
        "input_ids": item_f["input_ids"].unsqueeze(0).to(device),
        "attention_mask": item_f["attention_mask"].unsqueeze(0).to(device),
    }
    labels = item_f["labels"].unsqueeze(0).to(device)

    # 1. Backward through training item
    out_train = model(**inputs)
    _next_token_cross_entropy(out_train.logits, labels).backward()
    grad_param = model.lm_head.weight.grad.clone()
    model.zero_grad(set_to_none=True)

    # 2. Backward through manual eval next-token loss
    prefix = "The spelling used was "
    div = align_clm_divergence(gpt2_tok, prefix, f, a)
    prefix_ids = gpt2_tok.encode(prefix.rstrip(), add_special_tokens=True)
    eval_ids = prefix_ids + list(div.shared_ids)
    eval_inputs = {"input_ids": torch.tensor([eval_ids], dtype=torch.long, device=device)}
    out_eval = model(**eval_inputs)
    eval_last_logits = out_eval.logits[0, -1, :]
    manual_loss = -torch.log_softmax(eval_last_logits, dim=-1)[div.factual_id]
    manual_loss.backward()
    grad_manual = model.lm_head.weight.grad.clone()
    model.zero_grad(set_to_none=True)

    # In float64 the two gradients agree up to rounding at 1e-15, far below the tolerance
    assert torch.allclose(grad_param, grad_manual, atol=1e-6)


def test_k0_bare_word_vocab_not_overriding_leading_space(gpt2_tok):
    """When alternative_target is given and k=0, the label must stay the leading-space token, not bare vocab."""
    text = "We saw the [MASK]"
    f = "colour"
    a = "color"
    df = pd.DataFrame([{"masked": text, "factual": f, "alternative": a, "label": 1, "feature_class_id": 0}])
    ds = TextTrainingDataset(df, gpt2_tok, batch_size=1, is_decoder_only_model=True, max_length=32, balance_column=None, label_token_protocol="first_differing_token")

    bare_id = gpt2_tok.encode("colour", add_special_tokens=False)[0]
    lead_id = gpt2_tok.encode(" colour", add_special_tokens=False)[0]
    assert bare_id != lead_id, "Bare and leading-space ids should differ for colour in BPE"
    ds._vocab_ids = {"colour": bare_id, "color": gpt2_tok.encode("color", add_special_tokens=False)[0]}

    item = ds._create_item(text, f, alternative_target=a)
    div = align_clm_divergence(gpt2_tok, "We saw the ", f, a)
    assert div.k == 0, "colour/color in GPT2 has k=0"
    assert div.factual_id == lead_id, "Divergence id must be the leading-space token"

    last_pos = last_real_token_positions(item["attention_mask"].unsqueeze(0))[0].item()
    assigned_label = item["labels"][last_pos].item()
    assert assigned_label == lead_id, f"Expected leading-space id {lead_id}, got {assigned_label}"
    assert assigned_label != bare_id, f"Bare-word vocab id {bare_id} erroneously overrode leading-space id"


def test_k_gt_0_no_lone_space_token_in_context(gpt2_tok):
    """For a k > 0 pair, input ids before the slot equal tokenizer(prefix.rstrip()) + shared_ids, with no lone space."""
    prefix = "We saw the "
    text = f"{prefix}[MASK]"
    f = "anglicised"
    a = "anglicized"
    df = pd.DataFrame([{"masked": text, "factual": f, "alternative": a, "label": 1, "feature_class_id": 0}])
    ds = TextTrainingDataset(df, gpt2_tok, batch_size=1, is_decoder_only_model=True, max_length=32, balance_column=None, label_token_protocol="first_differing_token")

    div = align_clm_divergence(gpt2_tok, prefix, f, a)
    assert div.k > 0, "anglicised/anglicized should have k > 0 in GPT-2"

    item = ds._create_item(text, f, alternative_target=a)
    input_ids = item["input_ids"]
    mask = item["attention_mask"]

    real_tokens = input_ids[mask == 1].tolist()
    tokens_before_slot = real_tokens[:-1]

    expected_ctx = gpt2_tok(prefix.rstrip(), add_special_tokens=True)["input_ids"]
    expected_full = expected_ctx + list(div.shared_ids)

    assert tokens_before_slot == expected_full, (
        f"Tokens before slot {tokens_before_slot} must match prefix.rstrip() + shared_ids {expected_full}"
    )

    space_token_id = gpt2_tok.encode(" ", add_special_tokens=False)[0]
    assert tokens_before_slot[len(expected_ctx) - 1] != space_token_id, "Must not have a lone space token before shared continuation"


def test_identical_factual_alternative_neutral_creates_item(gpt2_tok):
    """When alternative == factual (neutral data), divergence alignment is bypassed without error."""
    text = "The word was [MASK]"
    f = "medium"
    df = pd.DataFrame([{"masked": text, "factual": f, "alternative": f, "label": 0, "feature_class_id": 0}])
    ds = TextTrainingDataset(df, gpt2_tok, batch_size=1, is_decoder_only_model=True, max_length=32, balance_column=None, label_token_protocol="first_differing_token")

    item = ds._create_item(text, f, alternative_target=f)
    assert item is not None
    assert "labels" in item
    assert (item["labels"] != -100).sum().item() == 1