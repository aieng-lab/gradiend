"""Instruct models are chat-formatted on every data path, in the chat format of their own family.

Before, ``InstructTokenizerWrapper`` hard-coded the Llama-3 chat tokens (wrong for every other family) and the
CLM first-differing-token rows unwrapped the tokenizer, so contrast rows were plain text while identity rows,
neutral rows, ``create_inputs`` and the language-model evaluation were chat prompts. Now one builder makes every row:

- :mod:`gradiend.trainer.text.common.chat_format` renders the prompt of the model's family: the tokenizer's own
  ``chat_template`` (there is no hand-written format of any family);
- a contrast row is ``chat prompt + shared answer tokens + [slot]`` with the label on the slot, an identity row is the
  same with nothing shared;
- a chat tokenizer supports only ``label_token_protocol="first_differing_token"``; combined with
  ``"no_leading_space"`` (the plain-text rule, which labels the last prompt token) it fails early with a ValueError.

- which models are Instruct models is decided by one name rule, which ``instruction_model`` can override;
- a template that writes the current date into the prompt gets a fixed date (a default, or ``chat_template_date``), which is stored with the trained model.

The tokenizers are ``gpt2`` with the chat tokens and the ``chat_template`` of a family added, so no model weights are
needed. The "Llama-3-style" template is a small synthetic one in the layout of the real Llama-3 template (role
headers followed by a blank line, ``<|eot_id|>`` after every turn).
"""

import datetime
import json
import logging
import random
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from gradiend import create_model_with_gradiend
from gradiend.model.model_with_gradiend import ModelWithGradiend
from gradiend.trainer.text.common import chat_format as chat_format_module
from gradiend.trainer.text.common.chat_format import (
    INSTRUCTION_NAME_WORDS,
    ChatFormat,
    TemplateChatFormat,
    chat_format_for_tokenizer,
    chat_prompt,
    chat_prompt_ids,
    chat_user_text_start,
    configure_chat_tokenizer,
    instruction_name_marker,
    legacy_instruction_name_rule,
    raw_tokenizer,
    register_chat_format,
    require_first_differing_token_for_chat,
    uses_chat_format,
)
from gradiend.trainer.text.common.lm_eval import _process_batch_for_clm, _select_clm_instruction_target_positions
from gradiend.trainer.text.common import loading as loading_module
from gradiend.trainer.text.common.loading import AutoTokenizerForLM, InstructTokenizerWrapper
from gradiend.trainer.text.common.model_base import (
    TextModelWithGradiend,
    resolve_chat_template_date,
    resolve_instruction_model,
)
from gradiend.trainer.text.prediction.clm_alignment import (
    align_clm_divergence,
    answer_ids,
    audit_clm_alignment,
    build_shared_prefix_item,
    clm_context_ids,
)
from gradiend.trainer.text.prediction.dataset import (
    TextTrainingDataset,
    build_first_differing_token_item,
    create_masked_pair_from_text,
)
from gradiend.trainer.text.prediction.decoder_eval_utils import compute_probability_shift_score_row_wise
from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
from gradiend.trainer.text.prediction.trainer import TextPredictionTrainer
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.trainer import Trainer
from gradiend.util.tokenization import offset_pairs, tokenize_with_offsets

FDT = "first_differing_token"
NLS = "no_leading_space"
MAX_LEN = 96

LLAMA3_TOKENS = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
LLAMA3_STYLE_TEMPLATE = (
    "{{- '<|begin_of_text|>' }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}"
)
# The system block of the Llama 3.1+ template in miniature: it asks for the current date, falling back to a constant.
DATED_TEMPLATE = (
    "{{- '<|begin_of_text|>' }}"
    "{%- if not date_string is defined %}"
    "{%- if strftime_now is defined %}{%- set date_string = strftime_now('%d %b %Y') %}"
    "{%- else %}{%- set date_string = '26 Jul 2024' %}{%- endif %}"
    "{%- endif %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|start_header_id|>system<|end_header_id|>\\n\\nToday Date: ' + date_string + '\\n\\n' + messages[0]['content'] + '<|eot_id|>' }}"
    "{%- endif %}"
    "{%- for message in messages if message['role'] != 'system' %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}"
)
CHATML_TEMPLATE = (
    "{%- for message in messages %}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

# The system prompt of the wrapper (indentation and trailing spaces are part of the text; a template format
# normalises its whitespace).
SYSTEM_PROMPT = (
    "\n    You are a language model that completes sentences. Predict the next word that naturally follows the "
    "given text. \n    Return only that word — no punctuation, no quotes, and no explanations.\n    "
)


def _llama3_prompt(user_text, system=SYSTEM_PROMPT):
    """The prompt of ``LLAMA3_STYLE_TEMPLATE``, spelled out token by token."""
    system = " ".join(system.split())
    system_part = f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>" if system else ""
    return (
        f"<|begin_of_text|>{system_part}<|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _chatml_prompt(user_text, system=SYSTEM_PROMPT):
    system = " ".join(system.split())
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"


def _gpt2(extra_tokens=()):
    tok = AutoTokenizer.from_pretrained("gpt2")
    if extra_tokens:
        tok.add_special_tokens({"additional_special_tokens": list(extra_tokens)})
    tok.pad_token = tok.eos_token
    return tok


def _llama3_tokenizer():
    tok = _gpt2(LLAMA3_TOKENS)
    tok.chat_template = LLAMA3_STYLE_TEMPLATE
    return tok


def _dated_tokenizer():
    tok = _gpt2(LLAMA3_TOKENS)
    tok.chat_template = DATED_TEMPLATE
    return tok


def _chatml_tokenizer():
    tok = _gpt2(["<|im_start|>", "<|im_end|>"])
    tok.chat_template = CHATML_TEMPLATE
    return tok


FAMILIES = {
    "llama3": (_llama3_tokenizer, _llama3_prompt),
    "chatml": (_chatml_tokenizer, _chatml_prompt),
}


@pytest.fixture(scope="module", params=sorted(FAMILIES))
def family(request):
    make_tokenizer, prompt = FAMILIES[request.param]
    return request.param, make_tokenizer, prompt


@pytest.fixture(scope="module")
def chat_tok(family):
    """The Instruct tokenizer wrapper of each family (Llama-3-style / ChatML template)."""
    _, make_tokenizer, _ = family
    return InstructTokenizerWrapper(make_tokenizer())


@pytest.fixture(scope="module")
def prompt_text(family):
    return family[2]


@pytest.fixture(scope="module")
def tiny_model(chat_tok):
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer, config.n_head, config.n_embd, config.vocab_size = 2, 2, 64, len(chat_tok.tokenizer)
    return AutoModelForCausalLM.from_config(config).eval()


def _items(tokenizer, protocol, text, factual, alternative, max_length=MAX_LEN):
    df = pd.DataFrame([{
        "masked": text, "factual": factual, "alternative": alternative, "label": 1,
        "feature_class_id": 0, "factual_id": "A", "alternative_id": "B",
    }])
    ds = TextTrainingDataset(
        df, tokenizer, batch_size=1, is_decoder_only_model=True, max_length=max_length,
        balance_column=None, label_token_protocol=protocol,
    )
    return ds[0]


def _real(item):
    """(real input ids, label position within them, label id) of a training item, whatever the padding side."""
    mask = item["attention_mask"].bool()
    ids = item["input_ids"][mask].tolist()
    labels = item["labels"][mask].tolist()
    positions = [i for i, label in enumerate(labels) if label != -100]
    assert len(positions) == 1
    return ids, positions[0], labels[positions[0]]


# ----------------------------------------------------------------------------- the chat formats

def test_llama3_style_prompt_is_rendered_by_the_template_format():
    """A Llama-3-shaped template is a template like any other: there is no hand-written Llama-3 format."""
    wrapper = InstructTokenizerWrapper(_llama3_tokenizer())
    assert type(wrapper.chat_format) is TemplateChatFormat
    for text in ["The woman said ", "The woman said", "", "Hello, world!\nSecond line", "  leading and trailing  "]:
        assert wrapper._wrap_prompt(text) == _llama3_prompt(text)
    assert wrapper._wrap_prompt(["The woman said ", "The woman said"]) == [
        _llama3_prompt("The woman said "), _llama3_prompt("The woman said"),
    ]
    assert wrapper._wrap_prompt(["only one"]) == _llama3_prompt("only one")
    with pytest.raises(TypeError):
        wrapper._wrap_prompt(3)
    wrapper.system_prompt = ""
    assert wrapper._wrap_prompt("The woman said ") == _llama3_prompt("The woman said ", system="")


def test_wrapper_keeps_its_tokenizer_behaviour():
    raw = _llama3_tokenizer()
    wrapper = InstructTokenizerWrapper(raw)
    text = "The woman said "
    golden = _llama3_prompt(text)
    assert list(wrapper(text)["input_ids"]) == list(raw(golden, add_special_tokens=False)["input_ids"])
    # a call that asks for no special tokens tokenizes the text as it is
    assert list(wrapper(text, add_special_tokens=False)["input_ids"]) == list(raw(text, add_special_tokens=False)["input_ids"])
    # the wrapper never lets the tokenizer add special tokens of its own
    assert list(wrapper(text, add_special_tokens=True)["input_ids"]) == list(wrapper(text)["input_ids"])
    assert wrapper.tokenize("Hi there") == raw.tokenize(golden.replace("The woman said ", "Hi there"))
    batch = wrapper([text, "Hi"], return_tensors="pt", padding=True)
    assert batch["input_ids"].shape[0] == 2
    assert wrapper.decode(raw("Hi")["input_ids"]) == "Hi"
    assert wrapper.convert_tokens_to_ids("<|eot_id|>") == raw.convert_tokens_to_ids("<|eot_id|>")
    wrapper.pad_token = "<|eot_id|>"  # unknown attributes go to the raw tokenizer
    assert raw.pad_token == "<|eot_id|>"


def test_wrapper_keeps_chat_state_to_itself():
    """The chat format and the headers are the wrapper's own; the raw tokenizer stays a plain tokenizer."""
    raw = _llama3_tokenizer()
    wrapper = InstructTokenizerWrapper(raw)
    assert not hasattr(raw, "chat_format")
    assert not hasattr(raw, "BEGIN") and not hasattr(raw, "user_prompt_header")
    assert uses_chat_format(wrapper) and not uses_chat_format(raw)
    assert raw_tokenizer(wrapper) is raw and raw_tokenizer(raw) is raw
    assert not uses_chat_format(object())


def test_format_is_the_chat_template_of_the_tokenizer():
    assert type(chat_format_for_tokenizer(_llama3_tokenizer())) is TemplateChatFormat
    assert type(chat_format_for_tokenizer(_chatml_tokenizer())) is TemplateChatFormat
    # a tokenizer without a chat template has no chat format, whatever tokens its vocabulary has
    for tokenizer in (_gpt2(), _gpt2(LLAMA3_TOKENS)):
        with pytest.raises(ValueError, match="gpt2") as error:
            chat_format_for_tokenizer(tokenizer)
        assert "register_chat_format" in str(error.value)
        with pytest.raises(ValueError, match="gpt2"):
            InstructTokenizerWrapper(tokenizer)


def test_the_wrapper_has_no_llama3_header_arguments():
    with pytest.raises(TypeError):
        InstructTokenizerWrapper(_llama3_tokenizer(), user_prompt_header="human")
    assert not hasattr(chat_format_module, "Llama3ChatFormat")


def test_a_new_family_plugs_in_without_editing_existing_code():
    class BracketFormat(ChatFormat):
        def frame(self, system_prompt):
            return f"[SYS]{system_prompt}[/SYS][USER]", "[/USER][ASSISTANT]"

    def factory(tokenizer):
        return BracketFormat() if getattr(tokenizer, "name_or_path", "") == "gpt2" else None

    register_chat_format(factory)
    try:
        wrapper = InstructTokenizerWrapper(_gpt2())
        assert isinstance(wrapper.chat_format, BracketFormat)
        wrapper.system_prompt = "S"
        assert wrapper._wrap_prompt("hi") == "[SYS]S[/SYS][USER]hi[/USER][ASSISTANT]"
        assert chat_prompt(wrapper, "hi") == "[SYS]S[/SYS][USER]hi[/USER][ASSISTANT]"
    finally:
        chat_format_module._CHAT_FORMAT_FACTORIES.remove(factory)
    with pytest.raises(ValueError):
        InstructTokenizerWrapper(_gpt2())


def test_template_format_is_the_chat_template_of_the_tokenizer():
    raw = _chatml_tokenizer()
    wrapper = InstructTokenizerWrapper(raw)
    system = " ".join(wrapper.system_prompt.split())
    for text in ["The woman said ", "", "Two\nlines  and  gaps", "  padded  "]:
        expected = raw.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True,
        )
        assert wrapper._wrap_prompt(text) == expected
        assert expected == _chatml_prompt(text)
    assert list(wrapper("The woman said ")["input_ids"]) == list(
        raw(_chatml_prompt("The woman said "), add_special_tokens=False)["input_ids"]
    )
    wrapper.system_prompt = ""
    assert wrapper._wrap_prompt("hi") == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


def test_template_without_a_system_role_carries_the_system_prompt_in_the_user_turn():
    raw = _chatml_tokenizer()
    raw.chat_template = (
        "{%- if messages[0]['role'] == 'system' %}{{- raise_exception('System role not supported') }}{%- endif %}"
        + CHATML_TEMPLATE
    )
    wrapper = InstructTokenizerWrapper(raw)
    wrapper.system_prompt = "Answer with one word."
    assert wrapper._wrap_prompt("The woman said ") == (
        "<|im_start|>user\nAnswer with one word.\n\nThe woman said <|im_end|>\n<|im_start|>assistant\n"
    )


def test_template_that_does_not_render_the_user_turn_is_an_error():
    raw = _chatml_tokenizer()
    raw.chat_template = "{{- 'no user turn here' }}"
    with pytest.raises(ValueError, match="gpt2"):
        InstructTokenizerWrapper(raw).chat_format.frame("S")


def test_the_prompt_has_exactly_one_beginning_of_text_token():
    """A template that writes BOS into its text, on a tokenizer that would also add BOS itself."""
    from tokenizers.processors import TemplateProcessing

    raw = _gpt2(["<s>", "<|im_start|>", "<|im_end|>"])
    raw.bos_token = "<s>"
    bos = raw.convert_tokens_to_ids("<s>")
    raw._tokenizer.post_processor = TemplateProcessing(
        single="<s> $A", pair="<s> $A <s> $B", special_tokens=[("<s>", bos)]
    )
    raw.chat_template = "{{- bos_token }}" + CHATML_TEMPLATE
    assert raw("hello")["input_ids"][0] == bos  # the tokenizer does add BOS by itself
    wrapper = InstructTokenizerWrapper(raw)
    for kwargs in ({}, {"add_special_tokens": True}):
        ids = list(wrapper("The woman said ", **kwargs)["input_ids"])
        assert ids[0] == bos and ids.count(bos) == 1
    ids = chat_prompt_ids(wrapper, "The woman said ")
    assert ids[0] == bos and ids.count(bos) == 1
    assert ids == list(wrapper("The woman said ")["input_ids"])
    # the row builder adds no second BOS either
    row = build_shared_prefix_item(wrapper, "The woman said ", [], 5, max_length=MAX_LEN, pad_to_max_length=False)
    assert row["input_ids"][0].tolist().count(bos) == 1


def test_user_text_start_is_the_length_of_the_frame_before_the_text(chat_tok, prompt_text):
    ids = chat_prompt_ids(chat_tok, "The woman said")
    start = chat_user_text_start(chat_tok)
    raw = raw_tokenizer(chat_tok)
    frame_before_text = prompt_text("The woman said").split("The woman said")[0].rstrip()
    assert ids[:start] == list(raw(frame_before_text, add_special_tokens=False)["input_ids"])
    after_header = raw.decode(ids[start:])
    assert after_header.startswith("\n") and after_header.lstrip("\n").startswith("The woman said")


# ----------------------------------------------------------------------------- rows: contrast and identity

CONTRAST_ROWS = {
    "k0": ("We saw the [MASK]", "colour", "color"),
    "k_gt0": ("The word is [MASK]", "anglicised", "anglicized"),
}


@pytest.mark.parametrize("row", sorted(CONTRAST_ROWS))
def test_contrast_row_is_chat_prompt_plus_shared_tokens_plus_slot(chat_tok, row):
    text, factual, alternative = CONTRAST_ROWS[row]
    prefix = text.split("[MASK]")[0]
    raw = raw_tokenizer(chat_tok)
    div = align_clm_divergence(chat_tok, prefix, factual, alternative)
    # the answer words are tokenized on their own, without a leading space
    assert list(div.shared_ids) + [div.factual_id] == answer_ids(chat_tok, factual)[: div.k + 1]
    assert list(div.shared_ids) + [div.alternative_id] == answer_ids(chat_tok, alternative)[: div.k + 1]
    item = _items(chat_tok, FDT, text, factual, alternative)
    context = chat_prompt_ids(chat_tok, prefix.rstrip())
    for side, label in (("factual", div.factual_id), ("alternative", div.alternative_id)):
        ids, position, label_id = _real(item[side])
        assert ids[:-1] == context + list(div.shared_ids)
        assert ids[-1] == raw.eos_token_id  # the slot token only carries the label
        assert position == len(ids) - 1 and label_id == label
    # both sides share the context: only the label differs
    assert item["factual"]["input_ids"].tolist() == item["alternative"]["input_ids"].tolist()
    assert item["factual"]["labels"].tolist() != item["alternative"]["labels"].tolist()


def test_chat_prompt_ends_the_context_so_the_label_is_predicted_from_the_generation_prompt(chat_tok, prompt_text):
    item = _items(chat_tok, FDT, *CONTRAST_ROWS["k0"])["factual"]
    ids, position, _ = _real(item)
    raw = raw_tokenizer(chat_tok)
    assert raw.decode(ids[:position]) == prompt_text("We saw the")
    # the position the decoder evaluation reads (the last token of the context) is the one that predicts the label
    last_prompt_token = raw.decode(ids[position - 1])
    assert last_prompt_token and last_prompt_token.strip() == "" and last_prompt_token.endswith("\n")


@pytest.mark.parametrize("row", ["k0", "k_gt0"])
def test_identity_and_neutral_rows_use_the_same_builder_with_nothing_shared(chat_tok, row):
    text, factual, _ = CONTRAST_ROWS[row]
    prefix = text.split("[MASK]")[0]
    first = answer_ids(chat_tok, factual)[0]
    for alternative in (factual, None):
        item = build_first_differing_token_item(
            chat_tok, prefix, factual, alternative, max_length=MAX_LEN, pad_to_max_length=True
        )
        ids = item["attention_mask"][0].bool()
        real = item["input_ids"][0][ids].tolist()
        assert real[:-1] == chat_prompt_ids(chat_tok, prefix.rstrip())
        assert item["labels"][0][ids].tolist() == [-100] * (len(real) - 1) + [first]
    # through the dataset: a row whose alternative equals the factual is an identity transition
    ids, position, label_id = _real(_items(chat_tok, FDT, text, factual, factual)["factual"])
    assert ids[:-1] == chat_prompt_ids(chat_tok, prefix.rstrip()) and label_id == first and position == len(ids) - 1


def test_the_label_of_a_chat_row_has_no_leading_space(chat_tok):
    raw = raw_tokenizer(chat_tok)
    ids = answer_ids(chat_tok, "colour")
    assert raw.decode(ids) == "colour" and raw.convert_ids_to_tokens(ids[0]) == "colour"
    plain = list(raw(" colour", add_special_tokens=False)["input_ids"])
    assert ids != plain  # the plain leading-space variant would be "Gcolour"
    assert answer_ids(chat_tok, "") == []


@pytest.mark.parametrize("side", ["right", "left"])
def test_label_sits_on_the_last_real_token_under_either_padding_side(side):
    tok = InstructTokenizerWrapper(_llama3_tokenizer())
    tok.padding_side = side
    item = _items(tok, FDT, *CONTRAST_ROWS["k0"])["factual"]
    mask = item["attention_mask"]
    assert int(mask.sum()) < MAX_LEN
    real_positions = mask.nonzero().flatten().tolist()
    labelled = (item["labels"] != -100).nonzero().flatten().tolist()
    assert labelled == [real_positions[-1]]
    assert (side == "left") == (int(mask[0]) == 0)


def test_empty_prefix_is_a_real_row_in_chat_format(chat_tok, prompt_text):
    """A sentence that starts with the target has an empty prefix; in a chat prompt that is still a prompt."""
    item = _items(chat_tok, FDT, "[MASK] is nice", "good", "bad")["factual"]
    ids, position, label_id = _real(item)
    assert ids[:-1] == chat_prompt_ids(chat_tok, "") and label_id == answer_ids(chat_tok, "good")[0]
    assert raw_tokenizer(chat_tok).decode(ids[:position]) == prompt_text("")


def test_overlong_prefix_is_shortened_from_the_left_and_keeps_the_chat_frame(chat_tok):
    long_prefix = " ".join(f"word{i}" for i in range(200)) + " and then"
    item = build_shared_prefix_item(chat_tok, long_prefix, [], 7, max_length=MAX_LEN, pad_to_max_length=False)
    ids = item["input_ids"][0].tolist()
    assert len(ids) <= MAX_LEN
    raw = raw_tokenizer(chat_tok)
    empty_frame = chat_prompt_ids(chat_tok, "")
    start = chat_user_text_start(chat_tok)
    assert ids[:start] == empty_frame[:start]  # system prompt and user header are kept
    text = raw.decode(ids)
    assert "and then" in text and "word0 " not in text  # the end of the sentence is kept
    assert item["labels"][0, -1].item() == 7
    with pytest.raises(ValueError, match="too small"):
        build_shared_prefix_item(chat_tok, "hi", [], 7, max_length=10, pad_to_max_length=False)


TRAILING_SPACE_ROWS = {
    "k0": ("We saw the", "colour", "color"),
    "k_gt0": ("The word is", "anglicised", "anglicized"),
    "identity": ("The word is", "medium", "medium"),
}


@pytest.mark.parametrize("row", sorted(TRAILING_SPACE_ROWS))
def test_a_trailing_space_of_the_prefix_does_not_change_a_chat_row(chat_tok, row):
    """The answer is a separate turn: a prefix with and without trailing whitespace gives the same row."""
    base, factual, alternative = TRAILING_SPACE_ROWS[row]
    reference = _items(chat_tok, FDT, base + "[MASK]", factual, alternative)
    for suffix in (" ", "  ", " " + chr(10)):
        item = _items(chat_tok, FDT, base + suffix + "[MASK]", factual, alternative)
        for side in ("factual", "alternative"):
            for key in ("input_ids", "attention_mask", "labels"):
                assert item[side][key].tolist() == reference[side][key].tolist(), (suffix, side, key)
    # and it is the row of the stripped prefix
    ids, position, _ = _real(reference["factual"])
    assert ids[:-1] == chat_prompt_ids(chat_tok, base) + list(align_clm_divergence(chat_tok, base, factual, alternative).shared_ids if factual != alternative else [])


def test_create_inputs_ignores_a_trailing_space_of_the_prefix_in_chat_format(wrapper_model):
    model = wrapper_model
    model.label_token_protocol = FDT
    for label, alternative in (("colour", "color"), ("anglicised", None), ("medium", "medium")):
        with_space = model.create_inputs("We saw the [MASK]", label, alternative=alternative)
        without = model.create_inputs("We saw the[MASK]", label, alternative=alternative)
        bare = model.create_inputs("We saw the ", label, alternative=alternative)
        for key in ("input_ids", "attention_mask", "labels"):
            assert with_space[key].tolist() == without[key].tolist() == bare[key].tolist()


def test_left_truncation_works_on_a_stripped_prefix(chat_tok):
    long_prefix = " ".join(f"word{i}" for i in range(200)) + " and then "
    stripped = long_prefix.rstrip()
    assert clm_context_ids(chat_tok, long_prefix, max_length=90) == chat_prompt_ids(chat_tok, stripped, max_length=90)
    ids = clm_context_ids(chat_tok, long_prefix, max_length=90)
    assert len(ids) <= 90
    assert raw_tokenizer(chat_tok).decode(ids).count("and then") == 1
    # a prefix of blanks leaves only the frame
    assert clm_context_ids(chat_tok, "   ") == chat_prompt_ids(chat_tok, "")


# ----------------------------------------------------------------------------- create_inputs

@pytest.fixture
def wrapper_model(chat_tok, tiny_model):
    model = create_model_with_gradiend(tiny_model, TextPredictionModelWithGradiend, tokenizer=chat_tok)
    model.tokenizer = chat_tok  # create_model_with_gradiend loads the tokenizer of the model name
    return model


def test_create_inputs_is_the_training_row(wrapper_model, chat_tok):
    model = wrapper_model
    assert uses_chat_format(model.tokenizer)
    text, factual, alternative = CONTRAST_ROWS["k_gt0"]
    item = _items(chat_tok, FDT, text, factual, alternative)
    for protocol in (None, FDT):
        if protocol is not None:
            model.label_token_protocol = protocol
        got = model.create_inputs(text, factual, alternative=alternative)
        ids, position, label_id = _real(item["factual"])
        assert got["input_ids"][0].tolist() == ids
        assert got["labels"][0].tolist() == [-100] * position + [label_id]
        # the alternative gives the other side of the contrast
        got_alt = model.create_inputs(text, alternative, alternative=factual)
        assert got_alt["labels"][0, -1].item() == _real(item["alternative"])[2]


def test_create_inputs_supervises_the_first_token_of_a_multi_token_label(wrapper_model, chat_tok):
    model = wrapper_model
    model.label_token_protocol = FDT
    got = model.create_inputs("The word is [MASK]", "anglicised")  # several tokens: 0.2.5 raised here
    ids = got["input_ids"][0].tolist()
    prefix_ids = chat_prompt_ids(chat_tok, "The word is")
    assert ids[:-1] == prefix_ids
    assert got["labels"][0, -1].item() == answer_ids(chat_tok, "anglicised")[0]
    # an alternative equal to the label is the same plain row
    same = model.create_inputs("The word is [MASK]", "anglicised", alternative="anglicised")
    assert same["input_ids"].tolist() == got["input_ids"].tolist()
    # text without the placeholder is the prefix as it is
    bare = model.create_inputs("The word is ", "anglicised")
    assert bare["input_ids"].tolist() == got["input_ids"].tolist()


# ----------------------------------------------------------------------------- the masked target

def test_masked_target_comes_from_the_sentence_not_from_the_chat_template(chat_tok):
    sentence = "The woman said that she was happy to see them again today"
    raw = raw_tokenizer(chat_tok)
    sentence_tokens = set(raw.tokenize(sentence))
    for seed in range(40):
        random.seed(seed)
        pair = create_masked_pair_from_text(sentence, chat_tok, True, min_prefix_tokens=3)
        assert pair is not None
        masked, target = pair
        assert target in sentence_tokens
        assert masked.startswith("The woman said") and "assistant" not in masked and "<|" not in masked
        assert masked.endswith("[MASK]")
    # the same pair as with the raw tokenizer, seed for seed
    for seed in range(10):
        random.seed(seed)
        via_wrapper = create_masked_pair_from_text(sentence, chat_tok, True, min_prefix_tokens=3)
        random.seed(seed)
        via_raw = create_masked_pair_from_text(sentence, raw, True, min_prefix_tokens=3)
        assert via_wrapper == via_raw


def test_offsets_of_a_chat_tokenizer_point_into_the_sentence(chat_tok):
    text = "The woman said hello"
    encoded, raw_offsets = tokenize_with_offsets(chat_tok, text, add_special_tokens=True, truncation=False)
    assert raw_offsets is not None
    offsets = offset_pairs(raw_offsets)
    ids = list(encoded["input_ids"])
    assert len(offsets) == len(ids)
    spans = [(s, e) for s, e in offsets if e > s]
    assert [text[s:e] for s, e in spans] == ["The", " woman", " said", " hello"]
    # every token of the chat frame has an empty span
    assert sum(1 for s, e in offsets if e <= s) == len(ids) - 4
    # tensor and batch outputs are rebased as well
    batch = chat_tok(["The woman said hello", "Hi"], return_offsets_mapping=True, return_tensors="pt", padding=True)
    for row, sentence in enumerate(["The woman said hello", "Hi"]):
        covered = [sentence[s:e] for s, e in batch["offset_mapping"][row].tolist() if e > s]
        assert "".join(covered) == sentence
    # a call without special tokens tokenizes the text itself: offsets are untouched
    plain = chat_tok(text, add_special_tokens=False, return_offsets_mapping=True)
    assert [text[s:e] for s, e in plain["offset_mapping"]] == ["The", " woman", " said", " hello"]


# ----------------------------------------------------------------------------- language-model evaluation

def _header_based_positions(input_ids, attention_mask, end_header_id):
    """The selection of evaluation positions that hard-codes the Llama-3 header token (the second-to-last one)."""
    positions = []
    for i in range(input_ids.shape[0]):
        token_positions = torch.where(attention_mask[i] != 0)[0].tolist()
        matches = [j for j, t in enumerate(input_ids[i].tolist()) if t == end_header_id]
        start = matches[-2] + 1 if len(matches) >= 2 else 0
        remaining = len(token_positions) - start
        if remaining > 1:
            positions.append(random.choice(token_positions[start + remaining // 2:]) - 1)
        else:
            positions.append(None)
    return positions


SENTENCES = ["The woman said that she was happy to see them", "Hi there my friend", "Short one"]


def test_evaluation_positions_do_not_depend_on_the_llama3_layout():
    """The selection only reads the chat format's frame: for a Llama-3-shaped template it equals the header-based one."""
    tok = InstructTokenizerWrapper(_llama3_tokenizer())
    enc = tok(SENTENCES, return_tensors="pt", padding=True)
    end_header = tok.tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    for seed in range(25):
        random.seed(seed)
        expected = _header_based_positions(enc["input_ids"], enc["attention_mask"], end_header)
        random.seed(seed)
        got = _select_clm_instruction_target_positions(enc["input_ids"], enc["attention_mask"], len(SENTENCES), tok)
        assert got == expected


def test_evaluation_positions_follow_the_chat_format_of_the_family(chat_tok):
    enc = chat_tok(SENTENCES, return_tensors="pt", padding=True)
    start = chat_user_text_start(chat_tok)
    for seed in range(25):
        random.seed(seed)
        positions = _select_clm_instruction_target_positions(
            enc["input_ids"], enc["attention_mask"], len(SENTENCES), chat_tok
        )
        for i, pos in enumerate(positions):
            n = int(enc["attention_mask"][i].sum())
            assert pos is not None and start <= pos + 1 < n  # the true token is inside the user turn or after it


def test_evaluation_scores_the_next_token_inside_the_user_turn(chat_tok, tiny_model):
    random.seed(0)
    enc = chat_tok(SENTENCES[:2], return_tensors="pt", padding=True)
    positions = _select_clm_instruction_target_positions(enc["input_ids"], enc["attention_mask"], 2, chat_tok)
    stats, correct, total, true_labels, predicted = _process_batch_for_clm(
        tiny_model, chat_tok, SENTENCES[:2], positions, tiny_model.device
    )
    assert total == 2
    for row, pos in zip(stats, positions):
        assert row["token_index"] == pos
        assert row["true"] == chat_tok.decode([enc["input_ids"][stats.index(row), pos + 1]], skip_special_tokens=True)


# ----------------------------------------------------------------------------- audit and decoder evaluation

def test_alignment_audit_checks_the_answer_words_of_a_chat_tokenizer(chat_tok):
    """In a chat prompt the answer is the first thing the assistant writes: the words are tokenized without a space."""
    raw = raw_tokenizer(chat_tok)
    pairs = [("anglicised", "anglicized"), ("bride", "bridegroom"), ("good", "bad")]
    audit = audit_clm_alignment(chat_tok, pairs)
    assert audit["status"].tolist() == ["aligned", "invalid", "single"]
    assert audit["k"].tolist() == [2, 2, 0]
    assert audit["factual_ids"].iloc[0] == answer_ids(chat_tok, "anglicised")
    assert audit["alternative_ids"].iloc[0] == answer_ids(chat_tok, "anglicized")
    # the plain audit of the same words after a space uses other tokens (a leading-space variant)
    plain = audit_clm_alignment(raw, pairs[:1], context_prefix="The word is ")
    assert plain["factual_ids"].iloc[0] != audit["factual_ids"].iloc[0]
    assert plain["factual_ids"].iloc[0] == list(raw(" anglicised", add_special_tokens=False)["input_ids"])
    # the context is irrelevant in chat format: the answer starts the assistant turn
    with_prefix = audit_clm_alignment(chat_tok, pairs[:1], context_prefix="The word is ")
    assert with_prefix["factual_ids"].iloc[0] == audit["factual_ids"].iloc[0]
    assert with_prefix["k"].tolist() == [2]


def test_hub_id_audit_loads_the_chat_tokenizer(monkeypatch, chat_tok):
    from gradiend.trainer.text.common.loading import AutoTokenizerForLM
    from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
    from gradiend.trainer.core.arguments import TrainingArguments

    monkeypatch.setattr(AutoTokenizerForLM, "from_pretrained", classmethod(lambda cls, name, *a, **k: chat_tok))
    data = pd.DataFrame([{
        "masked": "The word was [MASK]", "factual": "anglicised", "alternative": "anglicized",
        "split": "train", "label_class": "UK", "alternative_class": "US",
    }])
    cfg = TextPredictionConfig(
        target_classes=["UK", "US"], data=data, masked_col="masked", label_col="factual",
        alternative_col="alternative", label_class_col="label_class",
        alternative_class_col="alternative_class", split_col="split",
    )
    trainer = TextPredictionTrainer(model="some-model-instruct", config=cfg, training_args=TrainingArguments(label_token_protocol=FDT))
    assert trainer._tokenizer_for_clm_audit() is chat_tok


def _next_token_probs(model, ids):
    with torch.no_grad():
        return torch.softmax(model(input_ids=torch.tensor([ids], device=model.device)).logits[0, -1], dim=-1)


def test_decoder_evaluation_reads_the_position_that_training_supervises(chat_tok, tiny_model):
    """Rows with shared tokens: P(factual) / P(alternative) are read at the position that predicts the label."""
    text, factual, alternative = CONTRAST_ROWS["k_gt0"]
    df = pd.DataFrame([{"factual_id": "A", "alternative_id": "B", "masked": text, "factual": factual, "alternative": alternative}])
    _, per_row = compute_probability_shift_score_row_wise(tiny_model, chat_tok, df, return_per_row_df=True)
    item = _items(chat_tok, FDT, text, factual, alternative)
    for side, column in (("factual", "P_factual"), ("alternative", "P_alternative")):
        ids, position, label_id = _real(item[side])
        probs = _next_token_probs(tiny_model, ids[:position])
        assert per_row[column].iloc[0] == pytest.approx(float(probs[label_id]), rel=1e-4)
    assert clm_context_ids(chat_tok, "The word is ") == chat_prompt_ids(chat_tok, "The word is")


def test_decoder_evaluation_of_unshared_rows_reads_the_end_of_the_generation_prompt(chat_tok, tiny_model):
    text, factual, alternative = "We saw the [MASK]", "colour", "color"  # the trailing space is dropped in training and eval
    df = pd.DataFrame([{"factual_id": "A", "alternative_id": "B", "masked": text, "factual": factual, "alternative": alternative}])
    _, per_row = compute_probability_shift_score_row_wise(tiny_model, chat_tok, df, return_per_row_df=True)
    ids, position, label_id = _real(_items(chat_tok, FDT, text, factual, alternative)["factual"])
    trained = _next_token_probs(tiny_model, ids[:position])
    evaluated = _next_token_probs(tiny_model, chat_prompt_ids(chat_tok, "We saw the"))
    assert torch.allclose(trained, evaluated, atol=1e-7)
    # the evaluated probability counts the trained label token (and other spellings of the word)
    assert per_row["P_factual"].iloc[0] >= float(trained[label_id]) - 1e-9


def test_chat_k0_row_scores_exactly_the_probability_of_the_training_label(chat_tok, tiny_model):
    """The row-wise eval of a chat row with nothing shared is the training row: same context, same label id."""
    text, factual, alternative = "We saw the [MASK]", "colour", "color"
    df = pd.DataFrame([{"factual_id": "A", "alternative_id": "B", "masked": text, "factual": factual, "alternative": alternative}])
    _, per_row = compute_probability_shift_score_row_wise(tiny_model, chat_tok, df, return_per_row_df=True)
    item = _items(chat_tok, FDT, text, factual, alternative)
    for side, column in (("factual", "P_factual"), ("alternative", "P_alternative")):
        ids, position, label_id = _real(item[side])
        probs = _next_token_probs(tiny_model, ids[:position])
        assert per_row[column].iloc[0] == pytest.approx(float(probs[label_id]), rel=1e-4)


def _tiny_lm(kind, vocab_size):
    torch.manual_seed(0)
    if kind == "gpt2":
        config = AutoConfig.from_pretrained("gpt2")
        config.n_layer, config.n_head, config.n_embd = 2, 2, 64
    else:
        from transformers import LlamaConfig

        config = LlamaConfig(
            vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=256,
        )
    config.vocab_size = vocab_size
    return AutoModelForCausalLM.from_config(config).eval()


@pytest.mark.parametrize("kind", ["gpt2", "llama"])
@pytest.mark.parametrize("row", ["k0", "k_gt0"])
def test_logits_at_the_supervised_position_do_not_depend_on_the_padding_side(family, kind, row):
    """A chat row padded on the left and on the right predicts its label from the same logits.

    The supervised position is the one before the label slot: the last token of the chat prompt (row without
    shared tokens) or the last shared token. A real forward pass through a tiny random model, with the positions
    of the real tokens counted from the first real token (GPT-2 has absolute positions and needs them;
    RoPE models, as Llama, are run the way the package runs them, with the attention mask only).

    The model is float64: the two paddings are the same maths with the real tokens at other positions, and float32
    rounds them differently (a few 1e-7 here, more on another machine's kernels). In float64 GPT-2 agrees to 1e-15;
    Llama to about 1e-7, the float32 precision of its rotary tables (cos/sin are always computed in float32), still
    1000 times below the tolerance.
    """
    _, make_tokenizer, _ = family
    text, factual, alternative = CONTRAST_ROWS[row]
    model = _tiny_lm(kind, len(make_tokenizer())).double()
    logits_by_side = {}
    for side in ("right", "left"):
        tok = InstructTokenizerWrapper(make_tokenizer())
        tok.padding_side = side
        assert tok.tokenizer.padding_side == side
        item = _items(tok, FDT, text, factual, alternative)["factual"]
        mask = item["attention_mask"]
        assert int(mask.sum()) < MAX_LEN and (int(mask[0]) == 0) == (side == "left")
        inputs = {"input_ids": item["input_ids"][None], "attention_mask": mask[None]}
        if kind == "gpt2":
            inputs["position_ids"] = (mask.cumsum(-1) - 1).clamp(min=0)[None]
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        first_real = int(mask.nonzero()[0])
        label_position = int((item["labels"] != -100).nonzero()[0])
        supervised = label_position - 1  # a causal model predicts the label at ``label_position`` from here
        prefix = text.split("[MASK]")[0].rstrip()
        div = align_clm_divergence(tok, prefix, factual, alternative)
        context_length = len(chat_prompt_ids(tok, prefix))
        # the last token of the chat prompt, or the last shared token after it
        assert supervised - first_real == context_length + len(div.shared_ids) - 1
        logits_by_side[side] = logits[supervised]
    assert torch.allclose(logits_by_side["left"], logits_by_side["right"], atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------------------------- chat tokenizer + no_leading_space

def test_the_chat_check_is_one_helper_and_never_rejects_a_plain_tokenizer(chat_tok):
    require_first_differing_token_for_chat(chat_tok, FDT)
    for plain in (raw_tokenizer(chat_tok), object(), None):
        for protocol in (FDT, NLS):
            require_first_differing_token_for_chat(plain, protocol)
    with pytest.raises(ValueError) as error:
        require_first_differing_token_for_chat(chat_tok, NLS)
    message = str(error.value)
    assert "gpt2" in message and "first_differing_token" in message and "no_leading_space" in message
    assert "retrain" in message


def test_dataset_rejects_no_leading_space_for_a_chat_tokenizer(chat_tok):
    with pytest.raises(ValueError, match="first_differing_token"):
        _items(chat_tok, NLS, *CONTRAST_ROWS["k0"])
    _items(chat_tok, FDT, *CONTRAST_ROWS["k0"])
    # a plain tokenizer keeps both rules, untouched
    for protocol in (FDT, NLS):
        _items(raw_tokenizer(chat_tok), protocol, *CONTRAST_ROWS["k0"])


def test_create_inputs_rejects_no_leading_space_for_a_chat_tokenizer(wrapper_model):
    model = wrapper_model
    model.label_token_protocol = NLS
    for alternative in (None, "color"):
        with pytest.raises(ValueError, match="first_differing_token"):
            model.create_inputs("We saw the [MASK]", "colour", alternative=alternative)
    model.label_token_protocol = FDT
    assert model.create_inputs("We saw the [MASK]", "colour")["labels"][0, -1].item() == answer_ids(model.tokenizer, "colour")[0]


def test_row_wise_decoder_evaluation_rejects_no_leading_space_for_a_chat_tokenizer(chat_tok, tiny_model):
    text, factual, alternative = CONTRAST_ROWS["k_gt0"]
    df = pd.DataFrame([{"factual_id": "A", "alternative_id": "B", "masked": text, "factual": factual, "alternative": alternative}])
    with pytest.raises(ValueError, match="first_differing_token"):
        compute_probability_shift_score_row_wise(tiny_model, chat_tok, df, label_token_protocol=NLS)


def _model_the_trainer_hands_out(trainer_protocol, saved_training_args, tokenizer):
    """A model through the real creation path (``create_model_with_gradiend``), with an optional training.json record."""

    class _Model:
        @classmethod
        def from_pretrained(cls, load_directory, **kwargs):
            training = None if saved_training_args is None else {"training_args": saved_training_args}
            return SimpleNamespace(
                gradiend=SimpleNamespace(kwargs={} if training is None else {"training": training}),
                tokenizer=tokenizer,
            )

    class _Trainer(TextPredictionTrainer):
        def __init__(self, protocol):  # bypass the heavy constructor
            self._training_args = TrainingArguments(label_token_protocol=protocol)

        @property
        def model_with_gradiend_cls(self):
            return _Model

    return _Trainer(trainer_protocol).create_model_with_gradiend("ignored")


@pytest.mark.parametrize(
    "trainer_protocol,saved",
    [
        (FDT, {"label_token_protocol": NLS}),  # an Instruct model trained with the old rule
        (FDT, {"label_token_protocol": "legacy"}),  # the same record under the old name
        (FDT, {"max_steps": 5}),  # a record from before the key existed
        (NLS, None),  # a fresh model with training arguments that ask for the plain-text rule
    ],
)
def test_trainer_rejects_a_chat_model_with_no_leading_space(chat_tok, trainer_protocol, saved):
    with pytest.raises(ValueError, match="first_differing_token"):
        _model_the_trainer_hands_out(trainer_protocol, saved, chat_tok)


@pytest.mark.parametrize(
    "trainer_protocol,saved,expected",
    [
        (FDT, None, FDT),
        (NLS, {"label_token_protocol": FDT}, FDT),  # the record of the model wins
        (FDT, {"label_token_protocol": "canonical"}, FDT),
    ],
)
def test_trainer_accepts_a_chat_model_with_first_differing_token(chat_tok, trainer_protocol, saved, expected):
    model = _model_the_trainer_hands_out(trainer_protocol, saved, chat_tok)
    assert model.label_token_protocol == expected


@pytest.mark.parametrize("saved", [None, {"label_token_protocol": NLS}, {"max_steps": 5}])
def test_trainer_still_accepts_a_plain_model_with_either_rule(chat_tok, saved):
    for trainer_protocol in (FDT, NLS):
        model = _model_the_trainer_hands_out(trainer_protocol, saved, raw_tokenizer(chat_tok))
        assert model.label_token_protocol in (FDT, NLS)


# ----------------------------------------------------------------------------- which models are Instruct models

@pytest.mark.parametrize(
    "name,marker",
    [
        ("Qwen/Qwen2.5-1.5B-Instruct", "instruct"),
        ("meta-llama/Llama-3.2-1B-Instruct", "instruct"),
        ("google/gemma-2-2b-it", "it"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "chat"),
        ("meta-llama/Llama-2-7b-chat-hf", "chat"),
        ("mistralai/Mistral-7B-Instruct-v0.2", "instruct"),
        ("QWEN2.5-INSTRUCT", "instruct"),
        ("/models/gemma-2-2b-it", "it"),  # local paths: the base name counts
        ("C:\\models\\gemma-2-2b-it", "it"),
        ("./checkpoints/my_model_it/", "it"),
    ],
)
def test_name_rule_marks_instruction_models(name, marker):
    assert instruction_name_marker(name) == marker


@pytest.mark.parametrize(
    "name",
    [
        "Qwen/Qwen2.5-1.5B",
        "gpt2",
        "EleutherAI/pythia-70m",
        "google/gemma-2-2b",
        "bert-base-cased",
        "fitness",  # ``it`` only inside a word
        "org/fitness-model",
        "alit",
        "org/alit-7b",
        "/models/instruct/llama-3-8b",  # only a parent directory says ``instruct``
        "C:\\chat\\llama-3-8b",
        "chat-org/llama-3-8b",  # the organisation is not part of the base name
        "",
    ],
)
def test_name_rule_ignores_base_models(name):
    assert instruction_name_marker(name) is None


def test_name_rule_words_are_the_documented_ones():
    assert INSTRUCTION_NAME_WORDS == ("instruct", "chat", "it")


class _FakeAutoTokenizer:
    """Stands in for ``transformers.AutoTokenizer`` inside the loader: every model has a ChatML tokenizer."""

    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        return _chatml_tokenizer()


@pytest.fixture
def fake_hub(monkeypatch):
    monkeypatch.setattr(loading_module, "AutoTokenizer", _FakeAutoTokenizer)


@pytest.mark.parametrize(
    "name,instruction_model,chat",
    [
        ("org/model-Instruct", None, True),
        ("org/model-it", None, True),
        ("org/model-chat", None, True),
        ("org/model", None, False),
        ("org/model-Instruct", False, False),  # forced to plain text although the name matches
        ("org/model-Instruct", True, True),
        ("org/model", True, True),  # forced to chat format although the name does not match
        ("org/model", False, False),
    ],
)
def test_loader_chat_formats_by_name_unless_overridden(fake_hub, name, instruction_model, chat):
    tokenizer = AutoTokenizerForLM.from_pretrained(name, instruction_model=instruction_model)
    assert uses_chat_format(tokenizer) is chat


def test_loader_logs_which_rule_fired_for_a_chat_tokenizer(fake_hub, caplog):
    with caplog.at_level(logging.INFO, logger=loading_module.logger.name):
        AutoTokenizerForLM.from_pretrained("org/model-Instruct")
        AutoTokenizerForLM.from_pretrained("org/model")  # plain text: nothing to say
        AutoTokenizerForLM.from_pretrained("org/model", instruction_model=True)
        AutoTokenizerForLM.from_pretrained("org/model-Instruct", instruction_model=False)
    messages = [record.getMessage() for record in caplog.records if record.name == loading_module.logger.name]
    assert len(messages) == 2
    by_name, forced = messages
    assert "org/model-Instruct" in by_name and "'instruct'" in by_name and "TemplateChatFormat" in by_name
    assert "instruction_model=False" in by_name  # how to override
    assert "org/model" in forced and "forced by instruction_model=True" in forced and "TemplateChatFormat" in forced


def test_forcing_the_chat_format_without_a_chat_template_fails_naming_the_model_and_the_way_out(monkeypatch):
    class _PlainHub:
        @classmethod
        def from_pretrained(cls, name, *args, **kwargs):
            return _gpt2()

    monkeypatch.setattr(loading_module, "AutoTokenizer", _PlainHub)
    # the name alone is only a hint: no chat format means plain text (see test_instruction_name_rule_fallback.py)
    assert not uses_chat_format(AutoTokenizerForLM.from_pretrained("org/model-Instruct"))
    with pytest.raises(ValueError, match="gpt2") as error:
        AutoTokenizerForLM.from_pretrained("org/model-Instruct", instruction_model=True)
    assert "instruction_model=False" in str(error.value)
    assert not uses_chat_format(AutoTokenizerForLM.from_pretrained("org/model-Instruct", instruction_model=False))


# ----------------------------------------------------------------------------- instruction_model: arguments and plumbing

def test_instruction_model_and_chat_template_date_are_training_arguments():
    default = TrainingArguments()
    assert default.instruction_model is None and default.chat_template_date is None
    args = TrainingArguments(instruction_model=False, chat_template_date="2024-07-26")
    restored = TrainingArguments.from_dict(args.to_dict())
    assert restored.instruction_model is False and restored.chat_template_date == "2024-07-26"
    # a dict saved before the keys existed loads as "decided by name" / "no date"
    old = TrainingArguments.from_dict({"max_steps": 5})
    assert old.instruction_model is None and old.chat_template_date is None
    for bad in ("yes", 1, 0):
        with pytest.raises(TypeError, match="instruction_model"):
            TrainingArguments(instruction_model=bad)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Qwen/Qwen2.5-1.5B-Instruct", True),
        ("meta-llama/Llama-3.2-1B-INSTRUCT", True),
        ("/models/instruct/llama-3-8b", True),  # the old rule also looked at the organisation and parent directories
        ("instruct-org/llama-3-8b", True),
        ("google/gemma-2-2b-it", False),  # ``it`` and ``chat`` are words of the new rule only
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", False),
        ("Qwen/Qwen2.5-1.5B", False),
        ("", False),
    ],
)
def test_the_legacy_name_rule_is_instruct_anywhere_in_the_name(name, expected):
    assert legacy_instruction_name_rule(name) is expected


def test_the_recorded_decision_wins_and_a_record_without_it_is_read_by_the_old_rule():
    def record(**training_args):
        return {"training": {"training_args": training_args}}

    # a recorded decision wins over what is asked for now and over the name
    assert resolve_instruction_model(record(instruction_model=False), True, "org/model-Instruct") is False
    assert resolve_instruction_model(record(instruction_model=False), None, "org/model-Instruct") is False
    assert resolve_instruction_model(record(instruction_model=True), False, "org/model") is True
    # a record without it was written before the decision was stored: the old rule decides, whatever is asked for
    assert resolve_instruction_model(record(max_steps=3), True, "org/model-it") is False
    assert resolve_instruction_model(record(max_steps=3), None, "org/model-Instruct") is True
    assert resolve_instruction_model(record(max_steps=3), False, "org/model-Instruct") is True
    assert resolve_instruction_model(record(instruction_model=None), True, "org/model-chat") is False  # None = not stored
    # no record at all (a model that is not a trained checkpoint): the setting asked for
    assert resolve_instruction_model(None, True, "org/model") is True
    assert resolve_instruction_model(None, None, "org/model-Instruct") is None  # the current name rule decides later
    assert resolve_instruction_model({}, False, "org/model-Instruct") is False
    assert resolve_instruction_model({"training": "no training_args"}, True, "org/model") is True


class _RecordingTokenizerLoader:
    calls = []

    @classmethod
    def from_pretrained(cls, name, *args, **kwargs):
        cls.calls.append((name, kwargs))
        return SimpleNamespace(name_or_path=name)


@pytest.fixture
def recording_loader(monkeypatch):
    from gradiend.trainer.text.common import model_base as model_base_module

    class _Backbone:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return SimpleNamespace(name_or_path=name)

    _RecordingTokenizerLoader.calls = []
    monkeypatch.setattr(model_base_module, "AutoTokenizerForLM", _RecordingTokenizerLoader)
    monkeypatch.setattr(model_base_module, "AutoModelForLM", _Backbone)
    return _RecordingTokenizerLoader.calls


def test_load_model_hands_the_setting_to_the_tokenizer_loader(recording_loader):
    TextModelWithGradiend._load_model(SimpleNamespace(name_or_path="org/model"))
    TextModelWithGradiend._load_model(SimpleNamespace(name_or_path="org/model"), instruction_model=False)
    TextModelWithGradiend._load_model("org/model", instruction_model=True, base_model_offload="allow")
    assert [kwargs["instruction_model"] for _, kwargs in recording_loader] == [None, False, True]


def test_load_model_of_a_checkpoint_keeps_the_value_it_was_trained_with(recording_loader):
    def load(record, asked):
        recording_loader.clear()
        TextModelWithGradiend._load_model(
            "checkpoint", base_model_id="org/model", gradiend_kwargs=record, instruction_model=asked,
            base_model_offload="allow",
        )
        (name, kwargs), = recording_loader
        assert name == record.get("tokenizer", "org/model")  # the tokenizer id of the record, else the backbone's
        return kwargs["instruction_model"]

    assert load({"training": {"training_args": {"instruction_model": False}}}, True) is False
    assert load({"training": {"training_args": {"instruction_model": True}}}, None) is True
    assert load({"tokenizer": "org/other"}, True) is True  # no training record
    assert load({"tokenizer": "org/other"}, None) is None  # the current name rule decides in the tokenizer loader


@pytest.mark.parametrize(
    "name,decided",
    [
        ("org/model-Instruct", True),
        ("org/INSTRUCT-7b", True),
        ("instruct-org/model", True),
        ("org/model-it", False),  # an old plain-trained ``...-it`` model stays plain
        ("org/model-chat", False),
        ("org/model", False),
    ],
)
def test_a_record_without_the_decision_is_loaded_by_the_old_name_rule(recording_loader, name, decided):
    record = {"tokenizer": name, "training": {"training_args": {"max_steps": 1}}}
    for asked in (None, True, False):  # a trainer setting does not override the record
        recording_loader.clear()
        TextModelWithGradiend._load_model(
            "checkpoint", base_model_id="org/model", gradiend_kwargs=record, instruction_model=asked,
            base_model_offload="allow",
        )
        (loaded_name, kwargs), = recording_loader
        assert loaded_name == name and kwargs["instruction_model"] is decided


@pytest.fixture
def stub_backbone(monkeypatch):
    """``_load_model`` with a backbone stub; the tokenizer loader stays the real one (see ``fake_hub``)."""
    from gradiend.trainer.text.common import model_base as model_base_module

    class _Backbone:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return SimpleNamespace(name_or_path=name)

    monkeypatch.setattr(model_base_module, "AutoModelForLM", _Backbone)


def _is_chat_after_load(name, training_args=None, asked=None):
    """Whether the tokenizer of a model loaded from a checkpoint (record ``training_args``) or fresh (None) is chat."""
    if training_args is None:
        _, tokenizer = TextModelWithGradiend._load_model(SimpleNamespace(name_or_path=name), instruction_model=asked)
    else:
        record = {"tokenizer": name, "training": {"training_args": training_args}}
        _, tokenizer = TextModelWithGradiend._load_model(
            "checkpoint", base_model_id=name, gradiend_kwargs=record, instruction_model=asked, base_model_offload="allow"
        )
    return uses_chat_format(tokenizer)


def test_a_recorded_decision_wins_over_a_changed_name_rule(fake_hub, stub_backbone, monkeypatch):
    # today's rule: ``-Instruct`` is chat, the plain name is not; a record holds the decision that was made
    assert _is_chat_after_load("org/model-Instruct", {"instruction_model": False}) is False
    assert _is_chat_after_load("org/model", {"instruction_model": True}) is True
    # the name rule changes: recorded decisions do not move
    monkeypatch.setattr(loading_module, "instruction_name_marker", lambda name: "instruct")
    assert _is_chat_after_load("org/model-Instruct", {"instruction_model": False}) is False
    assert _is_chat_after_load("org/model", {"instruction_model": False}) is False
    monkeypatch.setattr(loading_module, "instruction_name_marker", lambda name: None)
    assert _is_chat_after_load("org/model-Instruct", {"instruction_model": True}) is True
    # a record without the decision is read by the OLD rule, which the new rule never touches
    assert _is_chat_after_load("org/model-Instruct", {"max_steps": 1}) is True
    monkeypatch.setattr(loading_module, "instruction_name_marker", lambda name: "instruct")
    assert _is_chat_after_load("org/model-it", {"max_steps": 1}) is False
    # positive control: a model without a record follows the (changed) name rule
    assert _is_chat_after_load("org/model") is True
    monkeypatch.setattr(loading_module, "instruction_name_marker", lambda name: None)
    assert _is_chat_after_load("org/model-Instruct") is False


def test_the_setting_travels_from_training_arguments_to_the_tokenizer(fake_hub):
    """``from_pretrained(training_args=...)`` hands ``instruction_model`` down to the tokenizer loader."""
    def model_for(name, **arguments):
        lm = _tiny_lm("gpt2", len(_chatml_tokenizer()))
        lm.name_or_path = name
        return create_model_with_gradiend(lm, TextPredictionModelWithGradiend, training_args=TrainingArguments(**arguments))

    assert uses_chat_format(model_for("org/tiny-Instruct").tokenizer)
    assert not uses_chat_format(model_for("org/tiny-Instruct", instruction_model=False).tokenizer)
    assert not uses_chat_format(model_for("org/tiny").tokenizer)
    assert uses_chat_format(model_for("org/tiny", instruction_model=True).tokenizer)


def test_instruction_model_false_reads_plain_text_on_every_path(fake_hub):
    """An ``...-Instruct`` model with ``instruction_model=False`` is a plain model: no chat anywhere."""
    def model_for(name, **arguments):
        lm = _tiny_lm("gpt2", len(_chatml_tokenizer()))
        lm.name_or_path = name
        return create_model_with_gradiend(lm, TextPredictionModelWithGradiend, training_args=TrainingArguments(**arguments))

    forced_plain = model_for("org/tiny-Instruct", instruction_model=False, label_token_protocol=NLS)
    reference = model_for("org/tiny", label_token_protocol=NLS)
    for model in (forced_plain, reference):
        model.label_token_protocol = NLS
    tokenizer = forced_plain.tokenizer
    assert not uses_chat_format(tokenizer) and raw_tokenizer(tokenizer) is tokenizer
    # the trainer accepts the plain-text rule for it (a chat model would be rejected)
    _bare_trainer(instruction_model=False, label_token_protocol=NLS)._configure_model(forced_plain)
    # training rows, create_inputs and the wrapper-free tokenizer are those of the plain model
    for protocol in (NLS, FDT):
        for contrast in CONTRAST_ROWS.values():
            ours = _items(tokenizer, protocol, *contrast)
            theirs = _items(reference.tokenizer, protocol, *contrast)
            for side in ("factual", "alternative"):
                for key in ("input_ids", "attention_mask", "labels"):
                    assert ours[side][key].tolist() == theirs[side][key].tolist()
            decoded = tokenizer.decode(ours["factual"]["input_ids"][ours["factual"]["attention_mask"].bool()])
            assert "<|im_start|>" not in decoded and "assistant" not in decoded
    got = forced_plain.create_inputs("We saw the [MASK]", "colour")
    expected = reference.create_inputs("We saw the [MASK]", "colour")
    assert got["input_ids"].tolist() == expected["input_ids"].tolist()
    assert got["labels"].tolist() == expected["labels"].tolist()
    assert "<|im_start|>" not in tokenizer.decode(got["input_ids"][0])


# ----------------------------------------------------------------------------- the trainer applies the chat settings

def _bare_trainer(**arguments):
    """A text trainer with only its training arguments (the heavy constructor is bypassed)."""

    class _Trainer(TextPredictionTrainer):
        def __init__(self):
            self._training_args = TrainingArguments(**arguments)

    return _Trainer()


def _configured_model(trainer_arguments, saved_training_args, tokenizer):
    """The model the trainer hands out for a (fresh or saved) checkpoint, and the trainer."""

    class _Model:
        @classmethod
        def from_pretrained(cls, load_directory, **kwargs):
            training = None if saved_training_args is None else {"training_args": saved_training_args}
            return SimpleNamespace(
                gradiend=SimpleNamespace(kwargs={} if training is None else {"training": training}),
                tokenizer=tokenizer,
            )

    class _Trainer(TextPredictionTrainer):
        def __init__(self):
            self._training_args = TrainingArguments(**trainer_arguments)

        @property
        def model_with_gradiend_cls(self):
            return _Model

    trainer = _Trainer()
    return trainer.create_model_with_gradiend("ignored"), trainer


def _instruction_model_warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and "instruction_model" in r.getMessage()]


def test_the_loaded_models_decision_wins_over_the_trainers_setting(caplog):
    """The tokenizer was loaded by the record (see ``_load_model``); the trainer adopts the decision it is in."""
    plain = _gpt2()

    def chat():
        return InstructTokenizerWrapper(_chatml_tokenizer())

    # recorded False (plain tokenizer): a trainer that says True is overruled, with a warning
    with caplog.at_level(logging.WARNING):
        _, trainer = _configured_model({"instruction_model": True}, {"label_token_protocol": FDT, "instruction_model": False}, plain)
    assert trainer._training_args.instruction_model is False
    assert _instruction_model_warnings(caplog)
    # recorded True (chat tokenizer): a trainer that says False is overruled, with a warning
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _, trainer = _configured_model({"instruction_model": False}, {"label_token_protocol": FDT, "instruction_model": True}, chat())
    assert trainer._training_args.instruction_model is True
    assert _instruction_model_warnings(caplog)
    # a trainer that agrees, or asked for nothing, gets no warning; the unset value is filled in
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _, agreeing = _configured_model({"instruction_model": True}, {"label_token_protocol": FDT, "instruction_model": True}, chat())
        _, silent = _configured_model({}, {"label_token_protocol": FDT, "instruction_model": True}, chat())
    assert agreeing._training_args.instruction_model is True and silent._training_args.instruction_model is True
    assert not _instruction_model_warnings(caplog)
    # a record without the decision (saved before it was stored): the old name rule already decided in the loader,
    # and whatever tokenizer came out is what the trainer adopts
    _, trainer = _configured_model({"instruction_model": True}, {"label_token_protocol": FDT, "max_steps": 5}, plain)
    assert trainer._training_args.instruction_model is False
    _, trainer = _configured_model({}, {"label_token_protocol": FDT, "max_steps": 5}, chat())
    assert trainer._training_args.instruction_model is True
    # a fresh model has no record: the trainer's setting stays, whatever the tokenizer is
    _, trainer = _configured_model({"instruction_model": True}, None, plain)
    assert trainer._training_args.instruction_model is True
    _, trainer = _configured_model({}, None, plain)
    assert trainer._training_args.instruction_model is None


# ----------------------------------------------------------------------------- dates inside chat templates

DATED_PROMPT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nToday Date: {date}\n\n"
    + " ".join(SYSTEM_PROMPT.split())
    + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
DEFAULT_DATE = "2024-07-26"
DEFAULT_DATE_IN_PROMPT = "26 Jul 2024"
CHAT_LOGGER = chat_format_module.logger.name


def _date_messages(caplog):
    """The INFO lines the chat format logged about the date."""
    return [
        record.getMessage() for record in caplog.records
        if record.name == CHAT_LOGGER and record.levelno == logging.INFO
    ]


@pytest.fixture
def dated_hub(monkeypatch):
    """Every tokenizer the loader fetches has the template that asks for the current date."""

    class _DatedAutoTokenizer:
        @classmethod
        def from_pretrained(cls, name, *args, **kwargs):
            return _dated_tokenizer()

    monkeypatch.setattr(loading_module, "AutoTokenizer", _DatedAutoTokenizer)


def _model_for(name, tokenizer, **arguments):
    """A model of the given name whose tokenizer the (faked) hub loader decides on."""
    lm = _tiny_lm("gpt2", len(tokenizer))
    lm.name_or_path = name
    return create_model_with_gradiend(lm, TextPredictionModelWithGradiend, training_args=TrainingArguments(**arguments))


def test_the_default_date_is_one_constant():
    assert chat_format_module.DEFAULT_CHAT_TEMPLATE_DATE == DEFAULT_DATE
    assert "ChatTemplateDateRequired" not in dir(chat_format_module)  # no date is never an error
    assert TrainingArguments().chat_template_date is None  # unset: the default is not copied into every argument set


def test_a_template_that_asks_for_the_date_gets_the_default_and_says_so_once(caplog):
    wrapper = InstructTokenizerWrapper(_dated_tokenizer())  # loading the tokenizer alone renders nothing
    assert wrapper.chat_format.inserts_date is False
    with caplog.at_level(logging.INFO, logger=CHAT_LOGGER):
        first = wrapper._wrap_prompt("hi")
        second = wrapper._wrap_prompt("The woman said ")
        third = wrapper._wrap_prompt("hi")
        wrapper.chat_format.frame("another system prompt")  # a new frame is no reason to say it again
    assert first == third == DATED_PROMPT.format(date=DEFAULT_DATE_IN_PROMPT, text="hi")
    assert second == DATED_PROMPT.format(date=DEFAULT_DATE_IN_PROMPT, text="The woman said ")
    assert wrapper.chat_format.inserts_date is True
    (message,) = _date_messages(caplog)
    assert "gpt2" in message  # which model
    assert "inserts the current date" in message
    assert DEFAULT_DATE in message and "the default date" in message
    assert "chat_template_date" in message  # how to change it


def test_a_given_date_wins_over_the_default(caplog):
    wrapper = InstructTokenizerWrapper(_dated_tokenizer(), chat_template_date="2025-03-04")
    with caplog.at_level(logging.INFO, logger=CHAT_LOGGER):
        prompt = wrapper._wrap_prompt("hi")
    assert prompt == DATED_PROMPT.format(date="04 Mar 2025", text="hi")
    (message,) = _date_messages(caplog)
    assert "2025-03-04" in message and "the configured date" in message and DEFAULT_DATE not in message


def test_the_default_date_makes_the_prompt_the_same_every_time():
    first = InstructTokenizerWrapper(_dated_tokenizer())
    second = InstructTokenizerWrapper(_dated_tokenizer())
    expected = DATED_PROMPT.format(date=DEFAULT_DATE_IN_PROMPT, text="The woman said ")
    assert first._wrap_prompt("The woman said ") == first._wrap_prompt("The woman said ") == expected
    assert second._wrap_prompt("The woman said ") == expected
    assert first.chat_format.chat_template_date == DEFAULT_DATE  # the date in force is readable


class _Clock:
    """A system clock the test moves by hand."""

    day = datetime.datetime(2031, 2, 3)

    @classmethod
    def now(cls):
        return cls.day


@pytest.mark.parametrize("given", [None, "2025-03-04"])
def test_the_prompt_does_not_follow_the_system_clock(monkeypatch, given):
    import transformers.utils.chat_template_utils as chat_template_utils

    monkeypatch.setattr(chat_template_utils, "datetime", _Clock)
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
    # positive control: the tokenizer's own ``strftime_now`` does read the clock
    _Clock.day = datetime.datetime(2031, 2, 3)
    assert "Today Date: 03 Feb 2031" in _dated_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts = []
    for day in (datetime.datetime(2031, 2, 3), datetime.datetime(2040, 9, 9)):
        _Clock.day = day
        wrapper = InstructTokenizerWrapper(_dated_tokenizer(), chat_template_date=given)
        prompts.append(wrapper._wrap_prompt("hi"))
    expected = DATED_PROMPT.format(date=DEFAULT_DATE_IN_PROMPT if given is None else "04 Mar 2025", text="hi")
    assert prompts[0] == prompts[1] == expected


@pytest.mark.parametrize("bad", ["2024-13-01", "2024-02-30", "26 Jul 2024", "20240726", "2024-7-26", "", "yesterday", 20240726])
def test_an_invalid_date_is_rejected(bad):
    with pytest.raises(ValueError, match="chat_template_date"):
        TrainingArguments(chat_template_date=bad)
    with pytest.raises(ValueError, match="chat_template_date"):
        InstructTokenizerWrapper(_dated_tokenizer(), chat_template_date=bad)
    with pytest.raises(ValueError, match="chat_template_date"):
        InstructTokenizerWrapper(_dated_tokenizer()).chat_format.configure(bad)
    with pytest.raises(ValueError, match="chat_template_date"):
        configure_chat_tokenizer(InstructTokenizerWrapper(_dated_tokenizer()), bad)


def test_a_template_that_never_asks_for_the_date_logs_nothing(caplog):
    with caplog.at_level(logging.INFO, logger=CHAT_LOGGER):
        plain = InstructTokenizerWrapper(_chatml_tokenizer())
        prompt = plain._wrap_prompt("hi")
        assert configure_chat_tokenizer(plain) is None
        dated = InstructTokenizerWrapper(_chatml_tokenizer(), chat_template_date="2024-07-26")
        assert dated._wrap_prompt("hi") == prompt
    assert plain.chat_format.inserts_date is False and dated.chat_format.inserts_date is False
    assert _date_messages(caplog) == []


def test_frames_are_cached_per_system_prompt_and_date():
    wrapper = InstructTokenizerWrapper(_dated_tokenizer(), chat_template_date="2024-07-26")
    frame = wrapper.chat_format.frame(wrapper.system_prompt)
    assert wrapper.chat_format.frame(wrapper.system_prompt) is frame
    wrapper.chat_format.configure("2025-03-04")
    other = wrapper.chat_format.frame(wrapper.system_prompt)
    assert other != frame and "04 Mar 2025" in other[0]
    wrapper.chat_format.configure("2024-07-26")
    assert wrapper.chat_format.frame(wrapper.system_prompt) is frame
    assert wrapper.chat_format.chat_template_date == "2024-07-26"
    wrapper.chat_format.configure(None)  # back to the default
    assert wrapper.chat_format.chat_template_date == DEFAULT_DATE


def test_configure_chat_tokenizer_renders_the_frame_now_and_reports_the_date_in_use():
    wrapper = InstructTokenizerWrapper(_dated_tokenizer())
    assert wrapper.chat_format.inserts_date is False
    assert configure_chat_tokenizer(wrapper) == DEFAULT_DATE  # no date given: the default, and the frame is rendered
    assert wrapper.chat_format.inserts_date is True
    assert configure_chat_tokenizer(wrapper, "2025-03-04") == "2025-03-04"
    assert configure_chat_tokenizer(wrapper, None) == "2025-03-04"  # no date given: the one in force stays
    # nothing to report for a template without a date and for anything that is not a chat tokenizer
    assert configure_chat_tokenizer(InstructTokenizerWrapper(_chatml_tokenizer()), "2025-03-04") is None
    for not_a_chat_tokenizer in (_gpt2(), object(), None):
        assert configure_chat_tokenizer(not_a_chat_tokenizer, "2025-03-04") is None


def test_a_template_that_cannot_frame_a_system_prompt_fails_in_the_configuration_too():
    raw = _chatml_tokenizer()
    raw.chat_template = "{{- 'no user turn here' }}"
    with pytest.raises(ValueError, match="gpt2"):
        configure_chat_tokenizer(InstructTokenizerWrapper(raw), "2024-07-26")


# ----------------------------------------------------------------------------- the date is stored with the model

def test_the_recorded_date_wins_and_a_record_without_one_leaves_the_request():
    def record(**training_args):
        return {"training": {"training_args": training_args}}

    assert resolve_chat_template_date(record(chat_template_date="2024-07-26"), "2025-03-04") == ("2024-07-26", True)
    assert resolve_chat_template_date(record(chat_template_date="2024-07-26"), None) == ("2024-07-26", True)
    assert resolve_chat_template_date(record(max_steps=3), "2025-03-04") == ("2025-03-04", False)  # nothing recorded
    assert resolve_chat_template_date(record(chat_template_date=None), "2025-03-04") == ("2025-03-04", False)
    assert resolve_chat_template_date(record(max_steps=3), None) == (None, False)  # the default applies at render time
    # no record at all (a fresh model)
    assert resolve_chat_template_date(None, "2025-03-04") == ("2025-03-04", False)
    assert resolve_chat_template_date({}, None) == (None, False)
    assert resolve_chat_template_date({"training": "no training_args"}, "2025-03-04") == ("2025-03-04", False)


@pytest.fixture
def dated_loader(monkeypatch):
    """``_load_model`` with a backbone stub and a tokenizer loader that returns a chat tokenizer with the dated template."""
    from gradiend.trainer.text.common import model_base as model_base_module

    class _Backbone:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return SimpleNamespace(name_or_path=name)

    class _Loader:
        @classmethod
        def from_pretrained(cls, name, *args, **kwargs):
            return InstructTokenizerWrapper(_dated_tokenizer())

    monkeypatch.setattr(model_base_module, "AutoTokenizerForLM", _Loader)
    monkeypatch.setattr(model_base_module, "AutoModelForLM", _Backbone)


def _prompt_after_load(*, record=None, asked=None, from_checkpoint=True):
    """What the tokenizer of a model loaded by ``_load_model`` renders for the user text ``hi``."""
    if from_checkpoint:
        kwargs = {"base_model_id": "org/model", "gradiend_kwargs": record}
        _, tokenizer = TextModelWithGradiend._load_model(
            "checkpoint", chat_template_date=asked, base_model_offload="allow", **kwargs
        )
    else:
        _, tokenizer = TextModelWithGradiend._load_model("org/model", chat_template_date=asked, base_model_offload="allow")
    return tokenizer._wrap_prompt("hi")


def test_load_model_gives_the_chat_tokenizer_the_date(dated_loader):
    recorded = {"training": {"training_args": {"chat_template_date": "2024-07-26"}}}
    unrecorded = {"training": {"training_args": {"max_steps": 3}}}
    # the date a model was trained with wins over the one asked for now
    assert _prompt_after_load(record=recorded, asked="2025-03-04") == DATED_PROMPT.format(date="26 Jul 2024", text="hi")
    assert _prompt_after_load(record=recorded) == DATED_PROMPT.format(date="26 Jul 2024", text="hi")
    # a record without a date: the date asked for, else the default
    assert _prompt_after_load(record=unrecorded, asked="2025-03-04") == DATED_PROMPT.format(date="04 Mar 2025", text="hi")
    assert _prompt_after_load(record=unrecorded) == DATED_PROMPT.format(date=DEFAULT_DATE_IN_PROMPT, text="hi")
    # a model that is not a checkpoint
    assert _prompt_after_load(asked="2025-03-04", from_checkpoint=False) == DATED_PROMPT.format(date="04 Mar 2025", text="hi")
    assert _prompt_after_load(from_checkpoint=False) == DATED_PROMPT.format(date=DEFAULT_DATE_IN_PROMPT, text="hi")


def test_a_recorded_date_survives_a_change_of_the_default(dated_loader, monkeypatch):
    recorded = {"training": {"training_args": {"chat_template_date": "2024-07-26"}}}
    unrecorded = {"training": {"training_args": {"max_steps": 3}}}
    monkeypatch.setattr(chat_format_module, "DEFAULT_CHAT_TEMPLATE_DATE", "2030-01-01")
    assert _prompt_after_load(record=recorded) == DATED_PROMPT.format(date="26 Jul 2024", text="hi")
    # positive control: what has no record follows the (new) default
    assert _prompt_after_load(record=unrecorded) == DATED_PROMPT.format(date="01 Jan 2030", text="hi")
    assert _prompt_after_load(from_checkpoint=False) == DATED_PROMPT.format(date="01 Jan 2030", text="hi")


def test_the_date_travels_from_training_arguments_to_the_tokenizer(dated_hub):
    """``from_pretrained(training_args=...)`` hands ``chat_template_date`` down to the tokenizer."""
    tokenizer = _dated_tokenizer()
    with_date = _model_for("org/tiny-Instruct", tokenizer, chat_template_date="2025-03-04")
    assert with_date.tokenizer._wrap_prompt("hi") == DATED_PROMPT.format(date="04 Mar 2025", text="hi")
    default = _model_for("org/tiny-Instruct", tokenizer)
    assert default.tokenizer._wrap_prompt("hi") == DATED_PROMPT.format(date=DEFAULT_DATE_IN_PROMPT, text="hi")
    # a date for a model that is not an Instruct model changes nothing
    plain = _model_for("org/tiny", tokenizer, chat_template_date="2025-03-04")
    assert not uses_chat_format(plain.tokenizer)


def test_a_model_resolves_the_date_only_when_its_template_asks_for_it(dated_hub, monkeypatch):
    tokenizer = _dated_tokenizer()
    default = _model_for("org/tiny-Instruct", tokenizer)
    assert default.resolved_training_settings() == {"instruction_model": True, "chat_template_date": DEFAULT_DATE}
    given = _model_for("org/tiny-Instruct", tokenizer, chat_template_date="2025-03-04")
    assert given.resolved_training_settings() == {"instruction_model": True, "chat_template_date": "2025-03-04"}
    # the two settings are recorded with the training arguments; they are not GRADIEND construction metadata
    assert not {"instruction_model", "chat_template_date"} & set(given.gradiend.kwargs)
    assert not uses_chat_format(_model_for("org/tiny", tokenizer).tokenizer)
    # plain text: the decision is stored too (False, never None), there is no date
    assert _model_for("org/tiny", tokenizer).resolved_training_settings() == {"instruction_model": False}
    forced = _model_for("org/tiny", tokenizer, instruction_model=True)
    assert forced.resolved_training_settings() == {"instruction_model": True, "chat_template_date": DEFAULT_DATE}
    off = _model_for("org/tiny-Instruct", tokenizer, instruction_model=False)
    assert off.resolved_training_settings() == {"instruction_model": False}
    monkeypatch.setattr(loading_module, "AutoTokenizer", _FakeAutoTokenizer)  # a ChatML template never asks
    chatml = _model_for("org/tiny-Instruct", _chatml_tokenizer(), chat_template_date="2025-03-04")
    assert uses_chat_format(chatml.tokenizer) and chatml.resolved_training_settings() == {"instruction_model": True}
    assert ModelWithGradiend.resolved_training_settings(None) == {}  # the base class resolves nothing


def test_the_trainer_stores_the_resolved_settings_in_the_training_arguments(dated_hub):
    model = _model_for("org/tiny-Instruct", _dated_tokenizer())
    args = TrainingArguments()
    assert args.instruction_model is None
    Trainer._record_resolved_model_settings(model, args)
    assert args.instruction_model is True and args.chat_template_date == DEFAULT_DATE
    restored = TrainingArguments.from_dict(args.to_dict())  # they are serialised
    assert restored.instruction_model is True and restored.chat_template_date == DEFAULT_DATE
    explicit = TrainingArguments(chat_template_date="2025-03-04")
    Trainer._record_resolved_model_settings(_model_for("org/tiny-Instruct", _dated_tokenizer(), chat_template_date="2025-03-04"), explicit)
    assert explicit.chat_template_date == "2025-03-04"
    # a plain model stores the decision False, and no date
    plain = TrainingArguments()
    Trainer._record_resolved_model_settings(_model_for("org/tiny", _dated_tokenizer()), plain)
    assert plain.instruction_model is False and plain.chat_template_date is None
    # nothing to record for something that is not a model of this package
    untouched = TrainingArguments()
    Trainer._record_resolved_model_settings(SimpleNamespace(resolved_training_settings=lambda: {"instruction_model": True}), untouched)
    assert untouched.instruction_model is None and untouched.chat_template_date is None


def test_the_log_line_says_when_the_date_is_recorded_with_the_model(dated_loader, caplog):
    record = {"training": {"training_args": {"chat_template_date": "2024-07-26"}}}
    with caplog.at_level(logging.INFO, logger=CHAT_LOGGER):
        _, tokenizer = TextModelWithGradiend._load_model(
            "checkpoint", base_model_id="org/model", gradiend_kwargs=record, chat_template_date="2025-03-04",
            base_model_offload="allow",
        )
        tokenizer._wrap_prompt("hi")
    (message,) = _date_messages(caplog)
    assert "2024-07-26" in message and "recorded with the model" in message
    assert "configured" not in message and "default" not in message and "chat_template_date" not in message
    # the same format object, asked by hand: the flag only words the log, the date is the one given
    caplog.clear()
    wrapper = InstructTokenizerWrapper(_dated_tokenizer())
    wrapper.chat_format.configure("2025-03-04", recorded=True)
    with caplog.at_level(logging.INFO, logger=CHAT_LOGGER):
        assert wrapper._wrap_prompt("hi") == DATED_PROMPT.format(date="04 Mar 2025", text="hi")
    assert "recorded with the model" in _date_messages(caplog)[0]
    wrapper.chat_format.configure(None, recorded=True)  # no date: the default, which is never "recorded"
    assert wrapper.chat_format.chat_template_date == DEFAULT_DATE


def test_the_trainer_adopts_the_recorded_date_and_warns_only_when_it_replaces_one(caplog):
    saved = {"label_token_protocol": FDT, "chat_template_date": "2024-07-26"}
    with caplog.at_level(logging.WARNING):
        _, trainer = _configured_model({"chat_template_date": "2025-03-04"}, saved, InstructTokenizerWrapper(_dated_tokenizer()))
    assert trainer._training_args.chat_template_date == "2024-07-26"
    assert [r for r in caplog.records if "chat_template_date" in r.getMessage()]
    caplog.clear()
    with caplog.at_level(logging.WARNING):  # the trainer asked for nothing: nothing to warn about
        _, trainer = _configured_model({}, saved, InstructTokenizerWrapper(_dated_tokenizer()))
    assert trainer._training_args.chat_template_date == "2024-07-26"
    assert not [r for r in caplog.records if "chat_template_date" in r.getMessage()]
    # a record without a date leaves the trainer's own
    _, trainer = _configured_model({"chat_template_date": "2025-03-04"}, {"label_token_protocol": FDT}, _gpt2())
    assert trainer._training_args.chat_template_date == "2025-03-04"


def test_a_model_the_trainer_hands_out_has_its_frame_rendered(caplog):
    wrapper = InstructTokenizerWrapper(_dated_tokenizer())
    with caplog.at_level(logging.INFO, logger=CHAT_LOGGER):
        model, _ = _configured_model({}, None, wrapper)
    assert model.tokenizer.chat_format.inserts_date is True  # rendered now, not at the first row
    assert len(_date_messages(caplog)) == 1
    # a template that cannot frame a system prompt fails before training
    raw = _chatml_tokenizer()
    raw.chat_template = "{{- 'no user turn here' }}"
    with pytest.raises(ValueError, match="gpt2"):
        _configured_model({}, None, InstructTokenizerWrapper(raw))


def _hub_audit_trainer(training_args):
    from gradiend.trainer.text.prediction.trainer import TextPredictionConfig

    data = pd.DataFrame([{
        "masked": "The word was [MASK]", "factual": "anglicised", "alternative": "anglicized",
        "split": "train", "label_class": "UK", "alternative_class": "US",
    }])
    cfg = TextPredictionConfig(
        target_classes=["UK", "US"], data=data, masked_col="masked", label_col="factual",
        alternative_col="alternative", label_class_col="label_class",
        alternative_class_col="alternative_class", split_col="split",
    )
    return TextPredictionTrainer(model="some-model-instruct", config=cfg, training_args=training_args)


def test_hub_id_audit_applies_instruction_model_and_the_date(monkeypatch):
    seen = []

    def fake_from_pretrained(cls, name, *args, **kwargs):
        seen.append(kwargs)
        return InstructTokenizerWrapper(_dated_tokenizer())

    monkeypatch.setattr(AutoTokenizerForLM, "from_pretrained", classmethod(fake_from_pretrained))
    trainer = _hub_audit_trainer(
        TrainingArguments(label_token_protocol=FDT, instruction_model=True, chat_template_date="2025-03-04")
    )
    tokenizer = trainer._tokenizer_for_clm_audit()
    assert seen[0]["instruction_model"] is True
    assert tokenizer.chat_format.chat_template_date == "2025-03-04"
    # without a date the audit uses the default, as every other path does
    default = _hub_audit_trainer(TrainingArguments(label_token_protocol=FDT))._tokenizer_for_clm_audit()
    assert default.chat_format.chat_template_date == DEFAULT_DATE and default.chat_format.inserts_date is True


def test_a_recreated_model_is_configured_too(monkeypatch):
    """Pre-pruning re-creates the model; it must get the same configuration as every other model the trainer hands out."""
    import gradiend.trainer.trainer as trainer_module

    wrapper = InstructTokenizerWrapper(_dated_tokenizer())
    model = SimpleNamespace(gradiend=SimpleNamespace(kwargs={}), tokenizer=wrapper)
    monkeypatch.setattr(trainer_module, "create_model_with_gradiend", lambda *args, **kwargs: model)

    class _Trainer(TextPredictionTrainer):
        base_model_path = "org/model"

        def __init__(self):
            self._training_args = TrainingArguments(label_token_protocol=FDT)

        @property
        def model_with_gradiend_cls(self):
            return object

        def resolve_model_path(self, path):
            return path

        def _ensure_data_for_training(self):
            pass

    old = SimpleNamespace(gradiend=SimpleNamespace(_lazy_init=False), base_model=None)
    training_args = SimpleNamespace(pre_prune_config=object(), trust_remote_code=False)
    assert wrapper.chat_format.inserts_date is False
    assert _Trainer()._prepare_model_for_pre_prune_if_needed(old, training_args) is model
    assert wrapper.chat_format.inserts_date is True  # its chat frame was rendered by the configuration
    assert model.label_token_protocol == FDT


# ----------------------------------------------------------------------------- end to end: training stores the date

def _train_tiny_model(tmp_path, name, **arguments):
    """Train a few steps of a tiny GPT-2 (with the dated chat template) saved as ``name``.

    Returns:
        ``(base dir, trained dir, the training_args stored in training.json)``.
    """
    from gradiend.trainer.text.prediction.trainer import TextPredictionConfig

    base_dir = tmp_path / name
    tokenizer = _dated_tokenizer()
    tokenizer.save_pretrained(base_dir)
    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer, config.n_head, config.n_embd, config.vocab_size = 2, 2, 32, len(tokenizer)
    AutoModelForCausalLM.from_config(config).save_pretrained(base_dir)

    sentences = [
        "The doctor said [MASK] was tired", "The nurse told me [MASK] was late", "My friend knew [MASK] was away",
        "The teacher thought [MASK] was ready", "The pilot felt [MASK] was lucky", "The singer hoped [MASK] was right",
    ]
    rows = []
    for split, repeats in (("train", 3), ("validation", 1), ("test", 1)):
        for _ in range(repeats):
            for text in sentences:
                rows.append({"masked": text, "factual": "she", "alternative": "he", "split": split,
                             "label_class": "F", "alternative_class": "M"})
                rows.append({"masked": text, "factual": "he", "alternative": "she", "split": split,
                             "label_class": "M", "alternative_class": "F"})
    cfg = TextPredictionConfig(
        target_classes=["F", "M"], data=pd.DataFrame(rows), masked_col="masked", label_col="factual",
        alternative_col="alternative", label_class_col="label_class",
        alternative_class_col="alternative_class", split_col="split",
    )
    args = TrainingArguments(
        experiment_dir=str(tmp_path / "experiment"), max_steps=3, eval_steps=3, train_batch_size=2, do_eval=False,
        use_cache=False, learning_rate=1e-4, **arguments,
    )
    TextPredictionTrainer(model=str(base_dir), config=cfg, args=args).train()
    trained_dir = tmp_path / "experiment" / "model"
    return base_dir, trained_dir, json.loads((trained_dir / "training.json").read_text(encoding="utf-8"))["training_args"]


@pytest.mark.parametrize(
    "name,arguments,chat,recorded_date",
    [
        ("tiny-instruct", {}, True, DEFAULT_DATE),  # the date the template got is stored, although none was asked for
        ("tiny-instruct", {"chat_template_date": "2025-03-04"}, True, "2025-03-04"),
        ("tiny", {}, False, None),  # a plain model stores the decision False, and has no date
        ("tiny-it", {}, True, DEFAULT_DATE),  # the name rule marks ``-it``
        ("tiny-instruct", {"instruction_model": False}, False, None),  # forced to plain text
        ("tiny", {"instruction_model": True}, True, DEFAULT_DATE),  # forced to the chat format
    ],
)
def test_training_stores_the_decision_and_the_date_and_a_reload_keeps_them(
    tmp_path, monkeypatch, name, arguments, chat, recorded_date
):
    base_dir, trained_dir, training_args = _train_tiny_model(tmp_path, name, **arguments)
    assert training_args["instruction_model"] is chat  # the resolved decision, never None
    assert training_args.get("chat_template_date") == recorded_date

    # the default date and the name rule change later: a trained model keeps what it was trained with,
    # a fresh model (no record) follows the new defaults
    monkeypatch.setattr(chat_format_module, "DEFAULT_CHAT_TEMPLATE_DATE", "2030-01-01")
    monkeypatch.setattr(loading_module, "instruction_name_marker", lambda _name: None if chat else "instruct")
    trained = TextPredictionModelWithGradiend.from_pretrained(str(trained_dir))
    fresh = TextPredictionModelWithGradiend.from_pretrained(str(base_dir))
    assert uses_chat_format(trained.tokenizer) is chat
    assert uses_chat_format(fresh.tokenizer) is (not chat)
    if chat:
        trained_date = datetime.date.fromisoformat(recorded_date).strftime("%d %b %Y")
        assert trained.tokenizer._wrap_prompt("hi") == DATED_PROMPT.format(date=trained_date, text="hi")
    else:
        assert fresh.tokenizer._wrap_prompt("hi") == DATED_PROMPT.format(date="01 Jan 2030", text="hi")


@pytest.fixture
def old_rule_dir():
    """A temporary directory with no ``instruct`` in its path.

    The old name rule reads the whole path, parent directories included, so a test directory below a checkout named
    ``...-instruct-...`` would decide for the model in it.
    """
    path = Path(tempfile.mkdtemp(prefix="gradiend_old_rule_"))
    assert not legacy_instruction_name_rule(path)
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.parametrize("name,chat", [("tiny-it", False), ("tiny-instruct", True)])
def test_a_model_saved_before_the_decision_was_stored_is_read_by_the_old_name_rule(old_rule_dir, name, chat):
    base_dir, trained_dir, training_args = _train_tiny_model(old_rule_dir, name)
    assert training_args["instruction_model"] is True  # both are chat models under today's rule (``-it`` is a marker)
    # make the record look like one written before the decision was stored
    path = trained_dir / "training.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["training_args"]["instruction_model"]
    path.write_text(json.dumps(record), encoding="utf-8")
    model = TextPredictionModelWithGradiend.from_pretrained(str(trained_dir))
    assert uses_chat_format(model.tokenizer) is chat  # ``-it`` was a plain model before, ``-Instruct`` a chat model
