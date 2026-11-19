# Decoder-Only Models

This page focuses on decoder-only caveats. For the full list of prediction
objectives across BERT-like, GPT-like, and seq2seq models, see
[Token prediction methods](token-prediction-methods.md).

Decoder-only (causal) language models are supported in two ways:

1. **Next-token mode (default):** gradients from the left context up to `[MASK]`
   (`prediction_objective="auto"` or `"clm_next_token"` on GPT-style models).
2. **Optional auxiliary MLM head:** a lightweight head for masked-token gradients
   when left context alone is insufficient (`clm_mlm_head`).

## Next-token mode (`clm_next_token`)

You can create a [`TextPredictionTrainer`][gradiend.trainer.text.prediction.trainer.TextPredictionTrainer] with a decoder-only model (e.g. GPT-2)
and run the same training and evaluation pipeline. The model predicts the target
token from the **left context** before `[MASK]` (there is no true bidirectional mask).

When generating decoder-only data, make sure the target has enough left context.
[`TextFilterConfig`][gradiend.data.text.filter_config.TextFilterConfig].min_left_context_words is designed for this. Very short
prefixes often make the next-token objective noisy and can hurt convergence.

### Code outline (English gender, next-token only)

```python
from gradiend import TextPredictionTrainer, TrainingArguments

args = TrainingArguments(
    experiment_dir="runs/gender_en_decoder_only",
    prediction_objective="clm_next_token",
    train_batch_size=8,
    eval_steps=25,
    max_steps=100,
    source="factual",
    target="diff",
    learning_rate=1e-4,
    use_cache=False,
)

trainer = TextPredictionTrainer(model="gpt2", ..., args=args)
trainer.train()
enc_eval = trainer.evaluate_encoder(max_size=100, use_cache=False, return_df=True)
dec_results = trainer.evaluate_decoder()
```

[:material-file-code-outline: `train_gender_en.py`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_en.py)

**Runnable example:** [train_gender_en.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_en.py)
— set `DECODER_ONLY = True` in the script or pass `--decoder-only` (see that file).

## Multi-token targets and divergence-point supervision

**An example.** Take the sentence "In my opinion the spelling is [MASK]", where the factual word is `anglicised` and the alternative is `anglicized`. A GPT-2 tokenizer splits them like this:

| word | tokens |
|---|---|
| `anglicised` (factual) | ` ang` `lic` `ised` |
| `anglicized` (alternative) | ` ang` `lic` `ized` |

A causal model predicts one token at a time. If GRADIEND supervised only the first token, factual and alternative would both be ` ang`: the same label on both sides, so no contrast and nothing to learn. The two words only differ at the third token. So GRADIEND treats the shared start as context and supervises the first token where they differ:

- **model input:** "In my opinion the spelling is ang lic"
- **factual label:** `ised`, **alternative label:** `ized`

This is **divergence-point alignment**, and it happens automatically. In general, let $F$ and $A$ be the token sequences of the factual and alternative continuations, and let $k$ be the length of their longest common prefix. When $k < 	ext{len}(F)$ and $k < 	ext{len}(A)$:

1. **Shared prefix as context:** The model input is formed by the preceding context plus the $k$ shared tokens $F[:k]$ (in the example: ` ang` `lic`, $k = 2$).
2. **Divergence-point contrast:** The factual label is $F[k]$ and the alternative label is $A[k]$, supervising next-token prediction at token position $k+1$ (`ised` vs `ized`).
3. When $k = 0$ (nothing is shared, as for `she` vs `he`, or any multi-token targets with different first tokens), the label is the first token of each target, i.e. the leading-space variant (`tokenizer(" " + word)`) when the target follows whitespace. This is the token `create_inputs` already used; gradiend 0.2.5 and earlier trained on the no-space token instead (see the label token protocol below).

### Label token protocol

The `label_token_protocol` field of [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments] decides which token a decoder-only training row supervises. For the sentence "The doctor said that [MASK]" with the target `she`, the model really produces the word with its leading space (` she`, as in "that she"). `first_differing_token` supervises that token; the older rule supervised `she` without the space, a word piece that hardly ever follows a space. The two options:

- `"first_differing_token"` (default): the divergence-point rule above, applied to every row that has an alternative (also when $k = 0$). Rows without an alternative follow the prefix like a contrast row: the leading-space variant after a prefix that ends in whitespace, the literal token after a glued or empty prefix (for example `The word is[MASK]`). Training and `create_inputs(text, label, alternative=...)` use it for every row. Row-wise decoder evaluation uses it for rows whose targets share a token prefix and applies `clm_unalignable_rows` to rows that cannot be aligned; all other rows keep the scoring of gradiend 0.2.5.
- `"no_leading_space"`: the rule of gradiend 0.2.5 and earlier (`vocab[label]`, the no-space token), without alignment and without the pre-training audit. Use it only to reproduce models with a plain tokenizer trained with those releases; Instruct models (chat tokenizer) do not support it, see [Instruct models](#instruct-models).

A trained model keeps the rule it was trained with: when a trainer loads a saved model, the label rule recorded in that model's `training.json` is used for its datasets and evaluation instead of the trainer's own `label_token_protocol` (a warning names the difference). A saved setting without the key (a model from gradiend 0.2.5 or earlier) loads as `no_leading_space`, and the old names `canonical` / `legacy` still load (as `first_differing_token` / `no_leading_space`).

### Unalignable targets and pre-training audit

Alignment needs a place where the two words disagree. If one target is a token prefix of the other (e.g. ` bride` vs ` bride` `groom`: the shorter word ends exactly where the longer one goes on) there is no such place, and if both words tokenize identically there is none either. Formally, if one target is a token prefix of the other (e.g. `bride` vs `bridegroom` where $k = \min(\text{len}(F), \text{len}(A))$) or if both sides tokenize to identical token sequences ($F = A$), contrastive next-token supervision is invalid because the shorter side has no valid next-token alternative.

GRADIEND performs a pre-training audit before training, pruning, or learning-rate search. You can configure the behavior via [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments]:

- `clm_unalignable_rows="error"` (default): fails fast with an aggregated [`ClmAlignmentError`][gradiend.trainer.text.prediction.clm_alignment.ClmAlignmentError] listing all offending pairs, token IDs, and counts.
- `clm_unalignable_rows="drop"`: automatically drops unalignable rows from all data splits and logs the dropped count per feature class.

You can also run [`audit_clm_alignment`][gradiend.trainer.text.prediction.clm_alignment.audit_clm_alignment] directly to inspect dataset alignment:

```python
from gradiend import audit_clm_alignment

summary_df = audit_clm_alignment(tokenizer, dataframe_or_pairs)
print(summary_df[["factual", "alternative", "status", "reason", "k"]])
```

## Instruct models

**What changes.** Instruct models (for example `Qwen/Qwen2.5-1.5B-Instruct` or `google/gemma-2-2b-it`) are trained to answer in a chat, not to continue raw text, so a bare sentence prefix is not what they expect. For such a model GRADIEND puts every row into a chat: the sentence before `[MASK]` becomes the user message, and the first tokens of the assistant's reply are what is supervised. You do not have to do anything: the model is recognised by its name (details below).

```python
args = TrainingArguments(prediction_objective="clm_next_token", experiment_dir="runs/instruct", ...)
# "Instruct" in the name: the rows are chat prompts automatically
trainer = TextPredictionTrainer(model="Qwen/Qwen2.5-1.5B-Instruct", ..., args=args)
```

**An example.** The row "The word is [MASK]" with factual `anglicised` and alternative `anglicized` (see the section above) is shown to a Qwen model like this (other families look different, because the layout is the one of the model's own tokenizer):

```
<|im_start|>system
You are a language model that completes sentences. Predict the next word ... Return only that word ...<|im_end|>
<|im_start|>user
The word is<|im_end|>
<|im_start|>assistant
ang lic <slot>
```

(The token split is the GPT-2 one from above, for illustration; a Qwen tokenizer may split the words differently.) The assistant's answer starts right after `assistant`, so the answer tokens have no leading space. The shared tokens `ang` `lic` are the start of the reply, and `<slot>` is a placeholder position whose label is `ised` (factual) or `ized` (alternative): the model predicts it from the end of the shared reply, exactly the divergence-point rule above. A row without shared tokens has just the prompt and the slot, and the label is the first token of the answer.

The same chat format is used on all data paths: training rows (contrast, identity and neutral rows), `create_inputs`, row-wise decoder evaluation, the language-model score and the alignment audit. The details:

- **Which models:** the base name of the model id or path (the part after the last `/`) is split at every character that is not a letter or a digit; if one of the words is `instruct`, `chat` or `it` (any case), the model is an Instruct model. `Qwen/Qwen2.5-1.5B-Instruct`, `google/gemma-2-2b-it`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and a local path `/models/my-model-it` match; `Qwen/Qwen2.5-1.5B`, `google/gemma-2-2b`, `fitness` (the word must stand alone) and a local path whose parent directory says `instruct` do not. When a tokenizer is chat-wrapped, one INFO line says which rule fired, which chat format is used and how to override it.
- **Override:** [`TrainingArguments(instruction_model=True)`][gradiend.trainer.core.arguments.TrainingArguments] chat-formats a model whose name does not match, `instruction_model=False` reads plain text even if the name matches (the model is then a plain model: plain rows, plain `create_inputs`, both label rules available). `None`, the default, decides by name. The decision in force (`True` or `False`, never `None`) is stored with every trained model, also with a plain one, and a reloaded model keeps it, whatever the name rule says later. A model saved before the decision was stored was trained under the old rule, `"instruct"` anywhere in the name (any case), and is loaded by that rule: an old plain-trained `gemma-2-2b-it` stays plain, an old Instruct model is chat-formatted again and then rejected (see **Label rule**). A value you set on the trainer for a loaded model does not override its record: the record wins and a warning says so.
- **Format:** the `chat_template` of the tokenizer (Llama, Qwen, Mistral, Gemma, ...); GRADIEND has no hand-written format of any family. A tokenizer without a chat template raises a `ValueError` that names the model; a family without a template plugs in with [`register_chat_format`][gradiend.trainer.text.common.chat_format.register_chat_format] and does not need any change in GRADIEND, or the model is read as plain text with `instruction_model=False`.
- **Date in the template:** some templates (Llama 3.1 and later, gpt-oss) write the current date into the system prompt by calling `strftime_now`. The template never reads the system clock, because a prompt that changes every day would make the rows differ with the day they are built. It gets a fixed date instead: `2024-07-26`, the date the Llama templates themselves fall back to, unless you pass another one with [`TrainingArguments(chat_template_date="2025-03-04")`][gradiend.trainer.core.arguments.TrainingArguments] (ISO `YYYY-MM-DD`). The template's own format applies (`26 Jul 2024` for `%d %b %Y`). These models work out of the box. The first time a template asks for the date, one INFO line names the model, the date used and how to change it; a template that never asks is not affected and logs nothing. When a model is trained, the date its template used is stored with it like `instruction_model` (also when it was the default), and a reloaded model keeps it, so a later change of the default never changes a trained model; a model stored without a date uses the current default.
- **Reasoning-style templates:** templates such as DeepSeek-R1 or gpt-oss render fine, but their generation prompt ends in a thinking or channel opener. "The answer is the first thing the assistant says" is therefore only an approximation for these models.
- **Rows:** the text before `[MASK]`, without its trailing whitespace, is the user message (a prefix with and without a trailing space gives the same row, in training, `create_inputs` and decoder evaluation). A training row is the chat prompt, then the $k$ shared answer tokens (if any), then one slot token that only carries the label; the label is the first differing token of the factual and the alternative answer and is predicted from the end of the generation prompt, the position at which the model starts its answer. Identity and neutral rows are built the same way with nothing shared.
- **Answer tokens:** the answer starts the assistant turn, so the words are tokenized without a leading space, and the alignment audit compares these tokens.
- **Long rows:** a context that does not fit into `max_length` is shortened from the left, inside the user message; the chat frame is always kept, and a `max_length` too small for the frame raises an error.
- **Label rule:** chat-formatted models use the `first_differing_token` rule only. A chat tokenizer combined with `label_token_protocol="no_leading_space"` raises a `ValueError` as early as possible: when a dataset is built, when the trainer sets up a model (also a loaded model whose record says `no_leading_space`), in `create_inputs` and in row-wise decoder evaluation. The Instruct rule of 0.2.5 (the whole prompt in Llama-3 tokens for every family, single-token labels, the label on the last prompt token) is not supported: a model trained with it cannot be reproduced and must be retrained.

## Optional: train a custom MLM head (`clm_mlm_head`)

Decoder-only models naturally only see the left context of the `[MASK]`. For GRADIEND
training, this can yield weak or unstable gradients when the left context does not
provide enough information to predict the target token (e.g. German articles where
the target depends on the following noun).

The optional custom head pools hidden states around the `[MASK]` position and learns
a classifier over the target tokens. It is trained via
[`TextPredictionTrainer`][gradiend.trainer.text.prediction.trainer.TextPredictionTrainer].train_decoder_only_mlm_head(). Key details:

- The head learns one classifier output per unique label string in the training data
  (multi-token strings are one class).
- Pooling length of typically 3–5 tokens after `[MASK]` is usually sufficient to
  approximate MLM-like gradients.
- The head is a lightweight classifier and does not replace the base decoder.

Trade-off: GRADIEND encodings are typically less sharp than for full encoder-only/MLM
models, but this approach works broadly and yields stable GRADIEND training.

> Once a custom head is trained, it is used for GRADIEND training automatically when
> `experiment_dir` is set (reload picks up `decoder_mlm_head/` under the run).

### Code outline (German gender, auxiliary MLM head)

```python
from gradiend import TextPredictionTrainer, TrainingArguments

base_model = "dbmdz/german-gpt2"
args = TrainingArguments(
    experiment_dir="runs/german_de_decoder_only",
    train_batch_size=8,
    eval_steps=25,
    max_steps=100,
    source="alternative",
    target="diff",
    learning_rate=1e-3,
    use_cache=False,
    add_identity_for_other_classes=True,
)

trainer = TextPredictionTrainer(
    model=base_model,
    run_id="gender_de_masc_nom_fem_nom",
    data="aieng-lab/de-gender-case-articles",
    target_classes=["masc_nom", "fem_nom"],
    masked_col="masked",
    split_col="split",
    eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
    args=args,
)

mlm_head_path = trainer.train_decoder_only_mlm_head(base_model, epochs=3, batch_size=4, max_size=1000)
trainer.train()

enc_eval = trainer.evaluate_encoder(max_size=100, use_cache=False, return_df=True)
dec_results = trainer.evaluate_decoder()
```

**Runnable example:** [train_gender_de_decoder_only.py](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_gender_de_decoder_only.py)

## Notes

- **eval_neutral_data** is optional. When omitted, decoder evaluation uses training-like
  data (test split with factual masks filled in) for LMS; target tokens are
  auto-ignored. See
  [Evaluation (intra-model)](../tutorials/evaluation-intra-model.md#neutral-data-for-decoder-evaluation-lms).
- Under `clm_mlm_head`, encoder analysis on neutral data uses **CLM** gradients (not
  the auxiliary head), since neutral targets are outside the head's label set.
- Decoder evaluation (probabilities of predicting target tokens) uses the base decoder
  even when the model includes a custom MLM head. This keeps evaluation grounded in
  the actual base model behavior rather than the auxiliary head.
