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
