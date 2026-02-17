# Decoder-only models

Decoder-only (causal) language models are supported in two ways:

1. **Default mode (no custom head):** use the standard workflow; gradients are computed with CLM context up to `[MASK]`.
2. **Optional custom MLM head:** train a lightweight head to improve masked-token gradients when left-context alone is insufficient.

## How to use decoder-only models

You can create a `TextPredictionTrainer` with a decoder-only model (e.g. GPT-2) and run the same training and evaluation pipeline. By default, the model predicts the target token from the left context of the `[MASK]`. This is often enough for training, but it can be too little context for accurate token prediction in some datasets.

## Optional: train a custom MLM head

Decoder-only models naturally only see the left context of the `[MASK]`. For GRADIEND training, this can yield weak or unstable gradients when the left context does not provide enough information to predict the target token.

The optional custom head pools hidden states around the `[MASK]` position and learns a classifier over the target tokens. It is trained via `TextPredictionTrainer.train_decoder_only_mlm_head()`. Key details:

- Target token set is restricted to the labels present in training data (single-token labels only).
- Pooling length of typically 3–5 tokens after `[MASK]` is usually sufficient to approximate MLM-like gradients.
- The head is a lightweight classifier and does not replace the base decoder.

Trade-off: GRADIEND encodings are typically less sharp than for full encoder-only/MLM models, but this approach works broadly and yields stable GRADIEND training.

> Once a custom head is trained, it is used for GRADIEND training and evaluation *automatically* (if 'experiment_dir' is given).

## Code outline

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

## Notes

- Encoder analysis on neutral data uses CLM gradients, since the custom MLM head only provides gradients for its target tokens (neutral targets are not defined there).
- Decoder evaluation (probabilities of predicting target tokens) uses the base decoder even when the model includes a custom MLM head. This keeps evaluation grounded in the actual base model behavior rather than the auxiliary head.
