"""
German DE decoder-only workflow: DecoderModelWithMLMHead replaces AutoModelForMaskedLM.
Create Trainer(model=base_model, ...); optionally train the MLM head first with
train_decoder_only_mlm_head(); then trainer.train() uses the MLM head path when it exists.
"""

from gradiend import TextPredictionTrainer, TrainingArguments, PrePruneConfig, PostPruneConfig


base_model = "dbmdz/german-gpt2"

args = TrainingArguments(
    experiment_dir="runs/examples/german_de_decoder_only",
    train_batch_size=8,
    encoder_eval_max_size=10,
    train_max_size=500,
    decoder_eval_max_size_training_like=100,
    decoder_eval_max_size_neutral=100,
    eval_steps=100,
    max_steps=1000,
    source="alternative",
    target="diff",
    learning_rate=1e-3,
    add_identity_for_other_classes=True,
    use_cache=True,
    pre_prune_config=PrePruneConfig(n_samples=16, topk=0.01),
    post_prune_config=PostPruneConfig(topk=0.01),
)

pair = ("masc_nom", "fem_nom")
trainer = TextPredictionTrainer(
    model=base_model,
    run_id="gender_de_masc_nom_fem_nom",
    data="aieng-lab/de-gender-case-articles",
    target_classes=list(pair),
    masked_col="masked",
    split_col="split",
    eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
    args=args,
)
print("=== German DE decoder-only: MLM head is drop-in for MaskedLM; resolve automatically ===\n")

# 1) Optionally train decoder-only MLM head (same data as GRADIEND; masked must contain [MASK])
#    Returns path; trainer.train() will use it automatically when it exists
print("1) Training decoder-only MLM head (optional)...")
mlm_head_path = trainer.train_decoder_only_mlm_head(base_model, epochs=3, batch_size=4, max_size=1000)
print(f"   MLM head saved at {mlm_head_path}\n")

# 2) trainer.train() resolves to MLM head path when it exists; returns self (trainer)
trainer.train()

max_size = 100
enc_eval = trainer.evaluate_encoder(max_size=max_size, return_df=True)
enc_df = enc_eval.get("encoder_df")
if enc_df is not None:
    trainer.plot_encoder_distributions(encoder_df=enc_df)
print(f"  encoder metrics: {enc_eval}")

# 3) Decoder evaluation uses CLM (base decoder) only when base is DecoderModelWithMLMHead
print("2) Decoder evaluation...")
dec = trainer.evaluate_decoder()
summary = dec.get("summary", {})
for key in list(summary.keys())[:5]:
    if isinstance(summary[key], dict):
        b = summary[key]
        print(f"   best {key}: value={b.get('value')}, feature_factor={b.get('feature_factor')}, lr={b.get('learning_rate')}")
print(f"   Grid: {len(dec.get('grid', {}))} (feature_factor, lr) combinations\n")

