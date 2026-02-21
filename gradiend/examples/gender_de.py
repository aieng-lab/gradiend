from gradiend import TextPredictionTrainer, TrainingArguments

model_name = "bert-base-german-cased"
args = TrainingArguments(
    experiment_dir="runs/examples/german_de",
    train_batch_size=8,
    encoder_eval_max_size=20,
    eval_steps=100,
    num_train_epochs=1,
    max_steps=500,
    source="alternative",
    target="diff",
    eval_batch_size=8,
    learning_rate=1e-4,
    use_cache=False,
    add_identity_for_other_classes=True,
    max_seeds=3,
    min_convergent_seeds=2,
)

pair = ("masc_nom", "fem_nom") # nominative male vs female (along a gender axis)
trainer = TextPredictionTrainer(
    model=model_name,
    run_id=f"gender_de_{pair[0]}_{pair[1]}",
    data="aieng-lab/de-gender-case-articles",
    target_classes=list(pair),
    masked_col="masked",
    split_col="split",
    eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
    args=args,
    img_format='png',
)

print(f"=== German gender: {pair[0]} vs {pair[1]} (BERT) ===")
trainer.train()
print(f"Using cached model at {trainer.model_path}" if getattr(trainer, "_last_train_used_cache", False) else f"Model saved at {trainer.model_path}")
trainer.plot_training_convergence()
ts = (trainer.get_training_stats() or {}).get("training_stats", {})
print(f"  correlation={ts.get('correlation')}, mean_by_class={ts.get('mean_by_class')}")

enc_results = trainer.evaluate_encoder(max_size=100, return_df=True, plot=True, use_cache=False)
print(f"  encoder metrics: {enc_results}")

dec_results = trainer.evaluate_decoder()
print(f"  decoder summary: {list(dec_results['summary'].keys())}")
print(f"  decoder grid size: {len(dec_results['grid'])}")

trainer.rewrite_base_model(
    decoder_results=dec_results,
    output_dir="./output",
    metric_key="masc_nom",
)
