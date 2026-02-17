from gradiend import TextPredictionTrainer, TrainingArguments

model_name = "distilbert-base-cased"

args = TrainingArguments(
    train_batch_size=8,
    encoder_eval_max_size=10,
    decoder_eval_max_size_training_like=100,
    decoder_eval_max_size_neutral=100,
    eval_steps=25,
    max_steps=100,
    source="alternative",
    target="diff",
    learning_rate=1e-4,
)

race_white_black_config = ("race", ("white", "black"), ["asian"])
religion_christian_muslim_config = ("religion", ("christian", "muslim"), ["jewish"])
for bias_type, pair, other_classes in [race_white_black_config, religion_christian_muslim_config]:
    trainer = TextPredictionTrainer(
        model=model_name,
        run_id=f"{bias_type}_{pair[0]}_{pair[1]}",
        data=f"aieng-lab/gradiend_{bias_type}_data",
        target_classes=pair,
        masked_col="masked",
        eval_neutral_data="aieng-lab/biasneutral",
        args=args,
    )

    print(f"=== {bias_type}: {pair[0]} vs {pair[1]} ===")
    trainer.train()

    stats = trainer.get_training_stats()
    ts = stats["training_stats"]
    corr = ts["correlation"]
    print(f"  correlation={corr}, mean_by_class={ts['mean_by_class']}")

    max_size = 100
    enc_eval = trainer.evaluate_encoder(max_size=max_size, return_df=True)
    enc_df = enc_eval["encoder_df"]
    trainer.plot_encoder_distributions(encoder_df=enc_df)
    # Encoder metrics without cache: pass encoder_df explicitly. No experiment_dir or use_cache needed.
    enc_metrics = trainer.get_encoder_metrics(encoder_df=enc_df)
    print(f"  encoder metrics: {enc_metrics['all_data']}")

    dec_results = trainer.evaluate_decoder()
    stats = dec_results['summary'][f"prob::{pair[0]}"]
    print(f"  decoder {pair[0]} statistics: {stats}")

    # Save one changed model biasing towards the first class in the pair
    changed_model = trainer.select_changed_model(
        decoder_results=dec_results,
        metric_key=pair[0],  # or metric_key=list(pair) to save one model per class
    )

    # do something with the changed model
