"""German decoder-only workflow with auxiliary MLM head (``clm_mlm_head``).

For plain decoder-only next-token prediction (``clm_next_token``), see
``train_gender_en.py`` with ``DECODER_ONLY = True`` or ``--decoder-only``.
"""

from gradiend import TextPredictionTrainer, TrainingArguments, PrePruneConfig, PostPruneConfig


if __name__ == "__main__":
    pair = ("masc_nom", "fem_nom")
    args = TrainingArguments(
        experiment_dir="runs/examples/german_de_decoder_only",
        train_batch_size=8,
        encoder_eval_max_size=10,
        decoder_eval_max_size_training_like=100,
        decoder_eval_max_size_neutral=100,
        train_max_size=500,
        eval_steps=100,
        max_steps=500,
        source="alternative",
        target="diff",
        learning_rate=1e-4,
        add_identity_for_other_classes=True,
        use_cache=True,
        fail_on_non_convergence=True,
        prediction_objective="auto",
        pre_prune_config=PrePruneConfig(n_samples=16, topk=0.01),
        post_prune_config=PostPruneConfig(topk=0.01),
    )
    trainer = TextPredictionTrainer(
        model="dbmdz/german-gpt2",
        run_id="gender_de_masc_nom_fem_nom",
        data="aieng-lab/de-gender-case-articles",
        target_classes=list(pair),
        masked_col="masked",
        split_col="split",
        eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
        args=args,
    )

    print("=== German decoder-only example ===")
    mlm_head_path = trainer.train_decoder_only_mlm_head("dbmdz/german-gpt2", epochs=3, batch_size=4, max_size=1000)
    print(f"MLM head saved at {mlm_head_path}")

    trainer.train()
    enc_eval = trainer.evaluate_encoder(max_size=100, return_df=True)
    enc_df = enc_eval.pop("encoder_df")
    trainer.plot_encoder_distributions(encoder_df=enc_df)
    print(f"  encoder metrics: {enc_eval}")

    dec = trainer.evaluate_decoder()
    for key, value in list({k: v for k, v in dec.items() if k not in {'grid', 'plot_path', 'plot_paths'}}.items())[:5]:
        if isinstance(value, dict):
            print(
                f"  best {key}: value={value.get('value')}, "
                f"feature_factor={value.get('feature_factor')}, lr={value.get('learning_rate')}"
            )
    print(f"  grid size: {len(dec.get('grid', {}))}")
