from gradiend import TextPredictionTrainer, TrainingArguments, PrePruneConfig, PostPruneConfig


if __name__ == "__main__":
    pair = ("masc_nom", "fem_nom")
    args = TrainingArguments(
        experiment_dir="runs/examples/german_de",
        train_batch_size=8,
        encoder_eval_max_size=20,
        train_max_size=500,
        decoder_eval_max_size_training_like=100,
        decoder_eval_max_size_neutral=500,
        eval_steps=25,
        num_train_epochs=1,
        max_steps=250,
        source="alternative",
        target="diff",
        eval_batch_size=8,
        learning_rate=1e-4,
        use_cache=False,
        add_identity_for_other_classes=True,
        max_seeds=5,
        min_convergent_seeds=2,
        fail_on_non_convergence=True,
        pre_prune_config=PrePruneConfig(n_samples=16, topk=0.1, source="diff"),
        post_prune_config=PostPruneConfig(topk=0.001, part="decoder-weight"),
    )
    trainer = TextPredictionTrainer(
        model="bert-base-german-cased",
        run_id=f"gender_de_{pair[0]}_{pair[1]}",
        data="aieng-lab/de-gender-case-articles",
        target_classes=list(pair),
        masked_col="masked",
        split_col="split",
        eval_neutral_data="aieng-lab/wortschatz-leipzig-de-grammar-neutral",
        args=args,
        img_format="png",
    )

    print(f"=== German gender: {pair[0]} vs {pair[1]} (BERT) ===")
    trainer.train()
    print(f"Model: {trainer.model_path}")

    trainer.plot_training_convergence()
    enc_results = trainer.evaluate_encoder(
        max_size=100,
        return_df=True,
        plot=True,
        plot_kwargs={"target_and_neutral_only": False},
    )
    print(f"  encoder metrics: {enc_results}")

    dec_results = trainer.evaluate_decoder()
    print(f"  decoder grid size: {len(dec_results['grid'])}")
    trainer.rewrite_base_model(
        decoder_results=dec_results,
        output_dir="./output",
        target_class="masc_nom",
    )
    trainer.plot_probability_shifts(decoder_results=dec_results)
