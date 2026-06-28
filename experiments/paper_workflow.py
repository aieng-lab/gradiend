from gradiend import TextFilterConfig, TextPredictionDataCreator, TrainingArguments, TextPredictionTrainer, \
    PrePruneConfig, PostPruneConfig, check_plot_environment

check_plot_environment()


creator = TextPredictionDataCreator(
    base_data='wikimedia/wikipedia',
    hf_config='20231101.en',
    feature_targets=[
        TextFilterConfig(targets=["he", "she", "it"], id="3SG"),
        TextFilterConfig(targets=["they"], id="3PL"),
    ],
    min_left_context_words=10,
    use_cache=True,
)
training_data = creator.generate_training_data(max_size_per_class=1000)
neutral_data = creator.generate_neutral_data(
    additional_excluded_words=["i", "we", "you"],
    max_size=1000,
)

model="bert-base-cased"

args = TrainingArguments(
    train_batch_size=16,
    max_steps=200,
    eval_steps=20,
    learning_rate=1e-5,
    experiment_dir=f'runs/demonstration-{model}',
    use_cache=True,
    pre_prune_config=PrePruneConfig(n_samples=8, topk=0.1),
    post_prune_config=PostPruneConfig(topk=0.01),
)
trainer = TextPredictionTrainer(
    model=model,
    data=training_data,
    eval_neutral_data=neutral_data,
    img_format="pdf", # different from paper script, to persist the figures
    args=args,
)

trainer.train()
trainer.plot_training_convergence(class_spread="iqr")

enc_result = trainer.evaluate_encoder(plot=True, plot_kwargs={'figsize': (5, 2), 'legend_fontsize': 7})
print("Correlation:", enc_result["correlation"])
print("Mean by class:", enc_result["mean_by_feature_class"])
dec = trainer.evaluate_decoder(plot=True, target_class="3SG", use_cache=True, plot_kwargs={'figsize': (7, 4.5)}) #, lrs=[-10, -1, -0.1, -0.01, -0.001])
print(dec["3SG"]["learning_rate"])
changed_base_model = trainer.rewrite_base_model(decoder_results=dec, target_class="3SG")
