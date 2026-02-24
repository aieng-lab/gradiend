from gradiend import TextFilterConfig, TextPredictionDataCreator, TrainingArguments, TextPredictionTrainer, \
    PrePruneConfig, PostPruneConfig

NEUTRAL_EXCLUDE = ["i", "we", "you", "he", "she", "it", "they", "me", "us", "him", "her", "them"]

creator = TextPredictionDataCreator(
    base_data='wikipedia',
    hf_config='20220301.en',
    feature_targets=[
        TextFilterConfig(targets=["he", "she", "it"], id="3SG"),
        TextFilterConfig(targets=["they"], id="3PL"),
    ],
)
training_data = creator.generate_training_data(max_size_per_class=2500)
neutral_data = creator.generate_neutral_data(
    additional_excluded_words=NEUTRAL_EXCLUDE,
    max_size=1000,
)

args = TrainingArguments(
    train_batch_size=8,
    max_steps=250,
    eval_steps=25,
    num_train_epochs=1,
    learning_rate=1e-5,
    experiment_dir='runs/demonstration',
    use_cache=True,
    pre_prune_config=PrePruneConfig(n_samples=16, topk=0.01),
    post_prune_config=PostPruneConfig(topk=0.01),
)
trainer = TextPredictionTrainer(
    model="bert-base-cased",
    data=training_data,
    #eval_neutral_data=neutral_data,
    args=args,
)
trainer.train()
trainer.plot_training_convergence()

enc_result = trainer.evaluate_encoder(plot=True)
print("Correlation:", enc_result.get("correlation"))
dec = trainer.evaluate_decoder(plot=True, increase_target_probabilities=False)
print("Decoder cell for 3SG", dec["3SG"])
changed_base_model = trainer.rewrite_base_model(decoder_results=dec, target_class="3SG")

