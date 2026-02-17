"""
Basic GRADIEND workflow using English pronoun data.

Generates training and neutral data with TextPredictionDataCreator, saves to an
output folder, then loads from that generated data for GRADIEND training
(3SG he/she/it vs 3PL they).

Requires: pip install gradiend[data] transformers torch datasets
"""

from pathlib import Path

from gradiend import (
    TextPredictionTrainer,
    TextPredictionConfig,
    TrainingArguments,
)

# Output folder for generated data; training and trainer both use this
DATA_DIR = "data/english_pronouns"

def main():
    base = Path(DATA_DIR)
    training_path = base / "training.csv"
    neutral_path = base / "neutral.csv"

    config = TextPredictionConfig(
        run_id="pronoun_3sg_3pl",
        data=training_path,
        target_classes=["3SG", "3PL"],
        eval_neutral_data=neutral_path,
    )

    args = TrainingArguments(
        experiment_dir="runs/pronoun_workflow",
        train_batch_size=8,
        eval_steps=100,
        num_train_epochs=2,
        max_steps=500,
        source="factual",
        target="diff",
        eval_batch_size=4,
        learning_rate=1e-4,
    )

    print("\n=== Training ===")
    trainer = TextPredictionTrainer(
        model="bert-base-uncased",
        config=config,
        args=args,
    )
    trainer.train()
    trainer.plot_training_convergence()

    stats = trainer.get_training_stats()
    ts = stats.get("training_stats", {}) if stats else {}
    print(f"  correlation={ts.get('correlation')}")

    print("\n=== Encoder evaluation ===")
    trainer.evaluate_encoder(plot=True)
    enc_metrics = trainer.get_encoder_metrics(split="train")
    if enc_metrics:
        print(f"  {enc_metrics}")

    print("\n=== Decoder evaluation ===")
    dec = trainer.evaluate_decoder()
    summary = dec.get("summary", {})
    if summary:
        for k, v in summary.items():
            if isinstance(v, dict) and "value" in v:
                print(f"  {k}: {v.get('value')}")
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
