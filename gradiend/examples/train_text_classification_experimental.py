"""
Experimental TextClassificationTrainer example.

This is intentionally small and self-contained. Text classification support is
preliminary; TextPrediction remains the main documented workflow.
"""

from __future__ import annotations

from pathlib import Path

from gradiend import TrainingArguments
from gradiend.data.text.classification import TextClassificationDataCreator
from gradiend.trainer.text.classification import TextClassificationConfig, TextClassificationTrainer

DATA_DIR = Path("data/tweet_eval_sentiment")
POS_WORDS = ("love", "great", "amazing", "best", "excellent", "incredible", "awesome", "fantastic")
NEG_WORDS = ("hate", "terrible", "awful", "worst", "disappointed", "poor", "waste", "horrible")


def label_sentiment(text: str) -> str | None:
    text = text.lower()
    if any(word in text for word in POS_WORDS):
        return "positive"
    if any(word in text for word in NEG_WORDS):
        return "negative"
    return None


if __name__ == "__main__":
    creator = TextClassificationDataCreator(
        base_data="tweet_eval",
        hf_config="sentiment",
        split="train",
        label_fn=label_sentiment,
        seed=42,
        output_dir=str(DATA_DIR),
        output_format="csv",
        use_cache=True,
        neutral_filter_fn=lambda text: label_sentiment(text) is None,
    )

    train_df = creator.generate_training_data(max_size=5000, min_rows_for_split=50, balance="try")
    neutral_df = creator.generate_neutral_data(max_size=500)
    print(f"Training rows: {len(train_df)} ({train_df['label'].value_counts().to_dict()})")
    print(f"Neutral rows: {len(neutral_df)}")

    config = TextClassificationConfig(
        run_id="tweet_eval_sentiment_demo",
        data=DATA_DIR / "training.csv",
        target_classes=["negative", "positive"],
        eval_neutral_data=DATA_DIR / "neutral.csv",
        eval_neutral_max_rows=1000,
    )
    args = TrainingArguments(
        experiment_dir="runs/examples/tweet_eval_sentiment",
        train_batch_size=8,
        eval_batch_size=8,
        max_steps=1000,
        learning_rate=5e-4,
        use_cache=False,
        fail_on_non_convergence=True,
    )
    trainer = TextClassificationTrainer(
        model="distilbert-base-uncased",
        config=config,
        args=args,
    )
    trainer.train_classification_head(epochs=1, batch_size=8, max_length=128)
    trainer.train()
    trainer.plot_training_convergence()
    trainer.evaluate_encoder(plot=True)
    trainer.evaluate_decoder(plot=True, use_cache=False)
