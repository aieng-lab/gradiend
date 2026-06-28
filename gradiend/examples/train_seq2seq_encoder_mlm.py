"""
T5 seq2seq GRADIEND with encoder-side MLM (``seq2seq_encoder_mlm``).

Same English 3SG/3PL pronoun data as ``train_english_pronouns.py`` / ``start_workflow.py``,
but on ``t5-small``. With ``prediction_objective="auto"``, seq2seq models resolve to
encoder-side MLM — BERT-like ``[MASK]`` scoring on the encoder stack.

Run:
    python -m gradiend.examples.train_seq2seq_encoder_mlm
"""

from __future__ import annotations

from pathlib import Path

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.examples.create_english_pronoun_data import ensure_english_pronoun_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "english_pronouns"
EXPERIMENT_DIR = PROJECT_ROOT / "runs" / "examples" / "t5_encoder_mlm"
TARGET_CLASSES = ("3SG", "3PL")


if __name__ == "__main__":
    training_path, neutral_path = ensure_english_pronoun_data(output_dir=str(DATA_DIR))
    print(f"=== English pronoun data: using CSVs in {DATA_DIR} ===")

    args = TrainingArguments(
        experiment_dir=str(EXPERIMENT_DIR),
        prediction_objective="auto",  # → seq2seq_encoder_mlm for T5
        train_batch_size=4,
        train_max_size=1000,
        eval_steps=100,
        max_steps=500,
        learning_rate=1e-5,
        use_cache=False,
        fail_on_non_convergence=True,
    )
    trainer = TextPredictionTrainer(
        model="t5-small",
        run_id="t5_3sg_3pl_encoder_mlm",
        data=training_path,
        target_classes=list(TARGET_CLASSES),
        eval_neutral_data=neutral_path,
        args=args,
    )

    print("\n=== T5 pronouns: 3SG vs 3PL (seq2seq_encoder_mlm via auto) ===")
    trainer.train()
    print(f"Model saved at {trainer.model_path}")

    trainer.plot_training_convergence()
    trainer.evaluate_encoder(split="test", plot=True)
    dec = trainer.evaluate_decoder(plot=True)
    for cls in TARGET_CLASSES:
        if cls in dec:
            print(f"  decoder {cls}: {dec[cls]}")
