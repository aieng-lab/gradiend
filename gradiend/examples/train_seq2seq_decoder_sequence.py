"""
T5 seq2seq GRADIEND workflow with multi-token decoder sequence cloze.

**Experimental** — ``seq2seq_decoder_sequence_cloze`` does not converge reliably for
all setups. Prefer ``train_seq2seq_encoder_mlm.py`` (encoder-side MLM, default for
``prediction_objective="auto"`` on T5/BART).

Uses the same English pronoun data as ``train_english_pronouns.py`` /
``create_english_pronoun_data.py``.

Run:
    python -m gradiend.examples.train_seq2seq_decoder_sequence
"""

from __future__ import annotations

from pathlib import Path

from gradiend import PostPruneConfig, TextPredictionTrainer, TrainingArguments
from gradiend.examples.create_english_pronoun_data import ensure_english_pronoun_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "english_pronouns"
EXPERIMENT_DIR = PROJECT_ROOT / "runs" / "examples" / "t5_decoder_sequence"
TARGET_CLASSES = ("3SG", "3PL")


if __name__ == "__main__":
    training_path, neutral_path = ensure_english_pronoun_data(output_dir=str(DATA_DIR))
    print(f"=== English pronoun data: using CSVs in {DATA_DIR} ===")
    print(f"  {training_path}")
    print(f"  {neutral_path}")

    args = TrainingArguments(
        experiment_dir=str(EXPERIMENT_DIR),
        prediction_objective="seq2seq_decoder_sequence_cloze",
        decoder_sequence_cloze_rhs_window=0,
        train_batch_size=4,
        train_max_size=1000,
        eval_steps=100,
        max_steps=1000,
        source="alternative",
        target="diff",
        learning_rate=1e-5,
        use_cache=False,
        fail_on_non_convergence=True,
        add_identity_for_other_classes=False,
        post_prune_config=PostPruneConfig(topk=0.001, part="decoder-weight"),
    )
    trainer = TextPredictionTrainer(
        model="t5-small",
        run_id="t5_3sg_3pl_sequence_cloze",
        data=training_path,
        target_classes=list(TARGET_CLASSES),
        eval_neutral_data=neutral_path,
        args=args,
    )

    print("\n=== T5 pronouns: 3SG vs 3PL (seq2seq_decoder_sequence_cloze) ===")
    trainer.train()
    print(f"Model saved at {trainer.model_path}")

    trainer.plot_training_convergence()
    stats = trainer.get_training_stats() or {}
    print(f"  correlation={stats.get('training_stats', {}).get('correlation')}")

    print("\n=== Decoder evaluation ===")
    dec = trainer.evaluate_decoder(plot=True)
    for cls in TARGET_CLASSES:
        if cls in dec:
            print(f"  {cls}: {dec[cls]}")

    print("\n=== Done ===")
