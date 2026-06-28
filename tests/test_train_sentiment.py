import pandas as pd

from gradiend.examples import train_sentiment


def _minimal_sentiment_merged() -> pd.DataFrame:
    rows = []
    for i, (word, cls) in enumerate(
        [
            ("happy", "positive"),
            ("sad", "negative"),
            ("glad", "positive"),
            ("mad", "negative"),
            ("cool", "positive"),
            ("bad", "negative"),
            ("good", "positive"),
            ("sick", "negative"),
            ("nice", "positive"),
            ("fake", "negative"),
            ("real", "positive"),
            ("late", "negative"),
        ]
    ):
        other_cls = "negative" if cls == "positive" else "positive"
        other_word = "sad" if cls == "positive" else "happy"
        rows.append(
            {
                "masked": f"text {i} [MASK]",
                "label_class": cls,
                "label": word,
                "alternative_class": other_cls,
                "alternative": other_word,
            }
        )
    return pd.DataFrame(rows)


def test_apply_vocabulary_held_out_split_assigns_all_splits():
    split_df = train_sentiment.apply_vocabulary_held_out_split(
        _minimal_sentiment_merged(),
        seed=0,
    )
    splits = set(split_df["split"].astype(str).unique())
    assert {"train", "validation", "test"}.issubset(splits)


def test_train_sentiment_seed_controls_manual_split_seed(monkeypatch, tmp_path):
    captured = {}

    class DummyTrainer:
        def __init__(self, *, model, config, args):
            captured["model"] = model
            captured["config"] = config
            captured["args"] = args

        def train(self):
            captured["trained"] = True

        def plot_training_convergence(self):
            captured["plotted"] = True

        def get_training_stats(self):
            return {"training_stats": {}}

        def evaluate_encoder(self, *, plot):
            captured["encoder_plot"] = plot
            return {}

        def get_encoder_metrics(self, *, use_cache):
            captured["metrics_use_cache"] = use_cache
            return {}

    def fake_load(_path, *, seed=0, **kwargs):
        captured["split_seed"] = seed
        return train_sentiment.apply_vocabulary_held_out_split(
            _minimal_sentiment_merged(),
            seed=seed,
        )

    monkeypatch.setattr(train_sentiment, "TextPredictionTrainer", DummyTrainer)
    monkeypatch.setattr(train_sentiment, "load_and_split_sentiment_training_data", fake_load)
    monkeypatch.setattr(train_sentiment, "_evaluate_split_stability", lambda trainer, experiment_dir: None)

    train_sentiment.train(
        training_path=tmp_path / "training.csv",
        neutral_path=tmp_path / "neutral.csv",
        experiment_dir=tmp_path / "runs",
        seed=123,
    )

    assert captured["args"].seed == 123
    assert captured["split_seed"] == 123
    assert captured["config"].split_col is None
    assert captured["trained"] is True
