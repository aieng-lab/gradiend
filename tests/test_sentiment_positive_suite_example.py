from pathlib import Path

from gradiend.examples import train_sentiment_positive_suite as example


def test_sentiment_positive_suite_resolves_expected_pairs(monkeypatch):
    def fake_tweet_eval_texts():
        texts = []
        for positive, negative in example.SENTIMENT_PAIRS:
            for word in (positive, negative):
                texts.extend(
                    f"The tweet author said the update was {word} in example {i}."
                    for i in range(8)
                )
        texts.extend(f"This neutral tweet mentions a product update {i}." for i in range(12))
        return texts

    monkeypatch.setattr(example, "_load_tweet_eval_texts", fake_tweet_eval_texts)

    training, neutral = example.create_data(max_size_per_class=8, neutral_max_size=12)
    suite = example.build_suite(training, neutral)

    expected_pairs = list(example.SENTIMENT_PAIRS)

    assert list(suite.pair_by_id.values()) == expected_pairs
    assert set(training) == {word for pair in expected_pairs for word in pair}
    assert not neutral.empty


def test_sentiment_positive_suite_only_convergent_reuses_artifact_cache(monkeypatch):
    monkeypatch.setattr(example, "_load_tweet_eval_texts", lambda: [
        f"The update felt {word} in example {i}."
        for positive, negative in example.SENTIMENT_PAIRS
        for word in (positive, negative)
        for i in range(8)
    ])

    training, neutral = example.create_data(max_size_per_class=8, neutral_max_size=0)
    suite = example.build_suite(training, neutral)

    trainer = next(iter(suite.trainers.values()))
    assert trainer._resolve_artifact_use_cache() is True


def test_sentiment_positive_suite_write_docs_images_copies_plots(tmp_path, monkeypatch):
    def fake_tweet_eval_texts():
        return [
            f"The update felt {word} in example {i}."
            for positive, negative in example.SENTIMENT_PAIRS
            for word in (positive, negative)
            for i in range(8)
        ]

    monkeypatch.setattr(example, "_load_tweet_eval_texts", fake_tweet_eval_texts)

    experiment_dir = tmp_path / "sentiment_positive_suite"
    docs_dir = tmp_path / "docs" / "img"
    monkeypatch.setattr(example, "EXPERIMENT_DIR", experiment_dir)
    monkeypatch.setattr(example, "DOCS_IMG_DIR", docs_dir)

    training, neutral = example.create_data(max_size_per_class=8, neutral_max_size=0)
    suite = example.build_suite(training, neutral)

    def fake_plot_similarity(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"similarity")

    def fake_plot_cross(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"cross")

    monkeypatch.setattr(suite, "evaluate_encoder", lambda **kwargs: None)
    monkeypatch.setattr(suite, "plot_similarity_heatmap", fake_plot_similarity)
    monkeypatch.setattr(suite, "plot_cross_encoding_heatmap", fake_plot_cross)

    example.plot_suite_outputs(suite, write_docs_images=True)

    assert (docs_dir / example.DOCS_SIMILARITY).read_bytes() == b"similarity"
    assert (docs_dir / example.DOCS_CROSS_ENCODING).read_bytes() == b"cross"
