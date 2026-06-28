from pathlib import Path

from gradiend.examples import train_race_symmetric_suite as example


def test_race_symmetric_suite_resolves_three_pairwise_children():
    suite = example.build_suite(max_steps=10)

    assert len(suite.trainers) == 3
    assert set(suite.pair_by_id.values()) == {
        ("asian", "black"),
        ("asian", "white"),
        ("black", "white"),
    }
    assert set(suite.pair_by_id) == {"asian__black", "asian__white", "black__white"}


def test_race_symmetric_suite_write_docs_images_copies_plots(tmp_path, monkeypatch):
    suite = example.build_suite(max_steps=10)
    experiment_dir = tmp_path / "race_symmetric_suite"
    docs_dir = tmp_path / "docs" / "img"
    monkeypatch.setattr(example, "EXPERIMENT_DIR", experiment_dir)
    monkeypatch.setattr(example, "DOCS_IMG_DIR", docs_dir)

    def fake_plot_topk(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"topk")

    def fake_plot_cross(*args, **kwargs):
        Path(kwargs["output_path"]).write_bytes(b"cross")

    monkeypatch.setattr(suite, "plot_topk_overlap_heatmap", fake_plot_topk)
    monkeypatch.setattr(suite, "plot_cross_encoding_heatmap", fake_plot_cross)
    monkeypatch.setattr(suite, "evaluate_encoder", lambda **kwargs: None)

    example.plot_suite_outputs(suite, write_docs_images=True)

    assert (docs_dir / example.DOCS_TOPK_OVERLAP).read_bytes() == b"topk"
    assert (docs_dir / example.DOCS_CROSS_ENCODING).read_bytes() == b"cross"
