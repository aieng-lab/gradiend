import pytest


@pytest.fixture(scope="module")
def ablation():
    from experiments import gender_de_pre_prune_topk_ablation as mod

    return mod


def test_default_mode_is_mask_recall(ablation):
    args = ablation.build_arg_parser().parse_args([])
    assert not args.full_grid


def test_recall_only_flag_is_accepted(ablation):
    args = ablation.build_arg_parser().parse_args(["--recall-only"])
    assert not args.full_grid


def test_default_grid_schedules_decade_pre_topk(ablation):
    args = ablation.build_arg_parser().parse_args([])

    assert args.pre_topk_values == [1.0, 0.1, 0.01, 0.001]
    missing = ablation._missing_grid_cells(
        [],
        pairs=[ablation.DER_DIE_PAIRS[0]],
        sources=["diff"],
        n_samples_values=[16],
        pre_topk_values=args.pre_topk_values,
    )
    assert any(cell["pre_topk"] == pytest.approx(0.1) for cell in missing)
    assert not any(cell["pre_topk"] == pytest.approx(0.3) for cell in missing)


def test_decade_pre_topk_helper(ablation):
    assert ablation._is_decade_pre_topk(0.1)
    assert ablation._is_decade_pre_topk(0.01)
    assert not ablation._is_decade_pre_topk(0.3)
    assert ablation._decade_pre_topk_values([1.0, 0.3, 0.1, 0.03, 0.01]) == [1.0, 0.1, 0.01]


def test_ref_recall_metrics_matches_topk_intersection(ablation):
    ref = set(range(1000))
    heuristic = set(range(400, 1400))
    recall, precision = ablation._ref_recall_metrics(heuristic, ref)
    assert recall == pytest.approx(0.6)
    assert precision == pytest.approx(600 / 1000)


def test_has_success_recall_only_does_not_require_converged(ablation, monkeypatch):
    run_id = ablation._grid_run_id(ablation.DER_DIE_PAIRS[0], "diff", 16, 0.1)
    pair_key = ablation._pair_slug(ablation.DER_DIE_PAIRS[0])
    idx_path = "indices.json"
    monkeypatch.setattr(ablation.os.path, "isfile", lambda path: path == idx_path)
    results = [
        ablation.GridResult(
            pair=pair_key,
            run_id=run_id,
            pre_topk=0.1,
            pre_source="diff",
            pre_n_samples=16,
            ref_recall=0.42,
            topk_indices_file=idx_path,
            mask_recall=True,
            converged=False,
        )
    ]
    assert ablation._has_success(results, pair_key, run_id, recall_only=True)
    assert not ablation._has_success(results, pair_key, run_id, recall_only=False)


def test_row_plottable_allows_high_recall_for_mask_recall_rows(ablation):
    row = ablation.GridResult(
        pair="masc_nom_fem_nom",
        run_id="test",
        pre_topk=0.1,
        ref_recall=0.995,
        topk_indices_file="indices.json",
        converged=True,
        kept_dim=100,
        mask_recall=True,
    )
    assert ablation._row_plottable(row, require_converged=True)


def test_ref_recall_plot_uses_one_inline_titled_legend(ablation, monkeypatch):
    summaries = [
        ablation.ConfigSummary(
            pre_topk=topk,
            pre_source=source,
            pre_n_samples=n_samples,
            mean_ref_recall=recall,
            mean_kept_dim=100.0,
            mean_cross_overlap=0.2,
            mean_encoder_correlation=0.9,
            n_pairs=2,
        )
        for source in ("factual", "alternative", "diff")
        for n_samples, recall in ((1, 0.4), (2, 0.5))
        for topk in (0.1,)
    ]
    captured = {}
    monkeypatch.setattr(
        ablation,
        "_save_figure",
        lambda fig, output_path, **kwargs: captured.setdefault("fig", fig),
    )

    ablation.plot_ref_recall_vs_pre_topk(
        summaries,
        output_path="unused.pdf",
        sources=["factual", "alternative", "diff"],
        pre_topk_values=[1.0, 0.1, 0.01, 0.001],
    )

    fig = captured["fig"]
    tick_labels = {label.get_text() for ax in fig.axes for label in ax.get_xticklabels()}
    assert "0.3" not in tick_labels
    assert "0.03" not in tick_labels
    assert len(fig.legends) == 1
    legend = fig.legends[0]
    assert legend.get_title().get_text() == ""
    assert [text.get_text() for text in legend.get_texts()] == [
        "n_samples",
        "1",
        "2",
    ]
    assert legend._ncols == 3
    assert all(ax.get_legend() is None for ax in fig.axes)
    assert fig._supylabel.get_text() == "Recall"
