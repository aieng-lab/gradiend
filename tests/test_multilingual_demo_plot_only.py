"""Plot-only helpers for multilingual_gradiend_demo."""

import json
import os
import tempfile

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def demo():
    import experiments.multilingual_gradiend_demo as mod

    return mod


# Small Hugging Face text corpus (8.5k short reviews) with plenty of first/second/third-person pronouns.
_SMALL_BASE_DATASET = "cornell-movie-review-data/rotten_tomatoes"


@pytest.fixture(scope="module", autouse=True)
def _small_probe_corpus(demo, tmp_path_factory):
    """Generate the probe-pool pronoun data from a small Hugging Face corpus in a temp directory.

    These tests check plot-only wiring with the heavy steps mocked, but ``plot_cross_encoding`` still
    builds probe trainers from the English pronoun data and generates it when missing. From the real
    source (``wikimedia/wikipedia``, 50,000 articles, spaCy over every segment) that took tens of
    minutes and wrote into ``data/`` of the working directory. The same generation code path (Hugging
    Face loading, spaCy tag filtering, splits) runs here on a small corpus, so it stays covered.
    """
    import gradiend.examples.create_english_pronoun_data as pronoun_data

    data_dir = tmp_path_factory.mktemp("english_pronouns")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pronoun_data, "WIKIPEDIA_DATASET", _SMALL_BASE_DATASET)
        patch.setattr(pronoun_data, "WIKIPEDIA_HF_CONFIG", None)
        patch.setattr(pronoun_data, "WIKIPEDIA_BASE_MAX_SIZE", 8000)
        patch.setattr(pronoun_data, "MAX_SIZE_PER_CLASS", 10)
        patch.setattr(pronoun_data, "NEUTRAL_MAX_SIZE", 20)
        patch.setattr(demo, "PRONOUN_DATA_DIR", str(data_dir))
        yield


def _write_minimal_gradiend_checkpoint(model_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"architecture": {"input_dim": 4}}, handle)
    with open(os.path.join(model_dir, "model.safetensors"), "wb") as handle:
        handle.write(b"")


class _StubTrainer:
    def __init__(self, *, run_id: str, experiment_dir: str) -> None:
        self.run_id = run_id
        self._experiment_dir = experiment_dir

    @property
    def experiment_dir(self) -> str:
        return self._experiment_dir


def test_trainer_has_cached_checkpoint_false_when_missing(demo):
    with tempfile.TemporaryDirectory() as temp:
        trainer = _StubTrainer(run_id="gender_de_a_b", experiment_dir=temp)
        assert demo.trainer_has_cached_checkpoint(trainer) is False


def test_trainer_has_cached_checkpoint_true_when_model_present(demo):
    with tempfile.TemporaryDirectory() as temp:
        run_dir = os.path.join(temp, "gender_de_a_b")
        _write_minimal_gradiend_checkpoint(os.path.join(run_dir, "model"))
        trainer = _StubTrainer(run_id="gender_de_a_b", experiment_dir=run_dir)
        assert demo.trainer_has_cached_checkpoint(trainer) is True


def test_filter_trainers_with_checkpoints_keeps_only_cached(demo):
    with tempfile.TemporaryDirectory() as temp:
        cached_dir = os.path.join(temp, "race_white_black")
        _write_minimal_gradiend_checkpoint(os.path.join(cached_dir, "model"))
        trainers = {
            "race_white_black": _StubTrainer(run_id="race_white_black", experiment_dir=cached_dir),
            "race_white_asian": _StubTrainer(
                run_id="race_white_asian",
                experiment_dir=os.path.join(temp, "race_white_asian"),
            ),
        }
        kept = demo.filter_trainers_with_checkpoints(trainers, experiment_dir=temp)
        assert list(kept) == ["race_white_black"]


def test_filter_trainers_with_checkpoints_raises_when_empty(demo):
    with tempfile.TemporaryDirectory() as temp:
        trainers = {
            "race_white_black": _StubTrainer(
                run_id="race_white_black",
                experiment_dir=os.path.join(temp, "race_white_black"),
            ),
        }
        with pytest.raises(FileNotFoundError, match="no cached GRADIEND checkpoints"):
            demo.filter_trainers_with_checkpoints(trainers, experiment_dir=temp)


def test_discover_cached_run_ids_finds_saved_models(demo):
    with tempfile.TemporaryDirectory() as temp:
        run_dir = os.path.join(temp, "pronoun_1SG_1PL")
        _write_minimal_gradiend_checkpoint(os.path.join(run_dir, "model"))
        assert demo.discover_cached_run_ids(temp) == frozenset({"pronoun_1SG_1PL"})


def test_demo_problem_for_run_id_maps_race_and_religion(demo):
    assert demo._demo_problem_for_run_id("race_white_black") == "race"
    assert demo._demo_problem_for_run_id("religion_christian_muslim") == "religion"
    assert demo._demo_problem_for_run_id("pronoun_1SG_1PL") == "pronoun"


def test_filter_pair_definitions_by_cache(demo):
    definitions = [
        demo.SuitePairDefinition(
            target_classes=("a", "b"),
            child_id="run_a",
            label="a <-> b",
        ),
        demo.SuitePairDefinition(
            target_classes=("c", "d"),
            child_id="run_b",
            label="c <-> d",
        ),
    ]
    filtered = demo._filter_pair_definitions_by_cache(definitions, frozenset({"run_a"}))
    assert [definition.child_id for definition in filtered] == ["run_a"]


def test_build_experiment_config_honors_experiment_dir_override(demo, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["multilingual_gradiend_demo.py", "--experiment-dir", "runs/custom_demo_path"],
    )
    cli = demo.parse_args()
    config = demo.build_experiment_config(cli)
    assert config.args.experiment_dir == "runs/custom_demo_path"


def test_demo_plot_style_uses_latex_sans_serif_arrows(demo):
    assert demo.DEMO_PLOT_STYLE.use_latex is True
    assert demo.DEMO_PLOT_STYLE.font_family == "sans-serif"
    assert demo.DEMO_PLOT_STYLE.transition_arrows == "latex"


def test_recompute_encoder_cache_requires_plot_only(demo, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["multilingual_gradiend_demo.py", "--recompute-encoder-cache"],
    )
    cli = demo.parse_args()

    with pytest.raises(ValueError, match="require --plot-only"):
        demo.build_experiment_config(cli)


def test_repair_encoder_cache_requires_plot_only(demo, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["multilingual_gradiend_demo.py", "--repair-encoder-cache"],
    )
    cli = demo.parse_args()

    with pytest.raises(ValueError, match="require --plot-only"):
        demo.build_experiment_config(cli)


def test_repair_and_recompute_encoder_cache_are_mutually_exclusive(demo, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "multilingual_gradiend_demo.py",
            "--plot-only",
            "--repair-encoder-cache",
            "--recompute-encoder-cache",
        ],
    )
    cli = demo.parse_args()

    with pytest.raises(ValueError, match="mutually exclusive"):
        demo.build_experiment_config(cli)


def test_plot_incomplete_encoder_cache_requires_plot_only(demo, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["multilingual_gradiend_demo.py", "--plot-incomplete-encoder-cache"],
    )
    cli = demo.parse_args()

    with pytest.raises(ValueError, match="require --plot-only"):
        demo.build_experiment_config(cli)


def test_plot_incomplete_encoder_cache_rejects_repair_or_recompute(demo, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "multilingual_gradiend_demo.py",
            "--plot-only",
            "--plot-incomplete-encoder-cache",
            "--repair-encoder-cache",
        ],
    )
    cli = demo.parse_args()

    with pytest.raises(ValueError, match="only for cache-only plotting"):
        demo.build_experiment_config(cli)


def test_plot_cross_encoding_plot_only_uses_cache_without_hf_pool(demo, monkeypatch, tmp_path):
    import pandas as pd
    import gradiend.comparison.cross_encoding as cross_encoding_module

    captured: dict = {}

    def _fail_hf_pool(*args, **kwargs):
        trainers = args[0] if args else kwargs.get("trainers", {})
        if any(cross_encoding_module.trainer_uses_hf_dataset(trainer) for trainer in trainers.values()):
            raise AssertionError("plot_only must not load HuggingFace trainer datasets")
        return pd.DataFrame()

    def _fake_encoder_summary(trainers, feature_classes, **kwargs):
        captured.update(kwargs)
        return {
            "sentiment_positive_negative": {
                "encoder_df": pd.DataFrame(
                    [
                        {
                            "encoded": 0.5,
                            "factual_id": "positive",
                            "alternative_id": "negative",
                            "transition_id": "positive->negative",
                            "type": "training",
                        }
                    ]
                )
            }
        }

    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.collect_unified_test_rows",
        _fail_hf_pool,
    )
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.build_cross_task_encoder_summary",
        _fake_encoder_summary,
    )
    monkeypatch.setattr(demo, "plot_gradiend_transition_cross_encoding_heatmap", lambda *a, **k: {})
    monkeypatch.setattr(demo, "plot_cross_encoding_heatmap", lambda *a, **k: {})
    # seed stability is requested but there are no trainers: the std matrices need at least one, and
    # computing them is not what this test is about
    import gradiend.comparison as comparison

    monkeypatch.setattr(
        comparison, "compute_gradiend_transition_cross_encoding_matrix", lambda *a, **k: {}, raising=False
    )
    monkeypatch.setattr(demo, "_plot_std_heatmap_from_cell_stats", lambda *a, **k: None)

    config = demo.ExperimentConfig(
        model_name="google-bert/bert-base-multilingual-cased",
        decoder_eval_mode=demo.DecoderEvalMode.NONE,
        mlm_head_scope=demo.MlmHeadScope.GLOBAL,
        args=demo.TrainingArguments(
            experiment_dir=str(tmp_path),
            encoder_eval_max_size=50,
            analyze_seed_stability=True,
        ),
        mlm_head_args={},
    )
    demo.plot_cross_encoding(
        config,
        {},
        trainer_order=[],
        trainer_pretty_groups={},
        feature_order=["positive", "negative"],
        feature_pretty_groups={"Sentiment": ["positive", "negative"]},
        plot_only=True,
    )
    assert captured.get("cache_only") is True
    assert captured.get("allow_incomplete_cache") is False
    assert isinstance(captured.get("eval_rows"), pd.DataFrame)
    assert captured["eval_rows"].empty


def test_plot_cross_encoding_plot_only_can_allow_incomplete_encoder_cache(
    demo, monkeypatch, tmp_path, capsys
):
    import pandas as pd

    captured: dict = {}

    def _fake_encoder_summary(trainers, feature_classes, **kwargs):
        captured.update(kwargs)
        return {
            "sentiment_positive_negative": {
                "encoder_df": pd.DataFrame(
                    [
                        {
                            "encoded": 0.5,
                            "factual_id": "positive",
                            "alternative_id": "negative",
                            "transition_id": "positive->negative",
                            "type": "training",
                        }
                    ]
                )
            }
        }

    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.collect_unified_test_rows",
        lambda *a, **k: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.build_cross_task_encoder_summary",
        _fake_encoder_summary,
    )
    monkeypatch.setattr(demo, "plot_gradiend_transition_cross_encoding_heatmap", lambda *a, **k: {})
    monkeypatch.setattr(demo, "plot_cross_encoding_heatmap", lambda *a, **k: {})

    config = demo.ExperimentConfig(
        model_name="google-bert/bert-base-multilingual-cased",
        decoder_eval_mode=demo.DecoderEvalMode.NONE,
        mlm_head_scope=demo.MlmHeadScope.GLOBAL,
        args=demo.TrainingArguments(experiment_dir=str(tmp_path), encoder_eval_max_size=50),
        mlm_head_args={},
    )
    demo.plot_cross_encoding(
        config,
        {},
        trainer_order=[],
        trainer_pretty_groups={},
        feature_order=["positive", "negative"],
        feature_pretty_groups={"Sentiment": ["positive", "negative"]},
        plot_only=True,
        plot_incomplete_encoder_cache=True,
    )

    assert captured.get("cache_only") is True
    assert captured.get("allow_incomplete_cache") is True
    assert "WARNING: --plot-incomplete-encoder-cache is enabled" in capsys.readouterr().out


def test_plot_cross_encoding_plot_only_recompute_encoder_cache_collects_eval_rows(
    demo, monkeypatch, tmp_path
):
    import pandas as pd

    captured: dict = {}

    def _fake_eval_rows(*args, **kwargs):
        captured["collected_eval_rows"] = True
        return pd.DataFrame(
            [
                {
                    "masked": "[MASK] good",
                    "split": "test",
                    "factual_class": "positive",
                    "alternative_class": "negative",
                    "factual": "good",
                    "alternative": "bad",
                    "transition": "positive->negative",
                }
            ]
        )

    def _fake_encoder_summary(trainers, feature_classes, **kwargs):
        captured.update(kwargs)
        return {
            "sentiment_positive_negative": {
                "encoder_df": pd.DataFrame(
                    [
                        {
                            "encoded": 0.5,
                            "factual_id": "positive",
                            "alternative_id": "negative",
                            "transition_id": "positive->negative",
                            "type": "training",
                        }
                    ]
                )
            }
        }

    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.collect_unified_test_rows",
        _fake_eval_rows,
    )
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.build_cross_task_encoder_summary",
        _fake_encoder_summary,
    )
    monkeypatch.setattr(demo, "plot_gradiend_transition_cross_encoding_heatmap", lambda *a, **k: {})
    monkeypatch.setattr(demo, "plot_cross_encoding_heatmap", lambda *a, **k: {})

    config = demo.ExperimentConfig(
        model_name="google-bert/bert-base-multilingual-cased",
        decoder_eval_mode=demo.DecoderEvalMode.NONE,
        mlm_head_scope=demo.MlmHeadScope.GLOBAL,
        args=demo.TrainingArguments(experiment_dir=str(tmp_path), encoder_eval_max_size=50),
        mlm_head_args={},
    )
    demo.plot_cross_encoding(
        config,
        {},
        trainer_order=[],
        trainer_pretty_groups={},
        feature_order=["positive", "negative"],
        feature_pretty_groups={"Sentiment": ["positive", "negative"]},
        plot_only=True,
        recompute_encoder_cache=True,
    )

    assert captured.get("collected_eval_rows") is True
    assert captured.get("cache_only") is False
    assert captured.get("force_recompute") is True
    assert isinstance(captured.get("eval_rows"), pd.DataFrame)


def test_plot_cross_encoding_plot_only_repair_encoder_cache_collects_eval_rows_without_force(
    demo, monkeypatch, tmp_path
):
    import pandas as pd

    captured: dict = {}

    def _fake_eval_rows(*args, **kwargs):
        captured["collected_eval_rows"] = True
        return pd.DataFrame(
            [
                {
                    "masked": "[MASK] good",
                    "split": "test",
                    "factual_class": "positive",
                    "alternative_class": "negative",
                    "factual": "good",
                    "alternative": "bad",
                    "transition": "positive->negative",
                }
            ]
        )

    def _fake_encoder_summary(trainers, feature_classes, **kwargs):
        captured.update(kwargs)
        return {
            "sentiment_positive_negative": {
                "encoder_df": pd.DataFrame(
                    [
                        {
                            "encoded": 0.5,
                            "factual_id": "positive",
                            "alternative_id": "negative",
                            "transition_id": "positive->negative",
                            "type": "training",
                        }
                    ]
                )
            }
        }

    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.collect_unified_test_rows",
        _fake_eval_rows,
    )
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.build_cross_task_encoder_summary",
        _fake_encoder_summary,
    )
    monkeypatch.setattr(demo, "plot_gradiend_transition_cross_encoding_heatmap", lambda *a, **k: {})
    monkeypatch.setattr(demo, "plot_cross_encoding_heatmap", lambda *a, **k: {})

    config = demo.ExperimentConfig(
        model_name="google-bert/bert-base-multilingual-cased",
        decoder_eval_mode=demo.DecoderEvalMode.NONE,
        mlm_head_scope=demo.MlmHeadScope.GLOBAL,
        args=demo.TrainingArguments(experiment_dir=str(tmp_path), encoder_eval_max_size=50),
        mlm_head_args={},
    )
    demo.plot_cross_encoding(
        config,
        {},
        trainer_order=[],
        trainer_pretty_groups={},
        feature_order=["positive", "negative"],
        feature_pretty_groups={"Sentiment": ["positive", "negative"]},
        plot_only=True,
        repair_encoder_cache=True,
    )

    assert captured.get("collected_eval_rows") is True
    assert captured.get("cache_only") is False
    assert captured.get("force_recompute") is False
    assert isinstance(captured.get("eval_rows"), pd.DataFrame)


def test_plot_cross_encoding_plot_only_skips_incomplete_encoder_caches(
    demo, monkeypatch, tmp_path, capsys
):
    import pandas as pd

    plot_calls = {"n": 0}

    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.collect_unified_test_rows",
        lambda *a, **k: pd.DataFrame(
            [
                {
                    "masked": "[MASK] good",
                    "split": "test",
                    "factual_class": "positive",
                    "alternative_class": "negative",
                    "factual": "good",
                    "alternative": "bad",
                    "transition": "positive->negative",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.build_cross_task_encoder_summary",
        lambda *a, **k: {
            "sentiment_positive_negative": {"encoder_df": pd.DataFrame()},
        },
    )
    monkeypatch.setattr(
        demo,
        "plot_gradiend_transition_cross_encoding_heatmap",
        lambda *a, **k: plot_calls.__setitem__("n", plot_calls["n"] + 1),
    )
    monkeypatch.setattr(
        demo,
        "plot_cross_encoding_heatmap",
        lambda *a, **k: plot_calls.__setitem__("n", plot_calls["n"] + 1),
    )

    config = demo.ExperimentConfig(
        model_name="google-bert/bert-base-multilingual-cased",
        decoder_eval_mode=demo.DecoderEvalMode.NONE,
        mlm_head_scope=demo.MlmHeadScope.GLOBAL,
        args=demo.TrainingArguments(experiment_dir=str(tmp_path), encoder_eval_max_size=50),
        mlm_head_args={},
    )

    demo.plot_cross_encoding(
        config,
        {"sentiment_positive_negative": object()},
        trainer_order=["sentiment_positive_negative"],
        trainer_pretty_groups={"Sentiment": ["sentiment_positive_negative"]},
        feature_order=["positive", "negative"],
        feature_pretty_groups={"Sentiment": ["positive", "negative"]},
        plot_only=True,
    )

    assert plot_calls["n"] == 0
    out = capsys.readouterr().out
    assert "WARNING: cached encoder CSVs are missing required local probe rows" in out
    assert "no usable cached encoder CSVs" in out


def test_plot_cross_encoding_plot_only_allows_partial_missing_cells_with_warning(
    demo, monkeypatch, tmp_path, capsys
):
    import pandas as pd
    import gradiend.comparison as comparison

    plot_calls = {"n": 0}

    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.collect_unified_test_rows",
        lambda *a, **k: pd.DataFrame(
            [
                {
                    "masked": "[MASK] good",
                    "split": "test",
                    "factual_class": "positive",
                    "alternative_class": "negative",
                    "factual": "good",
                    "alternative": "bad",
                    "transition": "positive->negative",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.build_cross_task_encoder_summary",
        lambda *a, **k: {
            "sentiment_positive_negative": {
                "encoder_df": pd.DataFrame(
                    [
                        {
                            "encoded": 0.7,
                            "label": 1.0,
                            "factual_id": "positive",
                            "alternative_id": "negative",
                            "transition_id": "positive->negative",
                            "type": "training",
                        }
                    ]
                )
            },
            "race_white_black": {"encoder_df": pd.DataFrame()},
        },
    )
    monkeypatch.setattr(
        comparison,
        "collect_unified_test_transitions",
        lambda *a, **k: ["positive->negative"],
        raising=False,
    )
    monkeypatch.setattr(
        comparison,
        "pair_by_id_from_trainers",
        lambda trainers: {
            "sentiment_positive_negative": ("positive", "negative"),
            "race_white_black": ("white", "black"),
        },
        raising=False,
    )
    monkeypatch.setattr(
        comparison,
        "source_by_id_from_trainers",
        lambda trainers: {key: "alternative" for key in trainers},
        raising=False,
    )
    monkeypatch.setattr(
        comparison,
        "compute_gradiend_transition_cross_encoding_matrix",
        lambda *a, **k: {
            "measure": "gradiend_transition_cross_encoding",
            "model_ids": ["sentiment_positive_negative"],
            "column_ids": ["positive->negative"],
            "matrix": [[0.7]],
        },
        raising=False,
    )
    monkeypatch.setattr(
        comparison,
        "compute_anchor_aligned_encoding_std_matrix",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("no std")),
        raising=False,
    )
    monkeypatch.setattr(
        "gradiend.visualizer.heatmaps.encoding.compute_anchor_aligned_encoding_matrix",
        lambda *a, **k: {
            "measure": "anchor_aligned_encoding_factual_mean",
            "model_ids": ["positive", "negative"],
            "column_ids": ["positive", "negative"],
            "matrix": [[0.7, None], [None, None]],
            "rows": ["positive", "negative"],
            "columns": ["positive", "negative"],
        },
    )
    monkeypatch.setattr(
        demo,
        "plot_gradiend_transition_cross_encoding_heatmap",
        lambda *a, **k: plot_calls.__setitem__("n", plot_calls["n"] + 1),
    )
    monkeypatch.setattr(
        demo,
        "plot_cross_encoding_heatmap",
        lambda *a, **k: plot_calls.__setitem__("n", plot_calls["n"] + 1),
    )

    class _Trainer:
        def __init__(self, target_classes):
            self.target_classes = target_classes

    config = demo.ExperimentConfig(
        model_name="google-bert/bert-base-multilingual-cased",
        decoder_eval_mode=demo.DecoderEvalMode.NONE,
        mlm_head_scope=demo.MlmHeadScope.GLOBAL,
        args=demo.TrainingArguments(
            experiment_dir=str(tmp_path), encoder_eval_max_size=50, analyze_seed_stability=True
        ),
        mlm_head_args={},
    )

    demo.plot_cross_encoding(
        config,
        {
            "sentiment_positive_negative": _Trainer(("positive", "negative")),
            "race_white_black": _Trainer(("white", "black")),
        },
        trainer_order=["sentiment_positive_negative", "race_white_black"],
        trainer_pretty_groups={"Demo": ["sentiment_positive_negative", "race_white_black"]},
        feature_order=["positive", "negative", "white", "black"],
        feature_pretty_groups={"Demo": ["positive", "negative", "white", "black"]},
        plot_only=True,
    )

    assert plot_calls["n"] > 0
    out = capsys.readouterr().out
    assert "WARNING: cached encoder CSVs are missing required local probe rows" in out
    assert "WARNING: skipping oriented cross-encoding std heatmap" in out
    assert "no usable cached encoder CSVs" not in out


def test_plot_results_recomputes_encoder_before_topk_when_requested(demo, monkeypatch, tmp_path):
    calls: list[str] = []

    monkeypatch.setattr(demo, "_validate_multiseed_plot_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        demo,
        "_build_plot_order_and_groups",
        lambda ids: (list(ids), {"Demo": list(ids)}),
    )
    monkeypatch.setattr(demo, "_pretty_label", lambda mid: str(mid))
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.plot_topk_overlap_heatmap",
        lambda *args, **kwargs: calls.append("topk"),
    )
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.plot_cross_encoding",
        lambda *args, **kwargs: calls.append("cross"),
    )
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.plot_topk_overlap_venn",
        lambda *args, **kwargs: None,
    )

    class _Trainer:
        def get_training_stats(self):
            return {"convergence_info": {"converged": True}}

    config = demo.ExperimentConfig(
        model_name="google-bert/bert-base-multilingual-cased",
        decoder_eval_mode=demo.DecoderEvalMode.NONE,
        mlm_head_scope=demo.MlmHeadScope.GLOBAL,
        args=demo.TrainingArguments(experiment_dir=str(tmp_path), encoder_eval_max_size=50),
        mlm_head_args={},
    )

    demo.plot_results(
        config,
        {"run_a": object()},
        {"run_a": _Trainer()},
        plot_only=True,
        recompute_encoder_cache=True,
    )

    assert calls[:2] == ["cross", "topk"]


def test_plot_results_passes_stable_ids_to_venn(demo, monkeypatch, tmp_path):
    captured: dict = {}

    monkeypatch.setattr(demo, "_validate_multiseed_plot_cache", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        demo,
        "_build_plot_order_and_groups",
        lambda ids: (list(ids), {"Pronoun": list(ids)}),
    )
    monkeypatch.setattr(demo, "_pretty_label", lambda mid: f"Pretty {mid}")
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.plot_topk_overlap_heatmap",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.plot_cross_encoding",
        lambda *args, **kwargs: None,
    )

    def _capture_venn(models, *args, **kwargs):
        captured["models"] = models
        captured.update(kwargs)

    monkeypatch.setattr(
        "experiments.multilingual_gradiend_demo.plot_topk_overlap_venn",
        _capture_venn,
    )

    class _Trainer:
        def __init__(self, converged):
            self._converged = converged

        def get_training_stats(self):
            return {"convergence_info": {"converged": self._converged}}

    ids = ["pronoun_1SG_3PL", "pronoun_1SG_3SG", "pronoun_3SG_3PL"]
    config = demo.ExperimentConfig(
        model_name="google-bert/bert-base-multilingual-cased",
        decoder_eval_mode=demo.DecoderEvalMode.NONE,
        mlm_head_scope=demo.MlmHeadScope.GLOBAL,
        args=demo.TrainingArguments(experiment_dir=str(tmp_path), encoder_eval_max_size=50),
        mlm_head_args={},
    )

    demo.plot_results(
        config,
        {mid: object() for mid in ids},
        {
            "pronoun_1SG_3PL": _Trainer(True),
            "pronoun_1SG_3SG": _Trainer(False),
            "pronoun_3SG_3PL": _Trainer(True),
        },
        plot_only=True,
    )

    assert list(captured["models"]) == ids
    assert captured["label_mapping"]["pronoun_1SG_3SG"] == "Pretty pronoun_1SG_3SG"
    assert captured["converged_by_id"]["pronoun_1SG_3SG"] is False
