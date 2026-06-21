import torch

import importlib.util
import sys
from pathlib import Path


def _load_experiment_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_pre_prune_topk = _load_experiment_module(
    "gender_de_pre_prune_topk_ablation_under_test",
    "experiments/gender_de_pre_prune_topk_ablation.py",
)
GridResult = _pre_prune_topk.GridResult
build_arg_parser = _pre_prune_topk.build_arg_parser
_default_pair_results_path = _pre_prune_topk._default_pair_results_path
_discover_result_paths = _pre_prune_topk._discover_result_paths
_load_results_many = _pre_prune_topk._load_results_many
_pair_slug = _pre_prune_topk._pair_slug
_seed_pair_results_from_combined = _pre_prune_topk._seed_pair_results_from_combined
_topk_base_global = _pre_prune_topk._topk_base_global


class _FakeGradiend:
    def get_topk_weights(self, part="decoder-weight", topk=1000):
        assert part == "decoder-weight"
        assert topk == 2
        return [2, 0]

    def _get_base_global_index_map(self):
        return torch.tensor([103124603, 7, 42], dtype=torch.long)


class _FakeModelWithGradiend:
    gradiend = _FakeGradiend()

    def get_topk_weights(self, part="decoder-weight", topk=1000):
        raise AssertionError("wrapper top-k is already base-global and must not be remapped")


def test_topk_base_global_maps_underlying_local_indices_once():
    assert _topk_base_global(
        _FakeModelWithGradiend(),
        topk=2,
        part="decoder-weight",
    ) == {42, 103124603}


def test_pair_cli_accepts_alias_and_source_target():
    parser = build_arg_parser()

    alias_args = parser.parse_args(["--pair", "masc_nom_fem_nom"])
    assert alias_args.pair == ("masc_nom", "fem_nom")

    explicit_args = parser.parse_args(["--pairs", "masc_nom:fem_nom,fem_acc:fem_gen"])
    assert explicit_args.pairs == [("masc_nom", "fem_nom"), ("fem_acc", "fem_gen")]


def test_single_pair_default_results_path_is_sharded(tmp_path):
    pair = ("masc_nom", "fem_nom")
    assert _pair_slug(pair) == "masc_nom_fem_nom"
    assert _default_pair_results_path(str(tmp_path), pair) == str(
        tmp_path / "pair_results" / "masc_nom_fem_nom.json"
    )


def test_plot_discovers_pair_result_shards(tmp_path):
    first = tmp_path / "pair_results" / "masc_nom_fem_nom.json"
    second = tmp_path / "pair_results" / "fem_acc_fem_gen.json"
    first.parent.mkdir()
    first.write_text(
        '[{"pair":"masc_nom_fem_nom","run_id":"a","pre_topk":1.0,"topk_indices_file":"x"}]',
        encoding="utf-8",
    )
    second.write_text(
        '[{"pair":"fem_acc_fem_gen","run_id":"b","pre_topk":1.0,"topk_indices_file":"y"}]',
        encoding="utf-8",
    )

    paths = _discover_result_paths(str(tmp_path), None)
    assert paths == [str(second), str(first)] or paths == [str(first), str(second)]

    results = _load_results_many(paths)
    assert [type(row) for row in results] == [GridResult, GridResult]
    assert {row.pair for row in results} == {"masc_nom_fem_nom", "fem_acc_fem_gen"}


def test_pair_shard_can_seed_from_legacy_combined_results(tmp_path):
    combined = tmp_path / "pre_topk_grid_results.json"
    combined.write_text(
        """
        [
          {"pair":"masc_nom_fem_nom","run_id":"keep","pre_topk":1.0,"topk_indices_file":"x"},
          {"pair":"fem_acc_fem_gen","run_id":"skip","pre_topk":1.0,"topk_indices_file":"y"}
        ]
        """,
        encoding="utf-8",
    )
    shard = tmp_path / "pair_results" / "masc_nom_fem_nom.json"

    _seed_pair_results_from_combined(
        output_dir=str(tmp_path),
        results_path=str(shard),
        pair=("masc_nom", "fem_nom"),
    )

    results = _load_results_many([str(shard)])
    assert [(row.pair, row.run_id) for row in results] == [("masc_nom_fem_nom", "keep")]
