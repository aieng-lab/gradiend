"""
Tests for backbone parameter selection (params / param_map) when building GRADIEND from a base model.

Covers _filter_params_by_include wildcard matching and build_gradiend_from_base_model with
params and param_map list arguments.
"""

import tempfile

import torch
import torch.nn as nn
from collections import OrderedDict

from gradiend.model.core.backbone import (
    _filter_params_by_include,
    _normalize_param_map_arg,
    build_gradiend_from_base_model,
    get_backbone_module,
    split_backbone_vs_head_params,
)


class _MinimalBackbone(nn.Module):
    """Minimal backbone with two linear layers for testing param selection."""

    def __init__(self, dim=4):
        super().__init__()
        self.layer0 = nn.Linear(dim, dim)
        self.layer1 = nn.Linear(dim, dim)


class _MinimalHFModel(nn.Module):
    """HF-style model with base_model submodule (get_backbone_module returns base_model)."""

    base_model_prefix = "base_model"

    def __init__(self, dim=4):
        super().__init__()
        self.base_model = _MinimalBackbone(dim=dim)


class TestFilterParamsByInclude:
    """Test _filter_params_by_include (exact and wildcard matching)."""

    def test_none_returns_unchanged(self):
        param_lookup = OrderedDict([
            ("a.weight", torch.randn(2, 2)),
            ("b.weight", torch.randn(2, 2)),
        ])
        out = _filter_params_by_include(param_lookup, None)
        assert out is param_lookup
        out = _filter_params_by_include(param_lookup, [])
        assert list(out.keys()) == ["a.weight", "b.weight"]

    def test_exact_match(self):
        param_lookup = OrderedDict([
            ("encoder.layer.0.weight", torch.randn(4, 4)),
            ("encoder.layer.1.weight", torch.randn(4, 4)),
            ("head.weight", torch.randn(2, 4)),
        ])
        out = _filter_params_by_include(param_lookup, ["encoder.layer.0.weight"])
        assert list(out.keys()) == ["encoder.layer.0.weight"]

    def test_wildcard_match(self):
        param_lookup = OrderedDict([
            ("encoder.layer.0.weight", torch.randn(4, 4)),
            ("encoder.layer.0.bias", torch.randn(4)),
            ("encoder.layer.1.weight", torch.randn(4, 4)),
            ("encoder.layer.1.bias", torch.randn(4)),
        ])
        out = _filter_params_by_include(param_lookup, ["encoder.layer.0.*"])
        assert list(out.keys()) == ["encoder.layer.0.weight", "encoder.layer.0.bias"]

    def test_multiple_patterns_preserve_order(self):
        param_lookup = OrderedDict([
            ("encoder.layer.0.weight", torch.randn(4, 4)),
            ("encoder.layer.1.weight", torch.randn(4, 4)),
            ("encoder.layer.2.weight", torch.randn(4, 4)),
        ])
        out = _filter_params_by_include(
            param_lookup,
            ["encoder.layer.2.*", "encoder.layer.0.*"],
        )
        assert list(out.keys()) == ["encoder.layer.0.weight", "encoder.layer.2.weight"]

    def test_literal_dot_in_pattern(self):
        param_lookup = OrderedDict([
            ("encoder.layer.0.weight", torch.randn(4, 4)),
        ])
        out = _filter_params_by_include(param_lookup, ["encoder.layer.0.weight"])
        assert list(out.keys()) == ["encoder.layer.0.weight"]


class TestNormalizeParamMapArg:
    """Test _normalize_param_map_arg."""

    def test_none_or_empty(self):
        assert _normalize_param_map_arg(None) == []
        assert _normalize_param_map_arg([]) == []

    def test_single_element_list_of_list(self):
        assert _normalize_param_map_arg([["a", "b"]]) == ["a", "b"]

    def test_plain_list(self):
        assert _normalize_param_map_arg(["a", "b"]) == ["a", "b"]


class TestBuildGradiendFromBaseModelParams:
    """Test build_gradiend_from_base_model with params and param_map (layer selection)."""

    def test_build_with_params_restricts_to_matching_layers(self):
        """params list filters backbone to matching parameter names (wildcards)."""
        model = _MinimalHFModel(dim=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            gradiend = build_gradiend_from_base_model(
                model,
                tmpdir,
                params=["base_model.layer0.*"],
                latent_dim=1,
            )
        param_names = list(gradiend.param_map.keys())
        assert all(n.startswith("base_model.layer0.") for n in param_names)
        assert "base_model.layer0.weight" in param_names
        assert "base_model.layer0.bias" in param_names
        assert not any(n.startswith("base_model.layer1.") for n in param_names)
        expected_dim = sum(
            torch.tensor(spec["shape"]).prod().item()
            for spec in gradiend.param_map.values()
        )
        assert gradiend.input_dim == expected_dim

    def test_build_with_param_map_list_restricts_to_list(self):
        """param_map as list of names restricts to those params."""
        model = _MinimalHFModel(dim=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            gradiend = build_gradiend_from_base_model(
                model,
                tmpdir,
                param_map=["base_model.layer1.weight", "base_model.layer1.bias"],
                latent_dim=1,
            )
        param_names = list(gradiend.param_map.keys())
        assert set(param_names) == {"base_model.layer1.weight", "base_model.layer1.bias"}
        expected_dim = 4 * 4 + 4
        assert gradiend.input_dim == expected_dim

    def test_build_with_params_and_param_map(self):
        """params filters first; param_map further restricts (both as lists)."""
        model = _MinimalHFModel(dim=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            gradiend = build_gradiend_from_base_model(
                model,
                tmpdir,
                params=["base_model.layer0.weight", "base_model.layer1.weight"],
                param_map=["base_model.layer1.weight"],
                latent_dim=1,
            )
        param_names = list(gradiend.param_map.keys())
        assert param_names == ["base_model.layer1.weight"]
        assert gradiend.input_dim == 4 * 4


