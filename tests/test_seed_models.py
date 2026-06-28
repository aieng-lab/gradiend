"""Tests for SeedModelGroup comparison container."""

from __future__ import annotations

import pytest

from gradiend.trainer.core.seed_models import SeedModelGroup


class _NamedModel:
    def __init__(self, name: str):
        self.name = name


def test_seed_model_group_iteration_and_primary():
    models = [_NamedModel("a"), _NamedModel("b")]
    group = SeedModelGroup(models, selection="all_convergent", seed_values=[10, 11])
    assert len(group) == 2
    assert list(group) == models
    assert group.primary is models[0]
    assert group.seed_values == (10, 11)
    assert group.selection == "all_convergent"
    assert bool(group) is True


def test_seed_model_group_empty_is_falsy():
    group = SeedModelGroup([])
    assert not group
    with pytest.raises(ValueError, match="empty"):
        _ = group.primary


def test_seed_model_group_default_seed_values():
    models = [_NamedModel("only")]
    group = SeedModelGroup(models)
    assert group.seed_values == (0,)
