"""Tests for pre-prune stratified sample selection."""

from __future__ import annotations

from gradiend.trainer.core.pruning import PrePruneConfig, _stratified_indices


class _FakeDataset:
    def __init__(self, feature_classes: list) -> None:
        self._feature_classes = feature_classes

    def __len__(self) -> int:
        return len(self._feature_classes)

    def __getitem__(self, idx: int) -> dict:
        return {"feature_class_id": self._feature_classes[idx]}


def _make_balanced_dataset(n_per_class: int = 20) -> _FakeDataset:
    return _FakeDataset(["a"] * n_per_class + ["b"] * n_per_class)


def test_pre_prune_config_seed_validation() -> None:
    cfg = PrePruneConfig(n_samples=4, topk=0.1, seed=7)
    assert cfg.seed == 7


def test_stratified_indices_reproducible_with_seed() -> None:
    data = _make_balanced_dataset()
    first = _stratified_indices(data, 8, "feature_class_id", ["a", "b"], seed=42)
    second = _stratified_indices(data, 8, "feature_class_id", ["a", "b"], seed=42)
    assert first == second


def test_stratified_indices_smaller_n_is_subset_of_larger_n() -> None:
    data = _make_balanced_dataset()
    sizes = [1, 2, 4, 8, 16, 32]
    previous: set[int] | None = None
    for n_samples in sizes:
        indices = set(
            _stratified_indices(
                data,
                n_samples,
                "feature_class_id",
                ["a", "b"],
                seed=42,
            )
        )
        assert len(indices) == n_samples
        if previous is not None:
            assert previous.issubset(indices)
        previous = indices
