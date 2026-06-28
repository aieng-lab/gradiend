import warnings

import pytest

from gradiend.util.deprecation import resolve_include_other_classes


def test_use_all_transitions_emits_deprecation_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_include_other_classes(
            include_other_classes=False,
            use_all_transitions=True,
        )
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_include_other_classes_without_use_all_transitions_no_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_include_other_classes(
            include_other_classes=True,
            use_all_transitions=False,
        )
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_plot_comparison_heatmap_fmt_emits_deprecation_warning():
    pytest.importorskip("matplotlib")
    from gradiend.visualizer.heatmaps.base import plot_comparison_heatmap

    data = {"matrix": [[1.0]], "model_ids": ["a"]}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plot_comparison_heatmap(data, fmt=".1f", show=False)
    assert any(
        issubclass(w.category, DeprecationWarning) and "fmt" in str(w.message).lower()
        for w in caught
    )
