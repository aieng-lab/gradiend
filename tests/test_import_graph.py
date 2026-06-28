"""Regression tests for trainer package import order and circular imports."""

import gradiend


def test_gradiend_public_api_all_exports_importable():
    for name in gradiend.__all__:
        obj = getattr(gradiend, name)
        assert obj is not None, name


def test_suite_imports_before_text_prediction_trainer_is_loaded():
    """TrainerSuite must not pull in the text prediction trainer at import time."""
    import sys

    for name in list(sys.modules):
        if name.startswith("gradiend.trainer.text.prediction.trainer"):
            del sys.modules[name]

    from gradiend.trainer.suite import SymmetricTrainerSuite  # noqa: F401

    assert "gradiend.trainer.text.prediction.trainer" not in sys.modules


def test_core_unified_data_import_is_lightweight():
    import sys

    from gradiend.trainer.core.unified_data import resolve_dataframe  # noqa: F401

    assert "gradiend.trainer.text.prediction.trainer" not in sys.modules
