"""
Shared pytest fixtures for GRADIEND tests.

Provides mock models, tokenizers, and common test utilities.
"""

import os
import sys
import gc
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Use non-interactive backend for any tests that use matplotlib (headless/CI)
os.environ.setdefault("MPLBACKEND", "Agg")
import tempfile
import shutil

import pytest

# Never rewrite scheduler-provided GPU visibility implicitly. CPU-only test runs
# must opt in before PyTorch is imported.
if os.environ.get("GRADIEND_TEST_USE_CUDA", "").strip().lower() in {"0", "false", "no", "off"}:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

def _close_matplotlib_figures() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.close("all")


@pytest.fixture(autouse=True)
def _release_test_memory():
    """Close matplotlib figures between tests (full suite can otherwise grow)."""
    yield
    _close_matplotlib_figures()
    if os.environ.get("GRADIEND_GC_EACH_TEST") == "1":
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@pytest.fixture(scope="session", autouse=True)
def _cache_hub_tokenizer_loads():
    """Load each Hub tokenizer once per session and hand out independent copies.

    transformers 5 sends a Hub request (additional_chat_templates) for every
    ``AutoTokenizer.from_pretrained("<hub id>")``. Unauthenticated, the Hub
    rate-limits these (HTTP 429, long retries), which made the suite very slow.
    Local directories are never cached, and every caller gets a deep copy
    because tests mutate tokenizers (special tokens, pad token, ...).
    """
    import copy

    from transformers import AutoTokenizer

    original = AutoTokenizer.__dict__["from_pretrained"].__func__
    cache = {}

    def cached_from_pretrained(cls, name, *args, **kwargs):
        key = None
        if isinstance(name, str) and not os.path.exists(name):
            try:
                key = (cls, name, args, tuple(sorted(kwargs.items())))
                hash(key)
            except TypeError:
                key = None
        if key is None:
            return original(cls, name, *args, **kwargs)
        if key not in cache:
            cache[key] = original(cls, name, *args, **kwargs)
        return copy.deepcopy(cache[key])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(AutoTokenizer, "from_pretrained", classmethod(cached_from_pretrained))
        yield


def pytest_configure(config):
    repo_basetemp = Path(__file__).resolve().parents[1] / ".pytest_tmp_local"
    repo_basetemp.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(repo_basetemp)

    if os.environ.get("GRADIEND_PROFILE_TEST_MEMORY") != "1":
        return
    config.pluginmanager.register(_MemoryProfiler(), "gradiend_memory_profiler")


def pytest_report_header(config):
    """Report CUDA inputs without initializing the runtime during pytest startup."""
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    return (
        "GRADIEND test CUDA environment preserved "
        f"(CUDA_VISIBLE_DEVICES={visible_devices!r}, torch.version.cuda={torch.version.cuda!r}); "
        "fresh-process preflight runs before the first real example"
    )


class _MemoryProfiler:
    """Log per-test RSS deltas when GRADIEND_PROFILE_TEST_MEMORY=1 is set."""

    def __init__(self):
        self._rss_before = 0

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        import gc

        try:
            import psutil
        except ImportError:
            yield
            return

        gc.collect()
        proc = psutil.Process()
        self._rss_before = proc.memory_info().rss
        yield
        gc.collect()
        rss_after = proc.memory_info().rss
        delta_mb = (rss_after - self._rss_before) / (1024 * 1024)
        total_mb = rss_after / (1024 * 1024)
        threshold = float(os.environ.get("GRADIEND_PROFILE_TEST_MEMORY_MB", "100"))
        if delta_mb >= threshold:
            print(
                f"\n[mem] +{delta_mb:.1f} MB (total {total_mb:.1f} MB): {item.nodeid}",
                flush=True,
            )


from tests.testing_mocks import MockTokenizer, SimpleMockModel


@pytest.fixture
def mock_model():
    """Fixture providing a simple mock base model."""
    return SimpleMockModel(name_or_path='mock-model', dtype=torch.float32)


@pytest.fixture
def mock_tokenizer():
    """Fixture providing a simple mock tokenizer."""
    return MockTokenizer(vocab_size=1000)


@pytest.fixture
def temp_dir():
    """Fixture providing a temporary directory that is cleaned up after the test."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def set_seed():
    """Fixture to set random seeds for reproducibility."""
    def _set_seed(seed: int = 42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
    return _set_seed


@pytest.fixture
def patch_model_loading():
    """Fixture to patch model loading to avoid HuggingFace calls."""
    from unittest.mock import patch
    from gradiend.trainer.text.common.model_base import TextModelWithGradiend
    
    def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
        """Mock _load_model to return mock objects instead of loading from HuggingFace."""
        # This will be used in tests that need to avoid HuggingFace loading
        # The actual mock_model and mock_tokenizer will be passed via closure
        # For now, return None - tests should provide their own mocks
        return None, None
    
    return patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model))
