"""
Smoke tests: run all gradiend.examples so they complete without verification.

These tests execute the example scripts as they are (no result checks) to ensure
the example workflows do not break. They use the examples' own iteration counts
for quick demonstration; the test bench verified runs use more iterations.

To avoid cache reuse across runs (e.g. same Docker, multiple test-bench runs),
runs/examples/ is removed before each example run, so use_cache=True still
exercises the full pipeline. Real user runs under runs/ (outside examples/) are not touched.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Project root (parent of test_bench)
TEST_BENCH_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = TEST_BENCH_DIR.parent

# Examples write to runs/examples/<name>; only this dir is cleaned before each run
EXAMPLE_RUNS_DIR = PROJECT_ROOT / "runs" / "examples"

# Example modules to run (under gradiend.examples)
# start_workflow is the self-contained tutorial script used by docs/start.md
# data_creation_pronouns must run before english_pronoun_singular_plural (creates data it needs)
EXAMPLE_MODULES = [
    "gradiend.examples.start_workflow",
    "gradiend.examples.race_religion",
    "gradiend.examples.gender_en",
    "gradiend.examples.gender_de",
    "gradiend.examples.gender_de_decoder_only",
    "gradiend.examples.data_creator_demo",
    "gradiend.examples.data_creation_pronouns",
    "gradiend.examples.english_pronoun_singular_plural",
    "gradiend.examples.gender_de_detailed",
]


def _run_example_module(module_name: str, timeout: int = 600) -> subprocess.CompletedProcess:
    """Run an example module's __main__ (as python -m module)."""
    return subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ},
    )


def _failure_log_path(module_name: str) -> Path:
    """Path for full stdout/stderr when a smoke test fails (so you can inspect full logs)."""
    safe_name = module_name.replace(".", "_").replace(" ", "_")
    log_dir = TEST_BENCH_DIR / "results"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"last_failure_{safe_name}.log"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
def test_example_runs_successfully(module_name: str):
    """
    Run each example module as-is; only check it exits successfully.

    No verification of metrics or outputs — just that the example runs to completion.
    runs/examples/ is removed before each example so use_cache=True does not skip execution.
    On failure, full stdout/stderr are written to test_bench/results/last_failure_<module>.log.
    """
    if EXAMPLE_RUNS_DIR.exists():
        shutil.rmtree(EXAMPLE_RUNS_DIR)
    result = _run_example_module(module_name)
    if result.returncode != 0:
        log_path = _failure_log_path(module_name)
        with open(log_path, "w") as f:
            f.write(f"=== {module_name} (returncode={result.returncode}) ===\n\n")
            f.write("--- stdout ---\n")
            f.write(result.stdout or "")
            f.write("\n--- stderr ---\n")
            f.write(result.stderr or "")
        print(f"\nFull log written to: {log_path}")
    assert result.returncode == 0, (
        f"Example {module_name} failed with return code {result.returncode}. "
        f"Full stdout/stderr in {_failure_log_path(module_name)}"
    )
