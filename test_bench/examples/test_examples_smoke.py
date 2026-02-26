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
# data_creation_pronouns must run before english_pronouns (creates data it needs)
EXAMPLE_MODULES = [
    "gradiend.examples.start_workflow",
    "gradiend.examples.race_religion",
    "gradiend.examples.gender_en",
    "gradiend.examples.gender_de",
    "gradiend.examples.gender_de_decoder_only",
    "gradiend.examples.data_creator_demo",
    "gradiend.examples.data_creation_pronouns",
    "gradiend.examples.english_pronouns",
    #"gradiend.examples.gender_de_detailed", # we exclude detailed run as it takes very long (many features)
]


def _run_example_module(module_name: str, timeout: int = 2000) -> subprocess.CompletedProcess:
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


def _write_failure_log(module_name: str, returncode: int, stdout: str, stderr: str, suffix: str = "") -> Path:
    """Write failure log and return the path. Handles both normal failures and timeouts."""
    log_path = _failure_log_path(module_name)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"=== {module_name} (returncode={returncode}){suffix} ===\n\n")
        f.write("--- stdout ---\n")
        f.write(stdout or "")
        f.write("\n--- stderr ---\n")
        f.write(stderr or "")
    print(f"\nFull log written to: {log_path}")
    return log_path


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
def test_example_runs_successfully(module_name: str):
    """
    Run each example module as-is; only check it exits successfully.

    No verification of metrics or outputs — just that the example runs to completion.
    runs/examples/ is removed before each example so use_cache=True does not skip execution.
    On failure, full stdout/stderr are written to test_bench/results/last_failure_<module>.log.
    Same for timeouts (subprocess.TimeoutExpired).
    """
    if EXAMPLE_RUNS_DIR.exists():
        shutil.rmtree(EXAMPLE_RUNS_DIR)
    try:
        result = _run_example_module(module_name)
    except subprocess.TimeoutExpired as e:
        def _to_str(b: object) -> str:
            return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")

        _write_failure_log(
            module_name,
            returncode=-1,
            stdout=_to_str(e.stdout),
            stderr=_to_str(e.stderr),
            suffix=" [TIMEOUT]",
        )
        pytest.fail(f"Example {module_name} timed out after 2000s. See {_failure_log_path(module_name)}")
    if result.returncode != 0:
        _write_failure_log(module_name, result.returncode, result.stdout or "", result.stderr or "")
    assert result.returncode == 0, (
        f"Example {module_name} failed with return code {result.returncode}. "
        f"Full stdout/stderr in {_failure_log_path(module_name)}"
    )
