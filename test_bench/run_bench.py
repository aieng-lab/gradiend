#!/usr/bin/env python
"""
Single-call test bench entry point.

Runs all integration tests and produces a comprehensive results overview at the end.

Usage:
  python test_bench/run_bench.py [--output-dir DIR] [--tb short|long|line|no]
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Resolve project and test_bench paths before any gradiend imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TEST_BENCH_DIR = SCRIPT_DIR

# Add project root so imports work when run as script
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_junit_xml(path: Path) -> list[dict]:
    """Parse pytest JUnit XML; return list of testcase dicts (name, status, time)."""
    if not path.exists():
        return []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        cases = []
        for suite in root.iter("testsuite"):
            for case in suite.iter("testcase"):
                name = case.get("name", "")
                classname = case.get("classname", "")
                time_val = float(case.get("time", 0))
                if case.find("failure") is not None:
                    status = "FAILED"
                elif case.find("skipped") is not None:
                    status = "SKIPPED"
                else:
                    status = "PASSED"
                full_name = f"{classname}::{name}" if classname else name
                cases.append({"name": full_name, "status": status, "time": time_val})
        return cases
    except Exception as e:
        return [{"name": f"(parse error: {e})", "status": "ERROR", "time": 0}]


def _print_results_overview(cases: list[dict], junit_path: Path | None) -> None:
    """Print a comprehensive results table and summary."""
    print()
    print("=" * 80)
    print("GRADIEND TEST BENCH – RESULTS OVERVIEW")
    print("=" * 80)
    if junit_path and junit_path.exists():
        print(f"Report: {junit_path}")
    print()

    if not cases:
        print("No test results (run with --junitxml to capture).")
        print("=" * 80)
        return

    # Table header
    name_width = min(70, max(50, max(len(c["name"]) for c in cases) + 2))
    print(f"{'Test':<{name_width}} {'Status':<10} {'Time (s)':>10}")
    print("-" * (name_width + 22))

    passed = failed = skipped = 0
    total_time = 0.0
    for c in cases:
        name = c["name"]
        if len(name) > name_width - 2:
            name = name[: name_width - 5] + "..."
        status = c["status"]
        t = c["time"]
        total_time += t
        if status == "PASSED":
            passed += 1
        elif status == "FAILED" or status == "ERROR":
            failed += 1
        else:
            skipped += 1
        print(f"{name:<{name_width}} {status:<10} {t:>10.2f}")

    print("-" * (name_width + 22))
    print(f"{'TOTAL':<{name_width}} {'':<10} {total_time:>10.2f}")
    print()
    print("Summary:")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total:   {len(cases)}")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run GRADIEND test bench and print results overview."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for results (junit XML). Default: test_bench/results/.",
    )
    parser.add_argument(
        "--no-overview",
        action="store_true",
        help="Do not print results overview (only pytest output).",
    )
    parser.add_argument(
        "--tb",
        default="short",
        choices=("short", "long", "line", "no"),
        help="Pytest traceback style (default: short). Use --tb=long for full tracebacks.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("GRADIEND Test Bench")
    print("=" * 70)

    output_dir = args.output_dir or (TEST_BENCH_DIR / "results")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    junit_path = output_dir / "junit.xml"

    # Discover and run tests
    test_files = list(TEST_BENCH_DIR.rglob("test_*.py"))
    if not test_files:
        print("No test files found!", file=sys.stderr)
        return 1

    test_paths = [str(f.resolve()) for f in test_files]
    pytest_args = test_paths + [
        "-v",
        "-s",
        f"--tb={args.tb}",
        "-m", "integration",
        f"--junitxml={junit_path}",
    ]

    import pytest
    exit_code = pytest.main(pytest_args)

    # Results overview
    if not args.no_overview:
        cases = _parse_junit_xml(junit_path)
        _print_results_overview(cases, junit_path)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
