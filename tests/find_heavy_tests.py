#!/usr/bin/env python
"""Measure peak RSS (MB) per test file (Windows/Linux)."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:
    print("psutil required: pip install psutil", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MARKER = "not slow and not integration"


def peak_rss_mb(test_file: Path) -> tuple[float, int]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_file),
        "-q",
        "-m",
        MARKER,
        "--tb=no",
    ]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    peak = 0
    while proc.poll() is None:
        try:
            p = psutil.Process(proc.pid)
            rss = p.memory_info().rss + sum(
                c.memory_info().rss for c in p.children(recursive=True)
            )
            peak = max(peak, rss)
        except (psutil.Error, psutil.NoSuchProcess):
            pass
        time.sleep(0.05)
    return peak / (1024 * 1024), proc.returncode


def main() -> int:
    files = sorted((PROJECT_ROOT / "tests").glob("test_*.py"))
    rows: list[tuple[float, str, int]] = []
    for path in files:
        if path.name == "find_heavy_tests.py":
            continue
        mb, rc = peak_rss_mb(path)
        rows.append((mb, path.name, rc))
        print(f"{mb:8.1f} MB  rc={rc}  {path.name}", flush=True)

    rows.sort(reverse=True)
    print("\n=== Top 25 by peak RSS ===")
    for mb, name, rc in rows[:25]:
        print(f"{mb:8.1f} MB  {name}" + (" FAIL" if rc else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
