#!/usr/bin/env python
"""
Run all test bench benchmarks.

This script runs all integration tests in the test bench.
It's designed to be run manually or in nightly CI jobs.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


def main():
    """Run all test bench tests."""
    test_bench_dir = Path(__file__).parent
    
    print("=" * 70)
    print("GRADIEND Test Bench")
    print("=" * 70)
    print(f"Running all benchmarks in: {test_bench_dir}")
    print()
    
    # Find all test files
    test_files = list(test_bench_dir.rglob("test_*.py"))
    
    if not test_files:
        print("No test files found!")
        return 1
    
    print(f"Found {len(test_files)} test file(s):")
    for test_file in test_files:
        print(f"  - {test_file.relative_to(test_bench_dir)}")
    print()
    
    # Run pytest on all test files (use absolute paths so package imports work)
    test_paths = [str(f.resolve()) for f in test_files]
    args = test_paths + [
        "-v",
        "-s",  # Show print statements
        "--tb=short",  # Short traceback format
        "-m", "integration",  # Only run integration tests
    ]
    
    print("Running tests...")
    print()
    
    exit_code = pytest.main(args)
    
    print()
    print("=" * 70)
    if exit_code == 0:
        print("✓ All benchmarks passed!")
    else:
        print("✗ Some benchmarks failed!")
    print("=" * 70)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
