#!/usr/bin/env python
"""
Test runner script that checks dependencies and runs tests.

This script:
1. Checks if pytest is available
2. Checks if required dependencies are installed
3. Runs the tests if everything is available
4. Provides helpful error messages if dependencies are missing
"""

import sys
import subprocess
from pathlib import Path

def check_dependency(module_name, package_name=None):
    """Check if a dependency is available."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def main():
    """Run tests with dependency checking."""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    print("=" * 60)
    print("GRADIEND Test Runner")
    print("=" * 60)
    print()
    
    # Check pytest
    print("Checking dependencies...")
    pytest_available, pytest_error = check_dependency("pytest", "pytest")
    
    if not pytest_available:
        print("❌ pytest is not installed")
        print(f"   Error: {pytest_error}")
        print()
        print("Install pytest with:")
        print("  pip install pytest")
        print("  or")
        print("  pip install -r requirements-dev.txt")
        return 1
    
    print("✅ pytest is available")
    
    # Check core dependencies
    core_deps = [
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ]
    
    missing_deps = []
    for module, package in core_deps:
        available, error = check_dependency(module, package)
        if available:
            print(f"✅ {package} is available")
        else:
            print(f"❌ {package} is not available: {error}")
            missing_deps.append(package)
    
    if missing_deps:
        print()
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print("Install with:")
        print(f"  pip install {' '.join(missing_deps)}")
        print("  or")
        print("  pip install -r requirements-dev.txt")
        return 1
    
    print()
    print("=" * 60)
    print("Running tests...")
    print("=" * 60)
    print()
    
    # Run pytest
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=short"],
            cwd=str(project_root),
            check=False
        )
        return result.returncode
    except Exception as e:
        print(f"Error running pytest: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
