#!/usr/bin/env python
"""
Check that all test files can be imported without errors.

This helps identify missing dependencies or import issues.
"""

import sys
import importlib.util
from pathlib import Path

def check_import(file_path):
    """Try to import a test file and report any errors."""
    file_path = Path(file_path)
    module_name = file_path.stem
    
    print(f"Checking {file_path.name}...", end=" ")
    
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"❌ Could not create spec")
            return False
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        
        # Try to execute the module
        spec.loader.exec_module(module)
        print("✅ OK")
        return True
    except ImportError as e:
        error_msg = str(e)
        # pytest is expected to be missing in some environments
        if "pytest" in error_msg.lower():
            print(f"⚠️  Missing pytest (expected for import check)")
            return True  # Don't fail on pytest
        print(f"❌ ImportError: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False

def main():
    """Check all test files."""
    test_dir = Path(__file__).parent
    test_files = sorted(test_dir.glob("test_*.py"))
    
    print("=" * 60)
    print("Checking test file imports...")
    print("=" * 60)
    print()
    
    results = []
    for test_file in test_files:
        if test_file.name == "check_test_imports.py":
            continue
        success = check_import(test_file)
        results.append((test_file.name, success))
    
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    failed = len(results) - passed
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    print()
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
