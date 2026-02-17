#!/usr/bin/env python
"""
Verify that pytest can discover all tests correctly.
Run this to check if pytest is working before configuring PyCharm.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Check pytest discovery."""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    print("=" * 60)
    print("Verifying pytest test discovery")
    print("=" * 60)
    print()
    
    # Check if pytest is available
    try:
        import pytest
        print(f"✅ pytest is installed (version: {pytest.__version__})")
    except ImportError:
        print("❌ pytest is not installed")
        print("Install with: pip install pytest")
        return 1
    
    # Try to collect tests
    print("\nCollecting tests...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dir), "--collect-only", "-q"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ pytest can discover tests:")
            print(result.stdout)
            return 0
        else:
            print("❌ pytest discovery failed:")
            print(result.stderr)
            return 1
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
