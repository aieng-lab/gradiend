#!/usr/bin/env python
"""
Run tests and save output to a file for debugging.
This helps capture test failures when running in environments where direct access is limited.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

def main():
    """Run tests and save output."""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    output_file = project_root / "test_output.txt"
    
    print("=" * 60)
    print("Running GRADIEND tests...")
    print("=" * 60)
    print(f"Output will be saved to: {output_file}")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_dir), "-v", "--tb=long"],
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Write output to file
        with open(output_file, 'w') as f:
            f.write(f"Test run at: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
            f.write("\n\nRETURN CODE:\n")
            f.write(str(result.returncode))
        
        # Also print to console
        print(result.stdout)
        if result.stderr:
            print("\nSTDERR:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
        
        print(f"\n{'=' * 60}")
        print(f"Test output saved to: {output_file}")
        print(f"Return code: {result.returncode}")
        print("=" * 60)
        
        return result.returncode
        
    except Exception as e:
        error_msg = f"Error running tests: {e}"
        print(error_msg, file=sys.stderr)
        with open(output_file, 'w') as f:
            f.write(error_msg)
        return 1

if __name__ == "__main__":
    sys.exit(main())
