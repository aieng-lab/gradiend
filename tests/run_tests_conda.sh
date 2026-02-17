#!/bin/bash
# Run tests using python -m pytest to ensure correct interpreter

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Checking Python and torch..."
python --version
python -c "import torch; print(f'torch version: {torch.__version__}')" || {
    echo "ERROR: torch not available in current Python interpreter"
    echo "Make sure conda environment is activated: conda activate gradiend-test"
    exit 1
}

echo ""
echo "Running tests with python -m pytest..."
python -m pytest tests/ -v --tb=long "$@"
