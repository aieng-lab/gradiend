#!/bin/bash
# Run tests using the same conda setup as PyCharm

cd /mnt/c/Git/gradiend || cd "$(dirname "$0")/.."

# Use the conda path from PyCharm output
CONDA_BIN="/home/drechsel/miniconda3/bin/conda"
ENV_NAME="gradiend"

if [ -f "$CONDA_BIN" ]; then
    echo "Running tests with conda..."
    "$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output pytest tests/ -v --tb=short
else
    echo "Conda not found at $CONDA_BIN, trying alternative methods..."
    # Try to find conda
    if command -v conda &> /dev/null; then
        conda run -n "$ENV_NAME" pytest tests/ -v --tb=short
    else
        echo "Error: Could not find conda. Please run manually:"
        echo "  conda activate gradiend"
        echo "  pytest tests/ -v --tb=short"
        exit 1
    fi
fi
