#!/bin/bash
# Run tests in WSL with conda env gradiend-test (or gradiend if gradiend-test missing).

cd /mnt/c/Git/gradiend 2>/dev/null || cd "$(dirname "$0")/.."

ENV_NAME="gradiend-test"
for CONDA_BIN in "/home/drechsel/miniconda3/bin/conda" "$(which conda 2>/dev/null)"; do
    [ -n "$CONDA_BIN" ] && [ -x "$CONDA_BIN" ] && break
done

if [ -n "$CONDA_BIN" ] && [ -x "$CONDA_BIN" ]; then
    if "$CONDA_BIN" env list | grep -q "^${ENV_NAME} "; then
        "$CONDA_BIN" run -n "$ENV_NAME" --no-capture-output pytest tests/ -v --tb=short "$@"
    else
        echo "Env $ENV_NAME not found, trying 'gradiend'..."
        "$CONDA_BIN" run -n gradiend --no-capture-output pytest tests/ -v --tb=short "$@"
    fi
else
    echo "Conda not found. From WSL run:"
    echo "  conda activate gradiend-test"
    echo "  pytest tests/ -v --tb=short"
    exit 1
fi
