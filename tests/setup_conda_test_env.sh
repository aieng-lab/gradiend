#!/bin/bash
# Setup conda environment for GRADIEND testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_NAME="gradiend-test"

echo "============================================================"
echo "GRADIEND Conda Test Environment Setup"
echo "============================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not available in PATH"
    echo "Please ensure conda is installed and initialized:"
    echo "  source ~/miniconda3/etc/profile.d/conda.sh"
    echo "  or"
    echo "  export PATH=\$HOME/miniconda3/bin:\$PATH"
    exit 1
fi

echo "✅ Conda is available: $(conda --version)"
echo ""

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Conda environment '${ENV_NAME}' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n "${ENV_NAME}" -y
    else
        echo "Using existing environment. Activate with:"
        echo "  conda activate ${ENV_NAME}"
        exit 0
    fi
fi

# Create conda environment with Python
echo "Creating conda environment '${ENV_NAME}' with Python 3.10..."
conda create -n "${ENV_NAME}" python=3.10 -y

# Activate environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core dependencies from requirements.txt
echo "Installing core dependencies from requirements.txt..."
pip install -r "${PROJECT_ROOT}/requirements.txt"

# Install development dependencies
echo "Installing development dependencies from requirements-dev.txt..."
pip install -r "${PROJECT_ROOT}/requirements-dev.txt"

# Install package in editable mode
echo "Installing GRADIEND package in editable mode..."
pip install -e "${PROJECT_ROOT}"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo "  or"
echo "  python tests/run_tests.py"
echo ""
