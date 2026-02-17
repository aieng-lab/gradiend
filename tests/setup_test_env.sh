#!/bin/bash
# Setup script for GRADIEND test environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/test_env"

echo "============================================================"
echo "GRADIEND Test Environment Setup"
echo "============================================================"
echo ""

# Check if venv already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at: $VENV_DIR"
    echo "Activate it with: source $VENV_DIR/bin/activate"
    exit 0
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR" || {
    echo "ERROR: Failed to create virtual environment."
    echo "You may need to install python3-venv:"
    echo "  sudo apt-get install python3.12-venv"
    exit 1
}

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install core dependencies
echo "Installing core dependencies from requirements.txt..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# Install pytest
echo "Installing pytest..."
pip install pytest

# Install package in editable mode
echo "Installing GRADIEND package in editable mode..."
pip install -e "$PROJECT_ROOT"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest tests/ -v"
echo "  or"
echo "  python tests/run_tests.py"
echo ""
