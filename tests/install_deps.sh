#!/bin/bash
# Install all dependencies for testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Installing dependencies for gradiend-test environment..."
echo ""

cd "$PROJECT_ROOT"

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install core dependencies
echo "Installing core dependencies from requirements.txt..."
pip install -r requirements.txt

# Install development dependencies
echo "Installing development dependencies from requirements-dev.txt..."
pip install -r requirements-dev.txt

# Install package in editable mode
echo "Installing GRADIEND package in editable mode..."
pip install -e .

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "Now run tests with:"
echo "  pytest tests/ -v"
