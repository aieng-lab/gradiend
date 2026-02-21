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

# Install gradiend with recommended + dev extras (source: pyproject.toml)
echo "Installing gradiend with recommended and dev extras..."
pip install -r requirements-dev.txt

echo ""
echo "✅ All dependencies installed!"
echo ""
echo "Now run tests with:"
echo "  pytest tests/ -v"
