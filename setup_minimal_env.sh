#!/bin/bash
# Setup script for minimal GRADIEND conda environment

set -e

ENV_NAME="gradiend-minimal"

echo "Creating minimal conda environment: $ENV_NAME"
echo "This will install only the core dependencies needed for GRADIEND."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install miniconda or anaconda first"
    exit 1
fi

# Create environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml -n $ENV_NAME

echo ""
echo "✅ Environment '$ENV_NAME' created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the installation, run:"
echo "  python test_imports_minimal.py"
