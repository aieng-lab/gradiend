#!/bin/bash
# Script to test training script and fix errors iteratively

set -e

cd "$(dirname "$0")"
export PYTHONPATH=.

echo "Testing training script..."
echo "=========================="

# Run the script and capture setup_for_model
python gradiend/setups/gender/en/training.py 2>&1 | tee training_test_output.log

echo ""
echo "=========================="
echo "Test completed!"
