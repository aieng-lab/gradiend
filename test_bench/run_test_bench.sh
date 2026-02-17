#!/usr/bin/env bash
#
# Single-call test bench runner.
# Use on non-SLURM setups: runs the full test bench and prints a results overview at the end.
#
# Usage:
#   ./test_bench/run_test_bench.sh              # run with Python (current env)
#   ./test_bench/run_test_bench.sh --apptainer   # build SIF if needed, run in container
#   ./test_bench/run_test_bench.sh --docker     # build image, run in container
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_BENCH_DIR="$REPO_DIR/test_bench"
cd "$REPO_DIR"

run_native() {
  echo "Running test bench with current Python..."
  exec python "$TEST_BENCH_DIR/run_bench.py"
}

run_apptainer() {
  SIF_DIR="${APPTAINER_SIF_DIR:-$REPO_DIR/test_bench/apptainer}"
  SIF="$SIF_DIR/gradiend-test-bench.sif"
  DEF="$REPO_DIR/test_bench/apptainer/gradiend-test-bench.def"

  if [[ ! -f "$SIF" ]]; then
    echo "Building Apptainer image (one-time): $SIF"
    mkdir -p "$SIF_DIR"
    apptainer build "$SIF" "$DEF"
  fi

  echo "Running test bench in Apptainer (fresh cache)..."
  apptainer exec --nv --bind "$REPO_DIR:/workspace" "$SIF" \
    bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate gradiend && cd /workspace && python test_bench/run_bench.py'
}

run_docker() {
  IMAGE_NAME="${GRADIEND_TEST_BENCH_IMAGE:-gradiend-test-bench}"
  echo "Building Docker image: $IMAGE_NAME"
  docker build -f "$TEST_BENCH_DIR/Dockerfile" -t "$IMAGE_NAME" "$REPO_DIR"
  mkdir -p "$REPO_DIR/test_bench/results"
  echo "Running test bench in Docker..."
  docker run --rm --gpus all \
    -v "$REPO_DIR/test_bench/results:/workspace/test_bench/results" \
    "$IMAGE_NAME"
}

case "${1:-}" in
  --apptainer) run_apptainer ;;
  --docker)    run_docker ;;
  "")          run_native ;;
  *)
    echo "Usage: $0 [--apptainer|--docker]"
    echo "  (no args)     Run with current Python"
    echo "  --apptainer   Run inside Apptainer (builds SIF if needed)"
    echo "  --docker      Build and run inside Docker"
    exit 1
    ;;
esac
