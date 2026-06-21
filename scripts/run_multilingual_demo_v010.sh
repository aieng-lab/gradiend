#!/usr/bin/env bash
# Reproduce multilingual_gradiend_demo.py from release v0.1.0 in an isolated conda env.
# Clones the repository if needed; run from any directory.
#
# Usage:
#   bash scripts/run_multilingual_demo_v010.sh
#   bash scripts/run_multilingual_demo_v010.sh --source alternative
#
# Optional env overrides:
#   GRADIEND_V010_SRC=~/Git/gradiend-v0.1.0   clone/checkout location (default)
#   GRADIEND_REPO_URL=https://github.com/aieng-lab/gradiend.git
#   GRADIEND_V010_ENV=gradiend-v010-py39
#   GRADIEND_V010_PYTHON=3.9
#   GRADIEND_V010_DATASETS_SPEC="datasets>=3.0.2,<4.0.0"
set -euo pipefail

TAG="v0.1.0"
REPO_URL="${GRADIEND_REPO_URL:-https://github.com/aieng-lab/gradiend.git}"
SRC_DIR="${GRADIEND_V010_SRC:-${HOME}/Git/gradiend-v0.1.0}"
ENV_NAME="${GRADIEND_V010_ENV:-gradiend-v010-py39}"
PYTHON_VERSION="${GRADIEND_V010_PYTHON:-3.9}"
# datasets 4.x removed trust_remote_code; v0.1.0 passes it to load_dataset.
DATASETS_SPEC="${GRADIEND_V010_DATASETS_SPEC:-datasets>=3.0.2,<4.0.0}"

ensure_source_tree() {
  if [[ -d "${SRC_DIR}/.git" ]]; then
    echo "Using existing clone: ${SRC_DIR}"
    git -C "${SRC_DIR}" fetch --depth 1 origin "refs/tags/${TAG}:refs/tags/${TAG}" 2>/dev/null \
      || git -C "${SRC_DIR}" fetch --tags origin
    git -C "${SRC_DIR}" checkout --detach "${TAG}"
    return
  fi

  if [[ -d "${SRC_DIR}" ]]; then
    echo "ERROR: ${SRC_DIR} exists but is not a git repository." >&2
    echo "Remove it or set GRADIEND_V010_SRC to another path." >&2
    exit 1
  fi

  echo "Cloning ${REPO_URL} (${TAG}) -> ${SRC_DIR}"
  mkdir -p "$(dirname "${SRC_DIR}")"
  git clone --depth 1 --branch "${TAG}" "${REPO_URL}" "${SRC_DIR}"
}

ensure_source_tree

if [[ ! -f "${SRC_DIR}/experiments/multilingual_gradiend_demo.py" ]]; then
  echo "ERROR: demo not found at ${SRC_DIR}/experiments/multilingual_gradiend_demo.py" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH; install Miniconda/Anaconda first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Creating conda env: ${ENV_NAME} (python ${PYTHON_VERSION})"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${ENV_NAME}"

python -m pip install -U pip wheel
# Editable core install only; skip [recommended] because spacy>=3.8 needs Python>=3.10.
# multilingual_gradiend_demo.py does not use spacy (only data-creation examples do).
python -m pip install -e "${SRC_DIR}"
python -m pip install \
  "${DATASETS_SPEC}" \
  "safetensors>=0.3.0" \
  "sentencepiece>=0.1.99" \
  "pyarrow>=14.0.0" \
  "matplotlib>=3.5.0" \
  "seaborn>=0.12.0" \
  "venn>=0.1.0" \
  "matplotlib_venn>=0.11.7"

python - <<EOF
import datasets
from datasets import load_dataset
import inspect
sig = inspect.signature(load_dataset)
assert "trust_remote_code" in sig.parameters, "datasets must accept trust_remote_code"
print(f"datasets {datasets.__version__} OK (trust_remote_code supported)")
EOF

cd "${SRC_DIR}"
echo "Running v0.1.0 demo from ${SRC_DIR} (env: ${ENV_NAME}, python ${PYTHON_VERSION})"
exec python experiments/multilingual_gradiend_demo.py "$@"
