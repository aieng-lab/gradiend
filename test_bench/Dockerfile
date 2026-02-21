# Test bench image: run full test bench with fresh cache in one call.
# Build: docker build -f test_bench/Dockerfile -t gradiend-test-bench .
#        Different Python: docker build -f test_bench/Dockerfile --build-arg BASE_IMAGE=continuumio/miniconda3:py311 -t gradiend-test-bench .
# Run:   docker run --rm -v "$(pwd)/test_bench/results:/workspace/test_bench/results" gradiend-test-bench

ARG BASE_IMAGE=continuumio/miniconda3:latest
FROM ${BASE_IMAGE}

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8 PYTHONUNBUFFERED=1

# Copy repo (build context = repo root)
WORKDIR /workspace
COPY . /workspace

# Create conda env with Python, install gradiend from pyproject.toml
RUN conda update -n base -c defaults conda -y \
 && conda create -n gradiend python=3.10 -y \
 && conda run -n gradiend pip install --no-cache-dir -e ".[recommended,dev]" \
 && conda clean -afy

# Default: run test bench with fresh cache; results go to test_bench/results/
# Mount that dir when running: -v "$(pwd)/test_bench/results:/workspace/test_bench/results"
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "gradiend", "python", "test_bench/run_bench.py"]
