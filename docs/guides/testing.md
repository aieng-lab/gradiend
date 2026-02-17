# Testing and coverage




## Unit tests (fast)

Run the test suite in `tests/` (excludes slow and integration tests by default in CI):

```bash
pytest tests/ -v --tb=short
```

Exclude slow/integration tests explicitly:

```bash
pytest tests/ -v -m "not slow and not integration"
```

Run a single test file or test:

```bash
pytest tests/test_data_generation.py -v
pytest tests/test_img_format.py::TestImgFormatTrainerForwarding::test_plot_encoder_distributions_receives_img_format_from_trainer -v
```

## Test coverage

Install dev dependencies (includes `pytest-cov`):

```bash
pip install -r requirements-dev.txt
# or: pip install -e ".[dev]"
```

Run tests with coverage report (terminal + optional HTML):

```bash
# Coverage for package code only (exclude tests and examples)
pytest tests/ -v --cov=gradiend --cov-report=term-missing --cov-report=html -m "not slow and not integration"
```

- **Terminal:** `--cov-report=term-missing` shows missed lines per file.
- **HTML report:** `--cov-report=html` writes `htmlcov/index.html`; open it in a browser to see line-by-line coverage and find untested code.

Useful options:

- `--cov-fail-under=60` — fail the run if total coverage is below 60%.
- Omit `-m "not slow and not integration"` to include all tests (slower).

## Test bench (integration, GPU)

The **test bench** runs real training and verification; it is slower and typically needs a GPU. **You do not need reference scores:** verified tests use built-in thresholds (e.g. correlation ≥ 0.35); run the bench the same way every time. See [test_bench/README.md](https://github.com/aieng-lab/gradiend/blob/main/test_bench/README.md) for first-time run and:

- **Running locally:** `python test_bench/run_bench.py`
- **Docker (recommended):** Build and run the test-bench image; it runs the full test bench including **verified runs** that assert scores (e.g. correlation ≥ threshold) and required files. Use `--build-arg BASE_IMAGE=continuumio/miniconda3:py311` for a different Python.
- **Unit tests in Docker (no GPU):** `docker build -f test_bench/Dockerfile.unit-tests --build-arg PYTHON_VERSION=3.11 -t gradiend-unit-tests .` then `docker run --rm gradiend-unit-tests`.

The test bench is **not** run in standard CI; run it manually or in a nightly job.
