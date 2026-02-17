# GRADIEND Test Bench

Integration tests that run actual training to verify expected behavior and catch regressions.

## Purpose

Unlike unit tests (in `tests/`), the test bench:
- Runs **actual training** (not mocked)
- Verifies **expected metrics** (e.g., correlation ≥ threshold, file existence)
- Catches **regressions** across setups
- Takes **longer to run** (minutes, not seconds)
- Requires **GPU resources**

## First-time run (no reference scores)

You **do not need reference scores** to run the test bench the first time. Verified tests use **built-in thresholds** (e.g. Pearson correlation ≥ 0.5 with tolerance 0.15, so effectively ≥ 0.35) and only check that required files exist (weights, `training.json`, etc.). Run the bench the same way every time:

```bash
# Docker (recommended)
docker build -f test_bench/Dockerfile -t gradiend-test-bench .
docker run --rm --gpus all -v "$(pwd)/test_bench/results:/workspace/test_bench/results" gradiend-test-bench

# Or with current Python
python test_bench/run_bench.py
```

The first run will **pass** if training completes and scores meet these default thresholds. No setup of expected values is required.

**Smoke-only (no score checks):** To only check that example workflows run to completion (no correlation or file checks), run only the example smoke tests:

```bash
pytest test_bench/examples/ -v -s -m integration
```

Smoke tests write to **`runs/examples/`** only. That directory is **deleted before each example** run, so reruns (e.g. same Docker image) do not reuse cache and always exercise the full pipeline. Real user runs under `runs/` (outside `runs/examples/`) are never touched.

**Recording your own baseline:** If you want to lock thresholds to your hardware (e.g. after a first run), run the full bench once and note the printed scores or the results overview; then adjust `DEFAULT_MIN_CORRELATION` / `DEFAULT_TOLERANCE` in `test_bench/verified/verify_utils.py` or the per-test `min_correlation` / `tolerance` arguments to match.

## Accessing full failing logs

- **Redirect to a file:** Run the bench and capture everything:  
  `python test_bench/run_bench.py 2>&1 | tee test_bench/results/run.log`  
  Then open `test_bench/results/run.log` for the full output (no truncation).

- **Long tracebacks:** Run the bench with full tracebacks:  
  `python test_bench/run_bench.py --tb=long`

- **Smoke test failures:** When an example smoke test fails (e.g. `gradiend.examples.gender_de_decoder_only`), the test writes **full stdout/stderr** to  
  `test_bench/results/last_failure_<module>.log`  
  so you can inspect the complete log without truncation.

## Single-call executable

One command runs the full test bench and prints a **comprehensive results overview** at the end (per-test status, time, summary).

### Direct (current Python)

```bash
python test_bench/run_bench.py
```

### Shell script (local or Apptainer)

```bash
./test_bench/run_test_bench.sh              # run with current Python
./test_bench/run_test_bench.sh --apptainer # build SIF if needed, run in Apptainer
./test_bench/run_test_bench.sh --docker    # build image, run in Docker (needs GPU)
```

### Docker (recommended)

Docker runs the **full test bench**: smoke tests and **verified runs** that assert training completes and that **scores/metrics meet expected thresholds** (e.g. Pearson correlation ≥ 0.5) and that required files exist (weights, `training.json`, etc.). Any failure (e.g. correlation too low or missing file) fails the run.

```bash
docker build -f test_bench/Dockerfile -t gradiend-test-bench .
mkdir -p test_bench/results
docker run --rm --gpus all -v "$(pwd)/test_bench/results:/workspace/test_bench/results" gradiend-test-bench
```

Results (JUnit XML and overview) are written to the mounted `test_bench/results/` directory.

**Python version:** Use a different base image to test with another Python (e.g. 3.11):

```bash
docker build -f test_bench/Dockerfile --build-arg BASE_IMAGE=continuumio/miniconda3:py311 -t gradiend-test-bench .
```

### Unit tests in Docker (no GPU, with coverage)

For fast unit tests and coverage in an isolated environment (Python easily configurable):

```bash
# Default Python 3.10
docker build -f test_bench/Dockerfile.unit-tests -t gradiend-unit-tests .
docker run --rm gradiend-unit-tests

# Python 3.11
docker build -f test_bench/Dockerfile.unit-tests --build-arg PYTHON_VERSION=3.11 -t gradiend-unit-tests .
docker run --rm gradiend-unit-tests

# Save coverage HTML report to host
mkdir -p htmlcov
docker run --rm -v "$(pwd)/htmlcov:/workspace/htmlcov" gradiend-unit-tests
```

See [docs/guides/testing.md](docs/guides/testing.md) for coverage and unit-test commands.

## Structure

```
test_bench/
├── README.md              # This file
├── run_bench.py           # Single-call entry (fresh cache, JUnit XML, results overview)
├── run_all.py             # Legacy: run pytest only
├── run_test_bench.sh      # One script: native / Apptainer / Docker
├── Dockerfile             # Docker image for test bench (GPU)
├── Dockerfile.unit-tests  # Docker image for unit tests + coverage (no GPU, PYTHON_VERSION arg)
├── results/               # Run artifacts (run_*, junit.xml) – gitignored
├── examples/              # Smoke tests
│   └── test_examples_smoke.py
├── verified/              # Verified runs (assert correlation thresholds and file existence)
│   ├── verify_utils.py
│   ├── test_race_religion.py
│   ├── test_gender_en.py
│   ├── test_gender_de.py
│   ├── test_gpt2_run.py
│   └── test_german_gpt2_mlm.py
└── apptainer/             # Apptainer definition (optional, for local/Apptainer runs)
```

## Use cases

- **Example smoke tests** (`examples/`): Run all `gradiend.setups.examples` (race_religion, gender_en, german_de, german_de_decoder_only) as they are; only assert they complete successfully (no metric verification). Use the examples’ own iteration counts for quick runs.
- **Verified runs** (`verified/`): Run with **more iterations** (e.g. 500) than examples to ensure proper conversion; verify **correlation ≥ threshold** and **existence of files** (config.json, weights, training.json).
  - One run for **race and religion** (BERT)
  - One run for **gender EN** (RoBERTa)
  - One run for **gender DE** (BERT German)
  - One run for **GPT-2** (race)
  - One run for **German GPT-2 + MLM head** for gender DE

## Usage (legacy / per-test)

Run a specific benchmark:
```bash
python -m pytest test_bench/verified/test_gender_en.py -v
```

Run all benchmarks (no fresh cache, no overview):
```bash
python test_bench/run_all.py
```

## Expected Metrics

Each setup defines `expected_metrics.json` with:
- **Minimum thresholds** (e.g., `pearson_correlation > 0.8`)
- **Tolerance ranges** (to account for hardware/version differences)
- **Required metrics** (what must be present)

## Adding New Setups

1. For **smoke only**: add a new example under `gradiend.setups.examples`; it will be run by `test_examples_smoke.py`.
2. For **verified**: create `test_bench/verified/test_<name>.py`, use `verify_utils.BENCH_TRAIN_CONFIG` and `assert_model_files_exist` / `assert_correlation_threshold`.
3. Ensure tests are marked `@pytest.mark.slow` and `@pytest.mark.integration` so `run_all.py` picks them up.

## Notes

- These tests are **not** part of standard CI/CD (too slow)
- Run them **manually** or in **nightly/weekly** CI jobs
- Metrics may vary slightly due to:
  - Different GPU hardware
  - Different package versions
  - Random initialization
  - Numerical precision
