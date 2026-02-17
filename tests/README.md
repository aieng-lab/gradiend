# GRADIEND Test Suite

This directory contains unit tests for the GRADIEND framework.

## Test Structure

Tests are organized by component:
- `test_model.py` - Core model tests (GradiendModel, ParamMappedGradiendModel)
- `test_training_arguments.py` - TrainingArguments validation and serialization
- `test_optional_dependencies.py` - Optional dependency handling (safetensors, matplotlib, seaborn, datasets, spacy)
- `conftest.py` - Shared pytest fixtures and utilities

## Running Tests

### Using pytest directly:
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run specific test
pytest tests/test_model.py::TestGradiendModel::test_gradiend_model_creation -v
```

### Using the test runner script:
```bash
python tests/run_tests.py
```

### In WSL with conda env gradiend-test:
```bash
# From project root in WSL (e.g. cd /mnt/c/Git/gradiend)
conda activate gradiend-test
pytest tests/ -v --tb=short

# Or run only data/trainer data-input tests:
pytest tests/test_data_generation.py tests/test_unified_data.py tests/test_trainer_data_inputs.py tests/test_data_module.py -v --tb=short

# Optional: use the helper script (uses gradiend-test if available)
bash tests/run_tests_wsl_gradiend_test.sh
```

### In PyCharm:
**IMPORTANT:** PyCharm must be configured to use pytest, not unittest!

1. Go to **Settings** → **Tools** → **Python Integrated Tools**
2. Set **Default test runner** to **pytest** (NOT unittest)
3. Click **Apply** and **OK**
4. Right-click on the `tests` folder → **Run 'pytest in tests'**

**If tests don't appear:**
- See `PYCHARM_SETUP.md` in project root for detailed instructions
- Run `python tests/verify_pytest_discovery.py` to verify pytest works
- Try **File** → **Invalidate Caches / Restart...**

## Test Configuration

Pytest configuration is in `pytest.ini` at the project root. The configuration:
- Sets `tests` as the test directory
- Looks for files matching `test_*.py`
- Uses verbose output by default
- Defines test markers (`slow`, `integration`)

## Fixtures

Shared fixtures are in `conftest.py`:
- `mock_model` - Simple mock base model
- `mock_tokenizer` - Mock tokenizer
- `temp_dir` - Temporary directory fixture
- `set_seed` - Seed setting utility

## Test Categories

- **Unit tests**: Fast tests using mocked/toy networks
- **Integration tests**: Marked with `@pytest.mark.integration` (run actual training)
- **Slow tests**: Marked with `@pytest.mark.slow` (can be skipped with `-m "not slow"`)
