"""The example smoke test builds the English pronoun data from a small corpus (no network, no training)."""

import importlib.util
import json
import os
import subprocess
import sys

import pytest

from gradiend.examples import create_english_pronoun_data as pronoun_data
from test_bench.examples import test_examples_smoke as smoke


_FAKE_EXAMPLE = """import json
from gradiend.examples.create_english_pronoun_data import english_pronoun_generation_config

if __name__ == "__main__":
    print("CONFIG=" + json.dumps(english_pronoun_generation_config()))
"""


def _source(module_name):
    return open(importlib.util.find_spec(module_name).origin, encoding="utf-8").read()


def test_every_example_using_the_pronoun_data_runs_on_the_small_corpus():
    uses_pronoun_data = {
        name for name in smoke.EXAMPLE_MODULES if "english_pronoun_data" in _source(name)
    }
    assert uses_pronoun_data == smoke.PRONOUN_DATA_EXAMPLES


def test_small_corpus_settings_are_constants_of_the_data_example():
    for name in smoke.SMALL_PRONOUN_DATA:
        assert hasattr(pronoun_data, name), name


def test_small_corpus_is_small():
    settings = smoke.SMALL_PRONOUN_DATA
    assert settings["WIKIPEDIA_BASE_MAX_SIZE"] < pronoun_data.WIKIPEDIA_BASE_MAX_SIZE
    assert settings["MAX_SIZE_PER_CLASS"] <= pronoun_data.MAX_SIZE_PER_CLASS
    assert settings["NEUTRAL_MAX_SIZE"] <= pronoun_data.NEUTRAL_MAX_SIZE


def test_other_examples_run_unchanged():
    assert smoke._example_command("gradiend.examples.start_workflow") == [
        sys.executable, "-m", "gradiend.examples.start_workflow",
    ]


def _run_runner(tmp_path, settings):
    (tmp_path / "fake_example.py").write_text(_FAKE_EXAMPLE)
    return subprocess.run(
        [sys.executable, "-c", smoke._SMALL_PRONOUN_DATA_RUNNER, "fake_example", json.dumps(settings)],
        cwd=str(tmp_path),
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), str(smoke.PROJECT_ROOT)])},
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_runner_gives_the_example_the_small_corpus(tmp_path):
    result = _run_runner(tmp_path, smoke.SMALL_PRONOUN_DATA)
    assert result.returncode == 0, result.stderr
    config = json.loads(next(line for line in result.stdout.splitlines() if line.startswith("CONFIG="))[len("CONFIG="):])
    assert config["base_data"] == smoke.SMALL_PRONOUN_DATA["WIKIPEDIA_DATASET"]
    assert config["hf_config"] is None
    assert config["base_max_size"] == smoke.SMALL_PRONOUN_DATA["WIKIPEDIA_BASE_MAX_SIZE"]
    assert config["max_size_per_class"] == smoke.SMALL_PRONOUN_DATA["MAX_SIZE_PER_CLASS"]
    assert config["neutral_max_size"] == smoke.SMALL_PRONOUN_DATA["NEUTRAL_MAX_SIZE"]


def test_runner_rejects_an_unknown_setting(tmp_path):
    result = _run_runner(tmp_path, {"NO_SUCH_SETTING": 1})
    assert result.returncode != 0
    assert "no setting 'NO_SUCH_SETTING'" in result.stderr


def test_pronoun_data_examples_use_the_runner():
    for name in smoke.PRONOUN_DATA_EXAMPLES:
        command = smoke._example_command(name)
        assert command[:3] == [sys.executable, "-c", smoke._SMALL_PRONOUN_DATA_RUNNER]
        assert command[3] == name
        assert json.loads(command[4]) == smoke.SMALL_PRONOUN_DATA
