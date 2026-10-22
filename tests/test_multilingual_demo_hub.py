"""A Hugging Face repo id as --experiment-dir is downloaded to runs/<name>; local folders stay as they are."""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def demo():
    import experiments.multilingual_gradiend_demo as mod

    return mod


def test_existing_local_folders_and_new_paths_are_kept(demo, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr("huggingface_hub.HfApi.repo_exists", lambda self, *a, **k: called.append(a) or False)
    assert demo.resolve_experiment_dir(str(tmp_path)) == str(tmp_path)  # exists: no lookup
    assert demo.resolve_experiment_dir("runs/new_run") == "runs/new_run"  # no such repo
    assert demo.resolve_experiment_dir("single_name") == "single_name"
    assert called == [("runs/new_run",)]


def test_hub_repo_is_downloaded_into_runs(demo, monkeypatch):
    seen = {}
    monkeypatch.setattr("huggingface_hub.HfApi.repo_exists", lambda self, *a, **k: True)

    def fake_download(repo_id, **kwargs):
        seen.update(repo_id=repo_id, **kwargs)
        return kwargs["local_dir"]

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    out = demo.resolve_experiment_dir("aieng-lab/gradiend-multilingual-bert-features")
    assert seen["repo_id"] == "aieng-lab/gradiend-multilingual-bert-features"
    assert out.replace("\\", "/") == "runs/gradiend-multilingual-bert-features"


def test_offline_lookup_falls_back_to_a_local_folder(demo, monkeypatch):
    def offline(self, *a, **k):
        raise OSError("no network")

    monkeypatch.setattr("huggingface_hub.HfApi.repo_exists", offline)
    assert demo.resolve_experiment_dir("aieng-lab/some-repo") == "aieng-lab/some-repo"


def test_hub_download_is_detected(demo, tmp_path):
    assert demo._is_hub_download("aieng-lab/gradiend-x", "runs/gradiend-x")
    assert not demo._is_hub_download("runs/new_run", "runs/new_run")  # nothing was downloaded
    assert not demo._is_hub_download(str(tmp_path), "elsewhere")  # an existing local folder
    assert not demo._is_hub_download(None, "runs/x")


def test_data_is_generated_by_default(demo, monkeypatch):
    monkeypatch.setattr(demo, "GENERATE_MISSING_DATA", True)
    monkeypatch.setattr(demo, "ensure_english_pronoun_data", lambda **kwargs: ("generated", "paths"))
    assert demo._pronoun_data_paths() == ("generated", "paths")


def test_no_data_generation_never_creates_data(demo, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(demo, "GENERATE_MISSING_DATA", False)

    def forbidden(**kwargs):
        raise AssertionError("data generation must not run")

    monkeypatch.setattr(demo, "ensure_english_pronoun_data", forbidden)
    monkeypatch.setattr("gradiend.examples.train_sentiment.generate_data", forbidden)

    pronoun_training, _ = demo._pronoun_data_paths()
    sentiment_training, _ = demo._sentiment_data_paths()
    assert not pronoun_training.exists() and not sentiment_training.exists()
    with pytest.raises(FileNotFoundError, match="--no-data-generation"):
        demo._cross_task_probe_trainers(None)  # the probe pool needs the data: callers fall back to the cache


def test_published_sentiment_data_joins_the_splits(demo, monkeypatch):
    import pandas as pd

    class Split:
        def __init__(self, frame):
            self.frame = frame

        def to_pandas(self):
            return self.frame

    splits = {"train": Split(pd.DataFrame({"x": [1]})), "test": Split(pd.DataFrame({"x": [2]}))}
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: splits)
    assert list(demo._published_sentiment_training_data()["x"]) == [1, 2]
