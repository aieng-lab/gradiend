"""A local checkpoint that fails to load is not a Transformers version problem.

Hugging Face reports a config with no usable ``model_type`` as "Unrecognized
model in <path>", which the loader matched and reported as "Transformers X does
not recognize the checkpoint architecture -- upgrade Transformers". For a
directory this repo wrote itself that is misleading: in the observed case an
ACTIEND tensors run promoted a checkpoint after *zero* seeds converged, and the
resulting artifact could not be reloaded. The message sent the reader to the
Transformers version instead of the non-convergence.
"""

from __future__ import annotations

import json

import pytest

from gradiend.trainer.text.common.loading import (
    _describe_local_checkpoint,
    _raise_unknown_transformers_architecture_error,
)


def _raise(path):
    with pytest.raises(ValueError) as excinfo:
        _raise_unknown_transformers_architecture_error(
            str(path), RuntimeError("Unrecognized model in ...")
        )
    return str(excinfo.value)


class TestLocalDirectoryDiagnosis:
    def test_missing_config_is_named(self, tmp_path):
        (tmp_path / "training.json").write_text("{}", encoding="utf-8")
        assert "no config.json" in _describe_local_checkpoint(tmp_path)

    def test_missing_model_type_is_named(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"hidden_size": 8}), encoding="utf-8"
        )
        detail = _describe_local_checkpoint(tmp_path)
        assert "no 'model_type'" in detail

    def test_missing_weights_is_named(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"model_type": "gpt_neox"}), encoding="utf-8"
        )
        assert "no weight files" in _describe_local_checkpoint(tmp_path)

    def test_unparseable_config_is_named(self, tmp_path):
        (tmp_path / "config.json").write_text("{not json", encoding="utf-8")
        assert "could not be parsed" in _describe_local_checkpoint(tmp_path)


class TestRaisedMessage:
    def test_local_path_does_not_blame_transformers(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"hidden_size": 8}), encoding="utf-8"
        )
        message = _raise(tmp_path)
        assert "upgrade transformers" not in message.lower()
        assert "released after the installed Transformers version" not in message

    def test_local_path_points_at_the_real_cause(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"hidden_size": 8}), encoding="utf-8"
        )
        message = _raise(tmp_path)
        assert "incomplete or" in message
        assert "converged" in message
        assert "no 'model_type'" in message

    def test_hub_id_keeps_the_version_guidance(self):
        """A genuine too-new architecture must still say so."""
        message = _raise("some-org/some-brand-new-model")
        assert "does not recognize the checkpoint architecture" in message
        assert "Upgrade Transformers" in message
