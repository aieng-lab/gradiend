"""Evaluation errors: skip the recoverable, abort on the unrecoverable.

A GRADIEND run on qwen3.5-9b hit CUDA OOM at step 0, logged it, continued
training with no evaluation, selected no checkpoint, and exited 0 -- so Slurm
reported COMPLETED for a job that trained nothing. The reported "79 GiB VRAM
peak" was the allocation failure, not a requirement.
"""

from __future__ import annotations

import pytest
import torch

from gradiend.trainer.core.callbacks import _is_unrecoverable_eval_error


class TestUnrecoverable:
    def test_torch_cuda_oom_type(self):
        oom = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if not isinstance(oom, type):
            pytest.skip("this torch has no torch.cuda.OutOfMemoryError")
        assert _is_unrecoverable_eval_error(oom("CUDA out of memory."))

    def test_plain_memory_error(self):
        assert _is_unrecoverable_eval_error(MemoryError("host ran out"))

    def test_message_fallback_when_the_type_is_unknown(self):
        """The OOM type has moved between torch versions."""
        exc = RuntimeError(
            "CUDA out of memory. Tried to allocate 2.58 GiB. GPU 0 has a total "
            "capacity of 79.25 GiB of which 567.94 MiB is free."
        )
        assert _is_unrecoverable_eval_error(exc)

    def test_generic_cuda_error_also_aborts(self):
        assert _is_unrecoverable_eval_error(RuntimeError("CUDA error: device-side assert"))


class TestRecoverable:
    def test_metric_failure_is_skipped_not_fatal(self):
        """A degenerate batch should not kill a long run."""
        assert not _is_unrecoverable_eval_error(ValueError("only one class present"))

    def test_key_error_is_skipped(self):
        assert not _is_unrecoverable_eval_error(KeyError("roc_auc"))

    def test_unrelated_runtime_error_is_skipped(self):
        assert not _is_unrecoverable_eval_error(RuntimeError("shape mismatch"))


class TestCallbackReRaises:
    def test_source_reraises_rather_than_returning_none(self):
        from pathlib import Path

        from gradiend.trainer.core import callbacks

        text = Path(callbacks.__file__).read_text(encoding="utf-8")
        guard = "if _is_unrecoverable_eval_error(e):"
        assert guard in text, "the OOM guard is missing entirely"
        after = text.split(guard, 1)[1]
        # The guarded branch must re-raise before the swallowing return.
        assert after.index("raise") < after.index("return None"), (
            "the guard logs the error but never re-raises it"
        )
