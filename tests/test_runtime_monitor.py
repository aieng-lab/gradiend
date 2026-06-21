import json

import torch

from gradiend.util.runtime_monitor import CudaMemorySpan, is_cuda_oom_error, write_cuda_oom_log


def test_cuda_oom_detection_requires_cuda_and_oom_text():
    assert is_cuda_oom_error(RuntimeError("CUDA out of memory. Tried to allocate 10 MiB"))
    assert is_cuda_oom_error(RuntimeError("cuda OutOfMemoryError: allocation failed"))
    assert not is_cuda_oom_error(RuntimeError("CPU out of memory"))
    assert not is_cuda_oom_error(RuntimeError("CUDA launch failed"))


def test_write_cuda_oom_log_jsonl(tmp_path):
    exc = RuntimeError("CUDA out of memory. Tried to allocate 10 MiB")

    path = write_cuda_oom_log(str(tmp_path), phase="train", exc=exc, monitor_path="monitor.jsonl")

    assert path is not None
    rows = (tmp_path / "cuda_oom_error.log").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert payload["phase"] == "train"
    assert payload["error_type"] == "RuntimeError"
    assert "CUDA out of memory" in payload["error"]
    assert payload["monitor_path"] == "monitor.jsonl"


def test_write_cuda_oom_log_ignores_non_oom(tmp_path):
    path = write_cuda_oom_log(str(tmp_path), phase="train", exc=RuntimeError("CUDA launch failed"))

    assert path is None
    assert not (tmp_path / "cuda_oom_error.log").exists()


def test_cuda_memory_span_collects_peak_stats(monkeypatch):
    calls = []

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda index: calls.append(("sync", index)))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda index: calls.append(("reset", index)))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda index: 10 + index)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda index: 20 + index)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda index: 100 + index)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda index: 200 + index)

    with CudaMemorySpan() as span:
        assert span.stats == []

    assert span.max_allocated_bytes == 101
    assert span.max_reserved_bytes == 201
    assert span.stats == [
        {
            "index": 0,
            "allocated_bytes": 10,
            "reserved_bytes": 20,
            "max_allocated_bytes": 100,
            "max_reserved_bytes": 200,
        },
        {
            "index": 1,
            "allocated_bytes": 11,
            "reserved_bytes": 21,
            "max_allocated_bytes": 101,
            "max_reserved_bytes": 201,
        },
    ]
    assert ("reset", 0) in calls
    assert ("reset", 1) in calls
