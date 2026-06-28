import sys

from gradiend.util.tqdm_utils import (
    _TqdmSafeWriter,
    gradiend_tqdm,
    patch_sys_stderr_for_tqdm,
)


class _BlockingStream:
    def __init__(self) -> None:
        self.writes: list[str] = []

    def write(self, s: str) -> int:
        self.writes.append(s)
        raise BlockingIOError(11, "write would block")

    def flush(self) -> None:
        raise BlockingIOError(11, "flush would block")

    def isatty(self) -> bool:
        return True


def test_tqdm_safe_writer_swallows_blocking_io_error() -> None:
    stream = _BlockingStream()
    writer = _TqdmSafeWriter(stream)
    assert writer.write("progress") == len("progress")
    writer.flush()


def test_patch_sys_stderr_for_tqdm_swallows_blocking_io_error(monkeypatch) -> None:
    stream = _BlockingStream()
    monkeypatch.setattr("gradiend.util.tqdm_utils.sys.stderr", stream)
    monkeypatch.setattr("gradiend.util.tqdm_utils._safe_stderr", None)
    monkeypatch.setattr("gradiend.util.tqdm_utils._stderr_patched", False)

    patch_sys_stderr_for_tqdm()

    assert isinstance(sys.stderr, _TqdmSafeWriter)
    sys.stderr.write("progress")
    sys.stderr.flush()


def test_gradiend_tqdm_swallows_blocking_io_error(monkeypatch) -> None:
    stream = _BlockingStream()
    monkeypatch.setattr(sys, "stderr", stream)
    monkeypatch.setattr("gradiend.util.tqdm_utils._safe_stderr", None)
    monkeypatch.setattr("gradiend.util.tqdm_utils._stderr_patched", False)
    monkeypatch.setattr("gradiend.util.tqdm_utils.tqdm_disabled", lambda: False)

    patch_sys_stderr_for_tqdm()

    values = []
    for value in gradiend_tqdm(range(3), desc="test", mininterval=0):
        values.append(value)

    assert values == [0, 1, 2]
