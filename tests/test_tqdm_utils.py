from gradiend.util.tqdm_utils import _TqdmSafeWriter


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
