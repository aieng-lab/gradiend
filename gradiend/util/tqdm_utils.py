"""Safe tqdm defaults for non-TTY / non-blocking stderr (Docker, remote runners)."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Iterable, Iterator, Optional, TextIO, TypeVar

T = TypeVar("T")


class _TqdmSafeWriter:
    """Wrap a text stream so tqdm progress writes never raise BlockingIOError."""

    def __init__(self, stream: TextIO) -> None:
        self.stream = stream

    def write(self, s: str) -> int:
        if not s:
            return 0
        try:
            self.stream.write(s)
            return len(s)
        except BlockingIOError:
            return len(s)

    def flush(self) -> None:
        try:
            self.stream.flush()
        except BlockingIOError:
            pass

    def isatty(self) -> bool:
        try:
            return self.stream.isatty()
        except Exception:
            return False


_safe_stderr: Optional[_TqdmSafeWriter] = None


def tqdm_stderr() -> TextIO:
    global _safe_stderr
    if _safe_stderr is None:
        _safe_stderr = _TqdmSafeWriter(sys.stderr)
    return _safe_stderr  # type: ignore[return-value]


def tqdm_disabled() -> bool:
    for name in ("TQDM_DISABLE", "GRADIEND_DISABLE_TQDM"):
        if os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}:
            return True
    return False


def tqdm_kwargs(**extra: Any) -> Dict[str, Any]:
    """Default kwargs for tqdm that tolerate non-blocking stderr."""
    kwargs: Dict[str, Any] = {
        "file": tqdm_stderr(),
        "dynamic_ncols": False,
        "ascii": True,
        "mininterval": 0.5,
        "disable": tqdm_disabled() or not sys.stderr.isatty(),
    }
    kwargs.update(extra)
    if tqdm_disabled():
        kwargs["disable"] = True
    return kwargs


def gradiend_tqdm(iterable: Optional[Iterable[T]] = None, **kwargs: Any):
    from tqdm import tqdm

    return tqdm(iterable, **tqdm_kwargs(**kwargs))


def iter_gradiend_tqdm(iterable: Iterable[T], **kwargs: Any) -> Iterator[T]:
    return iter(gradiend_tqdm(iterable, **kwargs))
