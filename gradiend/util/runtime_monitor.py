"""Optional persistent runtime monitor for crash diagnostics."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
import traceback
from typing import Any, Dict, Optional

import torch


class NullRuntimeMonitor:
    enabled = False
    path: Optional[str] = None

    def __enter__(self) -> "NullRuntimeMonitor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def mark(self, phase: str, **payload: Any) -> None:
        return None


class CudaMemorySpan:
    """Measure per-device PyTorch CUDA peak memory within a code span."""

    def __init__(self, *, reset_peak: bool = True, synchronize: bool = True) -> None:
        self.reset_peak = bool(reset_peak)
        self.synchronize = bool(synchronize)
        self.enabled = torch.cuda.is_available()
        self.stats: list = []

    def __enter__(self) -> "CudaMemorySpan":
        if not self.enabled:
            return self
        for index in range(torch.cuda.device_count()):
            try:
                if self.synchronize:
                    torch.cuda.synchronize(index)
                if self.reset_peak:
                    torch.cuda.reset_peak_memory_stats(index)
            except Exception:
                pass
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.enabled:
            self.stats = []
            return None
        rows = []
        for index in range(torch.cuda.device_count()):
            try:
                if self.synchronize:
                    torch.cuda.synchronize(index)
                rows.append(
                    {
                        "index": index,
                        "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                        "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                    }
                )
            except Exception as span_exc:
                rows.append({"index": index, "error": str(span_exc)})
        self.stats = rows
        return None

    @property
    def max_allocated_bytes(self) -> Optional[int]:
        values = [
            int(row["max_allocated_bytes"])
            for row in self.stats
            if isinstance(row, dict) and isinstance(row.get("max_allocated_bytes"), int)
        ]
        return max(values) if values else None

    @property
    def max_reserved_bytes(self) -> Optional[int]:
        values = [
            int(row["max_reserved_bytes"])
            for row in self.stats
            if isinstance(row, dict) and isinstance(row.get("max_reserved_bytes"), int)
        ]
        return max(values) if values else None


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Return True for the common CUDA out-of-memory exception shapes."""
    message = str(exc).lower()
    if "cuda" not in message:
        return False
    return "out of memory" in message or "outofmemory" in message


def write_cuda_oom_log(
    root: Optional[str],
    *,
    phase: str,
    exc: BaseException,
    monitor_path: Optional[str] = None,
) -> Optional[str]:
    """Write a compact CUDA OOM diagnostic log when an experiment/output dir exists."""
    if not root or not is_cuda_oom_error(exc):
        return None
    try:
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, "cuda_oom_error.log")
        row = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "time_unix": time.time(),
            "pid": os.getpid(),
            "phase": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "torch_cuda": RuntimeMonitor.collect_torch_cuda_stats(),
            "nvidia_smi": RuntimeMonitor.collect_nvidia_smi_stats(),
            "monitor_path": monitor_path,
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, default=str, sort_keys=True) + "\n")
        return path
    except Exception:
        return None


class RuntimeMonitor:
    """Append-only JSONL monitor with optional periodic system stats."""

    enabled = True

    def __init__(
        self,
        path: str,
        *,
        interval: float = 5.0,
        system_stats: bool = True,
    ) -> None:
        self.path = path
        self.interval = float(interval)
        self.system_stats = bool(system_stats)
        self._phase = "init"
        self._phase_payload: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fh = None

    @classmethod
    def from_training_args(cls, args: Any, output_dir: Optional[str]) -> "RuntimeMonitor | NullRuntimeMonitor":
        if not bool(getattr(args, "runtime_monitor", False)):
            return NullRuntimeMonitor()
        root = output_dir or getattr(args, "output_dir", None) or getattr(args, "experiment_dir", None)
        if not root:
            root = os.getcwd()
        os.makedirs(root, exist_ok=True)
        path = os.path.join(root, "runtime_monitor.jsonl")
        return cls(
            path,
            interval=float(getattr(args, "runtime_monitor_interval", 5.0)),
            system_stats=bool(getattr(args, "runtime_monitor_system_stats", True)),
        )

    def __enter__(self) -> "RuntimeMonitor":
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        self._write({"event": "monitor_start", "system": self._system_identity()})
        if self.system_stats and self.interval > 0:
            self._thread = threading.Thread(target=self._run, name="gradiend-runtime-monitor", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.mark("exception", exc_type=getattr(exc_type, "__name__", str(exc_type)), exc=str(exc))
            if exc is not None:
                write_cuda_oom_log(
                    os.path.dirname(os.path.abspath(self.path)),
                    phase=self._phase,
                    exc=exc,
                    monitor_path=self.path,
                )
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(max(self.interval, 1.0), 10.0))
        self._write({"event": "monitor_stop"})
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def mark(self, phase: str, **payload: Any) -> None:
        with self._lock:
            self._phase = str(phase)
            self._phase_payload = dict(payload)
        row = {"event": "mark", "phase": phase}
        row.update(payload)
        self._write(row)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                phase = self._phase
                phase_payload = dict(self._phase_payload)
            self._write(
                {
                    "event": "heartbeat",
                    "phase": phase,
                    "phase_payload": phase_payload,
                    "stats": self._collect_stats(),
                }
            )

    def _write(self, row: Dict[str, Any]) -> None:
        if self._fh is None:
            return
        row = dict(row)
        row.setdefault("time", time.strftime("%Y-%m-%d %H:%M:%S"))
        row.setdefault("time_unix", time.time())
        row.setdefault("pid", os.getpid())
        try:
            self._fh.write(json.dumps(row, default=str, sort_keys=True) + "\n")
            self._fh.flush()
        except Exception:
            pass

    def _system_identity(self) -> Dict[str, Any]:
        return {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }

    def _collect_stats(self) -> Dict[str, Any]:
        return {
            "memory": self._process_memory(),
            "torch_cuda": self.collect_torch_cuda_stats(),
            "nvidia_smi": self.collect_nvidia_smi_stats(),
        }

    def _process_memory(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_kb = int(usage.ru_maxrss)
            if platform.system().lower() == "darwin":
                out["max_rss_bytes"] = rss_kb
            else:
                out["max_rss_bytes"] = rss_kb * 1024
        except Exception:
            pass
        if hasattr(os, "sysconf"):
            try:
                pages = os.sysconf("SC_AVPHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                out["available_ram_bytes"] = int(pages) * int(page_size)
            except Exception:
                pass
        return out

    @staticmethod
    def collect_torch_cuda_stats() -> list:
        if not torch.cuda.is_available():
            return []
        stats = []
        for i in range(torch.cuda.device_count()):
            try:
                stats.append(
                    {
                        "index": i,
                        "allocated_bytes": int(torch.cuda.memory_allocated(i)),
                        "reserved_bytes": int(torch.cuda.memory_reserved(i)),
                        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(i)),
                        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(i)),
                    }
                )
            except Exception as exc:
                stats.append({"index": i, "error": str(exc)})
        return stats

    @staticmethod
    def collect_nvidia_smi_stats() -> list:
        query = "index,memory.used,memory.free,utilization.gpu,power.draw"
        try:
            proc = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2.0,
                check=False,
            )
        except Exception:
            return []
        if proc.returncode != 0:
            return []
        rows = []
        keys = ["index", "memory_used_mib", "memory_free_mib", "gpu_util_percent", "power_draw_w"]
        for line in proc.stdout.splitlines():
            values = [v.strip() for v in line.split(",")]
            if len(values) != len(keys):
                continue
            row: Dict[str, Any] = {}
            for key, value in zip(keys, values):
                try:
                    row[key] = float(value) if "." in value else int(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
        return rows
