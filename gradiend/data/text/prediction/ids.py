"""Class id helpers for text-prediction data creation."""

from __future__ import annotations

from gradiend.data.text import TextFilterConfig


def _class_id(cfg: TextFilterConfig, index: int) -> str:
    """Get class id from config; use id if set, else first target, else index fallback."""
    if cfg.id is not None:
        return cfg.id
    first = next((t for t in cfg.targets or [] if isinstance(t, str)), None)
    return first if first is not None else f"_class_{index}"

__all__ = ["_class_id"]
