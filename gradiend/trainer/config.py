"""
Trainer configuration: modality-agnostic parameters shared across trainer types.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerConfig:
    """
    Base configuration for GRADIEND trainers.

    Holds parameters that are independent of modality (text, image, etc.),
    such as visualization/plotting options used by all trainers.
    """

    img_format: str = "pdf"
    """Image format for plots (e.g. encoder distributions, training convergence). Forwarded to visualizer."""

    img_dpi: Optional[int] = None
    """DPI for saved plots (e.g. 600 for publication). None = use visualizer default."""

    def __str__(self) -> str:
        return f"TrainerConfig(img_format={self.img_format!r}, img_dpi={self.img_dpi})"
