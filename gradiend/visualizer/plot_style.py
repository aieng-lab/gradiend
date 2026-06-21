"""
Matplotlib style defaults for GRADIEND plots.

Environment variables:

``GRADIEND_PLOT_USE_LATEX``
    ``auto`` (default): enable ``text.usetex`` when a LaTeX executable is on ``PATH``.
    ``1``/``true``/``yes``: force LaTeX rendering (falls back with a warning if unavailable).
    ``0``/``false``/``no``: disable LaTeX rendering.

``GRADIEND_PLOT_FONT_PATH``
    Path to a ``.ttf``/``.otf`` font file registered with matplotlib and used as ``font.family``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from gradiend.util.logging import get_logger

logger = get_logger(__name__)

ENV_USE_LATEX = "GRADIEND_PLOT_USE_LATEX"
ENV_FONT_PATH = "GRADIEND_PLOT_FONT_PATH"

_LATEX_COMMANDS = ("latex", "pdflatex", "xelatex", "lualatex")
_CONFIGURED = False


def _parse_use_latex_env() -> Optional[bool]:
    raw = os.environ.get(ENV_USE_LATEX)
    if raw is None or not str(raw).strip():
        return None
    value = str(raw).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    if value in {"auto", "default"}:
        return None
    raise ValueError(
        f"{ENV_USE_LATEX} must be auto, 1, or 0; got {raw!r}"
    )


def _latex_on_path() -> bool:
    return any(shutil.which(command) for command in _LATEX_COMMANDS)


def _latex_usable() -> bool:
    if not _latex_on_path():
        return False
    try:
        from matplotlib.texmanager import TexManager

        TexManager()
        return True
    except Exception as exc:
        logger.debug("LaTeX found on PATH but matplotlib TexManager failed: %s", exc)
        return False


def _resolve_use_latex() -> bool:
    preference = _parse_use_latex_env()
    if preference is False:
        return False
    if preference is True:
        if _latex_usable():
            return True
        logger.warning(
            "%s=true but LaTeX is not usable; continuing without usetex",
            ENV_USE_LATEX,
        )
        return False
    return _latex_usable()


def _apply_custom_font(font_path: str) -> None:
    from matplotlib import font_manager

    path = Path(font_path).expanduser()
    if not path.is_file():
        logger.warning("%s is not a file: %s", ENV_FONT_PATH, path)
        return
    resolved = str(path.resolve())
    font_manager.fontManager.addfont(resolved)
    name = font_manager.FontProperties(fname=resolved).get_name()
    import matplotlib as mpl

    mpl.rcParams["font.family"] = name
    sans = mpl.rcParams.get("font.sans-serif", [])
    if isinstance(sans, str):
        sans = [sans]
    mpl.rcParams["font.sans-serif"] = [name, *[f for f in sans if f != name]]
    logger.info("Using plot font %r from %s", name, resolved)


def configure_matplotlib_style(*, force: bool = False) -> None:
    """Apply GRADIEND matplotlib defaults once per process.

    Args:
        force: Re-apply style even when it has already been configured.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    import matplotlib as mpl

    mpl.rcParams["text.usetex"] = bool(_resolve_use_latex())

    font_path = os.environ.get(ENV_FONT_PATH)
    if font_path and str(font_path).strip():
        _apply_custom_font(str(font_path).strip())

    _CONFIGURED = True


def disable_usetex_for_axis_text(ax: Any = None) -> None:
    """Render tick/group labels literally when global usetex is enabled.

    Args:
        ax: Matplotlib axis whose tick/text labels should bypass usetex.
    """
    import matplotlib as mpl

    if not mpl.rcParams.get("text.usetex"):
        return
    if ax is None:
        return
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_usetex(False)
    for text in ax.texts:
        text.set_usetex(False)
    title = ax.get_title()
    if title:
        ax.set_title(title, usetex=False)


def reset_matplotlib_style_config() -> None:
    """Reset one-time configuration (for tests)."""
    global _CONFIGURED
    _CONFIGURED = False
