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
from typing import Any, Dict, Optional

from gradiend.util.logging import get_logger

logger = get_logger(__name__)

ENV_USE_LATEX = "GRADIEND_PLOT_USE_LATEX"
ENV_FONT_PATH = "GRADIEND_PLOT_FONT_PATH"

_LATEX_COMMANDS = ("latex", "pdflatex", "xelatex", "lualatex")
_CONFIGURED = False
_LATEX_USABLE: Optional[bool] = None


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


def _latex_command_paths() -> Dict[str, Optional[str]]:
    return {command: shutil.which(command) for command in _LATEX_COMMANDS}


def _latex_usable() -> bool:
    global _LATEX_USABLE
    if _LATEX_USABLE is not None:
        return _LATEX_USABLE
    if not _latex_on_path():
        _LATEX_USABLE = False
        return False
    try:
        import tempfile

        import matplotlib as mpl
        from matplotlib import pyplot as plt
        from matplotlib.texmanager import TexManager

        TexManager()

        # TexManager() alone is insufficient: minimal TeX installs (e.g. cm-super-minimal
        # in Apptainer) can pass init yet fail on PDF save with missing PostScript fonts.
        with mpl.rc_context({"text.usetex": True}):
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.set_title(r"Test $\longleftrightarrow$")
            with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
                fig.savefig(tmp.name, bbox_inches="tight")
            plt.close(fig)
        _LATEX_USABLE = True
    except Exception as exc:
        logger.debug("LaTeX found on PATH but matplotlib usetex smoke test failed: %s", exc)
        _LATEX_USABLE = False
    return _LATEX_USABLE


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


def _check_configured_font(font_path: Optional[str]) -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "env_value": font_path,
        "path": None,
        "exists": False,
        "is_file": False,
        "valid_extension": None,
        "usable": font_path is None or not str(font_path).strip(),
        "font_name": None,
        "error": None,
    }
    if font_path is None or not str(font_path).strip():
        return status

    path = Path(str(font_path).strip()).expanduser()
    status["path"] = str(path)
    status["exists"] = path.exists()
    status["is_file"] = path.is_file()
    status["valid_extension"] = path.suffix.lower() in {".ttf", ".otf"}
    if not path.is_file():
        status["usable"] = False
        status["error"] = "Configured font path is not a file."
        return status
    if not status["valid_extension"]:
        status["usable"] = False
        status["error"] = "Configured font path should point to a .ttf or .otf file."
        return status

    try:
        from matplotlib import font_manager

        status["font_name"] = font_manager.FontProperties(fname=str(path.resolve())).get_name()
        status["path"] = str(path.resolve())
        status["usable"] = True
    except Exception as exc:
        status["usable"] = False
        status["error"] = str(exc)
    return status


def _font_family_text(font_family: Any) -> str:
    if isinstance(font_family, (list, tuple)):
        return ", ".join(str(value) for value in font_family)
    return str(font_family)


def _check_current_font() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "family": None,
        "available": False,
        "resolved_path": None,
        "error": None,
    }
    try:
        import matplotlib as mpl
        from matplotlib import font_manager

        family = mpl.rcParams.get("font.family")
        status["family"] = family
        font_path = font_manager.findfont(
            font_manager.FontProperties(family=family),
            fallback_to_default=True,
        )
        status["resolved_path"] = font_path
        status["available"] = bool(font_path)
    except Exception as exc:
        status["available"] = False
        status["error"] = str(exc)
    return status


def _format_plot_environment_status(status: Dict[str, Any]) -> str:
    latex = status["latex"]
    font = status["font"]
    mpl = status["matplotlib"]
    command_names = [name for name, path in latex["commands"].items() if path]
    command_text = ", ".join(command_names) if command_names else "none"
    font_env_text = font["env_value"] if font.get("env_value") else "unset"
    configured_font_text = (
        f"configured={font['font_name']} ({font['path']})"
        if font.get("font_name")
        else "configured=default matplotlib font"
        if font.get("usable")
        else f"configured=unusable ({font.get('error')})"
    )
    current_font = font["current"]
    current_font_text = (
        f"current={_font_family_text(current_font['family'])}, "
        f"current_available={current_font['available']}"
    )
    if current_font.get("resolved_path"):
        current_font_text += f", resolved={current_font['resolved_path']}"
    lines = [
        f"GRADIEND plot environment: {'OK' if status['ok'] else 'ISSUES'}",
        (
            "  LaTeX: "
            f"preference={latex['preference']}, on_path={latex['on_path']}, "
            f"usable={latex['usable']}, resolved_text_usetex={latex['resolved_text_usetex']}, "
            f"commands={command_text}"
        ),
        f"  Font: usable={font['usable']}, {font['env_var']}={font_env_text}, "
        f"{configured_font_text}, {current_font_text}",
        (
            "  Matplotlib: "
            f"available={mpl['available']}, version={mpl['version']}, "
            f"style_configured={mpl['style_configured']}, "
            f"current_text_usetex={mpl['current_text_usetex']}"
        ),
    ]
    if status["warnings"]:
        lines.append("  Warnings:")
        lines.extend(f"    - {warning}" for warning in status["warnings"])
    if status["info"]:
        lines.append("  Info:")
        lines.extend(f"    - {message}" for message in status["info"])
    return "\n".join(lines)


def check_plot_environment(*, print_status: bool = True, apply_style: bool = True) -> Dict[str, Any]:
    """Print and return GRADIEND plot rendering environment status.

    The returned dictionary reports LaTeX availability, the effective GRADIEND
    LaTeX preference, matplotlib's current ``rcParams`` font settings, and whether
    ``GRADIEND_PLOT_FONT_PATH`` points to a usable ``.ttf``/``.otf`` font file.

    Args:
        print_status: If True, print a compact human-readable status summary.
        apply_style: If True, apply the resolved GRADIEND ``text.usetex`` and
            configured font settings before reporting the active matplotlib state.
    """
    global _CONFIGURED
    raw_latex_preference = os.environ.get(ENV_USE_LATEX)
    raw_font_path = os.environ.get(ENV_FONT_PATH)
    latex_preference_error = None
    try:
        latex_preference = _parse_use_latex_env()
    except ValueError as exc:
        latex_preference = None
        latex_preference_error = str(exc)

    command_paths = _latex_command_paths()
    latex_on_path = any(path for path in command_paths.values())
    latex_usable = _latex_usable() if latex_on_path else False
    if latex_preference is False:
        resolved_usetex = False
    elif latex_preference is True:
        resolved_usetex = latex_usable
    else:
        resolved_usetex = latex_usable

    font_status = _check_configured_font(raw_font_path)

    matplotlib_status: Dict[str, Any] = {
        "available": False,
        "version": None,
        "current_text_usetex": None,
        "style_configured": _CONFIGURED,
        "current_font_family": None,
        "current_font_sans_serif": None,
        "error": None,
    }
    def _refresh_matplotlib_status() -> None:
        try:
            import matplotlib as mpl

            matplotlib_status.update(
                {
                    "available": True,
                    "version": getattr(mpl, "__version__", None),
                    "current_text_usetex": bool(mpl.rcParams.get("text.usetex")),
                    "style_configured": _CONFIGURED,
                    "current_font_family": mpl.rcParams.get("font.family"),
                    "current_font_sans_serif": mpl.rcParams.get("font.sans-serif"),
                    "error": None,
                }
            )
        except Exception as exc:
            matplotlib_status["error"] = str(exc)

    _refresh_matplotlib_status()

    if apply_style and matplotlib_status["available"] and latex_preference_error is None:
        try:
            import matplotlib as mpl

            mpl.rcParams["text.usetex"] = bool(resolved_usetex)
            if raw_font_path and str(raw_font_path).strip() and font_status["usable"]:
                _apply_custom_font(str(raw_font_path).strip())
            _CONFIGURED = True
            _refresh_matplotlib_status()
        except Exception as exc:
            font_status["usable"] = False
            font_status["error"] = str(exc)
            _refresh_matplotlib_status()

    current_font_status = _check_current_font() if matplotlib_status["available"] else {
        "family": None,
        "available": False,
        "resolved_path": None,
        "error": "matplotlib is not available.",
    }
    font_status["current"] = current_font_status
    forced_latex_but_unusable = latex_preference is True and not latex_usable
    active_usetex = matplotlib_status["current_text_usetex"]
    usetex_mismatch = matplotlib_status["available"] and active_usetex != resolved_usetex
    active_family = current_font_status.get("family")
    configured_font_name = font_status.get("font_name")
    font_style_not_applied = bool(
        font_status["usable"]
        and configured_font_name
        and not apply_style
        and not matplotlib_status["style_configured"]
    )
    expected_font_missing = bool(
        font_status["usable"]
        and configured_font_name
        and matplotlib_status["style_configured"]
        and configured_font_name not in (
            active_family if isinstance(active_family, (list, tuple)) else [active_family]
        )
    )
    font_unavailable = matplotlib_status["available"] and not current_font_status["available"]
    font_issue = font_unavailable or font_style_not_applied or expected_font_missing
    ok = (
        latex_preference_error is None
        and matplotlib_status["available"]
        and font_status["usable"]
        and not font_unavailable
        and not expected_font_missing
        and not forced_latex_but_unusable
        and not (usetex_mismatch and apply_style)
    )
    warnings = []
    info = []
    if latex_preference_error is not None:
        warnings.append(latex_preference_error)
    if forced_latex_but_unusable:
        warnings.append(f"{ENV_USE_LATEX}=true but LaTeX is not usable by matplotlib.")
    if not font_status["usable"] and raw_font_path and str(raw_font_path).strip():
        warnings.append(str(font_status["error"]))
    if font_unavailable:
        warnings.append(
            "Matplotlib cannot resolve the current font.family "
            f"({_font_family_text(current_font_status['family'])}): {current_font_status['error']}"
        )
    if font_style_not_applied:
        info.append(
            "GRADIEND style has not been applied in this process yet; "
            f"plots will register {ENV_FONT_PATH} as {configured_font_name!r} when rendering."
        )
    if expected_font_missing:
        warnings.append(
            f"{ENV_FONT_PATH} resolved to {configured_font_name!r}, but matplotlib is currently "
            f"using {_font_family_text(active_family)!r}."
        )
    if not matplotlib_status["available"]:
        warnings.append(f"matplotlib is not available: {matplotlib_status['error']}")
    if usetex_mismatch:
        if not apply_style and not matplotlib_status["style_configured"]:
            info.append(
                "GRADIEND style has not been applied in this process yet; "
                f"plots will set text.usetex={resolved_usetex} when rendering."
            )
        else:
            warnings.append(
                "Matplotlib text.usetex does not match the GRADIEND LaTeX preference "
                f"(current={active_usetex}, resolved={resolved_usetex})."
            )

    status = {
        "ok": ok,
        "warnings": warnings,
        "info": info,
        "latex": {
            "env_var": ENV_USE_LATEX,
            "env_value": raw_latex_preference,
            "preference": (
                "force_on"
                if latex_preference is True
                else "force_off"
                if latex_preference is False
                else "auto"
            ),
            "preference_error": latex_preference_error,
            "commands": command_paths,
            "on_path": latex_on_path,
            "usable": latex_usable,
            "resolved_text_usetex": resolved_usetex,
        },
        "font": {
            "env_var": ENV_FONT_PATH,
            **font_status,
        },
        "matplotlib": matplotlib_status,
    }
    if print_status:
        print(_format_plot_environment_status(status))
    return status


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
