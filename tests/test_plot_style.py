"""Tests for matplotlib plot style configuration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gradiend.visualizer.plot_style import (
    ENV_FONT_PATH,
    ENV_USE_LATEX,
    configure_matplotlib_style,
    reset_matplotlib_style_config,
)


@pytest.fixture(autouse=True)
def _reset_plot_style():
    reset_matplotlib_style_config()
    yield
    reset_matplotlib_style_config()


@pytest.fixture
def mpl_rc():
    pytest.importorskip("matplotlib")
    import matplotlib as mpl

    original = mpl.rcParams.copy()
    yield mpl.rcParams
    mpl.rcParams.update(original)


class TestPlotStyle:
    def test_auto_disables_usetex_when_latex_missing(self, mpl_rc, monkeypatch):
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        monkeypatch.delenv(ENV_FONT_PATH, raising=False)
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=False):
            configure_matplotlib_style(force=True)
        assert mpl_rc["text.usetex"] is False

    def test_auto_enables_usetex_when_latex_available(self, mpl_rc, monkeypatch):
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            configure_matplotlib_style(force=True)
        assert mpl_rc["text.usetex"] is True

    def test_force_usetex_off(self, mpl_rc, monkeypatch):
        monkeypatch.setenv(ENV_USE_LATEX, "0")
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            configure_matplotlib_style(force=True)
        assert mpl_rc["text.usetex"] is False

    def test_force_usetex_on_when_available(self, mpl_rc, monkeypatch):
        monkeypatch.setenv(ENV_USE_LATEX, "1")
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=True):
            configure_matplotlib_style(force=True)
        assert mpl_rc["text.usetex"] is True

    def test_force_usetex_on_when_unavailable_falls_back(self, mpl_rc, monkeypatch):
        monkeypatch.setenv(ENV_USE_LATEX, "true")
        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=False):
            configure_matplotlib_style(force=True)
        assert mpl_rc["text.usetex"] is False

    def test_custom_font_from_env_path(self, mpl_rc, monkeypatch, tmp_path):
        font_path = tmp_path / "DemoFont.ttf"
        font_path.write_bytes(b"font")

        monkeypatch.setenv(ENV_FONT_PATH, str(font_path))
        mock_fp = MagicMock()
        mock_fp.get_name.return_value = "Demo Font"

        with patch("gradiend.visualizer.plot_style._latex_usable", return_value=False), patch(
            "matplotlib.font_manager.fontManager"
        ) as mock_manager, patch(
            "matplotlib.font_manager.FontProperties",
            return_value=mock_fp,
        ):
            configure_matplotlib_style(force=True)

        mock_manager.addfont.assert_called_once_with(str(font_path.resolve()))
        assert mpl_rc["font.family"] in ("Demo Font", ["Demo Font"])
        assert mpl_rc["font.sans-serif"][0] == "Demo Font"

    def test_require_matplotlib_triggers_style_config(self, monkeypatch):
        pytest.importorskip("matplotlib")
        monkeypatch.delenv(ENV_USE_LATEX, raising=False)
        with patch("gradiend.visualizer.plot_optional.configure_matplotlib_style") as mock_configure:
            from gradiend.visualizer.plot_optional import _require_matplotlib

            _require_matplotlib()
        mock_configure.assert_called_once()
