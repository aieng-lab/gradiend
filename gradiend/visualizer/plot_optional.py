"""
Optional plot dependencies: matplotlib and seaborn.

Use _require_matplotlib() or _require_seaborn() at the start of plotting functions.
On failure, raises ImportError with install instructions (e.g. pip install gradiend[recommended]).

Plot styling (LaTeX, custom fonts) is configured on first import via
:func:`gradiend.visualizer.plot_style.configure_matplotlib_style` using
``GRADIEND_PLOT_USE_LATEX`` and ``GRADIEND_PLOT_FONT_PATH``.
"""

from gradiend.visualizer.plot_style import configure_matplotlib_style

_MSG_MATPLOTLIB = (
    "Plotting requires matplotlib. "
    "Install it with: pip install matplotlib "
    "Or install all recommended packages: pip install gradiend[recommended]"
)
_MSG_SEABORN = (
    "This plot requires seaborn (and matplotlib). "
    "Install with: pip install seaborn "
    "Or install all recommended packages: pip install gradiend[recommended]"
)


def _require_matplotlib():
    """Import matplotlib.pyplot; on failure raise an error with install instructions."""
    try:
        import matplotlib.pyplot as plt

        configure_matplotlib_style()
        return plt
    except ImportError:
        raise ImportError(_MSG_MATPLOTLIB) from None


def _require_seaborn():
    """Import seaborn; on failure raise an error with install instructions."""
    try:
        configure_matplotlib_style()
        import seaborn as sns

        return sns
    except ImportError:
        raise ImportError(_MSG_SEABORN) from None
