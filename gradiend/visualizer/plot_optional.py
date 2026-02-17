"""
Optional plot dependencies: matplotlib and seaborn.

Use _require_matplotlib() or _require_seaborn() at the start of plotting functions.
On failure, raises ImportError with install instructions (e.g. pip install gradiend[recommended]).
"""

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
        return plt
    except ImportError as e:
        raise ImportError(_MSG_MATPLOTLIB) from e


def _require_seaborn():
    """Import seaborn; on failure raise an error with install instructions."""
    try:
        import seaborn as sns
        return sns
    except ImportError as e:
        raise ImportError(_MSG_SEABORN) from e
