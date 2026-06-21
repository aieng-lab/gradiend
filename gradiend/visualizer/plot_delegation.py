"""Shared helpers for trainer/evaluator plot API delegation."""

from __future__ import annotations


def see_implementation(qualified_name: str) -> str:
    """Docstring suffix linking to the canonical visualizer implementation.

    Args:
        qualified_name: Fully qualified function name to reference.
    """
    return f"\n\nSee :func:`{qualified_name}` for the full parameter list and defaults.\n"
