"""
Build MkDocs documentation with griffe warnings suppressed.

Griffe emits warnings for missing type annotations in the source; these are
suppressed so that `mkdocs build --strict` can pass without requiring
full type hints across the codebase. Link and image warnings are still reported.

Usage: python scripts/build_docs.py build [--strict] [other mkdocs options]
"""
import logging
import runpy
import sys

# Suppress griffe type/annotation warnings before mkdocstrings (and thus griffe) is loaded
logging.getLogger("griffe").setLevel(logging.ERROR)

# Invoke mkdocs the same way as "python -m mkdocs"
sys.argv = ["mkdocs"] + (sys.argv[1:] or ["build"])
runpy.run_module("mkdocs", run_name="__main__")
