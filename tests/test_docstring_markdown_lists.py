"""Tests for scripts/fix_docstring_markdown_lists.py."""
from __future__ import annotations

from scripts.fix_docstring_markdown_lists import fix_docstring_content_lines, fix_file_text


def test_inserts_blank_line_before_bullet_list():
    lines = [
        "    This adapter:\n",
        "    - item one\n",
        "    - item two\n",
    ]
    fixed = fix_docstring_content_lines(lines)
    assert fixed[0] == "    This adapter:\n"
    assert fixed[1].strip() == ""
    assert fixed[2] == "    - item one\n"


def test_inserts_blank_line_after_bullet_list():
    lines = [
        "    - item one\n",
        "    Next paragraph.\n",
    ]
    fixed = fix_docstring_content_lines(lines)
    assert fixed[0] == "    - item one\n"
    assert fixed[1].strip() == ""
    assert fixed[2] == "    Next paragraph.\n"


def test_skips_lists_inside_fenced_code_block():
    lines = [
        "    Example:\n",
        "    ```python\n",
        "    x = [\n",
        "    - 1,\n",
        "    ]\n",
        "    ```\n",
    ]
    assert fix_docstring_content_lines(lines) == lines


def test_fix_file_text_updates_class_docstring():
    source = '''class Model:
    """
    Summary line.

    Features:
    - forward
    - backward
    """
'''
    updated, changes = fix_file_text(source)
    assert changes
    assert "Features:\n\n    - forward" in updated
