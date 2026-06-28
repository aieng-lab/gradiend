"""Ensure Markdown lists in docstrings are preceded by blank lines.

mkdocstrings renders docstring bodies as Markdown. Without a blank line before a
bullet/numbered list, CommonMark keeps list markers inside a single paragraph.

Usage:
    python scripts/fix_docstring_markdown_lists.py --check
    python scripts/fix_docstring_markdown_lists.py --fix
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATHS = (ROOT / "gradiend",)

_LIST_START = frozenset({"-", "*", "+"})
_GOOGLE_SECTIONS = frozenset({
    "args", "arguments", "params", "parameters",
    "returns", "return", "yields", "yield",
    "raises", "raise", "except", "exceptions",
    "warns", "warnings", "attributes", "attribute",
    "notes", "note", "examples", "example",
    "see also", "references", "references",
})


def _is_list_start(stripped: str) -> bool:
    if not stripped:
        return False
    if stripped[0] in _LIST_START:
        return len(stripped) > 1 and stripped[1] == " "
    if stripped[0].isdigit():
        dot = stripped.find(".")
        return dot > 0 and dot + 1 < len(stripped) and stripped[dot + 1] == " "
    return False


def _is_google_section_header(stripped: str) -> bool:
    if stripped.endswith(":") and ":" not in stripped[:-1]:
        return stripped[:-1].strip().lower() in _GOOGLE_SECTIONS
    return False


def _leading_ws(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _last_nonblank_index(lines: list[str]) -> int | None:
    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip():
            return idx
    return None


def _has_blank_since(lines: list[str], since_idx: int) -> bool:
    for line in lines[since_idx + 1 :]:
        if not line.strip():
            return True
    return False


def fix_docstring_content_lines(content_lines: list[str]) -> list[str]:
    """Return content lines with blank lines inserted around Markdown lists."""
    out: list[str] = []
    in_fence = False

    for line in content_lines:
        stripped = line.lstrip()

        if stripped.startswith("```"):
            in_fence = not in_fence

        if not in_fence and _is_list_start(stripped):
            prev_idx = _last_nonblank_index(out)
            if prev_idx is not None:
                prev_stripped = out[prev_idx].lstrip()
                if (
                    not _is_list_start(prev_stripped)
                    and not _is_google_section_header(prev_stripped)
                    and not _has_blank_since(out, prev_idx)
                ):
                    indent = _leading_ws(line) or _leading_ws(out[prev_idx])
                    out.append(f"{indent.rstrip()}\n" if indent.strip() else "\n")

        out.append(line)

        if not in_fence and stripped and not _is_list_start(stripped):
            prev_idx = _last_nonblank_index(out[:-1])
            if prev_idx is not None and _is_list_start(out[prev_idx].lstrip()):
                if not _has_blank_since(out[:-1], prev_idx):
                    indent = _leading_ws(line) or _leading_ws(out[prev_idx])
                    out.insert(len(out) - 1, f"{indent.rstrip()}\n" if indent.strip() else "\n")

    return out


def _docstring_expr_nodes(tree: ast.AST) -> list[ast.Expr]:
    nodes: list[ast.Expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr):
            continue
        value = first.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            nodes.append(first)
    return nodes


def _content_line_range(lines: list[str], expr: ast.Expr) -> tuple[int, int] | None:
    """Return [start, end) line indices for docstring interior lines."""
    start = expr.lineno - 1
    end = (expr.end_lineno or expr.lineno) - 1
    if start == end:
        return None
    if start + 1 >= end:
        return None
    return start + 1, end


def fix_file_text(source: str) -> tuple[str, list[str]]:
    """Return updated source and human-readable change descriptions."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, []

    lines = source.splitlines(keepends=True)
    changes: list[str] = []
    expr_nodes = sorted(
        _docstring_expr_nodes(tree),
        key=lambda node: node.lineno,
        reverse=True,
    )

    for expr in expr_nodes:
        span = _content_line_range(lines, expr)
        if span is None:
            continue
        start, end = span
        original_block = lines[start:end]
        fixed_block = fix_docstring_content_lines(original_block)
        if fixed_block == original_block:
            continue
        changes.append(f"line {start + 1}: insert blank line(s) before Markdown list")
        lines[start:end] = fixed_block

    return "".join(lines), changes


def iter_python_files(paths: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 1 if fixes are needed")
    mode.add_argument("--fix", action="store_true", help="rewrite files in place")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=DEFAULT_PATHS,
        help="files or directories (default: gradiend/)",
    )
    args = parser.parse_args(argv)

    files = iter_python_files(tuple(args.paths))
    violations: list[tuple[Path, list[str]]] = []

    for path in files:
        original = path.read_text(encoding="utf-8")
        updated, changes = fix_file_text(original)
        if changes:
            violations.append((path, changes))
            if args.fix:
                path.write_text(updated, encoding="utf-8", newline="\n")

    if not violations:
        print("All docstring Markdown lists are correctly spaced.")
        return 0

    rel_root = ROOT
    for path, changes in violations:
        rel = path.relative_to(rel_root)
        print(f"{rel}")
        for change in changes:
            print(f"  {change}")

    if args.check:
        print(f"\n{len(violations)} file(s) need blank lines before/after Markdown lists.")
        print("Run: python scripts/fix_docstring_markdown_lists.py --fix")
        return 1

    print(f"\nFixed {len(violations)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
