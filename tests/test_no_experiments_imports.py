from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOTS = ("experiments", "tests", "gradiend")


def _python_files():
    for root_name in SEARCH_ROOTS:
        root = ROOT / root_name
        if root.exists():
            yield from root.rglob("*.py")


class _ExperimentsImportVisitor(ast.NodeVisitor):
    """Flag module-level experiments imports (lazy imports inside functions are OK)."""

    def __init__(self, rel_path: str):
        self.rel_path = rel_path
        self.offenders: list[str] = []
        self._in_function = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._in_function += 1
        self.generic_visit(node)
        self._in_function -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._in_function += 1
        self.generic_visit(node)
        self._in_function -= 1

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._in_function:
            return
        if node.module == "experiments" or (node.module and node.module.startswith("experiments.")):
            self.offenders.append(f"{self.rel_path}:{node.lineno}")

    def visit_Import(self, node: ast.Import) -> None:
        if self._in_function:
            return
        for alias in node.names:
            if alias.name == "experiments" or alias.name.startswith("experiments."):
                self.offenders.append(f"{self.rel_path}:{node.lineno}")

    def visit_Call(self, node: ast.Call) -> None:
        if self._in_function:
            return
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                normalized = arg.value.replace("\\", "/")
                if normalized == "experiments" or normalized.startswith("experiments/"):
                    self.offenders.append(f"{self.rel_path}:{node.lineno}")
        self.generic_visit(node)


def test_no_imports_from_experiments_namespace():
    offenders: list[str] = []
    for path in _python_files():
        if path == Path(__file__).resolve():
            continue
        if path.relative_to(ROOT).parts[0] == "experiments":
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        rel = str(path.relative_to(ROOT))
        visitor = _ExperimentsImportVisitor(rel)
        visitor.visit(tree)
        offenders.extend(visitor.offenders)

    assert offenders == []
