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


def test_no_imports_from_experiments_namespace():
    offenders: list[str] = []
    for path in _python_files():
        if path == Path(__file__).resolve():
            continue
        if path.relative_to(ROOT).parts[0] == "experiments":
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "experiments":
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("experiments."):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "experiments" or alias.name.startswith("experiments."):
                        offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.Call):
                for arg in list(node.args) + [kw.value for kw in node.keywords]:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        normalized = arg.value.replace("\\", "/")
                        if normalized == "experiments" or normalized.startswith("experiments/"):
                            offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")

    assert offenders == []
