"""Every public module, class, function and method of the application must be documented."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "feesbot"
MIN_LENGTH = 15  # "Run it." is not documentation


def public_items(tree: ast.Module, prefix: str = ""):
    """Yield (qualified name, node) for every public class, function and method."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            yield prefix + node.name, node
            if isinstance(node, ast.ClassDef):
                yield from public_items(node, prefix + node.name + ".")


def all_items():
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        yield pytest.param(path.name, "(module)", tree, id=f"{path.name}::(module)")
        for name, node in public_items(tree):
            yield pytest.param(path.name, name, node, id=f"{path.name}::{name}")


@pytest.mark.parametrize("filename, name, node", list(all_items()))
def test_public_item_has_a_real_docstring(filename, name, node):
    doc = ast.get_docstring(node)
    assert doc, f"{filename}: {name} has no docstring"
    assert len(doc.strip()) >= MIN_LENGTH, f"{filename}: {name} has a docstring too short to help: {doc!r}"


def test_the_audit_really_finds_the_package():
    assert len(list(all_items())) > 80  # guards against the glob silently matching nothing
