"""No copy-pasted code: a function or class whose body (docstring aside) is identical to one in
another module of `raster_parser` belongs in one shared place (`commons/`, see
`docs/ARCHITECTURE.md`). Vendored upstream code (`_upstream/`) is exempt, and so are bodies that
are the same text but bind different module-level names (listed in `ALLOWED`)."""

from __future__ import annotations

import ast
import hashlib
from collections import defaultdict
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "raster_parser"
MIN_LINES = 4

# Same text, different meaning: each copy calls its own module's helpers / model classes.
ALLOWED = {
    "render_debug",  # each backend's own _compose_raster_layers / _page_layers / _PageState
    "_compose_raster_layers",  # each backend's own _RasterItem layout
    "_make_predict",
    "load_weights",
    "save_weights",
    "warmup",  # each backend's own model / loader
    "default_weights_path",  # each backend's own weight file name and env var
}


def _body_key(node: ast.AST) -> str:
    node = ast.parse(ast.unparse(node)).body[0]  # a copy we can strip
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        node.body = body[1:] or [ast.Pass()]
    node.decorator_list = []
    return hashlib.sha1(ast.unparse(node).encode()).hexdigest()


def find_duplicates() -> list[list[str]]:
    seen: dict[str, list[str]] = defaultdict(list)
    for path in sorted(PKG.rglob("*.py")):
        if "_upstream" in path.parts:
            continue
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if (
                isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name not in ALLOWED
                and node.end_lineno - node.lineno + 1 >= MIN_LINES
            ):
                seen[_body_key(node)].append(f"{path.relative_to(PKG).as_posix()}::{node.name}")
    return [sorted(v) for v in seen.values() if len({x.split("::")[0] for x in v}) > 1]


def test_no_function_is_copied_between_modules():
    assert find_duplicates() == []
