"""OldVectorClassification is a frozen, fully isolated snapshot: its only
`rastervec.*` imports outside its own folder may be `commons.models` (the
P3 interface) and the three debug-layer drawers in `commons.renderer`."""
from __future__ import annotations

import ast
from pathlib import Path

import rastervec.P3_Vector_Parsing.OldVectorClassification as old_pkg

OLD_DIR = Path(old_pkg.__file__).parent
OWN_PREFIX = "rastervec.P3_Vector_Parsing.OldVectorClassification"
ALLOWED_MODULES = {"rastervec.commons.models"}
ALLOWED_NAMES = {
    "rastervec.commons.renderer": {"render_boxes_pdf", "render_text_pdf", "render_vectors_pdf"},
}


def _violations(path: Path) -> list[str]:
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("rastervec") and not alias.name.startswith(OWN_PREFIX):
                    out.append(f"{path.name}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("rastervec"):
            mod = node.module
            if mod.startswith(OWN_PREFIX) or mod in ALLOWED_MODULES:
                continue
            names = {a.name for a in node.names}
            if not names <= ALLOWED_NAMES.get(mod, set()):
                out.append(f"{path.name}: from {mod} import {sorted(names)}")
    return out


def test_old_backend_imports_nothing_outside_its_allowlist():
    files = sorted(OLD_DIR.rglob("*.py"))
    assert files
    bad = [v for f in files for v in _violations(f)]
    assert bad == []


def test_every_old_module_carries_the_frozen_header():
    for f in sorted(OLD_DIR.rglob("*.py")):
        assert f.read_text(encoding="utf-8").startswith("# FROZEN"), f
