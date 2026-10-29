"""`docs/ARCHITECTURE.md` keeps its shared-code map complete: every `commons/` and `core/`
module is listed, and none that is gone."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "raster_parser"


def _shared_code_map() -> str:
    doc = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    start = doc.index("## Shared code map")
    end = doc.find("\n## ", start + 1)
    return doc[start : end if end > 0 else None]


def test_every_shared_module_is_in_the_map():
    section = _shared_code_map()
    modules = [
        p.relative_to(PKG).as_posix()
        for d in ("commons", "core")
        for p in (PKG / d).rglob("*.py")
        if p.name != "__init__.py"
    ]
    missing = [m for m in sorted(modules) if f"`{m}`" not in section]
    assert missing == []


def test_map_lists_no_module_that_is_gone():
    import re

    listed = re.findall(r"`((?:commons|core)/[\w/]+\.py)`", _shared_code_map())
    assert [m for m in listed if not (PKG / m).is_file()] == []
