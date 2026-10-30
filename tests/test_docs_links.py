"""The docs form one connected, link-checked set: every relative Markdown link resolves, every
Markdown file is reachable from the root README, and the handover pieces exist."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
FENCE = re.compile(r"```.*?```", re.S)
# generated or foreign Markdown that is not part of the handover set
IGNORED_PARTS = {"_upstream", "outputs", "references", ".venv", "node_modules", "baseline_tools"}


def _tracked_markdown() -> list[Path]:
    return sorted(
        p
        for p in ROOT.rglob("*.md")
        if not IGNORED_PARTS & set(p.relative_to(ROOT).parts)
        and not any(part.startswith(".") for part in p.relative_to(ROOT).parts)
        and not p.relative_to(ROOT).parts[0].endswith("-claude")
    )


DOCS = _tracked_markdown()


def _targets(path: Path, pattern: re.Pattern = LINK) -> list[tuple[str, Path]]:
    text = FENCE.sub("", path.read_text(encoding="utf-8"))
    out = []
    for raw in pattern.findall(text):
        if re.match(r"^(https?:|mailto:|#)", raw):
            continue
        out.append((raw, (path.parent / raw.split("#")[0]).resolve()))
    return out


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc):
    broken = [raw for raw, target in _targets(doc) if not target.exists()]
    assert not broken, f"{doc.relative_to(ROOT)}: {broken}"


def test_root_readme_exists_and_links_the_doc_index():
    readme = ROOT / "README.md"
    assert readme.is_file()
    assert (ROOT / "docs" / "README.md").resolve() in {t for _, t in _targets(readme)}


def test_every_markdown_file_is_reachable_from_the_root_readme():
    seen, queue = set(), [ROOT / "README.md"]
    while queue:
        doc = queue.pop()
        if doc in seen or doc.suffix != ".md" or not doc.is_file():
            continue
        seen.add(doc)
        queue.extend(t for _, t in _targets(doc))
        if doc.name == "README.md":
            queue.extend(t for _, t in _targets(doc, IMAGE))
    unreachable = [str(p.relative_to(ROOT)) for p in DOCS if p.resolve() not in seen]
    assert not unreachable, f"not linked from README.md: {unreachable}"


@pytest.mark.parametrize("name", ["HANDOVER.md", "README.md"])
def test_handover_docs_exist(name):
    assert (ROOT / "docs" / name).is_file()


def test_example_images_list_names_a_command_and_a_target_for_each_image():
    text = (ROOT / "docs" / "images" / "EXAMPLES.md").read_text(encoding="utf-8")
    rows = [ln for ln in text.splitlines() if re.match(r"^\|\s*\d+\s*\|", ln)]
    assert len(rows) >= 10
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) >= 4 and all(cells), row


def test_no_stale_names_in_docs():
    stale = (
        "dto_reconstruct_pipeline",
        "type4_pipeline",
        "master_label.py",
        "raster_label.py",
        "/vector_label.py",
    )
    offenders = []
    for doc in DOCS:
        if doc.name in {"HANDOVER.md"}:  # the handover may name what was removed
            continue
        text = doc.read_text(encoding="utf-8")
        offenders += [f"{doc.relative_to(ROOT)}: {s}" for s in stale if s in text]
    assert not offenders, offenders
