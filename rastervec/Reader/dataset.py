"""Dataset collection for the benchmarking suite: recursively walk a
directory tree for PDFs and sidecar label JSONs (the `manual_label.py` /
`auto_label.py` `LabelSet` format, see `Evaluation/Labelling/
label_schema.py`) and pair them into one flat list of `(pdf, page)` work
items, each carrying any human-entered `LabelEntry`s for that page.

Imported by `notebooks/benchmark_vector_classification.ipynb` -- it
replaces that notebook's old two-mode (`PDF_FOLDER` xor `LABELS_JSON`)
collection code with a single "point at a tree of mixed .pdf + .json"
mode. Pure filesystem + JSON parsing; the only `fitz` use is a page-count
probe for a PDF that has no sidecar label file (so its first N pages can
be auto-labelled).

A discovered `.json` that does not parse as a `LabelSet` is skipped (a
tree can hold unrelated JSON) with a warning. A label file's `pdf_path`
is resolved against, in order: an absolute path that exists, a path
relative to the JSON file's own directory, then a unique basename match
among the discovered PDFs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz

from rastervec.Evaluation.Labelling.label_schema import LabelEntry, LabelSet, load_labels
from rastervec.logging_setup import get_logger

_LOG = get_logger("reader.dataset")


@dataclass(frozen=True)
class DatasetPage:
    """One `(pdf, page)` benchmark work item. `manual_entries` is the
    `source="manual"` `LabelEntry`s a sidecar label file supplied for this
    exact page (empty when the PDF had no sidecar, or the sidecar had no
    manual entries for this page). Auto labels are NOT included here -- the
    benchmark derives those itself per page via `auto_label_pdf`, so ground
    truth stays independent of any pipeline run."""

    pdf_path: str
    page_index: int
    manual_entries: tuple[LabelEntry, ...] = ()


def find_pdfs(root: Path) -> list[Path]:
    """Every `*.pdf` under `root`, recursively, sorted."""
    return sorted(root.rglob("*.pdf"))


def find_label_sets(root: Path) -> list[tuple[Path, LabelSet]]:
    """Every `*.json` under `root` that parses as a `LabelSet`, as
    `(json_path, label_set)` pairs sorted by path. A `.json` that does not
    validate as a `LabelSet` is skipped with a warning."""
    out: list[tuple[Path, LabelSet]] = []
    for json_path in sorted(root.rglob("*.json")):
        try:
            out.append((json_path, load_labels(str(json_path))))
        except Exception as exc:  # noqa: BLE001 -- unrelated JSON in the tree is fine
            _LOG.warning("skipping %s: not a LabelSet (%s)", json_path, exc)
    return out


def resolve_pdf_path(
    raw: str, json_dir: Path, known_pdfs: list[Path],
) -> Path | None:
    """Resolve a label file's `pdf_path` string to a real PDF path:
    an absolute path that exists, else `json_dir / raw` if it exists, else
    a unique basename match among `known_pdfs`. `None` (logged) if none of
    those land."""
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p.resolve()

    relative = (json_dir / p).resolve()
    if relative.exists():
        return relative

    by_name = [pdf for pdf in known_pdfs if pdf.name == p.name]
    if len(by_name) == 1:
        return by_name[0].resolve()
    if len(by_name) > 1:
        _LOG.warning(
            "label pdf_path %r matches %d discovered PDFs by name; skipping",
            raw, len(by_name),
        )
        return None

    _LOG.warning("could not resolve label pdf_path %r (json dir: %s)", raw, json_dir)
    return None


def _page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    try:
        return doc.page_count
    finally:
        doc.close()


def collect_dataset(
    root: Path | str,
    *,
    pages_per_pdf: int | None = None,
    include_unlabelled: bool = True,
) -> list[DatasetPage]:
    """Recursively collect every PDF and every label file under `root` into
    one flat, sorted, de-duplicated list of `DatasetPage`s.

    - A PDF that a label file names: one `DatasetPage` per page that file
      references (any source), carrying that page's `source="manual"`
      entries.
    - A PDF with no label file (and `include_unlabelled`): one
      `DatasetPage` per page in `range(min(pages_per_pdf or n, n))` with no
      manual entries -- the benchmark auto-labels these.
    - A label file whose `pdf_path` resolves outside `root` is still
      included (its resolved path is added to the working set).
    """
    root = Path(root)
    known_pdfs = find_pdfs(root)
    known_by_path: dict[str, Path] = {str(p.resolve()): p for p in known_pdfs}

    # (pdf_path_str, page_index) -> list[LabelEntry] (manual only)
    manual_by_page: dict[tuple[str, int], list[LabelEntry]] = {}
    # pdf_path_str -> set of page indices a label file explicitly names
    labelled_pages: dict[str, set[int]] = {}

    for json_path, label_set in find_label_sets(root):
        resolved = resolve_pdf_path(label_set.pdf_path, json_path.parent, known_pdfs)
        if resolved is None:
            continue
        pdf_key = str(resolved)
        known_by_path.setdefault(pdf_key, resolved)

        for entry in label_set.entries:
            labelled_pages.setdefault(pdf_key, set()).add(entry.page_index)
            if entry.source == "manual":
                manual_by_page.setdefault((pdf_key, entry.page_index), []).append(entry)

    pages: list[DatasetPage] = []
    for pdf_key in sorted(known_by_path):
        if pdf_key in labelled_pages:
            for page_index in sorted(labelled_pages[pdf_key]):
                pages.append(
                    DatasetPage(
                        pdf_path=pdf_key,
                        page_index=page_index,
                        manual_entries=tuple(manual_by_page.get((pdf_key, page_index), [])),
                    )
                )
        elif include_unlabelled:
            n_pages = _page_count(Path(pdf_key))
            limit = n_pages if pages_per_pdf is None else min(pages_per_pdf, n_pages)
            for page_index in range(limit):
                pages.append(DatasetPage(pdf_path=pdf_key, page_index=page_index))

    _LOG.info(
        "collect_dataset(%s): %d (pdf, page) item(s), %d manual label(s)",
        root, len(pages), sum(len(v) for v in manual_by_page.values()),
    )
    return pages
