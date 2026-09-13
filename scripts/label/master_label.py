"""Master label workflow: runs `native_label` / `vector_label` / `raster_label`
end to end over one PDF, bundled into a single per-PDF folder --
`outputs/labels/<stem>_label/` -- instead of scattered ad-hoc files. Meant
for building a small, durable regression-testing label set (a handful of
PDFs, each labelled once and revisited over time).

    .venv/Scripts/python.exe scripts/label/master_label.py PATH/TO.pdf [--dpi 300]

Folder layout (all paths relative to `<stem>_label/`):

    original.pdf          -- verbatim copy of the source PDF
    vectorised.pdf         -- per-page `convert_page_text_only` renders, merged
    rasterised.pdf          -- per-page flattened-to-image renders, merged
    native_labels.json      -- native_label_pdf output (+ vector_signatures)
    vector_labels.json      -- VectorLabelApp's human-entered labels
    raster_labels.json      -- auto geometry (source="auto") + text re-synced
                                from vector_labels.json (source="raster",
                                label_id "vecsync:...") + hand-drawn entries
                                from RasterLabelApp
    manifest.json           -- per-page {"native_done", "rasterise_done",
                                "geometry_done"} flags driving the skip logic

Steps, in order:

1. **Native** (skippable per page via `native_done`): `native_label_pdf` +
   `convert_page_text_only` per not-yet-done page, assembled into
   `vectorised.pdf` (already-done pages are copied out of the existing file
   instead of recomputed); `attach_vector_signatures` per new page;
   `native_labels.json` merged (new entries only, keyed by `label_id`).
2. **Raster geometry** (skippable per page via `rasterise_done` /
   `geometry_done`, independently): `rasterised.pdf` assembled the same
   reuse-or-recompute way; `raster_geometry_for_page` replaces that page's
   `source="auto"` entries in `raster_labels.json`.
3. **Text re-sync** (always runs, no skip): `sync_text_from_vector_labels`
   against whatever `vector_labels.json` currently holds.
4. Opens `VectorLabelApp` on `original.pdf` / `vector_labels.json` --
   blocks until closed.
5. Re-syncs (step 3 again, now against the just-edited vector labels).
6. Opens `RasterLabelApp` (embedded-image picker) on `original.pdf` /
   `raster_labels.json` -- blocks until closed.
7. Prints a short summary.

Re-running on the same PDF: steps 1-2 skip already-done pages; steps 3-6
always run (so labelling can continue incrementally page by page, session
by session).

Not unit-testable end to end (chains three real Tk event loops). The
per-step assembly functions are plain-data/file orchestration and could be
unit-tested in isolation if this script grows further; kept as a manual
smoke test for now, per the existing convention for this package's
interactive tools.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pymupdf as fitz

from rastervec.Evaluation.conversion import convert_page_text_only
from rastervec.Evaluation.Labelling.label_schema import LabelSet, load_labels, save_labels
from rastervec.Evaluation.Labelling.native_label import attach_vector_signatures, native_label_pdf
from rastervec.Evaluation.Labelling.raster_label import (
    raster_geometry_for_page,
    sync_text_from_vector_labels,
)
from rastervec.logging_setup import configure_logging, get_logger
from rastervec.paths import output_dir

from raster_label import RasterLabelApp  # noqa: E402 -- sibling import, see module docstring
from vector_label import VectorLabelApp  # noqa: E402

_LOG = get_logger("master_label")

DEFAULT_DPI = 300


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"pages": {}}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _run_native_step(folder: Path, original_path: Path, page_count: int, manifest: dict) -> None:
    native_path = folder / "native_labels.json"
    vectorised_path = folder / "vectorised.pdf"
    native_labels = (
        load_labels(str(native_path)) if native_path.exists()
        else LabelSet(pdf_path=str(original_path))
    )
    known_ids = {e.label_id for e in native_labels.entries}
    existing_doc = fitz.open(str(vectorised_path)) if vectorised_path.exists() else None
    out_doc = fitz.open()
    new_pages = 0
    try:
        for p in range(page_count):
            state = manifest["pages"].setdefault(str(p), {})
            if state.get("native_done") and existing_doc is not None and p < existing_doc.page_count:
                out_doc.insert_pdf(existing_doc, from_page=p, to_page=p)
                continue
            new_pages += 1
            page_labels = native_label_pdf(str(original_path), p)
            page_bytes = convert_page_text_only(str(original_path), p)
            page_doc = fitz.open("pdf", page_bytes)
            out_doc.insert_pdf(page_doc)
            page_doc.close()

            fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
            os.close(fd)
            try:
                Path(tmp_name).write_bytes(page_bytes)
                attach_vector_signatures(page_labels, p, tmp_name)
            finally:
                os.unlink(tmp_name)

            for e in page_labels.entries:
                if e.label_id not in known_ids:
                    native_labels.entries.append(e)
                    known_ids.add(e.label_id)
            state["native_done"] = True
    finally:
        if existing_doc is not None:
            existing_doc.close()
    out_doc.save(str(vectorised_path))
    out_doc.close()
    save_labels(native_labels, str(native_path))
    _LOG.info("native step: %d page(s) newly processed, %d total labels", new_pages, len(native_labels.entries))


def _run_raster_step(
    folder: Path, original_path: Path, page_count: int, manifest: dict, dpi: int,
) -> None:
    rasterised_path = folder / "rasterised.pdf"
    raster_path = folder / "raster_labels.json"
    raster_labels = (
        load_labels(str(raster_path)) if raster_path.exists()
        else LabelSet(pdf_path=str(original_path))
    )
    existing_doc = fitz.open(str(rasterised_path)) if rasterised_path.exists() else None
    out_doc = fitz.open()
    src_doc = fitz.open(str(original_path))
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    rasterised_pages = 0
    geometry_pages = 0
    try:
        for p in range(page_count):
            state = manifest["pages"].setdefault(str(p), {})
            if state.get("rasterise_done") and existing_doc is not None and p < existing_doc.page_count:
                out_doc.insert_pdf(existing_doc, from_page=p, to_page=p)
            else:
                src_page = src_doc[p]
                pix = src_page.get_pixmap(matrix=matrix)
                new_page = out_doc.new_page(width=src_page.rect.width, height=src_page.rect.height)
                new_page.insert_image(new_page.rect, stream=pix.tobytes("png"))
                state["rasterise_done"] = True
                rasterised_pages += 1

            if not state.get("geometry_done"):
                fresh = raster_geometry_for_page(str(original_path), p)
                raster_labels.geometry_entries = [
                    g for g in raster_labels.geometry_entries
                    if not (g.page_index == p and g.source == "auto")
                ]
                raster_labels.geometry_entries.extend(fresh)
                state["geometry_done"] = True
                geometry_pages += 1
    finally:
        src_doc.close()
        if existing_doc is not None:
            existing_doc.close()
    out_doc.save(str(rasterised_path))
    out_doc.close()
    save_labels(raster_labels, str(raster_path))
    _LOG.info(
        "raster step: %d page(s) rasterised, %d page(s) geometry-extracted, %d geometry annotation(s) total",
        rasterised_pages, geometry_pages, len(raster_labels.geometry_entries),
    )


def _resync_raster_text(folder: Path, original_path: Path, page_count: int) -> None:
    raster_path = folder / "raster_labels.json"
    vector_path = folder / "vector_labels.json"
    raster_labels = (
        load_labels(str(raster_path)) if raster_path.exists()
        else LabelSet(pdf_path=str(original_path))
    )
    vector_labels = (
        load_labels(str(vector_path)) if vector_path.exists()
        else LabelSet(pdf_path=str(original_path))
    )
    sync_text_from_vector_labels(raster_labels, vector_labels, list(range(page_count)))
    save_labels(raster_labels, str(raster_path))


def _print_summary(folder: Path) -> None:
    def _count(name: str) -> tuple[int, int]:
        p = folder / name
        if not p.exists():
            return 0, 0
        labels = load_labels(str(p))
        return len(labels.entries), len(labels.geometry_entries)

    native_n, _ = _count("native_labels.json")
    vector_n, _ = _count("vector_labels.json")
    raster_n, geo_n = _count("raster_labels.json")
    print(f"\nLabel folder: {folder}")
    print(f"  native_labels.json  : {native_n} entries")
    print(f"  vector_labels.json  : {vector_n} entries")
    print(f"  raster_labels.json  : {raster_n} text entries, {geo_n} geometry annotations")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full native/vector/raster label workflow on one PDF.")
    parser.add_argument("pdf", help="Path to the source PDF.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Rasterise step DPI (default 300).")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)

    src_path = Path(args.pdf).resolve()
    stem = src_path.stem
    folder = output_dir("labels") / f"{stem}_label"
    original_path = folder / "original.pdf"
    if not original_path.exists():
        shutil.copy(str(src_path), str(original_path))
        _LOG.info("copied %s -> %s", src_path, original_path)

    manifest_path = folder / "manifest.json"
    manifest = _load_manifest(manifest_path)

    doc = fitz.open(str(original_path))
    page_count = doc.page_count
    doc.close()

    _run_native_step(folder, original_path, page_count, manifest)
    _save_manifest(manifest_path, manifest)

    _run_raster_step(folder, original_path, page_count, manifest, args.dpi)
    _save_manifest(manifest_path, manifest)

    _resync_raster_text(folder, original_path, page_count)

    _LOG.info("opening vector_label -- close its window to continue")
    VectorLabelApp(str(original_path), 0, str(folder / "vector_labels.json")).run()

    _resync_raster_text(folder, original_path, page_count)

    _LOG.info("opening raster_label -- close its window to finish")
    RasterLabelApp(str(original_path), str(folder / "raster_labels.json")).run()

    _print_summary(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
