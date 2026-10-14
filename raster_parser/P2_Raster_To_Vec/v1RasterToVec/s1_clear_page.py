"""Step 1 -- clear native text/drawings via the LibreOffice DeleteDrawings macro.

Moved from `delete_natives/delete_native.py`. `delete_drawings_from_pdf` (whole-document
clear) is not used by the raster pass; it stays public as `raster_parser.delete_drawings_from_pdf`.
"""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from subprocess import run

import numpy as np
import pymupdf
import pymupdf as pymupdf

from raster_parser.P2_Raster_To_Vec.v1RasterToVec.s1_libreoffice_macro import ensure_macro_registered

DEFAULT_OUTPUT_DUMP_DIR = Path("./tmp")


def delete_drawing_page(fp: str | Path) -> None:
    """Assumes that pdf file at ``fp`` is only a single page."""
    ensure_macro_registered()
    cmd = [
        "libreoffice",
        "--headless",
        "--invisible",
        "--nologo",
        "--norestore",
        "--nofirststartwizard",
        "macro:///Standard.DeleteDrawings.DeleteDrawingsAndExportPDF",
        str(fp),
    ]
    run(cmd)


def cleared_pdf_path(page_pdf: str | Path) -> Path:
    """Path LibreOffice DeleteDrawings writes next to the input (``*_cleared.pdf``)."""
    page_pdf = Path(page_pdf)
    return page_pdf.with_name(f"{page_pdf.stem}_cleared.pdf")


def write_single_page_pdf(src_pdf: str | Path, page_index: int, out_pdf: str | Path) -> Path:
    """Copy one page of ``src_pdf`` into a new single-page PDF at ``out_pdf``."""
    src_pdf = Path(src_pdf)
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(src_pdf) as doc:
        if page_index < 0 or page_index >= doc.page_count:
            raise IndexError(f"page_index {page_index} out of range (0..{doc.page_count - 1})")
        out = pymupdf.open()
        try:
            out.insert_pdf(doc, from_page=page_index, to_page=page_index)
            out.save(out_pdf)
        finally:
            out.close()
    return out_pdf


def clear_page(
    src_pdf: str | Path,
    page_index: int,
    *,
    dump_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Write one page under ``dump_dir``, run DeleteDrawings, return ``(work_dir, cleared_pdf)``.

    Creates a unique work subdirectory under ``dump_dir`` (default: ``./tmp``),
    writes ``page.pdf`` there, runs the LibreOffice macro (exports
    ``page_cleared.pdf`` beside it), and returns that work dir plus the cleared path.
    """
    parent = Path(dump_dir) if dump_dir is not None else DEFAULT_OUTPUT_DUMP_DIR
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="extract_lo_", dir=str(parent)))
    page_pdf = root / "page.pdf"
    write_single_page_pdf(src_pdf, page_index, page_pdf)
    page_pdf = page_pdf.resolve()
    delete_drawing_page(page_pdf)
    cleared = cleared_pdf_path(page_pdf)
    if not cleared.is_file():
        raise FileNotFoundError(
            f"LibreOffice clear did not produce {cleared.name}; "
            "is DeleteDrawings registered and LibreOffice installed?"
        )
    return root, cleared


def page_is_nonempty(
    pdf_path: str | Path,
    page_index: int = 0,
    *,
    probe_dpi: int = 36,
    ink_frac: float = 0.001,
    ink_thresh: int = 250,
) -> bool:
    """True if the page still has images, drawings, or meaningful non-white ink."""
    with pymupdf.open(str(pdf_path)) as doc:
        page = doc[page_index]
        if page.get_images():
            return True
        if page.get_drawings():
            return True
        pix = page.get_pixmap(dpi=probe_dpi)
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        if arr.size == 0:
            return False
        return float((arr < ink_thresh).sum()) / float(arr.size) > ink_frac


def delete_drawings_from_pdf(
    fp: str | Path,
    num_parallel: int = 1,
    *,
    dump_dir: str | Path | None = None,
) -> Path:
    """Split pages, run DeleteDrawings per page, recombine into ``*_cleared.pdf``.

    Per-page intermediates and the final cleared PDF are written under
    ``dump_dir`` (default: ``./tmp``). The LibreOffice macro exports beside each
    opened per-page PDF; Python chooses the dump root by placing those files there.
    """
    assert num_parallel >= 1, "num_parallel must be greater than or equal to 1"
    assert isinstance(num_parallel, int), "num_parallel must be an integer"
    fp = Path(fp)
    out_dir = Path(dump_dir) if dump_dir is not None else DEFAULT_OUTPUT_DUMP_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(fp)

    name = fp.stem
    out_paths = []
    for page in doc:
        out_path = out_dir / f"{name}_{page.number}.pdf"
        target_doc = pymupdf.open()
        target_doc.insert_pdf(doc, from_page=page.number, to_page=page.number)
        target_doc.save(out_path)
        out_paths.append(out_path)
        target_doc.close()
    doc.close()

    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        list(executor.map(delete_drawing_page, out_paths))

    cleared_out_doc = pymupdf.open()
    for out_fp in out_paths:
        cleared_out_fp = cleared_pdf_path(out_fp)
        cleared_out_page_doc = pymupdf.open(cleared_out_fp)
        cleared_out_doc.insert_pdf(cleared_out_page_doc, from_page=0, to_page=0)
        cleared_out_page_doc.close()
        cleared_out_fp.unlink(missing_ok=True)
        out_fp.unlink(missing_ok=True)

    cleared_out_fp = out_dir / f"{name}_cleared.pdf"
    cleared_out_doc.save(cleared_out_fp)
    cleared_out_doc.close()
    return cleared_out_fp
