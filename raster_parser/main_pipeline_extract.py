"""In-memory extraction: Type-2 native/fill OCR, then optional Type-4 on LO-cleared remainder."""

from __future__ import annotations

import shutil
from pathlib import Path

from raster_parser.models import NativePDFElements
from raster_parser.models import Text
from raster_parser.models import Vector


def _offset_text_blocks(texts: list[Text], block_offset: int) -> list[Text]:
    if block_offset <= 0:
        return texts
    for t in texts:
        t.block_no = int(t.block_no) + block_offset
    return texts


def _offset_vector_seqnos(vectors: list[Vector], seq_offset: int) -> list[Vector]:
    if seq_offset <= 0:
        return vectors
    for v in vectors:
        v.seqno = int(v.seqno) + seq_offset
    return vectors


def _max_block_no(words: list[Text]) -> int:
    if not words:
        return -1
    return max(int(t.block_no) for t in words)


def _max_seqno(vectors: list[Vector]) -> int:
    if not vectors:
        return -1
    return max(int(v.seqno) for v in vectors)


def _resolve_dump_dir(
    dump_dir: str | Path | None,
) -> Path:
    """Directory for LO + Type-4 dumps; default is ``./tmp`` (CWD-relative)."""
    from raster_parser.delete_natives.delete_native import DEFAULT_OUTPUT_DUMP_DIR

    root = Path(dump_dir) if dump_dir is not None else DEFAULT_OUTPUT_DUMP_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cleanup_dump_dir(dump_root: Path) -> None:
    """Remove the dump directory and all intermediates under it."""
    if dump_root.is_dir():
        shutil.rmtree(dump_root)


def extract(
    pdf_path: str | Path,
    page_index: int = 0,
    *,
    ocr_dpi: int = 300,
    use_angle_cls: bool = True,
    enable_raster_pass: bool = True,
    workers: int = 1,
    dump_dir: str | Path | None = None,
    debug: bool = False,
) -> NativePDFElements:
    """Native text + Type-2 fill OCR, then LibreOffice clear + Type-4 if residual content.

    Flow
    ----
    1. ``get_text`` / ``get_drawings`` — native words and all vectors.
    2. Type-2 — OCR grouped fill glyphs; keep non-fill vectors as drawings.
    3. LibreOffice ``DeleteDrawings`` — strip native text/vector shapes from a
       single-page copy (embedded images remain).
    4. If the cleared page is non-empty — run Type-4 raster OCR + autotrace and
       merge words/vectors into the result.

    Does **not** run native-vs-OCR comparison. Does **not** rasterize the original
    native page for Type-2; Type-4 only renders the LO-cleared residual page.

    ``enable_raster_pass=False`` skips steps 3–4 (useful for tests / Type-2-only).
    ``dump_dir``: directory for LO + Type-4 intermediate files
    (default: ``./tmp``; passed into ``clear_page`` / Type-4 dumps).
    ``debug``: when ``False`` (default), remove ``dump_dir`` after extraction;
    when ``True``, keep LO + Type-4 intermediates on disk. Type-4 autotrace is always on.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    dump_root = _resolve_dump_dir(dump_dir)

    try:
        return _extract_impl(
            pdf_path=pdf_path,
            page_index=page_index,
            ocr_dpi=ocr_dpi,
            use_angle_cls=use_angle_cls,
            enable_raster_pass=enable_raster_pass,
            workers=workers,
            dump_root=dump_root,
        )
    finally:
        if not debug:
            _cleanup_dump_dir(dump_root)


def _extract_impl(
    *,
    pdf_path: Path,
    page_index: int,
    ocr_dpi: int,
    use_angle_cls: bool,
    enable_raster_pass: bool,
    workers: int,
    dump_root: Path,
) -> NativePDFElements:
    from raster_parser.parsing.parser import PdfRasterParser
    from raster_parser.rendering.pdf_render.reconstruct import ocr_rotate_for_target, filter_text_vectors
    from raster_parser.ocr.wordgrouping import (
        Orientation,
        cluster_by_seqno,
        convert_vectors_to_glyphs,
        get_vectors,
    )
     # Actual Implementation of the extraction pipeline
    with PdfRasterParser(str(pdf_path), page_index) as parser:
        native_text = parser.extract_text()
        all_vectors = parser.extract_vectors()
        fill_vectors = filter_text_vectors(all_vectors)
        fill_ids = {v.seqno for v in fill_vectors}
        native_drawings = [v for v in all_vectors if v.seqno not in fill_ids]

        page_rotation = int(parser.page.rotation or 0)
        glyphs = convert_vectors_to_glyphs(fill_vectors)
        word_groups = cluster_by_seqno(glyphs, page_rotation)

        pairs = parser.extract_text_from_paddle_word_groups(
            [get_vectors(wg) for wg in word_groups],
            [(wg.bbox.x0, wg.bbox.y0, wg.bbox.x1, wg.bbox.y1) for wg in word_groups],
            dpi=ocr_dpi,
            use_angle_cls=use_angle_cls,
        )

        words: list[Text] = []
        for text, _ in pairs:
            wg = word_groups[text.block_no]
            text.glyph_orientation = wg.orientation
            target = 90 if wg.orientation == Orientation.VERTICAL else 0
            text.rotate = ocr_rotate_for_target(target, page_rotation)
            words.append(text)

        words.extend(native_text)
        words.sort(key=lambda t: t.block_no)
        vectors: list[Vector] = list(native_drawings)

    if enable_raster_pass:
        from raster_parser.delete_natives.delete_native import clear_page, page_is_nonempty
        from raster_parser.parsing.extract_raster_page import extract_raster_page

        work_dir, cleared = clear_page(pdf_path, page_index, dump_dir=dump_root)
        if page_is_nonempty(cleared, 0):
            t4_words, t4_vectors = extract_raster_page(
                cleared,
                0,
                dpi=ocr_dpi,
                workers=workers,
                output_dump_dir=work_dir,
            )
            block_off = _max_block_no(words) + 1
            seq_off = _max_seqno(vectors) + 1
            _offset_text_blocks(t4_words, block_off)
            _offset_vector_seqnos(t4_vectors, seq_off)
            words.extend(t4_words)
            vectors.extend(t4_vectors)
            words.sort(key=lambda t: t.block_no)

    return NativePDFElements.from_extract(words=words, vectors=vectors)
