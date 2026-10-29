"""Type-4 CLI: separate text from a drawing raster using OCR detection boxes.

PaddleOCR only *locates* text; it never decides pixels. ``text_final`` is all
non-white ink (gray < ink_thresh) inside each det bbox -- keeps thin strokes Otsu
would drop, including lines that cross the box. No second-tier filter. The drawing
layer keeps the full line mask plus ink outside those boxes, in the **original
page colours** (text glyphs erased to white). Recognition runs on the same
unfiltered grayscale text layer.

Step 2 (optional): colour-fill autotrace on ``drawing_final.png`` — cluster colours,
per-mask outline autotrace, combined coloured ``list[Vector]`` + PDF/JSON dumps.
Page-level adaptive BW PNG is written as QA.
Also writes ``*_ocr.txt`` / ``*_vectors.txt`` (same style as
``type2_dump_extraction_pipeline``).

Usage::

    uv run scripts/type4_pipeline.py page.pdf --page 0
    uv run scripts/type4_pipeline.py some_dir/ --output-dir out/
    uv run scripts/type4_pipeline.py some_dir/ --page-workers 5 --workers 1 --chunk 5
"""

import argparse
import gc
import multiprocessing as mp
import os
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from raster_parser.models import Text
from raster_parser.rendering.pdf_render.reconstruct import render_text_objects_to_pdf
from raster_parser.autotrace.drawing_colour_fill_autotrace import (
    _DEFAULT_BG_THRESH,
    _DEFAULT_GROUP_DIST,
    _DEFAULT_MIN_GROUP_PIXELS,
    colour_fill_autotrace,
)
from raster_parser.evaluation.native_vs_raster.native_vs_raster_text import (
    NativeVsRasterText,
    dedupe_stacked_texts,
    format_report,
)
from raster_parser.ocr.region_ocr import (
    recognize_boxes,
    recover_missed_glyphs,
    refine_boxes,
)
from raster_parser.parsing.text_drawing_parser import (
    OcrTextDrawingParser,
    filter_boxes_by_min_area,
    make_ocr,
    nms_boxes,
    ocr_boxes_rotated,
)
from raster_parser.parsing.extract_raster_page import (
    DEBUG_BOX_THICKNESS_PX,
    DEBUG_DET_BOX_BGR,
    DEBUG_GLYPH_OUTLINE_PAD_PX,
    DEBUG_RECOVERED_GLYPH_BGR,
    DUPLICATE_BOX_IOU,
    RECOVERED_GLYPH_MIN_CONF,
    page_to_bgr,
)

# Page labels in progress/log lines are cut to this many characters.
JOB_LABEL_MAX_CHARS = 40
# QA adaptive threshold: each pixel is compared with the Gaussian-weighted mean of its
# 51 x 51 px neighbourhood minus 10 gray levels.
ADAPTIVE_BLOCK_SIZE_PX = 51
ADAPTIVE_OFFSET = 10
# OCR word <-> native word linking: a pair links when bbox IoU >= MATCH_IOU, when the
# smaller box lies >= MATCH_CONTAIN inside the other, or when the boxes overlap and their
# centres are within MATCH_CENTER_TOL_PT.
MATCH_IOU = 0.2
MATCH_CONTAIN = 0.5
MATCH_CENTER_TOL_PT = 5.0
# CPU count assumed when os.cpu_count() cannot tell.
FALLBACK_CPU_COUNT = 4

# Per-process OCR for page ProcessPool workers (loaded once in initializer).
_PAGE_WORKER_OCR = None


def collect_pdfs(inputs) -> list:
    """Flatten a directory of PDFs into a list of all PDF paths."""
    pdfs = []
    for raw in inputs:
        path = Path(raw)
        if not path.exists():
            raise SystemExit(f"no such file or directory: {path}")
        if path.is_dir():
            pdfs.extend(sorted(path.rglob("*.pdf")))
        else:
            pdfs.append(path)
    # de-dup while preserving order
    seen, unique = set(), []
    for pdf in pdfs:
        key = pdf.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(pdf)
    return unique


def unique_stems(pdfs) -> dict:
    """Map each PDF path to a unique output stem.

    Defaults to the file stem; on collision (same name in different folders), prefixes
    with parent-dir names until unique so outputs can share one flat directory safely.
    """
    stems, used = {}, set()
    for pdf in pdfs:
        stem = pdf.stem
        parts = list(pdf.parent.parts)
        while stem in used and parts:
            stem = f"{parts.pop()}_{stem}"
        # if still colliding (identical paths shouldn't happen post de-dup), append index
        n = 1
        base = stem
        while stem in used:
            stem = f"{base}_{n}"
            n += 1
        used.add(stem)
        stems[pdf] = stem
    return stems


def native_texts(page) -> list:
    """Native words as Text in unrotated PDF points.

    ``get_text("words")`` already yields ``(x0, y0, x1, y1, word, block, line, word_no)`` in the
    page's *unrotated* coordinate space (mediabox user space) -- the same space the OCR boxes end
    up in, since ``page_to_bgr`` maps pixels back through ``~(rotation_matrix * scale)``.

    Stacked CAD overlays (same string, IoU >= 0.9) are de-duplicated so one visual label counts
    once for scoring and for ``text_only.pdf``.
    """
    out = []
    for x0, y0, x1, y1, word, bno, lno, wno in page.get_text("words"):
        r = pymupdf.Rect(x0, y0, x1, y1)
        r.normalize()
        out.append(
            Text(
                x0=r.x0,
                y0=r.y0,
                x1=r.x1,
                y1=r.y1,
                word=word,
                block_no=bno,
                line_no=lno,
                word_no=wno,
            )
        )
    return dedupe_stacked_texts(out)


def pdf_page_count(pdf_path: str) -> int:
    """Number of pages in a PDF."""
    doc = pymupdf.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def _page_pool_worker_init(min_conf: float) -> None:
    """
    Limit BLAS threads and load PaddleOCR once per page-worker process.
    """
    global _PAGE_WORKER_OCR
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # Multi-process: CPU only (avoids N workers fighting one GPU).
    _PAGE_WORKER_OCR = make_ocr("en", float(min_conf), use_gpu=False)


def _chunked(items: list, chunk_size: int) -> list[list]:
    """Split ``items`` into contiguous chunks (batch_test_script pattern)."""
    size = max(1, int(chunk_size))
    return [items[i : i + size] for i in range(0, len(items), size)]


def _make_parser(ocr, job: dict) -> OcrTextDrawingParser:
    """Build a parser from a picklable page-job dict."""
    return OcrTextDrawingParser(
        min_conf=float(job["min_conf"]),
        tile=int(job["tile"]),
        tile_overlap=int(job["tile_overlap"]),
        ocr_line_removal=str(job["ocr_line_removal"]),
        margin=int(job["margin"]),
        line_min=int(job["line_min"]),
        line_grow=int(job["line_grow"]),
        hough_threshold=int(job["hough_threshold"]),
        hough_max_gap=int(job["hough_max_gap"]),
        ink_thresh=int(job["ink_thresh"]),
        min_box_area=int(job["min_box_area"]),
        workers=int(job["workers"]),
        ocr=ocr,
    )


def _process_job(job: dict, parser: OcrTextDrawingParser) -> dict:
    """Run ``process_pdf`` for one job; return ``{label, sort_key, stats}`` only."""
    stats = process_pdf(
        Path(job["src"]),
        parser,
        int(job["dpi"]),
        int(job["page_index"]),
        rec_min_conf=float(job["rec_min_conf"]),
        out_dir=Path(job["out_dir"]) if job.get("out_dir") else None,
        out_stem=job.get("out_stem"),
        colour_autotrace=bool(job["colour_autotrace"]),
        workers=int(job["workers"]),
        temp_dir=Path(job["temp_dir"]) if job.get("temp_dir") else None,
        group_dist=float(job["group_dist"]),
        min_group_pixels=int(job["min_group_pixels"]),
        bg_thresh=int(job["bg_thresh"]),
        keep_colour_intermediates=bool(job["keep_colour_intermediates"]),
    )
    return {
        "label": str(job["label"]),
        "sort_key": tuple(job["sort_key"]),
        "stats": stats,
    }


def _run_page_job(job: dict) -> dict:
    """Spawn worker entry: reuse process OCR, process one page, return small stats."""
    global _PAGE_WORKER_OCR
    if _PAGE_WORKER_OCR is None:
        _PAGE_WORKER_OCR = make_ocr("en", float(job["min_conf"]), use_gpu=False)
    result = _process_job(job, _make_parser(_PAGE_WORKER_OCR, job))
    gc.collect()
    return result


def _run_page_chunk(
    jobs: list[dict],
    *,
    max_workers: int,
    min_conf: float,
) -> tuple[list[dict], int]:
    """Run one chunk of page jobs in a fresh spawn pool, then tear workers down.

    Closing the pool between chunks releases Paddle native allocations that
    ``gc.collect`` alone cannot return to the OS (batch_test chunk barrier).
    """
    page_results: list[dict] = []
    n_failed = 0
    if not jobs:
        return page_results, n_failed
    n_workers = min(max_workers, len(jobs))
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=_page_pool_worker_init,
        initargs=(float(min_conf),),
    ) as pool:
        futures = {pool.submit(_run_page_job, job): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                n_failed += 1
                print(
                    f"[page FAIL] {job.get('label', '?')}: {type(e).__name__}: {e}",
                    flush=True,
                )
                traceback.print_exc()
                continue
            page_results.append(result)
            done += 1
            print(
                f"[page done {done}/{len(jobs)}] {result['label']}",
                flush=True,
            )
    return page_results, n_failed


def collect_page_jobs(
    pdfs: list[Path],
    *,
    page: int | None,
    stems: dict,
    args,
    out_dir: Path | None,
    temp_dir: Path | None,
) -> list[dict]:
    """Build one picklable job dict per (pdf, page) to process."""
    jobs: list[dict] = []
    for pdf_i, src in enumerate(pdfs):
        if page is not None:
            pages = [page]
        else:
            pages = list(range(pdf_page_count(str(src))))
        for page_index in pages:
            label = f"{src.name} p{page_index}"[:JOB_LABEL_MAX_CHARS]
            jobs.append(
                {
                    "src": str(src.resolve()),
                    "page_index": int(page_index),
                    "label": label,
                    "sort_key": (pdf_i, int(page_index)),
                    "out_dir": str(out_dir.resolve()) if out_dir is not None else None,
                    "out_stem": stems.get(src),
                    "temp_dir": str(temp_dir.resolve()) if temp_dir is not None else None,
                    "dpi": int(args.dpi),
                    "rec_min_conf": float(args.rec_min_conf),
                    "min_conf": float(args.min_conf),
                    "tile": int(args.tile),
                    "tile_overlap": int(args.tile_overlap),
                    "ocr_line_removal": str(args.ocr_line_removal),
                    "margin": int(args.margin),
                    "ink_thresh": int(args.ink_thresh),
                    "min_box_area": int(args.min_box_area),
                    "line_min": int(args.line_min),
                    "line_grow": int(args.line_grow),
                    "hough_threshold": int(args.hough_threshold),
                    "hough_max_gap": int(args.hough_max_gap),
                    "workers": int(args.workers),
                    "colour_autotrace": bool(args.colour_autotrace),
                    "group_dist": float(args.group_dist),
                    "min_group_pixels": int(args.min_group_pixels),
                    "bg_thresh": int(args.bg_thresh),
                    "keep_colour_intermediates": bool(args.keep_colour_intermediates),
                }
            )
    return jobs


def process_pdf(
    src: Path,
    parser: OcrTextDrawingParser,
    dpi: int,
    page_index: int,
    rec_min_conf: float = 0.5,
    out_dir: Path | None = None,
    out_stem: str | None = None,
    *,
    colour_autotrace: bool = True,
    workers: int = 1,
    temp_dir: Path | None = None,
    group_dist: float = _DEFAULT_GROUP_DIST,
    min_group_pixels: int = _DEFAULT_MIN_GROUP_PIXELS,
    bg_thresh: int = _DEFAULT_BG_THRESH,
    keep_colour_intermediates: bool = True,
) -> dict:
    """Render one PDF page, run the parser + region OCR, and write outputs.

    ``workers``: process-pool size for tile det + box recog (ProcessPoolExecutor / spawn;
    one PaddleOCR per worker on CPU). ``workers == 1`` stays in-process.
    ``temp_dir``: parent for throwaway scratch (tile/box spills, colour autotrace temps);
    kept on disk for inspection when provided. Returns the native-vs-OCR comparison summary dict
    for this page so the caller can aggregate a grand total.
    """
    scratch: Path | None = None
    if temp_dir is not None or workers > 1:
        parent = temp_dir if temp_dir is not None else Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix=f"{src.stem}_p{page_index}_", dir=str(parent)))

    t0 = time.perf_counter()
    bgr, to_points = page_to_bgr(str(src), page_index, dpi)
    result = parser.parse(bgr, scratch_dir=scratch)
    print(f"First text and drawing separation took {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # text_final = ALL ink inside Paddle det boxes (no second-tier). Line crossings stay.
    text_final_gray = np.full_like(gray, 255)
    text_final_gray[result.text == 255] = gray[result.text == 255]
    # Drawing keeps original BGR colours: full lines + ink outside boxes; only non-line
    # text ink is erased to white.
    drawing_final_bgr = bgr.copy()
    erase = (result.text == 255) & (result.lines != 255)
    drawing_final_bgr[erase] = (255, 255, 255)

    # Clean-raster re-detection / recognition on the same unfiltered text layer.
    clean_rgb = cv2.cvtColor(text_final_gray, cv2.COLOR_GRAY2RGB)
    clean_scratch = (scratch / "clean_det") if scratch is not None else None
    if clean_scratch is not None:
        clean_scratch.mkdir(parents=True, exist_ok=True)
    clean_boxes = ocr_boxes_rotated(
        clean_rgb,
        parser.min_conf,
        parser.lang,
        parser.tile,
        parser.tile_overlap,
        ocr=parser.ocr,
        workers=workers,
        scratch_dir=clean_scratch,
    )
    del clean_rgb
    clean_boxes = filter_boxes_by_min_area(clean_boxes, parser.min_box_area)

    print(f"First OCR bbox detecton took {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()
    # Second pass: boxes far taller than the mode line height likely hold >=2 stacked lines
    # (which tanks recognition), so re-detect within them for finer per-line/word boxes.
    clean_boxes = refine_boxes(
        text_final_gray, clean_boxes, ocr=parser.ocr, lang=parser.lang, min_conf=parser.min_conf
    )
    print(f"Refining boxes took {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()
    # Drop near-duplicate boxes (upright+rotated overlaps, refine sub-boxes landing on a
    # neighbour) by IoU, without growing survivors the way merge_boxes would.
    clean_boxes = nms_boxes(clean_boxes, iou_thresh=DUPLICATE_BOX_IOU)
    clean_boxes = filter_boxes_by_min_area(clean_boxes, parser.min_box_area)
    print(f"Removing duplicate boxes took {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()
    rec_scratch = (scratch / "rec_boxes") if scratch is not None else None
    if rec_scratch is not None:
        rec_scratch.mkdir(parents=True, exist_ok=True)
    rec = recognize_boxes(
        text_final_gray,
        clean_boxes,
        ocr=parser.ocr,
        to_points=to_points,
        dpi=dpi,
        min_conf=rec_min_conf,
        workers=workers,
        lang=parser.lang,
        scratch_dir=rec_scratch,
    )
    print(f"Actual OCR on bboxes took {time.perf_counter() - t0:.2f} seconds")
    t0 = time.perf_counter()

    # Recovery: lone glyphs + grouped same-size nearby CCs the detector missed.
    recov = recover_missed_glyphs(
        gray,
        clean_boxes,
        ocr=parser.ocr,
        to_points=to_points,
        dpi=dpi,
        min_conf=RECOVERED_GLYPH_MIN_CONF,
    )
    print(
        f"Recovering missed glyphs took {time.perf_counter() - t0:.2f} seconds  "
        f"recovered={recov.n_text}/{recov.n_boxes} candidates"
    )
    t0 = time.perf_counter()
    if recov.n_text:
        text_final_gray[recov.kept_mask == 255] = gray[recov.kept_mask == 255]
        erase = (recov.kept_mask == 255) & (result.lines != 255)
        drawing_final_bgr[erase] = (255, 255, 255)
        rec.texts.extend(recov.texts)

    stem = out_stem or src.stem
    parent = out_dir if out_dir is not None else src.parent
    base = parent / f"{stem}_p{page_index}"
    page_stem = base.name  # e.g. foo_p0 — shared prefix for all page outputs
    drawing_final_path = base.with_name(page_stem + "_drawing_final.png")
    text_final_path = base.with_name(page_stem + "_text_final.png")
    initial_det_path = base.with_name(page_stem + "_initial_det.png")
    det_boxes_path = base.with_name(page_stem + "_det_boxes.png")
    page_adaptive_bw_path = base.with_name(page_stem + "_page_adaptive_bw.png")
    ocr_pdf_path = base.with_name(page_stem + "_ocr.pdf")
    ocr_txt_path = base.with_name(page_stem + "_ocr.txt")
    text_only_pdf_path = base.with_name(page_stem + "_text_only.pdf")
    # drawing = colour page with text glyphs erased; text = grayscale ink partition for OCR.
    cv2.imwrite(str(drawing_final_path), drawing_final_bgr)  # original colours, text -> white
    cv2.imwrite(str(text_final_path), text_final_gray)  # raw ink inside det boxes (+ recovered)
    # Colour step reloads from disk; free full-page colour + parse masks before autotrace peak.
    page_det_boxes = list(result.boxes)
    n_page_boxes = result.n_boxes
    del drawing_final_bgr, result

    # Preprocess QA on the whole input page (not drawing_final): adaptive BW.
    adaptive_ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE_PX,
        C=ADAPTIVE_OFFSET,
    )
    adaptive_bw = np.where(adaptive_ink > 0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(page_adaptive_bw_path), adaptive_bw)
    del adaptive_ink, adaptive_bw

    # Step 2: colour-fill autotrace on drawing_final (cluster -> per-mask outline -> combine).
    n_drawing_vectors = 0
    colour_paths: list = []
    drawing_vectors: list = []
    if colour_autotrace:
        t0 = time.perf_counter()
        colour = colour_fill_autotrace(
            drawing_final_path,
            out_dir=parent,
            stem=page_stem,
            dpi=dpi,
            bg_thresh=bg_thresh,
            group_dist=group_dist,
            min_group_pixels=min_group_pixels,
            keep_intermediates=keep_colour_intermediates,
            workers=workers,
        )
        n_groups = len(colour.groups)
        drawing_vectors = list(colour.combined_vectors)
        n_drawing_vectors = len(drawing_vectors)
        colour_paths = [
            colour.colours_dir,
            colour.combined_pdf_path,
            colour.vectors_json_path,
        ]
        del colour
        print(
            f"Step 2 colour autotrace took {time.perf_counter() - t0:.2f}s  "
            f"groups={n_groups} vectors={n_drawing_vectors}",
            flush=True,
        )
        if scratch is not None:
            print(f"scratch: {scratch.resolve()}")
    # Interim: PaddleOCR's first-pass detection boxes over the original page.
    initial_det_vis = bgr.copy()
    for x0, y0, x1, y1 in page_det_boxes:
        cv2.rectangle(
            initial_det_vis,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            DEBUG_DET_BOX_BGR,
            DEBUG_BOX_THICKNESS_PX,
        )
    cv2.imwrite(str(initial_det_path), initial_det_vis)
    del initial_det_vis, bgr, page_det_boxes
    # Visualize the clean-raster detection boxes (the word/line grouping) over the text.
    det_vis = cv2.cvtColor(text_final_gray, cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1 in clean_boxes:
        cv2.rectangle(
            det_vis,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            DEBUG_DET_BOX_BGR,
            DEBUG_BOX_THICKNESS_PX,
        )
    if recov.n_text:
        cnts, _ = cv2.findContours(recov.kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            p = DEBUG_GLYPH_OUTLINE_PAD_PX
            cv2.rectangle(
                det_vis,
                (x - p, y - p),
                (x + w + p, y + h + p),
                DEBUG_RECOVERED_GLYPH_BGR,
                DEBUG_BOX_THICKNESS_PX,
            )
    cv2.imwrite(str(det_boxes_path), det_vis)
    del det_vis, text_final_gray, gray

    # Grab the page's native words + vector count, then render OCR words to a PDF.
    doc = pymupdf.open(str(src))
    try:
        page = doc[page_index]
        native = native_texts(page)
        # One Vector-equivalent per native get_drawings() path (no stroke-split).
        n_native_vectors = len(page.get_drawings(extended=True))
        render_text_objects_to_pdf(page, rec.texts, ocr_pdf_path, dpi=dpi, glyph_boxes=True)
        # Pure text-only PDF: the page's NATIVE words on a blank same-size page, no glyph boxes.
        render_text_objects_to_pdf(page, native, text_only_pdf_path, dpi=dpi, glyph_boxes=False)
    finally:
        doc.close()

    # Dump OCR Text + drawing Vector objects (same style as type2_dump_extraction_pipeline).
    ocr_txt_path.write_text(
        "\n".join([f"=== TEXT (PaddleOCR @ {dpi} dpi) ===", *(str(t) for t in rec.texts)]) + "\n",
        encoding="utf-8",
    )
    vectors_txt_path = base.with_name(page_stem + "_vectors.txt")
    vectors_txt_path.write_text(
        "\n".join(f"Vector {i}: {v}" for i, v in enumerate(drawing_vectors)) + "\n",
        encoding="utf-8",
    )
    drawing_vectors.clear()
    del drawing_vectors

    # Stats: match OCR words to native words by bbox, then compare the strings. Containment
    # linking lets small native words link to the wide OCR line/word-group box containing them.
    comparison = NativeVsRasterText(native, rec.texts).compare(
        iou_thresh=MATCH_IOU, contain_thresh=MATCH_CONTAIN, center_tol=MATCH_CENTER_TOL_PT
    )
    stats = comparison.summary()
    print(f"Matching OCR words to native words took {time.perf_counter() - t0:.2f} seconds")

    results_path = base.with_name(page_stem + "_results.txt")
    results_path.write_text(
        format_report(comparison, page_label=f"{src.name} p{page_index}"), encoding="utf-8"
    )
    del comparison, native

    n_clean_boxes = rec.n_boxes
    n_recognized = rec.n_text
    n_recovered = recov.n_text
    del rec, recov, clean_boxes

    print(
        f"[{src.name} p{page_index}] page boxes: {n_page_boxes}   "
        f"clean boxes: {n_clean_boxes}   recognized: {n_recognized}   "
        f"recovered glyphs: {n_recovered}   "
        f"native vectors: {n_native_vectors}   drawing vectors: {n_drawing_vectors}"
    )
    print(
        f"    stats: native={stats['native_words']} ocr={stats['raster_words']} "
        f"matched={stats['matched']} agree={stats['agree']} disagree={stats['disagree']} "
        f"missed={stats['missed']} extra={stats['extra']} acc={stats['accuracy']:.1%} "
        f"word_acc={stats['word_accuracy']:.1%} "
        f"char_match={stats['char_match_rate']:.1%} cer={stats['cer']:.1%}"
    )
    for path in (
        drawing_final_path,
        page_adaptive_bw_path,
        text_final_path,
        initial_det_path,
        det_boxes_path,
        *colour_paths,
        ocr_pdf_path,
        ocr_txt_path,
        vectors_txt_path,
        text_only_pdf_path,
        results_path,
    ):
        if path is not None:
            print(path)
    print()
    return stats


def main() -> None:
    """CLI: render a PDF page, build line mask, OCR text boxes, erase glyph ink, write drawing + text PNGs."""
    p = argparse.ArgumentParser(
        description="Erase text from a drawing raster via OCR + within-box separation."
    )
    p.add_argument(
        "pdf",
        nargs="+",
        help="one or more PDF files and/or directories (scanned recursively for *.pdf)",
    )
    p.add_argument(
        "--output-dir",
        default="out/",
        help="write every PDF's outputs into this one directory (flat); default is next to each PDF",
    )
    p.add_argument(
        "--page",
        type=int,
        default=None,
        help="0-based page index to render; omit to process EVERY page of each PDF",
    )
    p.add_argument("--dpi", type=int, default=300, help="render DPI for get_pixmap")
    p.add_argument("--min-conf", type=float, default=0.0, help="OCR detection confidence cutoff")
    p.add_argument(
        "--rec-min-conf",
        type=float,
        default=0.0,
        help="recognition confidence gate for region OCR (drops low-conf/gibberish artefacts)",
    )
    p.add_argument(
        "--tile", type=int, default=512, help="OCR tile size in px (0 = whole image at once)"
    )
    p.add_argument(
        "--tile-overlap", type=int, default=64, help="tile overlap; must exceed largest label"
    )
    p.add_argument(
        "--ocr-line-removal",
        choices=["none", "inpaint"],
        default="none",
        help="optional line cleanup on the detection-only image (default: none)",
    )
    p.add_argument("--margin", type=int, default=1, help="grow each OCR box by this many px")
    p.add_argument(
        "--ink-thresh",
        type=int,
        default=250,
        help="text-mask soft threshold: gray < this counts as ink inside det boxes",
    )
    p.add_argument(
        "--min-box-area",
        type=int,
        default=150,
        help="drop Paddle det boxes with pixel area (w*h) below this (0 = disabled)",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help="ProcessPool size within a page for tile det, box recog, and "
        "colour autotrace (spawn; 1 = in-process)",
    )
    p.add_argument(
        "--page-workers",
        "--jobs",
        dest="page_workers",
        type=int,
        default=1,
        help="Parallel workers across pages (spawn; 1 = serial). Prefer "
        "--workers 1 when page-workers > 1 to avoid nested pools.",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=5,
        help="Pages per chunk when page-workers > 1 (lower = less peak RAM; "
        "pool is torn down between chunks, default: 5)",
    )
    p.add_argument(
        "--temp-dir",
        default=None,
        help="dump throwaway scratch here (tile/box spills, colour autotrace); default system temp",
    )
    p.add_argument(
        "--no-colour-autotrace",
        dest="colour_autotrace",
        action="store_false",
        help="skip colour-fill autotrace on drawing_final.png",
    )
    p.add_argument(
        "--group-dist",
        type=float,
        default=_DEFAULT_GROUP_DIST,
        help=f"colour cluster distance in HSV embedding (default {_DEFAULT_GROUP_DIST:g})",
    )
    p.add_argument(
        "--min-group-pixels",
        type=int,
        default=_DEFAULT_MIN_GROUP_PIXELS,
        help=f"drop colour groups smaller than this (default {_DEFAULT_MIN_GROUP_PIXELS})",
    )
    p.add_argument(
        "--bg-thresh",
        type=int,
        default=_DEFAULT_BG_THRESH,
        help="RGB channels >= this count as background white",
    )
    p.add_argument(
        "--no-keep-colour-intermediates",
        dest="keep_colour_intermediates",
        action="store_false",
        help="delete per-colour mask / isolate PNGs after tracing",
    )
    # global line mask
    p.add_argument(
        "--line-min", type=int, default=70, help="shortest straight run (px) that counts as a line"
    )
    p.add_argument(
        "--line-grow", type=int, default=3, help="line mask dilation / Hough line thickness (px)"
    )
    p.add_argument(
        "--hough-threshold",
        type=int,
        default=50,
        help="Hough accumulator votes a segment needs",
    )
    p.add_argument(
        "--hough-max-gap",
        type=int,
        default=3,
        help="largest gap (px) bridged inside one Hough segment",
    )
    args = p.parse_args()

    pdfs = collect_pdfs(args.pdf)
    if not pdfs:
        raise SystemExit("no PDFs found in the given inputs")

    out_dir = None
    stems = {}
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stems = unique_stems(pdfs)  # collision-safe names for the shared dir

    temp_dir = Path(args.temp_dir) if args.temp_dir else None
    page_workers = max(1, int(args.page_workers))
    if page_workers > 1 and args.workers > 1:
        print(
            f"warning: page-workers={page_workers} with workers={args.workers} nests "
            f"ProcessPools; forcing workers=1 for this run "
            f"(use --page-workers 1 if you want within-page workers={args.workers})",
            flush=True,
        )
        args.workers = 1

    jobs = collect_page_jobs(
        pdfs,
        page=args.page,
        stems=stems,
        args=args,
        out_dir=out_dir,
        temp_dir=temp_dir,
    )
    if not jobs:
        raise SystemExit("no pages to process")

    totals = {
        "native_words": 0,
        "raster_words": 0,
        "matched": 0,
        "agree": 0,
        "disagree": 0,
        "missed": 0,
        "extra": 0,
        "word_hits": 0,
        "native_chars": 0,
        "match_chars": 0,
        "edit_chars": 0,
    }
    cols = ["native_words", "raster_words", "matched", "agree", "disagree", "missed", "extra"]
    header = (
        f"{'page':<40} "
        + " ".join(f"{c:>12}" for c in cols)
        + f" {'acc':>7} {'word_acc':>8} {'char_match':>11} {'cer':>7}"
    )
    rows = [header, "-" * len(header)]

    page_results: list[dict] = []
    n_failed = 0
    chunk = max(1, int(args.chunk))
    if page_workers == 1:
        # Serial: one shared OCR/parser across pages (cheaper model load).
        ocr = make_ocr("en", args.min_conf)
        # Parser settings are identical across jobs; build once from the first job.
        parser = _make_parser(ocr, jobs[0])
        if args.workers > 1:
            print(
                f"workers={args.workers}: ProcessPoolExecutor (spawn) for tile det + "
                f"box recog + colour autotrace",
                flush=True,
            )
        for job in jobs:
            print(
                f"--- {job['src']} p{job['page_index']} ({job['sort_key'][0] + 1}/{len(pdfs)}) ---",
                flush=True,
            )
            try:
                page_results.append(_process_job(job, parser))
            except Exception as e:
                n_failed += 1
                print(f"[page FAIL] {job['label']}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue
            gc.collect()
    else:
        # Each page worker loads its own PaddleOCR (~1–2GB+). Cap aggressively.
        # Chunk + tear down pool between chunks so Paddle native RAM can release.
        max_pw = min(page_workers, len(jobs), max(1, (os.cpu_count() or FALLBACK_CPU_COUNT)))
        if max_pw < page_workers:
            print(
                f"warning: capping page-workers {page_workers} -> {max_pw} (cpu_count / job count)",
                flush=True,
            )
        chunks = _chunked(jobs, chunk)
        print(
            f"page-workers={max_pw}: {len(jobs)} page(s) in {len(chunks)} chunk(s) "
            f"of up to {chunk}; within-page workers={args.workers}",
            flush=True,
        )
        for chunk_index, page_chunk in enumerate(chunks, start=1):
            print(
                f"=== Chunk {chunk_index}/{len(chunks)} ({len(page_chunk)} page(s)) ===",
                flush=True,
            )
            chunk_results, chunk_failed = _run_page_chunk(
                page_chunk,
                max_workers=max_pw,
                min_conf=float(args.min_conf),
            )
            page_results.extend(chunk_results)
            n_failed += chunk_failed
            gc.collect()

    if not page_results:
        raise SystemExit(
            f"all {len(jobs)} page job(s) failed "
            f"(page-workers crash often = OOM / nested --workers / one bad page)"
        )
    if n_failed:
        print(
            f"warning: {n_failed}/{len(jobs)} page(s) failed; continuing with ok pages", flush=True
        )

    page_results.sort(key=lambda r: r["sort_key"])
    n_pages = len(page_results)
    for result in page_results:
        stats = result["stats"]
        for k in totals:
            totals[k] += stats[k]
        rows.append(
            f"{result['label']:<40} "
            + " ".join(f"{stats[c]:>12}" for c in cols)
            + f" {stats['accuracy']:>6.1%} {stats['word_accuracy']:>7.1%}"
            + f" {stats['char_match_rate']:>10.1%} {stats['cer']:>6.1%}"
        )

    nchars = totals["native_chars"] or 1
    nwords = totals["native_words"] or 1
    acc = totals["agree"] / totals["native_words"] if totals["native_words"] else 0.0
    word_acc = totals["word_hits"] / nwords
    char_match = totals["match_chars"] / nchars
    cer = totals["edit_chars"] / nchars
    print(
        f"=== TOTAL over {n_pages} page(s): "
        f"native={totals['native_words']} ocr={totals['raster_words']} "
        f"matched={totals['matched']} agree={totals['agree']} "
        f"disagree={totals['disagree']} missed={totals['missed']} "
        f"extra={totals['extra']} acc={acc:.1%} word_acc={word_acc:.1%} "
        f"char_match={char_match:.1%} cer={cer:.1%} ==="
    )

    rows.append("-" * len(header))
    rows.append(
        f"{('TOTAL (' + str(n_pages) + ' pages)'):<40} "
        + " ".join(f"{totals[c]:>12}" for c in cols)
        + f" {acc:>6.1%} {word_acc:>7.1%} {char_match:>10.1%} {cer:>6.1%}"
    )

    stats_path = (out_dir if out_dir is not None else Path(".")) / "stats.txt"
    stats_path.write_text("\n".join(rows) + "\n")
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()
