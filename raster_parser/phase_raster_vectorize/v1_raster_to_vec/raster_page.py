"""The raster pass on one (cleared) page: render, split text from drawing, OCR the text,
autotrace the drawing. Intermediate rasters / traces are written under ``output_dump_dir``
(default: ``./tmp``) and kept on disk.

Copied from `parsing/extract_raster_page.py` (`extract_raster_page`).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s2_render_page import page_to_bgr
from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s3_text_drawing_split import (
    OcrTextDrawingParser,
    filter_boxes_by_min_area,
    make_ocr,
    nms_boxes,
    ocr_boxes_rotated,
)
from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s4_region_ocr import (
    recognize_boxes,
    recover_missed_glyphs,
    refine_boxes,
)
from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s5_colour_autotrace import (
    colour_fill_autotrace,
)

# Detection boxes overlapping a larger box by more than this IoU are duplicates (e.g. the
# upright and the rotated pass finding the same label) and are dropped.
DUPLICATE_BOX_IOU = 0.6
# Recovered glyphs have no detection box backing them, so only near-certain reads are kept.
RECOVERED_GLYPH_MIN_CONF = 0.9
# Debug image styling: box outline thickness (px) and how far recovered-glyph outlines are
# drawn outside the glyph (px); colours are BGR.
DEBUG_BOX_THICKNESS_PX = 2
DEBUG_GLYPH_OUTLINE_PAD_PX = 3
DEBUG_DET_BOX_BGR = (0, 0, 255)
DEBUG_RECOVERED_GLYPH_BGR = (0, 170, 0)


def extract_raster_page(
    pdf_path: str | Path,
    page_index: int = 0,
    *,
    dpi: int = 300,
    min_conf: float = 0.0,
    rec_min_conf: float = 0.0,
    lang: str = "en",
    tile: int = 512,
    tile_overlap: int = 64,
    ocr_line_removal: str = "none",
    margin: int = 1,
    ink_thresh: int = 250,
    min_box_area: int = 150,
    line_min: int = 70,
    line_grow: int = 3,
    hough_threshold: int = 50,
    hough_max_gap: int = 3,
    workers: int = 1,
    output_dump_dir: str | Path | None = None,
    use_angle_cls: bool = True,
    keep_intermediates: bool = True,
) -> tuple[list, list]:
    """Run Type-4 on one page; return ``(texts, drawing_vectors)``.

    Colour-fill autotraces the drawing layer. Writes intermediate files under
    ``output_dump_dir`` (default: ``./tmp``) and leaves them on disk. With
    ``keep_intermediates=False`` only the returned texts and vectors remain: the debug images,
    dumps and every file the steps spill are removed before returning.
    """
    from raster_parser.phase_raster_vectorize.v1_raster_to_vec.s1_clear_page import (
        DEFAULT_OUTPUT_DUMP_DIR,
    )

    pdf_path = Path(pdf_path)
    dump_root = Path(output_dump_dir) if output_dump_dir is not None else DEFAULT_OUTPUT_DUMP_DIR
    dump_root.mkdir(parents=True, exist_ok=True)
    dump = Path(
        tempfile.mkdtemp(
            prefix=f"type4_{pdf_path.stem}_p{page_index}_",
            dir=str(dump_root),
        )
    )

    bgr, to_points = page_to_bgr(pdf_path, page_index, dpi)
    ocr = make_ocr(lang, min_conf, use_angle_cls=use_angle_cls)
    parser = OcrTextDrawingParser(
        min_conf=min_conf,
        lang=lang,
        tile=tile,
        tile_overlap=tile_overlap,
        ocr_line_removal=ocr_line_removal,
        margin=margin,
        line_min=line_min,
        line_grow=line_grow,
        hough_threshold=hough_threshold,
        hough_max_gap=hough_max_gap,
        ink_thresh=ink_thresh,
        min_box_area=min_box_area,
        workers=workers,
        ocr=ocr,
    )
    result = parser.parse(bgr, scratch_dir=dump)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    text_final_gray = np.full_like(gray, 255)
    text_final_gray[result.text == 255] = gray[result.text == 255]
    drawing_final_bgr = bgr.copy()
    erase = (result.text == 255) & (result.lines != 255)
    drawing_final_bgr[erase] = (255, 255, 255)

    clean_rgb = cv2.cvtColor(text_final_gray, cv2.COLOR_GRAY2RGB)
    clean_scratch = dump / "clean_det"
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
    clean_boxes = filter_boxes_by_min_area(clean_boxes, parser.min_box_area)
    clean_boxes = refine_boxes(
        text_final_gray,
        clean_boxes,
        ocr=parser.ocr,
        lang=parser.lang,
        min_conf=parser.min_conf,
    )
    clean_boxes = nms_boxes(clean_boxes, iou_thresh=DUPLICATE_BOX_IOU)
    clean_boxes = filter_boxes_by_min_area(clean_boxes, parser.min_box_area)

    rec_scratch = dump / "rec_boxes"
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
    recov = recover_missed_glyphs(
        gray,
        clean_boxes,
        ocr=parser.ocr,
        to_points=to_points,
        dpi=dpi,
        min_conf=RECOVERED_GLYPH_MIN_CONF,
    )
    if recov.n_text:
        text_final_gray[recov.kept_mask == 255] = gray[recov.kept_mask == 255]
        erase = (recov.kept_mask == 255) & (result.lines != 255)
        drawing_final_bgr[erase] = (255, 255, 255)
        rec.texts.extend(recov.texts)

    drawing_final_path = dump / "drawing_final.png"
    text_final_path = dump / "text_final.png"
    cv2.imwrite(str(drawing_final_path), drawing_final_bgr)  # the autotrace input
    if keep_intermediates:
        cv2.imwrite(str(text_final_path), text_final_gray)

        initial_det_vis = bgr.copy()
        for x0, y0, x1, y1 in result.boxes:
            cv2.rectangle(
                initial_det_vis,
                (int(x0), int(y0)),
                (int(x1), int(y1)),
                DEBUG_DET_BOX_BGR,
                DEBUG_BOX_THICKNESS_PX,
            )
        cv2.imwrite(str(dump / "initial_det.png"), initial_det_vis)

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
        cv2.imwrite(str(dump / "det_boxes.png"), det_vis)

    colour = colour_fill_autotrace(
        drawing_final_path,
        out_dir=dump,
        stem="drawing",
        dpi=dpi,
        keep_intermediates=keep_intermediates,
        workers=workers,
    )
    drawing_vectors = list(colour.combined_vectors)

    if keep_intermediates:
        (dump / "ocr.txt").write_text(
            "\n".join([f"=== TEXT (PaddleOCR @ {dpi} dpi) ===", *(str(t) for t in rec.texts)])
            + "\n",
            encoding="utf-8",
        )
        (dump / "vectors.txt").write_text(
            "\n".join(f"Vector {i}: {v}" for i, v in enumerate(drawing_vectors)) + "\n",
            encoding="utf-8",
        )
    else:
        shutil.rmtree(dump, ignore_errors=True)

    return list(rec.texts), list(drawing_vectors)
