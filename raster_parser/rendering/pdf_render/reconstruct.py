from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

from raster_parser.models import Text
from raster_parser.models import Vector

WHITE_FILL = (1.0, 1.0, 1.0)

# Box/rule test for 4-line filled paths (sizes in PDF points). A path smaller than this on both
# sides is glyph-sized, so it is never a box.
BOX_MIN_SIDE_PT = 10
# Longer side over shorter side: above RULE_MIN_ASPECT it is a rule (a long thin bar); below
# BOX_MAX_ASPECT it is a square-ish panel. Glyph strokes fall in between.
RULE_MIN_ASPECT = 10
BOX_MAX_ASPECT = 3
# Vector text is drawn in black, white or a pure primary; any other fill/stroke colour marks a
# layout box.
TEXT_COLOURS = (
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

# Text re-rendering (sizes in PDF points). Font size is capped at the larger of
# MAX_FONT_SIZE_FACTOR * font_size and MAX_FONT_SIZE_FLOOR_PT.
MAX_FONT_SIZE_FACTOR = 4.0
MAX_FONT_SIZE_FLOOR_PT = 96.0
# Word boxes thinner than this on either side are skipped.
MIN_TEXT_BOX_SIDE_PT = 0.25
# A native word is vertical when its box is this many times taller than wide per glyph
# (and at least TALL_RATIO overall); see `render_text_objects_to_page`.
VERTICAL_HEIGHT_PER_GLYPH = 0.9
TALL_RATIO = 1.25
# The first font size tried is this fraction of the box height (box width when sideways);
# each refit step multiplies the size by it again.
FONT_FILL_FRAC = 0.88
# Starting font size floor, and the size below which shrinking stops.
MIN_START_FONT_SIZE_PT = 4.0
MIN_FONT_SIZE_PT = 3.0
# The rendered word may run this much (fraction) past its box before it is shrunk.
TEXT_LENGTH_SLACK = 1.02
# Upper bound on shrink steps per word.
MAX_FONT_FIT_STEPS = 12


def _finish_kwargs_reconstruct(path: dict) -> dict:
    kwargs: dict = {
        "width": path.get("width") or 0,
        "even_odd": path.get("even_odd", False),
        "closePath": path.get("closePath", False),
        "lineJoin": path.get("lineJoin", 0),
        "stroke_opacity": path.get("stroke_opacity", 1),
        "fill_opacity": path.get("fill_opacity", 1),
    }
    fill = path.get("fill")
    if fill is not None:
        kwargs["fill"] = fill
    color = path.get("color")
    if color is not None:
        kwargs["color"] = color
    dashes = path.get("dashes")
    if dashes:
        kwargs["dashes"] = dashes
    line_cap = path.get("lineCap")
    if line_cap is not None:
        kwargs["lineCap"] = line_cap[0] if isinstance(line_cap, (list, tuple)) else line_cap
    return kwargs


def _replay_items(shape: fitz.Shape, items: list) -> None:
    """Draw one path's ``get_drawings()`` items: ``"l"`` line, ``"re"`` rectangle,
    ``"qu"`` quadrilateral, ``"c"`` cubic Bezier (start, two control points, end)."""
    for item in items:
        op = item[0]
        if op == "l":
            shape.draw_line(item[1], item[2])
        elif op == "re":
            shape.draw_rect(item[1])
        elif op == "qu":
            shape.draw_quad(item[1])
        elif op == "c":
            shape.draw_bezier(item[1], item[2], item[3], item[4])
        else:
            raise ValueError(f"unhandled drawing op {op!r}: {item!r}")


def filter_out_white_vectors(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if v.fill != WHITE_FILL]


# PyMuPDF path ``type``: "f" filled only, "s" stroked only, "fs" filled and stroked.
def filter_fill_vectors(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if v.type == "f"]


def filter_fs_vectors(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if v.type == "fs"]


def is_box_or_rule(vec: Vector) -> bool:
    x0, y0, x1, y1 = vec.rect
    w = x1 - x0
    h = y1 - y0
    if w == 0 or h == 0:
        return False
    if w < BOX_MIN_SIDE_PT and h < BOX_MIN_SIDE_PT:
        return False
    ratio = max(w, h) / min(w, h)
    is_rule = ratio > RULE_MIN_ASPECT
    is_square_ish = ratio < BOX_MAX_ASPECT
    is_not_text_fill = vec.fill is not None and vec.fill not in TEXT_COLOURS
    is_not_text_color = vec.color is not None and vec.color not in TEXT_COLOURS
    return is_rule or is_square_ish or is_not_text_fill or is_not_text_color


def _is_four_line_box(vec: Vector) -> bool:
    return len(vec.items) == 4 and all(seg[0] == "l" for seg in vec.items)


def _is_re_box(vec: Vector) -> bool:
    return len(vec.items) == 1 and vec.items[0][0] == "re"


def _is_quad_box(vec: Vector) -> bool:
    return len(vec.items) == 1 and vec.items[0][0] == "qu"


def is_box_v1(vec: Vector) -> bool:
    if vec.fill is None:
        return False
    # MuPDF rectangle ops are layout/panel fills, never Type-2 glyph ink.
    if _is_re_box(vec) or _is_quad_box(vec):
        return True
    return _is_four_line_box(vec) and is_box_or_rule(vec)


def filter_out_boxes(vectors: list[Vector]) -> list[Vector]:
    return [v for v in vectors if not is_box_v1(v)]


def filter_text_vectors(
    vectors: list[Vector],
    *,
    include_fs: bool = False,
    only_fs: bool = False,
) -> list[Vector]:
    if only_fs and include_fs:
        raise ValueError("filter_text_vectors: only_fs and include_fs are mutually exclusive")
    white_filtered = filter_out_white_vectors(vectors)
    if only_fs:
        fills = filter_fs_vectors(white_filtered)
    elif include_fs:
        fills = [v for v in white_filtered if v.type in ("f", "fs")]
    else:
        fills = filter_fill_vectors(white_filtered)
    fills = filter_out_boxes(fills)
    return fills


def _glyph_pdf_rotate(ocr_rotate: int, page_rotation: int) -> int:
    r = (int(ocr_rotate) - int(page_rotation)) % 360
    if int(page_rotation) % 360:
        r = (r + 180) % 360
    return r


def ocr_rotate_for_target(target_rotate: int, page_rotation: int) -> int:
    for cand in (0, 90, 180, 270):
        if _glyph_pdf_rotate(cand, page_rotation) == int(target_rotate) % 360:
            return cand
    return 0


def _text_anchor(rect: fitz.Rect, rotate: int) -> fitz.Point:
    """The text origin corner for ``rotate``: ``bl``/``br``/``tr`` are a Rect's bottom-left,
    bottom-right and top-right points."""
    r = int(rotate) % 360
    if r == 90:
        return rect.bl
    if r == 180:
        return rect.br
    if r == 270:
        return rect.tr
    return rect.bl


def drawing_from_vector(v: Vector) -> dict:
    path = {
        "type": v.type,
        "items": v.items,
        "fill": v.fill,
        "color": v.color,
        "dashes": v.dashes,
        "even_odd": v.even_odd,
        "closePath": v.close_path,
        "width": v.width,
        "stroke_opacity": v.stroke_opacity,
        "fill_opacity": v.fill_opacity,
        "seqno": v.seqno,
        "layer": v.layer,
        "level": v.level,
    }
    if v.line_cap is not None:
        path["lineCap"] = v.line_cap
    if v.line_join is not None:
        path["lineJoin"] = v.line_join
    if v.scissor is not None:
        path["scissor"] = v.scissor
    if v.blendmode is not None:
        path["blendmode"] = v.blendmode
    if v.isolated is not None:
        path["isolated"] = v.isolated
    if v.knockout is not None:
        path["knockout"] = v.knockout
    if v.opacity is not None:
        path["opacity"] = v.opacity
    return path


def reconstruct(
    vectors: list[Vector],
    page: fitz.Page,
    out_path: Path | str = "output.pdf",
    *,
    include_fs: bool = False,
    only_fs: bool = False,
    svg_coordinates: bool = False,
) -> Path:
    """Replay vectors onto a new PDF page.

    Use ``svg_coordinates=True`` when vectors were parsed from ``get_svg_image()``
    output (viewBox / ``page.rect`` space). Default expects ``get_drawings()`` coords
    (mediabox user space + page rotation metadata).
    """
    out_path = Path(out_path)
    rotation = int(page.rotation or 0)
    mediabox = page.mediabox

    outpdf = fitz.open()
    if svg_coordinates:
        view = page.rect
        outpage = outpdf.new_page(width=view.width, height=view.height)
    else:
        outpage = outpdf.new_page(width=mediabox.width, height=mediabox.height)
    vec = filter_text_vectors(
        vectors,
        include_fs=include_fs,
        only_fs=only_fs,
    )

    if rotation and not svg_coordinates:
        outpage.set_rotation(rotation)
    shape = outpage.new_shape()
    for v in vec:
        path = drawing_from_vector(v)
        _replay_items(shape, path["items"])
        shape.finish(**_finish_kwargs_reconstruct(path))

    shape.commit()
    outpdf.save(out_path)
    outpdf.close()
    return out_path


def vector_per_page(
    vectors: list[Vector],
    page: fitz.Page,
    out_path: Path | str = "vectors-per-page.pdf",
    *,
    per_page: int = 1,
    pages: int = 100,
    start: int = 0,
    include_fs: bool = False,
    only_fs: bool = False,
) -> Path:
    out_path = Path(out_path)
    rotation = int(page.rotation or 0)
    mediabox = page.mediabox
    fills = filter_text_vectors(
        vectors,
        include_fs=include_fs,
        only_fs=only_fs,
    )

    outpdf = fitz.open()
    for p in range(pages):
        batch = fills[p * per_page : (p + 1) * per_page]
        if not batch:
            break
        outpage = outpdf.new_page(width=mediabox.width, height=mediabox.height)
        if rotation:
            outpage.set_rotation(rotation)
        shape = outpage.new_shape()
        for v in batch:
            path = drawing_from_vector(v)
            _replay_items(shape, path["items"])
            shape.finish(**_finish_kwargs_reconstruct(path))
        shape.commit()

    outpdf.save(out_path)
    outpdf.close()
    return out_path


def render_text_objects_to_pdf(
    page: fitz.Page,
    texts: list[Text],
    output_path: Path | str,
    *,
    font_size: float = 12.0,
    dpi: int = 150,
    glyph_boxes: bool = True,
) -> Path:
    outpage = render_text_objects_to_page(
        page,
        texts,
        font_size=font_size,
        dpi=dpi,
        glyph_boxes=glyph_boxes,
    )
    out_path = Path(output_path)
    outpdf = outpage.parent
    outpdf.save(out_path)
    outpdf.close()
    return out_path


def render_text_objects_to_page(
    page: fitz.Page,
    texts: list[Text],
    *,
    font_size: float = 12.0,
    dpi: int = 150,
    glyph_boxes: bool = True,
) -> fitz.Page:
    _ = dpi
    rotation = int(page.rotation or 0)
    mediabox = page.mediabox

    outpdf = fitz.open()
    w, h = float(mediabox.width), float(mediabox.height)
    outpage = outpdf.new_page(width=w, height=h)
    if rotation:
        outpage.set_rotation(rotation)

    clip = fitz.Rect(0, 0, w, h)
    max_glyph = max(font_size * MAX_FONT_SIZE_FACTOR, MAX_FONT_SIZE_FLOOR_PT)
    # "helv" is PyMuPDF's name for the built-in (base-14) Helvetica font.
    font = fitz.Font("helv")

    for t in texts:
        word = (t.word or "").strip()
        if not word:
            continue

        r = fitz.Rect(float(t.x0), float(t.y0), float(t.x1), float(t.y1)) & clip
        if r.is_empty or r.width < MIN_TEXT_BOX_SIDE_PT or r.height < MIN_TEXT_BOX_SIDE_PT:
            continue

        page_rot = rotation % 360
        if glyph_boxes:
            ocr_rot = int(getattr(t, "rotate", 0) or 0) % 360
            text_rotate = _glyph_pdf_rotate(ocr_rot, page_rot)
        else:
            rd = r * page.rotation_matrix if page_rot else r
            # Native words carry no explicit rotation, so infer it from the bbox shape. A single
            # glyph is naturally taller than wide (digits "8"/"9", letters), so an aspect-only
            # test wrongly rotates them 90 degrees. Vertically-stacked text instead grows tall
            # in proportion to its length (height ~ n glyphs, width ~ one glyph), so gate on a
            # length-scaled ratio and never rotate 1-char words.
            n = len(word)
            vertical = n >= 2 and rd.height > rd.width * max(
                TALL_RATIO, VERTICAL_HEIGHT_PER_GLYPH * n
            )
            ocr_rot = 90 if vertical else 0
            text_rotate = _glyph_pdf_rotate(ocr_rot, page_rot)

        sideways = text_rotate in (90, 270)
        if sideways:
            fs = min(max(r.width * FONT_FILL_FRAC, MIN_START_FONT_SIZE_PT), max_glyph)
            limit = r.height * TEXT_LENGTH_SLACK
        else:
            fs = min(max(r.height * FONT_FILL_FRAC, MIN_START_FONT_SIZE_PT), max_glyph)
            limit = r.width * TEXT_LENGTH_SLACK

        for _ in range(MAX_FONT_FIT_STEPS):
            if font.text_length(word, fontsize=fs) <= limit:
                break
            fs *= FONT_FILL_FRAC
            if fs < MIN_FONT_SIZE_PT:
                break

        outpage.insert_text(
            _text_anchor(r, text_rotate),
            word,
            fontsize=fs,
            fontname="helv",
            rotate=text_rotate,
            color=(0, 0, 0),
        )

    return outpage
