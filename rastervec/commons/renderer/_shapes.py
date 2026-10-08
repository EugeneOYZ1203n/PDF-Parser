"""Shared shape-drawing helpers for the renderer package.

`replay_drawing_paths` is the accuracy-critical bit: PyMuPDF's
`get_drawings()` returns a filled glyph outline (an "o", "e", "8", "A", ...)
as *one* drawing whose `items` list holds the outer contour and the inner
counter, meant to be filled as a single even-odd path. Drawing each item
primitive on its own and calling `Shape.finish(closePath=True)` per
primitive fills every contour solid, so the counter disappears -- a direct
hit to OCR of vector text, which is this project's whole point. Since a
`Vector` is never decomposed (its `items` stay nested exactly as
`get_drawings()` returned them), this just replays every item of one
`Vector` into the shape, then calls `finish()` **once** per `Vector` with
its real `even_odd` / `line_join` / `line_cap` / opacity -- ported from
`archive/raster_parser/rendering/pdf_render/reconstruct.py`
(`_replay_items` + `_finish_kwargs_reconstruct`).

Blend mode + transparency-group opacity can't go through `Shape.finish()`
(it has no such parameter), so vectors are grouped into consecutive runs by
`(blendmode, opacity)` and each non-trivial run's committed content stream
is wrapped in a `/BM`+`/ca`+`/CA` ExtGState (`_wrap_run_gstate`). Without
this a Multiply-blended line reconstructs fully opaque, painting solid over
whatever is beneath it.

The same run split also keys on each Vector's `scissor` (its enclosing
clips' bbox, folded in at extraction) and `clips` (the exact shape of every
enclosing non-rectangular clip): a clipped run's stream gets a `re W n`
prefix plus one `<path> W n` (`W*` for even-odd) per exact clip, so hatching /
pattern fills -- or a rect clipped to a rounded/curved outline -- don't spill
past it. Runs are consecutive, so paint order is always the input order.
"""
from __future__ import annotations

from itertools import groupby

import pymupdf as fitz

from rastervec.commons.helpers.geometry import item_points
from rastervec.commons.models import Vector

_DEFAULT_PATH_COLOR = "#111827"


def path_color_hex(vector: Vector, default: str = _DEFAULT_PATH_COLOR) -> str:
    """A Vector's own stroke/fill color as a hex string -- callers should
    render the PDF's real color; any B/W-style simplification is purely an
    internal classification concern, never something substituted in its
    place for display."""
    color = vector.color if vector.color is not None else vector.fill
    if color is None:
        return default
    return "#%02x%02x%02x" % tuple(min(255, max(0, round(c * 255))) for c in color)


def _replay_item(shape: "fitz.Shape", item: tuple, dx: float, dy: float) -> bool:
    kind = item[0]
    pts = [(x + dx, y + dy) for x, y in item_points(item)]
    if kind == "l":
        shape.draw_line(pts[0], pts[1])
    elif kind == "re":
        shape.draw_rect(fitz.Rect(*pts[0], *pts[1]))
    elif kind == "qu":
        # item points are stored in cyclic box order (ul, ur, lr, ll) --
        # fitz.Quad's own constructor instead expects (ul, ur, ll, lr), so
        # passing pts straight through swaps the last two corners and draws
        # a crossed "hourglass" instead of a box.
        shape.draw_quad(fitz.Quad(pts[0], pts[1], pts[3], pts[2]))
    elif kind == "c":
        shape.draw_bezier(pts[0], pts[1], pts[2], pts[3])
    else:
        return False
    return True


def _is_trivial_blend(blendmode: str | None, opacity: float | None) -> bool:
    """True when a run needs no ExtGState wrap (normal blend, full opacity)."""
    return blendmode in (None, "Normal") and opacity in (None, 1.0)


def _resolve_extgstate_container(
    doc: "fitz.Document", page: "fitz.Page"
) -> "tuple[int, str] | None":
    """`(xref, key_prefix)` identifying where to add `/ExtGState` entries:
    call `doc.xref_set_key(xref, key_prefix + name, ...)`. That is the
    nested `/ExtGState` object when `/Resources` already carries one as an
    indirect ref, else the `/Resources` xref itself with an `"ExtGState/"`
    prefix (creating an empty subdict first). `None` if `/Resources` isn't
    an indirect object we can edit -- renderer-built pages always give one,
    so this is just a guard."""
    rtype, rvalue = doc.xref_get_key(page.xref, "Resources")
    if rtype != "xref":
        return None
    res_xref = int(rvalue.split()[0])
    gtype, gvalue = doc.xref_get_key(res_xref, "ExtGState")
    if gtype == "xref":
        return int(gvalue.split()[0]), ""
    if gtype != "dict":
        doc.xref_set_key(res_xref, "ExtGState", "<<>>")
    return res_xref, "ExtGState/"


def _clip_operator(page: "fitz.Page", clip: tuple, dx: float, dy: float) -> str:
    """`x y w h re W n` for page-space `clip` (offset by `(dx, dy)`), in
    the page's own PDF content-stream space."""
    rect = fitz.Rect(clip[0] + dx, clip[1] + dy, clip[2] + dx, clip[3] + dy)
    r = (rect * ~page.transformation_matrix).normalize()
    return f"{r.x0:g} {r.y0:g} {r.width:g} {r.height:g} re W n\n"


_CURRENT_POINT_EPS = 1e-3


def _clip_path_operator(
    page: "fitz.Page", items: tuple, even_odd: bool, dx: float, dy: float,
) -> str:
    """One exact clip path -- `items` (plain `Vector.items`-shaped tuples, page
    space, offset by `(dx, dy)`) as content-stream path operators in the page's
    own PDF space, then `W n` (`W* n` when `even_odd`). A new subpath (`m`)
    starts wherever an item doesn't begin at the current point."""
    inv = ~page.transformation_matrix

    def pt(p) -> str:
        q = fitz.Point(p[0] + dx, p[1] + dy) * inv
        return f"{q.x:g} {q.y:g}"

    ops: list[str] = []
    current = None
    for item in items:
        kind = item[0]
        pts = item_points(item)
        if kind == "re":
            r = (fitz.Rect(pts[0][0] + dx, pts[0][1] + dy, pts[1][0] + dx, pts[1][1] + dy)
                 * inv).normalize()
            ops.append(f"{r.x0:g} {r.y0:g} {r.width:g} {r.height:g} re")
            current = None
            continue
        if kind == "qu":
            ops.append(f"{pt(pts[0])} m {pt(pts[1])} l {pt(pts[2])} l {pt(pts[3])} l h")
            current = None
            continue
        if kind not in ("l", "c"):
            continue
        start = pts[0]
        if current is None or (abs(start[0] - current[0]) > _CURRENT_POINT_EPS
                               or abs(start[1] - current[1]) > _CURRENT_POINT_EPS):
            ops.append(f"{pt(start)} m")
        if kind == "l":
            ops.append(f"{pt(pts[1])} l")
        else:
            ops.append(f"{pt(pts[1])} {pt(pts[2])} {pt(pts[3])} c")
        current = pts[-1]
    if not ops:
        return ""
    return "\n".join(ops) + (" W* n\n" if even_odd else " W n\n")


def _extgstate_operator(page: "fitz.Page", blendmode: str | None, opacity: float | None) -> str:
    """`/GSx gs` for a fresh ExtGState carrying the run's blend mode and
    constant opacity, registered on `page`'s resources ("" if it can't be)."""
    doc = page.parent
    resolved = _resolve_extgstate_container(doc, page)
    if resolved is None:
        return ""
    container, key_prefix = resolved

    parts = ["/Type/ExtGState"]
    if blendmode not in (None, "Normal"):
        parts.append(f"/BM/{blendmode}")
    if opacity not in (None, 1.0):
        parts.append(f"/ca {opacity:.4f}")
        parts.append(f"/CA {opacity:.4f}")
    gs_xref = doc.get_new_xref()
    doc.update_object(gs_xref, "<<" + "".join(parts) + ">>")

    name = f"GSrv{gs_xref}"
    doc.xref_set_key(container, key_prefix + name, f"{gs_xref} 0 R")
    return f"/{name} gs\n"


def _wrap_run(
    page: "fitz.Page",
    new_xrefs: list[int],
    blendmode: str | None,
    opacity: float | None,
    clip: tuple | None,
    dx: float = 0.0,
    dy: float = 0.0,
    clips: tuple = (),
) -> None:
    """Wrap each content stream in `new_xrefs` (the ones a run's `commit`
    just added) in `q [/GSx gs] [clip re W n] [<path> W n ...] ... Q` --
    `/GSx` a fresh ExtGState carrying the run's blend mode and constant
    opacity, the rect clip the run's shared `scissor`, then one exact clip
    path per entry of the run's shared `clips` (nested clips intersect)."""
    if not new_xrefs:
        return
    prefix = "q\n"
    if not _is_trivial_blend(blendmode, opacity):
        prefix += _extgstate_operator(page, blendmode, opacity)
    if clip is not None:
        prefix += _clip_operator(page, clip, dx, dy)
    for items, even_odd in clips:
        prefix += _clip_path_operator(page, items, even_odd, dx, dy)
    doc = page.parent
    prefix = prefix.encode("latin-1")
    for xref in new_xrefs:
        stream = doc.xref_stream(xref)
        doc.update_stream(xref, prefix + stream + b"\nQ")


_BLACK = (0.0, 0.0, 0.0)


def _paint(v: Vector, monochrome: bool) -> tuple:
    """`(color, fill, stroke_opacity, fill_opacity)` to finish `v` with --
    its real paint, or (monochrome) solid black wherever it has any stroke /
    fill at all, opacities dropped."""
    if not monochrome:
        return v.color, v.fill, v.stroke_opacity, v.fill_opacity
    return (
        _BLACK if v.color is not None else None,
        _BLACK if v.fill is not None else None,
        None, None,
    )


def replay_drawing_paths(
    page: "fitz.Page",
    vectors: list[Vector],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
    monochrome: bool = False,
) -> None:
    """Replay `vectors` onto `page`: every item of a `Vector` is drawn (in
    its own stored order), then a single `shape.finish()` for that `Vector`
    carrying its real fill/stroke/width/dashes/closePath plus `even_odd`,
    `line_join`, `line_cap` and stroke/fill opacity. Points are offset by
    `(dx, dy)` (used to translate a cluster into its own isolated canvas).

    Vectors are processed in consecutive `(blendmode, opacity, scissor,
    clips)` runs -- each run gets its own `Shape` + `commit()`; a run with a
    non-Normal blend mode or a group opacity < 1 has its committed content
    stream wrapped in an ExtGState so it composites the way the source PDF
    did instead of painting fully opaque, and a run with a `scissor`
    (enclosing clip bbox) and/or `clips` (exact non-rectangular clip
    shapes) is clipped to them (`_wrap_run`). A run of ordinary
    vectors is one `Shape` + one `commit()`.

    A `Vector` carrying neither `color` nor `fill` is skipped outright:
    `Shape.finish()` emits a stroke operator whenever `fill` is `None` even
    with `color=None`, falling back to the default black graphics-state
    color instead of staying invisible.

    A stroke of width 0 (a PDF hairline -- what CAD exports use for almost
    every line) can't go through `Shape.finish(width=0)` either: PyMuPDF
    then discards the stroke colour and emits no `w`, so an `"s"` path paints
    in the default black at 1pt (burying small fills like arrowheads) and an
    `"fs"` path loses its stroke. `0 w` is written into the path's own
    content instead, with `finish` given width 1.

    `monochrome=True` (OCR / FAST inputs -- `png.py`) paints every stroke and
    fill solid black and drops opacity, blend mode and group opacity, so the
    image is black ink on white; geometry, widths (hairlines included),
    dashes, caps/joins and the fill rule are unchanged; clips are dropped
    too (a cluster render is an isolated canvas). Colour only belongs
    in the final reconstruction.
    """
    for (blendmode, opacity, clip, clips), run in groupby(
        vectors,
        key=lambda v: (None, None, None, ()) if monochrome
        else (v.blendmode, v.opacity, v.scissor, v.clips),
    ):
        run = list(run)
        shape = page.new_shape()
        drawn_any = False
        for v in run:
            if v.color is None and v.fill is None:
                continue

            drawn = False
            for item in v.items:
                drawn = _replay_item(shape, item, dx, dy) or drawn
            if not drawn:
                continue
            drawn_any = True

            color, fill, stroke_opacity, fill_opacity = _paint(v, monochrome)
            width = v.width
            if color is None:
                width = 0
            elif width is None:
                width = 1  # PDF's default line width
            elif width == 0:
                # PDF hairline. `Shape.finish(width=0)` would drop the stroke
                # colour and write no `w` -- the stroke then paints in the
                # default black at 1pt. Set `0 w` ourselves ahead of the path
                # and let `finish` see width 1 (no `w` of its own, colour kept).
                shape.draw_cont = "0 w\n" + shape.draw_cont
                width = 1
            kwargs: dict = {
                "width": width,
                "closePath": True if v.closePath is None else bool(v.closePath),
                "even_odd": bool(v.even_odd),
                "lineJoin": v.lineJoin or 0,
                "lineCap": v.lineCap or 0,
            }
            if color is not None:
                kwargs["color"] = color
            if fill is not None:
                kwargs["fill"] = fill
            if v.dashes:
                kwargs["dashes"] = v.dashes
            if stroke_opacity is not None:
                kwargs["stroke_opacity"] = stroke_opacity
            if fill_opacity is not None:
                kwargs["fill_opacity"] = fill_opacity
            shape.finish(**kwargs)

        if not drawn_any:
            continue

        contents_before = set(page.get_contents())
        shape.commit(overlay=True)
        if _is_trivial_blend(blendmode, opacity) and clip is None and not clips:
            continue
        new_xrefs = [x for x in page.get_contents() if x not in contents_before]
        _wrap_run(page, new_xrefs, blendmode, opacity, clip, dx, dy, clips)
