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
"""
from __future__ import annotations

from itertools import groupby

import pymupdf as fitz

from rastervec.helpers.geometry import item_points
from rastervec.models import Vector

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


def _wrap_run_gstate(
    page: "fitz.Page",
    new_xrefs: list[int],
    blendmode: str | None,
    opacity: float | None,
) -> None:
    """Wrap each content stream in `new_xrefs` (the ones a run's `commit`
    just added) in `q /GSx gs ... Q`, where `/GSx` is a fresh ExtGState
    carrying the run's blend mode and constant opacity."""
    if not new_xrefs:
        return
    doc = page.parent
    resolved = _resolve_extgstate_container(doc, page)
    if resolved is None:
        return
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

    prefix = f"q /{name} gs\n".encode("latin-1")
    for xref in new_xrefs:
        stream = doc.xref_stream(xref)
        doc.update_stream(xref, prefix + stream + b"\nQ")


def replay_drawing_paths(
    page: "fitz.Page",
    vectors: list[Vector],
    *,
    dx: float = 0.0,
    dy: float = 0.0,
) -> None:
    """Replay `vectors` onto `page`: every item of a `Vector` is drawn (in
    its own stored order), then a single `shape.finish()` for that `Vector`
    carrying its real fill/stroke/width/dashes/closePath plus `even_odd`,
    `line_join`, `line_cap` and stroke/fill opacity. Points are offset by
    `(dx, dy)` (used to translate a cluster into its own isolated canvas).

    Vectors are processed in consecutive `(blendmode, opacity)` runs -- each
    run gets its own `Shape` + `commit()`, and a run with a non-Normal blend
    mode or a group opacity < 1 has its committed content stream wrapped in
    an ExtGState (`_wrap_run_gstate`) so it composites the way the source
    PDF did instead of painting fully opaque. A run of ordinary vectors is
    one `Shape` + one `commit()`, exactly as before.

    A `Vector` carrying neither `color` nor `fill` is skipped outright:
    `Shape.finish()` emits a stroke operator whenever `fill` is `None` even
    with `color=None`, falling back to the default black graphics-state
    color instead of staying invisible.
    """
    for (blendmode, opacity), run in groupby(
        vectors, key=lambda v: (v.blendmode, v.opacity)
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

            kwargs: dict = {
                "width": v.width or 0,
                "closePath": True if v.closePath is None else bool(v.closePath),
                "even_odd": bool(v.even_odd),
                "lineJoin": v.lineJoin or 0,
                "lineCap": v.lineCap or 0,
            }
            if v.color is not None:
                kwargs["color"] = v.color
            if v.fill is not None:
                kwargs["fill"] = v.fill
            if v.dashes:
                kwargs["dashes"] = v.dashes
            if v.stroke_opacity is not None:
                kwargs["stroke_opacity"] = v.stroke_opacity
            if v.fill_opacity is not None:
                kwargs["fill_opacity"] = v.fill_opacity
            shape.finish(**kwargs)

        if not drawn_any:
            continue

        contents_before = set(page.get_contents())
        shape.commit(overlay=True)
        if _is_trivial_blend(blendmode, opacity):
            continue
        new_xrefs = [x for x in page.get_contents() if x not in contents_before]
        _wrap_run_gstate(page, new_xrefs, blendmode, opacity)
