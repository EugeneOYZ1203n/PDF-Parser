"""Renderer: rendering helpers, not a pipeline stage.

Turns pipeline data (vector paths, clusters, raster regions) into pixels
for OCR input. render_vector_cluster is implemented (isolates a cluster
of VectorPaths onto their own small PyMuPDF page and rasterizes it, so
OCR gets the PDF's own vector rendering rather than a hand-rolled
rasterizer); render_raster_region is still a stub since the Raster stage
itself isn't implemented yet. render_reconstructed_page is a debug-app-only
convenience (not OCR input, not evaluation.py's real reconstruction stage)
that redraws whatever a given stage has captured so far -- native words,
drawing vectors, OCR'd text -- onto a page-sized blank canvas, so the debug
app can show "does this look like the original" for one stage's output at
a time. Reconstructing pipeline output back into a real evaluation PDF is a
separate concern -- see evaluation.py, the pipeline's actual final stage.
"""
from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image

from rastervec.geometry import union_bbox
from rastervec.models import DrawingVector, Page, PageMeta, RasterImage, TextVectorResult, TextWord, VectorPath

_DEFAULT_PATH_COLOR = "#111827"

# Minimum padding (in PDF points) added around a cluster's bbox before
# rendering, so thin strokes right at the bbox edge aren't clipped.
_MIN_CLUSTER_PADDING = 4.0


def _draw_vector_path(shape: "fitz.Shape", path: VectorPath, dx: float = 0.0, dy: float = 0.0) -> None:
    """Draws one VectorPath onto `shape` with its own real stroke/fill/
    width/dashes, offset by (dx, dy) -- shared by render_vector_cluster
    (which translates into an isolated small canvas) and
    render_reconstructed_page (dx=dy=0, paths are already in the
    reconstruction's absolute page-space coordinates)."""
    if path.stroke_color is None and path.fill_color is None:
        # Shape.finish() always emits a stroke operator when `fill` is
        # None, even with `color=None` too -- it falls back to the
        # current (default black) graphics-state color rather than
        # staying invisible. A path with neither color set was never
        # meant to render at all, so skip it outright instead of relying
        # on finish() to no-op.
        return
    pts = [(x + dx, y + dy) for x, y in path.points]
    if path.kind == "l":
        shape.draw_line(pts[0], pts[1])
    elif path.kind == "re":
        shape.draw_rect(fitz.Rect(*pts[0], *pts[1]))
    elif path.kind == "qu":
        shape.draw_quad(fitz.Quad(pts))
    elif path.kind == "c":
        shape.draw_bezier(pts[0], pts[1], pts[2], pts[3])
    else:
        return
    shape.finish(
        color=path.stroke_color,
        fill=path.fill_color,
        width=path.stroke_width or 1.0,
        dashes=path.dashes,
        closePath=True if path.closed is None else bool(path.closed),
    )


def _text_color(color: int | None) -> tuple[float, float, float]:
    """Unpacks a PyMuPDF span-style packed sRGB int (as TextWord.color
    carries) into an (r, g, b) 0..1 tuple for insert_text's color param."""
    if color is None:
        return (0.0, 0.0, 0.0)
    return (
        ((color >> 16) & 255) / 255,
        ((color >> 8) & 255) / 255,
        (color & 255) / 255,
    )


class Renderer:
    """Rendering helpers shared by the debug app and the OCR input
    pipeline."""

    def path_color_hex(
        self, path: VectorPath, default: str = _DEFAULT_PATH_COLOR
    ) -> str:
        """A path's own stroke/fill color as a hex string -- callers
        should render the PDF's real color; any B/W-style simplification
        (e.g. Vector's background-fill heuristic) is purely an internal
        classification concern, never something substituted in its place
        for display."""
        color = path.stroke_color if path.stroke_color is not None else path.fill_color
        if color is None:
            return default
        return "#%02x%02x%02x" % tuple(min(255, max(0, round(c * 255))) for c in color)

    def render_vector_cluster(
        self, paths: list[VectorPath], page: Page, dpi: int
    ) -> "Image.Image":
        """High-resolution render of an isolated vector path cluster, used
        as OCR input. Builds a fresh single-page PyMuPDF document sized to
        the cluster's own bbox (plus padding for stroke overflow), redraws
        each path onto it with its real stroke/fill styling via
        fitz.Shape, then rasterizes at `dpi` -- reusing PyMuPDF's own
        rendering rather than re-implementing curve/fill rasterization by
        hand. `page` is unused by the geometry itself (paths are already
        in absolute page-space coordinates, translated here to the
        isolated canvas's origin) but kept in the signature for parity
        with render_raster_region and to match the interface other
        callers (RenderOCR.ocr_cluster) rely on."""
        if not paths:
            raise ValueError("render_vector_cluster requires at least one path")

        x0, y0, x1, y1 = union_bbox([p.bbox for p in paths])
        padding = max(
            _MIN_CLUSTER_PADDING, max((p.stroke_width or 0.0) for p in paths)
        )
        dx, dy = padding - x0, padding - y0
        width = (x1 - x0) + 2 * padding
        height = (y1 - y0) + 2 * padding

        doc = fitz.open()
        try:
            cluster_page = doc.new_page(width=width, height=height)
            shape = cluster_page.new_shape()
            for path in paths:
                _draw_vector_path(shape, path, dx=dx, dy=dy)
            shape.commit()

            zoom = dpi / 72.0
            pixmap = cluster_page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), alpha=False
            )
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            image.load()  # decode now -- the backing BytesIO doesn't outlive this call
            return image
        finally:
            doc.close()

    def render_raster_region(
        self, image: RasterImage, dpi: int
    ) -> "Image.Image":
        """High-resolution render of a raster image region, used as OCR
        input."""
        raise NotImplementedError

    def render_reconstructed_page(
        self,
        page_meta: PageMeta,
        *,
        native_words: list[TextWord] | None = None,
        drawing_vectors: list[DrawingVector] | None = None,
        ocr_results: list[TextVectorResult] | None = None,
        zoom: float = 1.0,
    ) -> "Image.Image":
        """Debug-app-only preview: redraws whatever has actually been
        captured so far -- one or more of native text words, drawing
        vectors (each drawn from its own real member VectorPaths, not
        just its aggregate bbox), OCR'd vector-text results -- onto a
        fresh blank page sized/rotated to match `page_meta`, then
        rasterizes at `zoom` the same way DebugApp.render() rasterizes the
        real page (so the two images are pixel-comparable at the same
        zoom level). Text reconstruction is necessarily approximate: font
        family isn't preserved (always the PyMuPDF base14 "helv"). Rotation
        is exact, at any angle -- page.insert_text's own `rotate` param only
        accepts multiples of 90, so text is rotated instead via a `morph`
        transform (a (fixpoint, rotation-matrix) pair applied as a `cm` op
        before drawing, PyMuPDF's own mechanism for arbitrary-angle text)
        -- this is still a "does this look roughly right" preview, not a
        byte-accurate reconstruction (that's evaluation.py's job once
        built)."""
        doc = fitz.open()
        try:
            page = doc.new_page(width=page_meta.width, height=page_meta.height)
            page.set_rotation(page_meta.rotation)

            if drawing_vectors:
                shape = page.new_shape()
                for dv in drawing_vectors:
                    for path in dv.paths:
                        _draw_vector_path(shape, path)
                shape.commit()

            if native_words:
                for word in native_words:
                    if not word.text.strip():
                        continue
                    origin = word.origin or (word.bbox[0], word.bbox[3])
                    page.insert_text(
                        origin, word.text,
                        fontsize=max(word.font_size, 1.0),
                        color=_text_color(word.color),
                        rotate=0,
                        morph=(fitz.Point(origin), fitz.Matrix(1, 1).prerotate(word.angle)),
                    )

            if ocr_results:
                for result in ocr_results:
                    if not result.text.strip():
                        continue
                    x0, _y0, _x1, y1 = result.bbox
                    fontsize = max(result.bbox[3] - result.bbox[1], 4.0)
                    origin = (x0, y1)
                    page.insert_text(
                        origin, result.text,
                        fontsize=fontsize,
                        rotate=0,
                        morph=(
                            fitz.Point(origin),
                            fitz.Matrix(1, 1).prerotate(result.rotation_used),
                        ),
                    )

            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            image.load()
            return image
        finally:
            doc.close()
