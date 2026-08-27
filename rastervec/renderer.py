"""Renderer: rendering helpers, not a pipeline stage.

Turns pipeline data (vector paths, clusters) into pixels for OCR input.
render_vector_cluster isolates a cluster of VectorPaths onto their own
small PyMuPDF page and rasterizes it, so OCR gets the PDF's own vector
rendering rather than a hand-rolled rasterizer -- pixel_to_page_bbox
inverts that same isolated-canvas transform (shared via the private
_cluster_frame helper) to map a detected-in-pixel-space bbox back into PDF
page space; render_page_paths is the whole-page counterpart (every given
path drawn onto one page-sized canvas, no isolation/padding), used as
FAST's own detection input by pipeline.py's fast_text_detect stage.
render_reconstructed_page is a debug-app-only convenience (not OCR input,
not evaluation.py's real reconstruction stage) that redraws whatever a
given stage has captured so far -- native words, drawing vectors, OCR'd
text -- onto a page-sized blank canvas, so the debug app can show "does
this look like the original" for one stage's output at a time.
Reconstructing pipeline output back into a real evaluation PDF is a
separate concern -- see evaluation.py, the pipeline's actual final stage.
"""
from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image

from rastervec.helpers.geometry import union_bbox
from rastervec.models import DrawingVector, PageMeta, TextVectorResult, TextWord, VectorPath

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
        # path.points is stored in cyclic box order (ul, ur, lr, ll) --
        # fitz.Quad's own constructor instead expects (ul, ur, ll, lr), so
        # passing pts straight through swaps the last two corners and draws
        # a crossed "hourglass" instead of a box.
        shape.draw_quad(fitz.Quad(pts[0], pts[1], pts[3], pts[2]))
    elif path.kind == "c":
        shape.draw_bezier(pts[0], pts[1], pts[2], pts[3])
    else:
        return
    shape.finish(
        color=path.stroke_color,
        fill=path.fill_color,
        width=path.stroke_width,
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

    def _cluster_frame(self, paths: list[VectorPath]) -> tuple[float, float, float]:
        """The isolated-canvas geometry shared by render_vector_cluster and
        pixel_to_page_bbox: `(x0, y0, padding)` where `x0`/`y0` are the
        cluster's own bbox origin (page space) and `padding` is the margin
        added around it (>= _MIN_CLUSTER_PADDING, or a member's own stroke
        width if wider) so edge strokes aren't clipped. A page-space point
        maps to canvas space via `(x - x0 + padding, y - y0 + padding)`."""
        x0, y0, _x1, _y1 = union_bbox([p.bbox for p in paths])
        padding = max(
            _MIN_CLUSTER_PADDING, max((p.stroke_width or 0.0) for p in paths)
        )
        return x0, y0, padding

    def cluster_frame_size(self, paths: list[VectorPath]) -> tuple[float, float]:
        """(width, height) in PDF points of the isolated canvas
        render_vector_cluster would build for `paths` (bbox plus
        _cluster_frame's padding) -- lets a caller (RenderOCR.ocr_cluster)
        pick a dpi that keeps the rendered pixel size above some minimum
        without duplicating _cluster_frame's own padding math."""
        x0, y0, x1, y1 = union_bbox([p.bbox for p in paths])
        _frame_x0, _frame_y0, padding = self._cluster_frame(paths)
        return (x1 - x0) + 2 * padding, (y1 - y0) + 2 * padding

    def render_vector_cluster(
        self, paths: list[VectorPath], dpi: int
    ) -> "Image.Image":
        """High-resolution render of an isolated vector path cluster, used
        as OCR input. Builds a fresh single-page PyMuPDF document sized to
        the cluster's own bbox (plus padding for stroke overflow, via
        _cluster_frame), redraws each path onto it with its real
        stroke/fill styling via fitz.Shape, then rasterizes at `dpi` --
        reusing PyMuPDF's own rendering rather than re-implementing
        curve/fill rasterization by hand."""
        if not paths:
            raise ValueError("render_vector_cluster requires at least one path")

        x0, y0, x1, y1 = union_bbox([p.bbox for p in paths])
        frame_x0, frame_y0, padding = self._cluster_frame(paths)
        dx, dy = padding - frame_x0, padding - frame_y0
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

    def pixel_to_page_bbox(
        self,
        paths: list[VectorPath],
        dpi: int,
        pixel_points: list[tuple[float, float]],
    ) -> tuple[float, float, float, float]:
        """Inverts render_vector_cluster's own (dx, dy, zoom) transform to
        map a set of pixel-space points (e.g. Paddle's detected polygon
        corners, from a render of this exact `paths`/`dpi` pair) back into
        PDF page space, returning their bbox. Used by RenderOCR.ocr_cluster
        to compute a TextVectorResult's ocr_bbox."""
        x0, y0, padding = self._cluster_frame(paths)
        zoom = dpi / 72.0
        xs = [px / zoom - padding + x0 for px, _py in pixel_points]
        ys = [py / zoom - padding + y0 for _px, py in pixel_points]
        return (min(xs), min(ys), max(xs), max(ys))

    def render_page_paths(
        self, paths: list[VectorPath], page_meta: PageMeta, dpi: int
    ) -> "Image.Image":
        """Renders a whole page-sized image containing only `paths` (no
        native text, no dropped/drawing content) -- used as FAST's own
        detection input: `pipeline.py`'s fast_text_detect stage draws every
        surviving vector-classification cluster's paths onto one shared
        page-sized canvas via this method, so FAST scans the whole page in
        a single pass instead of many small per-cluster collages. Stays in
        unrotated MediaBox space like every other stage (no rotation
        applied), consistent with `render_vector_cluster`."""
        doc = fitz.open()
        try:
            page = doc.new_page(width=page_meta.width, height=page_meta.height)
            shape = page.new_shape()
            for path in paths:
                _draw_vector_path(shape, path)
            shape.commit()

            zoom = dpi / 72.0
            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            image.load()
            return image
        finally:
            doc.close()

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

            base_font = fitz.Font("helv")
            font_span = base_font.ascender - base_font.descender

            if native_words:
                for word in native_words:
                    if not word.text.strip():
                        continue
                    if word.origin is not None:
                        origin = word.origin
                    else:
                        # No real origin recorded -- approximate the
                        # baseline from the bbox's top edge using the
                        # base14 font's own ascender metric (a font's
                        # em-square is taller than its bbox, and the
                        # baseline sits below the top edge by roughly
                        # ascender * fontsize, not at the bbox's bottom
                        # edge outright).
                        x0, y0, _x1, _y1 = word.bbox
                        origin = (x0, y0 + base_font.ascender * max(word.font_size, 1.0))
                    # Rotate around the word's own bbox center, not the
                    # baseline origin -- morph's fixpoint is what stays
                    # fixed under the transform, so using origin as the
                    # fixpoint swings the text around its own left edge
                    # instead of turning in place.
                    bx0, by0, bx1, by1 = word.bbox
                    center = fitz.Point((bx0 + bx1) / 2, (by0 + by1) / 2)
                    page.insert_text(
                        origin, word.text,
                        fontsize=max(word.font_size, 1.0),
                        color=_text_color(word.color),
                        rotate=0,
                        morph=(center, fitz.Matrix(1, 1).prerotate(word.angle)),
                    )

            def _place_text(text: str, bbox: tuple[float, float, float, float], rotation: float) -> None:
                if not text.strip():
                    return
                x0, y0, x1, y1 = bbox
                # A font's em-square (fontsize) is taller than the
                # rendered glyph bbox by ascender - descender (both
                # em-fractions); recover fontsize from the bbox height
                # via that ratio, then place the baseline
                # ascender*fontsize below the bbox's top edge, rather
                # than treating the bbox height as the fontsize and the
                # bbox's bottom edge as the baseline outright (both
                # wrong -- the old behavior overshot fontsize and
                # dropped the baseline below where the glyphs actually
                # sit).
                fontsize = max((y1 - y0) / font_span, 1.0)
                # Height alone doesn't guarantee the text actually fits
                # within its own bbox's width (e.g. a long OCR'd string
                # in a narrow cluster/word bbox) -- shrink fontsize
                # further, uniformly, so the rendered text_length never
                # exceeds the bbox width it was read from.
                bbox_width = x1 - x0
                if bbox_width > 0:
                    text_width = base_font.text_length(text, fontsize=fontsize)
                    if text_width > bbox_width:
                        fontsize = max(fontsize * bbox_width / text_width, 1.0)
                origin = (x0, y0 + base_font.ascender * fontsize)
                # Rotate around the bbox's own center, not the baseline
                # origin -- morph's fixpoint is what stays fixed under the
                # transform, so using origin as the fixpoint swings the
                # text around its own left edge instead of turning in
                # place around the bbox it was placed into.
                center = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
                page.insert_text(
                    origin, text,
                    fontsize=fontsize,
                    rotate=0,
                    morph=(center, fitz.Matrix(1, 1).prerotate(rotation)),
                )

            if ocr_results:
                for result in ocr_results:
                    if result.words:
                        # Word-level boxes available (e.g. Tesseract) --
                        # place/scale each word into its own detected
                        # bbox instead of stretching one string across
                        # the whole cluster bbox.
                        for word in result.words:
                            _place_text(word.text, word.bbox, result.rotation_used)
                    else:
                        _place_text(result.text, result.bbox, result.rotation_used)

            pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png")))
            image.load()
            return image
        finally:
            doc.close()
