"""Renderer: rendering helpers, not a pipeline stage.

Turns pipeline data (vector paths, clusters, raster regions) into pixels
for OCR input. render_vector_cluster is implemented (isolates a cluster
of VectorPaths onto their own small PyMuPDF page and rasterizes it, so
OCR gets the PDF's own vector rendering rather than a hand-rolled
rasterizer); render_raster_region is still a stub since the Raster stage
itself isn't implemented yet. Reconstructing pipeline output back into a
PDF for evaluation is a separate concern -- see evaluation.py, the
pipeline's actual final stage.
"""
from __future__ import annotations

import io

import pymupdf as fitz
from PIL import Image

from rastervec.geometry import union_bbox
from rastervec.models import Page, RasterImage, VectorPath

_DEFAULT_PATH_COLOR = "#111827"

# Minimum padding (in PDF points) added around a cluster's bbox before
# rendering, so thin strokes right at the bbox edge aren't clipped.
_MIN_CLUSTER_PADDING = 4.0


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
                if path.stroke_color is None and path.fill_color is None:
                    # Shape.finish() always emits a stroke operator when
                    # `fill` is None, even with `color=None` too -- it
                    # falls back to the current (default black) graphics
                    # state color rather than staying invisible. A path
                    # with neither color set was never meant to render at
                    # all, so skip it outright instead of relying on
                    # finish() to no-op.
                    continue
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
                    continue
                shape.finish(
                    color=path.stroke_color,
                    fill=path.fill_color,
                    width=path.stroke_width or 1.0,
                    dashes=path.dashes,
                    closePath=True if path.closed is None else bool(path.closed),
                )
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
