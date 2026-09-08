"""Notebook-only display plumbing for `pipeline_stage_visualization.ipynb`.

Not used by the real pipeline, and deliberately not re-exported through
`renderer/__init__.py` -- this module imports `matplotlib`, and every
pipeline module already does `from rastervec.renderer import ...` for
lightweight things (`render_vector_cluster`, `render_reconstructed_page`,
...), so folding this module into that package's own `__init__` would drag
matplotlib into every real pipeline run's import graph. Import it directly
(`from rastervec.renderer.notebook import ...`) instead -- the notebook
does this at the top; every stage module's own `render_<stage_name>`
function does it lazily, inside the function body, for the same reason.

The stage-specific part -- *what* to draw (which paths/polys/bboxes, what
color, what note) -- lives in each stage's own `render_<stage_name>`
function, next to that stage's code. This module only has the generic part
that's identical across every stage: turning that decision into pixels.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import pymupdf as fitz
from PIL import Image, ImageDraw

from rastervec.renderer._shapes import path_color_hex

if TYPE_CHECKING:
    from rastervec.models import VectorPath
    from rastervec.pipelines.result import PipelineResult

DEFAULT_PATH_COLOR = "#111827"


class RenderCategory(TypedDict, total=False):
    """One overlay category within a stage's `RenderResult`. `isolated`/
    `overlay` are a precomputed image pair (used when a category can't be
    expressed purely as paths/polys/bboxes over one shared color, e.g. a
    per-path custom coloring or a page reconstruction) -- when absent,
    `visualize()` builds them itself from `paths`/`polys`/`bboxes`."""

    name: str
    paths: "list[VectorPath]"
    path_color: str | None
    polys: list
    bboxes: list
    color: str
    width: int
    isolated: "Image.Image"
    overlay: "Image.Image"


@dataclass
class RenderResult:
    """One stage's notebook-visualization decision: the overlay
    categories to draw, plus an optional one-line note."""

    categories: "list[RenderCategory]" = field(default_factory=list)
    note: str | None = None


def page_setup(res: "PipelineResult", zoom: float) -> tuple["Image.Image", "fitz.Matrix"]:
    """Reopen `res`'s source page once: the rasterized original page at
    `zoom`, plus the rotation-baked display matrix (page/MediaBox space ->
    raster-pixel space) every `render_<stage_name>`/`draw_*` call below
    needs. `res.page.fitz_page` is None after a pipeline run (the pipeline
    closed its Reader), so this is the one place that reopens the PDF."""
    with res.open_page() as p:
        rotation_matrix = fitz.Matrix(p.fitz_page.rotation_matrix)
        pix = p.fitz_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        original = Image.open(io.BytesIO(pix.pil_tobytes(format="PNG"))).convert("RGB")
    return original, display_matrix(rotation_matrix, zoom)


def display_matrix(rotation_matrix: "fitz.Matrix", zoom: float) -> "fitz.Matrix":
    """page (unrotated MediaBox) space -> raster-pixel space, page rotation
    baked in -- the rule the deleted debug_app._get_display_matrix used."""
    return rotation_matrix * fitz.Matrix(zoom, zoom)


def _xf(x, y, matrix) -> tuple[float, float]:
    p = fitz.Point(x, y) * matrix
    return (p.x, p.y)


def draw_paths(img, paths, matrix, color=None, width: int = 2) -> None:
    """Polyline of each path's own points (curves as straight segments
    through control points). `color=None` -> each path in its own real PDF
    stroke/fill color."""
    d = ImageDraw.Draw(img)
    for p in paths:
        pts = [_xf(x, y, matrix) for x, y in p.points]
        if len(pts) < 2:
            continue
        c = color or path_color_hex(p)
        if p.kind in ("re", "qu"):
            d.polygon(pts, outline=c, width=width)
        else:
            d.line(pts, fill=c, width=width)


def draw_polys(img, polys, matrix, color: str = DEFAULT_PATH_COLOR, width: int = 2) -> None:
    d = ImageDraw.Draw(img)
    for poly in polys:
        pts = [_xf(x, y, matrix) for x, y in poly]
        if len(pts) >= 2:
            d.polygon(pts, outline=color, width=width)


def draw_bboxes(img, bboxes, matrix, color: str = DEFAULT_PATH_COLOR, width: int = 2) -> None:
    d = ImageDraw.Draw(img)
    for bb in bboxes:
        x0, y0 = _xf(bb[0], bb[1], matrix)
        x1, y1 = _xf(bb[2], bb[3], matrix)
        d.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], outline=color, width=width)


def blank_like(img) -> "Image.Image":
    return Image.new("RGB", img.size, "white")


def show_row(images, titles, height: float = 5.0) -> None:
    n = len(images)
    ar = images[0].height / images[0].width
    fig, axes = plt.subplots(1, n, figsize=(max(height / ar * n, 4.0), height))
    axes = [axes] if n == 1 else list(axes)
    for ax, im, t in zip(axes, images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def _paint(img, cat: "RenderCategory", matrix) -> None:
    if cat.get("paths"):
        draw_paths(img, cat["paths"], matrix, cat.get("path_color"), cat.get("width", 2))
    if cat.get("polys"):
        draw_polys(img, cat["polys"], matrix, cat.get("color", DEFAULT_PATH_COLOR), cat.get("width", 2))
    if cat.get("bboxes"):
        draw_bboxes(img, cat["bboxes"], matrix, cat.get("color", DEFAULT_PATH_COLOR), cat.get("width", 2))


def _iso_and_overlay(cat: "RenderCategory", original: "Image.Image", matrix):
    iso, ovl = cat.get("isolated"), cat.get("overlay")
    if iso is not None:
        iso = iso.convert("RGB")
    else:
        iso = blank_like(original)
        _paint(iso, cat, matrix)
    if ovl is not None:
        ovl = ovl.convert("RGB")
    else:
        ovl = original.copy()
        _paint(ovl, cat, matrix)
    return iso, ovl


def visualize(
    stage_key: str,
    result: "RenderResult",
    *,
    step_outputs: dict,
    original: "Image.Image",
    matrix,
) -> None:
    """Print `stage_key`'s status/note, show the original page, then an
    isolated/overlay image pair per `result.categories`."""
    out = step_outputs.get(stage_key)
    print(f"=== {stage_key} ===")
    if out is None:
        print(f"  (no step outcome for {stage_key!r} -- check STEP_NAMES / verbose run)")
        return
    if out.status == "error":
        print(f"  ERROR: {out.error}")
        return
    if result.note:
        print(" ", result.note)
    show_row([original], ["original page"])
    for cat in result.categories:
        iso, ovl = _iso_and_overlay(cat, original, matrix)
        show_row([iso, ovl], [f"{cat['name']} -- isolated", f"{cat['name']} -- overlay"])
    if not result.categories:
        print("  (no overlays for this stage)")
