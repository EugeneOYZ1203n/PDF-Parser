"""Helpers behind `gt_fidelity_showcase.ipynb` -- how faithfully the
vector -> raster ground truth (`visible_geometry.visible_geometry_for_page`) reproduces the page it was
taken from, as SSIM / PSNR between the original page render and a render of
the GT -- next to the old paint-blind GT (`_all_geometry_for_page` below,
every piece of every path) so the visibility filter's effect is visible.

Nothing re-implements a metric or a renderer: GT annotations go through
`vector_metrics.geometry_entries_from_annotations` -> `flatten_entries` ->
`vector_render.render_gray` (the benchmark's own SSIM/PSNR renders: real
gray, real width, white paper), and the scores are
`vector_metrics.image_similarity`. The original is rendered at the same dpi
in unrotated page space (the page's `/Rotate` cleared on an in-memory copy),
so every image shares one pixel frame.

Ink with no vector behind it (font text, embedded images) can never be in
the GT; with `mask_text` / `mask_images` their bboxes are painted white in
every image before scoring, so the score measures the drawing only.

Memory: one page at a time; at A1 / 150 dpi each gray image is ~17 MB and
an SSIM call peaks around 0.5 GB; the GT build itself (300 dpi) peaks around
1.3 GB. GT renders are cached per (pdf, page, dpi) as `.npz` under the run
folder's `cache/`, so re-running display cells never rebuilds them.
"""

from __future__ import annotations

import csv
import hashlib
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import pymupdf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from raster_parser.commons.paths import output_dir  # noqa: E402
from raster_parser.evaluation.evaluate import vector_render  # noqa: E402
from raster_parser.evaluation.evaluate.vector_metrics import (  # noqa: E402
    VectorMetricConfig,
    flatten_entries,
    geometry_entries_from_annotations,
    image_similarity,
)
from raster_parser.evaluation.labelling.visible_geometry import (
    pieces_for_vector,
    visible_geometry_for_page,
)  # noqa: E402
from raster_parser.phase_native_read.native_text import extract_native_text  # noqa: E402
from raster_parser.phase_native_read.reader import Reader  # noqa: E402
from raster_parser.phase_native_read.vector_extract import extract_vectors  # noqa: E402


def _all_geometry_for_page(pdf_path: str, page_index: int) -> list:
    """Every `Vector`'s pieces, visible or not -- the paint-blind GT before the visibility filter."""
    with Reader(pdf_path) as reader:
        vectors = extract_vectors(reader.get_page(page_index))
    return [piece for v in vectors for piece in pieces_for_vector(v)]


INK_THRESHOLD = 200  # gray below this = ink, for the diff map
DISPLAY_MAX_PX = 2000  # longest side of displayed / saved pictures


@dataclass
class PageFidelity:
    pdf: str
    page: int
    original: np.ndarray  # gray, masked
    gt_visible: np.ndarray
    gt_all: np.ndarray
    ssim_visible: float
    psnr_visible: float
    ssim_all: float
    psnr_all: float
    stats: dict = field(default_factory=dict)  # visible_geometry stats
    n_all: int = 0  # paint-blind pieces
    masked_boxes: int = 0
    seconds: float = 0.0

    @property
    def label(self) -> str:
        return f"{Path(self.pdf).stem} p{self.page}"


# --------------------------------------------------------------------------
# Renders
# --------------------------------------------------------------------------
def render_original_gray(pdf_path: str, page_index: int, dpi: float) -> np.ndarray:
    """(H, W) uint8 gray render in unrotated page space (CropBox size), the
    frame `vector_render.render_gray` draws in."""
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        page.set_rotation(0)  # in memory only; never saved
        zoom = dpi / 72.0
        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csGRAY, alpha=False
        )
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.stride)
        return img[:, : pix.width].copy()


def render_annotations_gray(
    annotations, page_meta, dpi: float, cfg: VectorMetricConfig
) -> np.ndarray:
    polys = flatten_entries(geometry_entries_from_annotations(annotations), cfg)
    return vector_render.render_gray(polys, page_meta, dpi)


def mask_boxes(pdf_path: str, page_index: int, *, text: bool, images: bool) -> list[tuple]:
    """Unrotated page-space bboxes of native words / embedded image placements."""
    boxes: list[tuple] = []
    with Reader(pdf_path) as reader:
        page = reader.get_page(page_index)
        if text:
            boxes += [tuple(t.bbox) for t in extract_native_text(page)]
        if images:
            boxes += [tuple(info["bbox"]) for info in page.pymupdf_page.get_image_info()]
    return boxes


def paint_white(img: np.ndarray, boxes: list[tuple], dpi: float, pad_px: int = 2) -> np.ndarray:
    out = img.copy()
    z = dpi / 72.0
    h, w = out.shape
    for x0, y0, x1, y1 in boxes:
        a, b = max(0, int(math.floor(x0 * z)) - pad_px), max(0, int(math.floor(y0 * z)) - pad_px)
        c, d = min(w, int(math.ceil(x1 * z)) + pad_px), min(h, int(math.ceil(y1 * z)) + pad_px)
        if c > a and d > b:
            out[b:d, a:c] = 255
    return out


def _fit(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Pad / crop by at most a pixel of rounding so both renders line up."""
    h, w = shape
    out = np.full((h, w), 255, np.uint8)
    hh, ww = min(h, img.shape[0]), min(w, img.shape[1])
    out[:hh, :ww] = img[:hh, :ww]
    return out


def _cache_key(pdf_path: str, page_index: int, dpi: float) -> str:
    p = Path(pdf_path)
    sig = f"{p.resolve()}|{p.stat().st_mtime_ns}|{page_index}|{dpi}"
    return f"{p.stem}_p{page_index}_{hashlib.md5(sig.encode()).hexdigest()[:10]}"


def _gt_renders(
    pdf_path: str, page_index: int, dpi: float, cfg: VectorMetricConfig, cache_dir: "Path | None"
) -> tuple[np.ndarray, np.ndarray, dict, int]:
    cache = cache_dir / f"{_cache_key(pdf_path, page_index, dpi)}.npz" if cache_dir else None
    if cache is not None and cache.is_file():
        z = np.load(cache, allow_pickle=False)
        stats = {k[len("stat_") :]: int(z[k]) for k in z.files if k.startswith("stat_")}
        return z["visible"], z["all"], stats, int(z["n_all"])
    with Reader(pdf_path) as reader:
        meta = reader.get_page(page_index).meta
    geom = visible_geometry_for_page(pdf_path, page_index)
    stats = {k: v for k, v in geom.stats.items() if isinstance(v, int)}
    visible = render_annotations_gray(geom.annotations, meta, dpi, cfg)
    del geom
    every = _all_geometry_for_page(pdf_path, page_index)
    all_img = render_annotations_gray(every, meta, dpi, cfg)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            visible=visible,
            all=all_img,
            n_all=len(every),
            **{f"stat_{k}": v for k, v in stats.items()},
        )
    return visible, all_img, stats, len(every)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_page(
    pdf_path: str,
    page_index: int,
    *,
    dpi: float = 150.0,
    mask_text: bool = True,
    mask_images: bool = True,
    cache_dir: "Path | None" = None,
) -> PageFidelity:
    t0 = time.perf_counter()
    cfg = VectorMetricConfig(render_dpi=dpi)
    original = render_original_gray(pdf_path, page_index, dpi)
    gt_visible, gt_all, stats, n_all = _gt_renders(pdf_path, page_index, dpi, cfg, cache_dir)
    gt_visible, gt_all = _fit(gt_visible, original.shape), _fit(gt_all, original.shape)
    boxes = mask_boxes(pdf_path, page_index, text=mask_text, images=mask_images)
    if boxes:
        original, gt_visible, gt_all = (
            paint_white(i, boxes, dpi) for i in (original, gt_visible, gt_all)
        )
    vis = image_similarity(original, gt_visible, cfg)
    every = image_similarity(original, gt_all, cfg)
    return PageFidelity(
        pdf=str(pdf_path),
        page=page_index,
        original=original,
        gt_visible=gt_visible,
        gt_all=gt_all,
        ssim_visible=vis.ssim,
        psnr_visible=vis.psnr,
        ssim_all=every.ssim,
        psnr_all=every.psnr,
        stats=stats,
        n_all=n_all,
        masked_boxes=len(boxes),
        seconds=time.perf_counter() - t0,
    )


def page_indices(pdf_path: str, pages: "list[int] | None", max_pages: "int | None") -> list[int]:
    with pymupdf.open(pdf_path) as doc:
        n = doc.page_count
    chosen = [p for p in (pages if pages is not None else range(n)) if 0 <= p < n]
    return chosen[:max_pages] if max_pages else chosen


# --------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------
def diff_map(original: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """RGB: black = ink in both, red = original ink with no GT, blue = GT over paper."""
    o, g = original < INK_THRESHOLD, gt < INK_THRESHOLD
    out = np.full((*original.shape, 3), 255, np.uint8)
    out[o & g] = (0, 0, 0)
    out[o & ~g] = (220, 30, 30)
    out[~o & g] = (30, 80, 230)
    return out


def _shrink(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = DISPLAY_MAX_PX / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)


def page_figure(r: PageFidelity):
    panels = [
        (
            r.original,
            "original" + (f" ({r.masked_boxes} text/image boxes masked)" if r.masked_boxes else ""),
        ),
        (r.gt_visible, f"visible GT: SSIM {r.ssim_visible:.3f}, PSNR {r.psnr_visible:.1f} dB"),
        (r.gt_all, f"all-vectors GT: SSIM {r.ssim_all:.3f}, PSNR {r.psnr_all:.1f} dB"),
        (diff_map(r.original, r.gt_visible), "visible GT diff (red = no GT, blue = GT on paper)"),
    ]
    h, w = r.original.shape
    fig, axes = plt.subplots(1, 4, figsize=(22, 22 * h / (4 * w) + 1.2))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(
            _shrink(img),
            cmap="gray" if img.ndim == 2 else None,
            vmin=0,
            vmax=255,
            interpolation="none",
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(r.label, fontsize=11)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Tables + output
# --------------------------------------------------------------------------
COLUMNS = (
    "pdf",
    "page",
    "vectors",
    "paint_dropped",
    "pieces_all",
    "pieces_invisible",
    "pieces_cut",
    "short_runs",
    "gt_pieces_visible",
    "ssim_visible",
    "psnr_visible",
    "ssim_all",
    "psnr_all",
    "masked_boxes",
    "seconds",
)


def summary_rows(results: list[PageFidelity]) -> list[dict]:
    """One row per page, then a mean row per PDF (and overall when > 1 PDF)."""
    rows = [
        {
            "pdf": Path(r.pdf).name,
            "page": r.page,
            "vectors": r.stats.get("vectors"),
            "paint_dropped": r.stats.get("paint_dropped_vectors"),
            "pieces_all": r.n_all,
            "pieces_invisible": r.stats.get("invisible_pieces"),
            "pieces_cut": r.stats.get("cut_pieces"),
            "short_runs": r.stats.get("short_runs"),
            "gt_pieces_visible": r.stats.get("annotations"),
            "ssim_visible": r.ssim_visible,
            "psnr_visible": r.psnr_visible,
            "ssim_all": r.ssim_all,
            "psnr_all": r.psnr_all,
            "masked_boxes": r.masked_boxes,
            "seconds": round(r.seconds, 1),
        }
        for r in results
    ]
    by_pdf: dict[str, list[dict]] = {}
    for row in rows:
        by_pdf.setdefault(row["pdf"], []).append(row)
    groups = list(by_pdf.items()) + ([("ALL", rows)] if len(by_pdf) > 1 else [])
    means = []
    for name, rs in groups:
        mean = {"pdf": name, "page": "mean"}
        for c in ("ssim_visible", "psnr_visible", "ssim_all", "psnr_all"):
            mean[c] = float(np.mean([x[c] for x in rs]))
        means.append(mean)
    return rows + means


def markdown_table(rows: list[dict]) -> str:
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
        return str(v)

    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in COLUMNS) + " |" for r in rows]
    return "\n".join(lines)


def new_run_dir(root: "Path | None" = None) -> Path:
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        return root
    return output_dir("gt_fidelity_showcase", datetime.now().strftime("%Y%m%d_%H%M%S"))


def save_page(r: PageFidelity, run_dir: Path, fig=None) -> Path:
    fig = fig or page_figure(r)
    path = run_dir / "pages" / f"{Path(r.pdf).stem}_p{r.page}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    return path


def save_summary(rows: list[dict], run_dir: Path) -> Path:
    path = run_dir / "summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in COLUMNS})
    return path
