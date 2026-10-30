"""Helpers behind `gt_problem_scan.ipynb` -- search PDFs for every case where
vector -> raster ground truth and the rendered page disagree: the problems
that made the old paint-blind GT wrong, the ones no vector GT can cover, and
the colour split's own blind spot.

The scan reuses the GT builder's parts (`raster_parser/evaluation/labelling/visible_geometry.py`:
render, the raster prelude's colour split, paint rule, dense visibility runs), in the
builder's order (image work first and freed, then vector extraction), so it
sees exactly what `visible_geometry_for_page` sees -- at the scan's own dpi
(150 by default: ~4x cheaper than the builder's 300, same categories).

Categories (`CATEGORIES`):

  paper_paint_near_ink  path whose stroke and fill both read as paper under the
                        split's ink rule (white, or a light fill -- which may still
                        be visibly tinted), lying within `NEAR_INK_PX` of real ink:
                        the old GT counted its outline (false GT); the visible GT
                        drops it, consistent with the training masks.
  paper_paint           other such paths, away from ink (harmless).
  light_fill            fill-only path lighter than ink but not white -- the old
                        DBSCAN split turned these into solid "ink" layers.
  clipped               path cut by its clip (`scissor` / non-rect `clips`) --
                        the old vote kept or dropped it whole.
  dashed                dashed path -- the old GT drew it solid.
  occluded              other paths with pieces cut or hidden by later paint. At
                        150 dpi the tips of tiny glyphs (< 1 px) can show up with
                        1-4 pieces -- resolution noise; ranked last by score.
  unbacked_text/_image/_other
                        ink with no vector behind it (native font text, embedded
                        images, anything else: shadings, annotations, Type3) --
                        no vector GT can exist; grouped into blobs.
  lost_dark_ink         dark pixels the colour split labels paper (its Otsu rule
                        can be pulled by a large light fill) -- these vanish
                        from the training masks *and* the GT.

Every finding is stored in unrotated page space (pt), the frame of
`get_drawings` / native text and of the overlays drawn here.
"""

from __future__ import annotations

import csv
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
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from raster_parser.commons.paths import output_dir  # noqa: E402
from raster_parser.evaluation.evaluate.liu_dori import is_dashed  # noqa: E402
from raster_parser.evaluation.labelling.visible_geometry import (
    pieces_for_vector as geometry_annotations_for_vector,
)  # noqa: E402
from raster_parser.evaluation.labelling.visible_geometry import (  # noqa: E402
    PAPER,
    _bezier,
    _compact_labels,
    _gc_paused,
    _pieces_px,
    color_layers,
    page_to_pixel_matrix,
    paint_reads_as_ink,
    _paper_rgb,
    render_page,
    rendered_ink,
    vector_paint_visible,
    visible_runs,
)
from raster_parser.phase_native_read.native_text import extract_native_text  # noqa: E402
from raster_parser.phase_native_read.reader import Reader  # noqa: E402
from raster_parser.phase_native_read.vector_extract import extract_vectors  # noqa: E402

DEFAULT_DPI = 150.0
NEAR_INK_PX = 2  # paper paint this close to ink was false GT before
LIGHT_FILL_MAX_LUMA = 0.97  # below this a non-ink fill is "light grey", above it "white"
CLIP_TOL_PT = 0.5
GT_MARGIN_PX = 2  # GT strokes drawn this much fatter when looking for unbacked ink
GROUP_PX = 3  # unbacked / lost ink blobs closer than this are one finding
MIN_COMPONENT_PX = 12  # smaller blobs ignored (antialiasing specks)
DARK_GRAY = 100
TEXT_OVERLAP = 0.3
DISPLAY_MAX_PX = 2000

CATEGORIES: dict[str, tuple[str, str]] = {  # key -> (colour, one-line meaning)
    "paper_paint_near_ink": (
        "#e11d48",
        "path the split reads as paper (white / light fill), next to ink (old GT: false stroke)",
    ),
    "paper_paint": ("#f59e0b", "path the split reads as paper, away from ink"),
    "light_fill": ("#eab308", "light non-white fill (old DBSCAN: blob layer)"),
    "clipped": ("#8b5cf6", "clipped path (old GT: kept/dropped whole)"),
    "dashed": ("#2563eb", "dashed path (old GT: drawn solid)"),
    "occluded": ("#06b6d4", "path cut / hidden by later paint"),
    "unbacked_text": ("#16a34a", "native text ink (no vector GT)"),
    "unbacked_image": ("#65a30d", "image ink (no vector GT)"),
    "unbacked_other": ("#0f766e", "other ink with no vector (shading / annot / Type3)"),
    "lost_dark_ink": ("#c026d3", "dark ink the colour split calls paper"),
}


@dataclass
class Finding:
    category: str
    bbox: tuple[float, float, float, float]  # unrotated page pt
    score: float  # ranking: px, pieces or area
    detail: str = ""


@dataclass
class PageScan:
    pdf: str
    page: int
    dpi: float
    findings: list[Finding] = field(default_factory=list)
    n_vectors: int = 0
    n_pieces: int = 0
    n_layers: int = 0
    otsu_l: "float | None" = None
    seconds: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{Path(self.pdf).stem} p{self.page}"

    def counts(self) -> dict[str, int]:
        out = {k: 0 for k in CATEGORIES}
        for f in self.findings:
            out[f.category] += 1
        return out


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------
def _px_box_to_page(x0, y0, x1, y1, inv) -> tuple[float, float, float, float]:
    pts = [pymupdf.Point(x, y) * inv for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    xs, ys = [p.x for p in pts], [p.y for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _overlap_frac(box, boxes: np.ndarray) -> float:
    """Largest fraction of `box`'s area covered by any one of `boxes` (N, 4)."""
    if not len(boxes):
        return 0.0
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    return float((ix * iy).max() / area)


def _luma(color) -> float:
    c = [float(x) for x in color]
    if len(c) == 1:
        return c[0]
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _is_clipped(v) -> bool:
    if v.clips:
        return True
    if v.scissor is None:
        return False
    r, s = v.rect, v.scissor
    return (
        r[0] < s[0] - CLIP_TOL_PT
        or r[1] < s[1] - CLIP_TOL_PT
        or r[2] > s[2] + CLIP_TOL_PT
        or r[3] > s[3] + CLIP_TOL_PT
    )


def _components(mask: np.ndarray, inv, group_px: int, min_px: int):
    """`(page bbox, ink px)` per blob of `mask`; blobs within `group_px` merge."""
    if not mask.any():
        return []
    grouped = (
        cv2.dilate(mask, np.ones((2 * group_px + 1, 2 * group_px + 1), np.uint8))
        if group_px
        else mask
    )
    n, lab, stats, _c = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    ink_px = np.bincount(lab[mask > 0], minlength=n)
    out = []
    for i in range(1, n):
        if ink_px[i] < min_px:
            continue
        x, y, w, h = stats[i, :4]
        out.append((_px_box_to_page(x, y, x + w, y + h, inv), int(ink_px[i])))
    return out


def _gt_mask(shape, pieces_px: np.ndarray, runs, widths_px: np.ndarray) -> np.ndarray:
    """Every visible run drawn at its width + 2 x `GT_MARGIN_PX`, batched per
    integer thickness."""
    mask = np.zeros(shape, np.uint8)
    if not len(runs):
        return mask
    t = runs.t0[:, None] + (runs.t1 - runs.t0)[:, None] * np.linspace(0, 1, 9)[None, :]
    p = pieces_px[runs.piece]
    pts = _bezier(np.repeat(p, 9, axis=0), t.reshape(-1)).reshape(len(runs), 9, 2)
    pts = np.round(pts).astype(np.int32)
    thick = np.maximum(1, np.round(widths_px[runs.piece] + 2 * GT_MARGIN_PX)).astype(int)
    for th in np.unique(thick):
        sel = pts[thick == th]
        cv2.polylines(mask, list(sel.reshape(-1, 9, 1, 2)), False, 255, int(th))
    return mask


# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------
def scan_page(pdf_path: str, page_index: int, dpi: float = DEFAULT_DPI) -> PageScan:
    """Every category of the module docstring for one page. The cyclic GC is
    paused throughout (as in the GT builder): the page's ~10^5-10^6 small
    objects otherwise cost ~40 % extra in collection passes."""
    with _gc_paused():
        return _scan_page(pdf_path, page_index, dpi)


def _scan_page(pdf_path: str, page_index: int, dpi: float) -> PageScan:
    res = PageScan(pdf=str(pdf_path), page=page_index, dpi=dpi)
    secs = res.seconds
    with Reader(str(pdf_path)) as reader:
        page = reader.get_page(page_index)
        t = time.perf_counter()
        rgb = render_page(page.pymupdf_page, dpi)
        matrix = page_to_pixel_matrix(page.pymupdf_page, dpi)
        inv = ~matrix
        scale = float(math.hypot(matrix.a, matrix.b))
        layers = color_layers(rgb)
        res.n_layers, res.otsu_l = layers.n_layers, layers.otsu_l
        labels = _compact_labels(layers.labels, layers.n_layers)
        paper = int(np.iinfo(labels.dtype).max)
        fringe = rendered_ink(rgb, _paper_rgb(rgb, layers.labels == PAPER))
        dark_lost = (
            (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) < DARK_GRAY) & (layers.labels == PAPER)
        ).view(np.uint8)
        del rgb, layers
        secs["render+layers"] = time.perf_counter() - t

        t = time.perf_counter()
        vectors = extract_vectors(page)
        words = np.array(
            [w.bbox for w in extract_native_text(page)] or np.zeros((0, 4)), float
        ).reshape(-1, 4)
        images = np.array(
            [i["bbox"] for i in page.pymupdf_page.get_image_info()] or np.zeros((0, 4)), float
        ).reshape(-1, 4)
        secs["extract"] = time.perf_counter() - t

    t = time.perf_counter()
    res.n_vectors = len(vectors)
    anns, owner, visible_paint = [], [], []
    for vi, v in enumerate(vectors):
        ok = vector_paint_visible(v, res.otsu_l)
        visible_paint.append(ok)
        for a in geometry_annotations_for_vector(v):
            if len(a.points) == (2 if a.kind == "l" else 4):
                anns.append(a)
                owner.append(vi)
    owner_a = np.array(owner, np.int64)
    pieces = _pieces_px(anns, matrix) if anns else np.zeros((0, 4, 2))
    widths = np.array([(a.width or 0.0) * scale for a in anns], float)
    res.n_pieces = len(anns)
    vis_paint = np.array(visible_paint, bool)
    piece_vis_paint = vis_paint[owner_a] if len(owner_a) else np.zeros(0, bool)

    # visibility of the pieces the GT builder keeps
    keep_idx = np.flatnonzero(piece_vis_paint)
    runs, _st = visible_runs(pieces[keep_idx], labels, res.n_layers, paper, extra_ink=fringe)
    runs.piece = keep_idx[runs.piece]
    n_v = len(vectors)
    has_run = np.zeros(len(anns), bool)
    has_run[runs.piece] = True
    cut_piece = np.zeros(len(anns), bool)
    cut_piece[runs.piece[~runs.full]] = True
    hidden_piece = piece_vis_paint & ~has_run
    pieces_per_v = np.bincount(owner_a, minlength=n_v)
    cut_per_v = np.bincount(owner_a[cut_piece], minlength=n_v)
    hidden_per_v = np.bincount(owner_a[hidden_piece], minlength=n_v)

    # paper paint: does it touch ink?
    drop_idx = np.flatnonzero(~piece_vis_paint)
    near_v = np.zeros(n_v, bool)
    if len(drop_idx):
        near, _ = visible_runs(
            pieces[drop_idx], labels, res.n_layers, paper, radius=NEAR_INK_PX, min_px=0.0
        )
        near_v[owner_a[drop_idx[near.piece]]] = True
    secs["visibility"] = time.perf_counter() - t

    t = time.perf_counter()
    for vi, v in enumerate(vectors):
        box = tuple(float(c) for c in v.rect)
        n_p = int(pieces_per_v[vi])
        if not vis_paint[vi]:
            cat = "paper_paint_near_ink" if near_v[vi] else "paper_paint"
            paint = v.fill if v.fill is not None else v.color
            res.findings.append(
                Finding(
                    cat,
                    box,
                    (box[2] - box[0]) * (box[3] - box[1]),
                    f"seq {v.seqno} type {v.type} paint {paint} ({n_p} pieces)",
                )
            )
        if (
            v.type == "f"
            and v.fill is not None
            and not paint_reads_as_ink(v.fill, res.otsu_l)
            and _luma(v.fill) < LIGHT_FILL_MAX_LUMA
        ):
            res.findings.append(
                Finding(
                    "light_fill",
                    box,
                    (box[2] - box[0]) * (box[3] - box[1]),
                    f"seq {v.seqno} fill {tuple(round(c, 3) for c in v.fill)}",
                )
            )
        if not vis_paint[vi]:
            continue
        cut, hidden = int(cut_per_v[vi]), int(hidden_per_v[vi])
        dashed = is_dashed(v.dashes)
        clipped = (cut or hidden) and _is_clipped(v)
        detail = f"seq {v.seqno}: {cut} piece(s) cut, {hidden} hidden"
        if dashed:
            res.findings.append(
                Finding("dashed", box, cut + hidden, f"seq {v.seqno} dashes {v.dashes}")
            )
        if clipped:
            res.findings.append(Finding("clipped", box, cut + hidden, detail))
        elif (cut or hidden) and not dashed:
            res.findings.append(Finding("occluded", box, cut + hidden, detail))

    # ink with no vector behind it
    ink = (labels != paper).view(np.uint8)
    unbacked = ink & ~_gt_mask(labels.shape, pieces, runs, widths).astype(bool)
    for box, px in _components(unbacked.astype(np.uint8), inv, GROUP_PX, MIN_COMPONENT_PX):
        if _overlap_frac(box, words) >= TEXT_OVERLAP:
            cat = "unbacked_text"
        elif _overlap_frac(box, images) >= TEXT_OVERLAP:
            cat = "unbacked_image"
        else:
            cat = "unbacked_other"
        res.findings.append(Finding(cat, box, px, f"{px} px"))
    for box, px in _components(dark_lost, inv, GROUP_PX, MIN_COMPONENT_PX):
        res.findings.append(
            Finding("lost_dark_ink", box, px, f"{px} px darker than {DARK_GRAY} on paper")
        )
    secs["findings"] = time.perf_counter() - t
    return res


def page_indices(pdf_path: str, pages: "list[int] | None", max_pages: "int | None") -> list[int]:
    with pymupdf.open(pdf_path) as doc:
        n = doc.page_count
    chosen = [p for p in (pages if pages is not None else range(n)) if 0 <= p < n]
    return chosen[:max_pages] if max_pages else chosen


def one_line(s: PageScan) -> str:
    c = {k: v for k, v in s.counts().items() if v}
    t = sum(s.seconds.values())
    return (
        f"{s.label}: {s.n_vectors} paths, {s.n_layers} layer(s), otsu "
        f"{'none' if s.otsu_l is None else round(s.otsu_l)} | "
        + (", ".join(f"{k} {v}" for k, v in c.items()) or "no findings")
        + f"  ({t:.1f}s)"
    )


# --------------------------------------------------------------------------
# Tables + output
# --------------------------------------------------------------------------
def summary_rows(scans: list[PageScan]) -> list[dict]:
    rows = []
    for s in scans:
        rows.append(
            {
                "pdf": Path(s.pdf).name,
                "page": s.page,
                "paths": s.n_vectors,
                "pieces": s.n_pieces,
                "layers": s.n_layers,
                "otsu_l": None if s.otsu_l is None else round(s.otsu_l, 1),
                **s.counts(),
                "seconds": round(sum(s.seconds.values()), 1),
            }
        )
    by_pdf: dict[str, list[dict]] = {}
    for r in rows:
        by_pdf.setdefault(r["pdf"], []).append(r)
    groups = list(by_pdf.items()) + ([("ALL", rows)] if len(by_pdf) > 1 else [])
    totals = []
    for name, rs in groups:
        tot = {"pdf": name, "page": "total"}
        for k in ("paths", "pieces", *CATEGORIES, "seconds"):
            tot[k] = round(sum(r[k] for r in rs), 1) if k == "seconds" else sum(r[k] for r in rs)
        totals.append(tot)
    return rows + totals


def findings_rows(scans: list[PageScan]) -> list[dict]:
    return [
        {
            "pdf": Path(s.pdf).name,
            "page": s.page,
            "category": f.category,
            "x0": round(f.bbox[0], 2),
            "y0": round(f.bbox[1], 2),
            "x1": round(f.bbox[2], 2),
            "y1": round(f.bbox[3], 2),
            "score": round(f.score, 2),
            "detail": f.detail,
        }
        for s in scans
        for f in s.findings
    ]


SUMMARY_COLUMNS = ("pdf", "page", "paths", "pieces", "layers", "otsu_l", *CATEGORIES, "seconds")
FINDING_COLUMNS = ("pdf", "page", "category", "x0", "y0", "x1", "y1", "score", "detail")


def markdown_table(rows: list[dict], columns=SUMMARY_COLUMNS) -> str:
    def fmt(v):
        return "" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))

    head = [c.replace("_", " ") for c in columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r.get(c)) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


def save_csv(rows: list[dict], path: Path, columns) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    return path


def new_run_dir(root: "Path | None" = None) -> Path:
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        return root
    return output_dir("gt_problem_scan", datetime.now().strftime("%Y%m%d_%H%M%S"))


# --------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------
def _render_unrotated(pdf_path: str, page_index: int, zoom: float, clip=None) -> np.ndarray:
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        page.set_rotation(0)  # in memory only
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip, alpha=False)
        return (
            np.frombuffer(pix.samples, np.uint8)
            .reshape(pix.height, pix.width, pix.n)[:, :, :3]
            .copy()
        )


def page_overlay(s: PageScan, categories=None):
    """The unrotated page with every finding outlined in its category colour."""
    cats = [c for c in (categories or CATEGORIES) if c in CATEGORIES]
    with pymupdf.open(s.pdf) as doc:
        page = doc[s.page]
        page.set_rotation(0)
        w, h = page.rect.width, page.rect.height
    zoom = min(DISPLAY_MAX_PX / max(w, h), 2.0)
    img = _render_unrotated(s.pdf, s.page, zoom)
    fig, ax = plt.subplots(figsize=(14, 14 * h / w + 0.6))
    ax.imshow(img)
    for f in s.findings:
        if f.category not in cats:
            continue
        x0, y0, x1, y1 = (c * zoom for c in f.bbox)
        ax.add_patch(
            Rectangle(
                (x0, y0),
                max(x1 - x0, 1),
                max(y1 - y0, 1),
                fill=False,
                edgecolor=CATEGORIES[f.category][0],
                linewidth=1.0,
            )
        )
    counts = s.counts()
    ax.legend(
        handles=[
            Patch(edgecolor=CATEGORIES[c][0], facecolor="none", label=f"{c} ({counts[c]})")
            for c in cats
            if counts[c]
        ],
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
    )
    ax.set_title(one_line(s), fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    return fig


def gallery(scans: list[PageScan], category: str, n: int = 12, pad_pt: float = 12.0, cols: int = 4):
    """Up to `n` crops of the highest-scoring findings of one category across
    every scanned page, finding outlined; None when there are none."""
    items = sorted(
        ((f, s) for s in scans for f in s.findings if f.category == category),
        key=lambda fs: -fs[0].score,
    )[:n]
    if not items:
        return None
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (f, s) in zip(axes.ravel(), items):
        x0, y0, x1, y1 = f.bbox
        clip = pymupdf.Rect(x0 - pad_pt, y0 - pad_pt, x1 + pad_pt, y1 + pad_pt)
        side = max(clip.width, clip.height, 1.0)
        zoom = min(400 / side, 8.0)
        with pymupdf.open(s.pdf) as doc:
            page = doc[s.page]
            page.set_rotation(0)
            clip &= page.rect
        if clip.is_empty:
            continue
        img = _render_unrotated(s.pdf, s.page, zoom, clip=clip)
        ax.imshow(img)
        ax.add_patch(
            Rectangle(
                ((x0 - clip.x0) * zoom, (y0 - clip.y0) * zoom),
                max((x1 - x0) * zoom, 1),
                max((y1 - y0) * zoom, 1),
                fill=False,
                edgecolor=CATEGORIES[category][0],
                lw=1.5,
            )
        )
        ax.set_title(f"{s.label} @ ({x0:.0f}, {y0:.0f})\n{f.detail[:60]}", fontsize=7)
    fig.suptitle(f"{category} -- {CATEGORIES[category][1]}", fontsize=11)
    fig.tight_layout()
    return fig


def save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
