"""Vector-geometry metrics for a raster (P2) run.

2 GT vector types (`VECTOR_TYPES`): `vector_to_raster` (`GeometryAnnotation`
entries with `source=="auto"`), `original_raster` (`source=="manual"`).
Predictions are the vectors of a pipeline run on the page's rasterised PDF
(P2's output -- that page has no native vectors). The page is scored against
the **union** of both GT types (that is what P2 sees); D_v is also reported
per type. `original_vector` is a `metrics.TEXT_TYPES` text type only, never
scored here. Full description: `docs/RasterToVectorEvaluation.md`.

- **Line detection** (`liu_dori.py`): the Liu & Dori (1997) vector-level
  protocol -- D_v / F_v / VRI, every per-overlap quality factor, per-line
  basic / fragmentation quality, fragmentation histograms, text/solid/type
  splits.
- **Pixel level** (`liu_dori.pixel_stats`): the same paper's D_p / F_p /
  PRI on ink masks of GT and predictions rendered at their floored widths
  (`vector_render.render_ink_pair`).
- **Image** (`image_similarity`): SSIM + PSNR between grayscale renders of GT
  and predictions at their real widths (`vector_render.py` renders; this
  module only compares arrays).
- **Length** (`length_stats`): sum of raw flattened segment lengths, GT vs
  predicted; ratio pred / GT -- a global sanity check.
- **Direction** (`direction_stats`): length-weighted histogram of raw
  flattened segment angles mod 180 deg (`direction_bins` bins); score =
  intersection of the two normalised histograms (1 = identical) -- a global
  sanity check.

Length/direction/image work off `flatten_entries`: each `GeometryEntry`
becomes points (`"l"` 2, `"c"` `curve_segments + 1` equal-t samples;
`re`/`qu` were already split into 4 lines upstream), and consecutive
touching entries of one group (one source `Vector`) are joined into one
polyline.

Pure -- no pipeline/label_schema/fitz import; callers (`adapters.py`) build
`GeometryEntry` lists via `geometry_entries_from_vector`/
`geometry_entries_from_annotations` (duck-typed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rastervec.Evaluation.Evaluate import liu_dori
from rastervec.Evaluation.Evaluate.liu_dori import LiuDoriConfig, LiuDoriResult, PixelStats
from rastervec.Evaluation.Evaluate.metrics import Ratio

VECTOR_TYPES: tuple[str, ...] = ("vector_to_raster", "original_raster")

Point = tuple[float, float]

_TOUCH_EPS = 1e-6


@dataclass(frozen=True)
class VectorMetricConfig:
    curve_segments: int = 10
    direction_bins: int = 36
    render_dpi: float = 150.0
    psnr_cap_db: float = 100.0
    # Width (pt) a fill-only entry (no stroke width) is rendered with.
    default_width_pt: float = 0.5
    liu_dori: LiuDoriConfig = LiuDoriConfig()
    # Used when a page's rasterised-PDF dpi is unknown.
    default_dpi: float = 300.0


# --------------------------------------------------------------------------
# GeometryEntry -- unifies GT GeometryAnnotations and predicted Vector items
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GeometryEntry:
    kind: str  # "l" | "c"
    points: tuple[Point, ...]
    color: "tuple[float, ...] | None" = None
    fill: "tuple[float, ...] | None" = None
    width: "float | None" = None
    dashes: "str | None" = None
    # Entries of one group (one source Vector / annotation) may be joined
    # into one polyline when consecutive ones touch.
    group: int = -1
    vector_type: str = ""

    @property
    def endpoints(self) -> tuple[Point, Point]:
        return (self.points[0], self.points[-1])


def geometry_entries_from_annotations(anns: list, *, vector_type: str = "") -> list[GeometryEntry]:
    """`anns`: `list[GeometryAnnotation]` (duck-typed); each annotation its
    own group."""
    return [
        GeometryEntry(
            kind=a.kind, points=tuple(tuple(p) for p in a.points), color=a.color, fill=a.fill,
            width=a.width, dashes=getattr(a, "dashes", None), group=i, vector_type=vector_type,
        )
        for i, a in enumerate(anns)
    ]


def geometry_entries_from_vector(v, *, group: int = -1) -> list[GeometryEntry]:
    """Decomposes one `Vector` (duck-typed) into `GeometryEntry` "l"/"c"
    primitives -- "re"/"qu" expand to their 4 edges, matching
    `label_schema.geometry_annotations_for_vector`'s decomposition."""
    from rastervec.commons.helpers.geometry import item_points

    common = dict(color=v.color, fill=v.fill, width=v.width, dashes=getattr(v, "dashes", None), group=group)
    out: list[GeometryEntry] = []
    for item in v.items:
        kind = item[0]
        pts = [tuple(p) for p in item_points(item)]
        if kind in ("l", "c"):
            out.append(GeometryEntry(kind=kind, points=tuple(pts), **common))
        elif kind in ("re", "qu"):
            if kind == "re":
                (x0, y0), (x1, y1) = pts
                corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            else:
                corners = pts
            for p1, p2 in zip(corners, corners[1:] + corners[:1]):
                out.append(GeometryEntry(kind="l", points=(p1, p2), **common))
    return out


def geometry_entries_from_vectors(vectors: list) -> list[GeometryEntry]:
    """Every `Vector` of a run, each its own group."""
    out: list[GeometryEntry] = []
    for i, v in enumerate(vectors):
        out.extend(geometry_entries_from_vector(v, group=i))
    return out


# --------------------------------------------------------------------------
# Flattening
# --------------------------------------------------------------------------
@dataclass
class FlatPolyline:
    points: np.ndarray  # (N, 2) page space, N >= 2
    gray: float  # 0 black .. 1 white
    width: float
    vector_type: str = ""


def _cubic_points(pts, n: int) -> np.ndarray:
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in pts)
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    u = 1.0 - t
    return u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3


def _entry_points(e: GeometryEntry, n_curve: int) -> np.ndarray:
    if e.kind == "c" and len(e.points) == 4:
        return _cubic_points(e.points, n_curve)
    return np.asarray(e.points, dtype=float).reshape(-1, 2)


def _gray(e: GeometryEntry) -> float:
    c = e.color if e.color is not None else e.fill
    if not c:
        return 0.0
    if len(c) >= 3:
        return float(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])
    return float(c[0])


def flatten_entries(entries: list[GeometryEntry], cfg: "VectorMetricConfig | None" = None) -> list[FlatPolyline]:
    """Entries -> polylines. Consecutive entries of one group with the same
    paint are joined when the previous end touches the next start."""
    cfg = cfg or VectorMetricConfig()
    out: list[FlatPolyline] = []
    cur: "list[np.ndarray] | None" = None
    cur_key = None
    for e in entries:
        pts = _entry_points(e, cfg.curve_segments)
        if len(pts) < 2:
            continue
        width = e.width if e.width is not None else cfg.default_width_pt
        key = (e.group, _gray(e), width, e.vector_type)
        if (
            cur is not None and key == cur_key and e.group >= 0
            and float(np.abs(cur[-1][-1] - pts[0]).max()) <= _TOUCH_EPS
        ):
            cur.append(pts[1:])
            continue
        if cur is not None:
            out.append(FlatPolyline(np.concatenate(cur), cur_key[1], cur_key[2], cur_key[3]))
        cur, cur_key = [pts], key
    if cur is not None:
        out.append(FlatPolyline(np.concatenate(cur), cur_key[1], cur_key[2], cur_key[3]))
    return out


def _segments(polys: list[FlatPolyline]) -> np.ndarray:
    """(M, 2) segment vectors of every polyline."""
    if not polys:
        return np.zeros((0, 2))
    return np.concatenate([np.diff(p.points, axis=0) for p in polys])


# --------------------------------------------------------------------------
# Length + direction (raw flattened geometry)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LengthStats:
    gt_length: float
    pred_length: float

    @property
    def ratio(self) -> Ratio:
        return Ratio(self.pred_length, self.gt_length)


def length_stats(gt: list[FlatPolyline], pred: list[FlatPolyline]) -> LengthStats:
    def total(polys):
        s = _segments(polys)
        return float(np.hypot(s[:, 0], s[:, 1]).sum()) if len(s) else 0.0

    return LengthStats(gt_length=total(gt), pred_length=total(pred))


def direction_histogram(polys: list[FlatPolyline], bins: int) -> np.ndarray:
    """Length-weighted histogram of segment angles in [0, 180) deg,
    unnormalised."""
    s = _segments(polys)
    if not len(s):
        return np.zeros(bins)
    length = np.hypot(s[:, 0], s[:, 1])
    angle = np.degrees(np.arctan2(s[:, 1], s[:, 0])) % 180.0
    idx = np.minimum((angle / (180.0 / bins)).astype(int), bins - 1)
    return np.bincount(idx, weights=length, minlength=bins).astype(float)


def histogram_intersection(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = float(a.sum()), float(b.sum())
    if sa <= 0 or sb <= 0:
        return math.nan
    return float(np.minimum(a / sa, b / sb).sum())


@dataclass(frozen=True)
class DirectionStats:
    gt_hist: tuple[float, ...]
    pred_hist: tuple[float, ...]

    @property
    def intersection(self) -> float:
        return histogram_intersection(np.asarray(self.gt_hist), np.asarray(self.pred_hist))


def direction_stats(gt: list[FlatPolyline], pred: list[FlatPolyline], bins: int) -> DirectionStats:
    return DirectionStats(
        gt_hist=tuple(direction_histogram(gt, bins)),
        pred_hist=tuple(direction_histogram(pred, bins)),
    )


# --------------------------------------------------------------------------
# Image similarity (renders come from vector_render.py)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ImageSimilarityStats:
    ssim: float
    mse: float
    n_pixels: int
    psnr_cap_db: float = 100.0

    @property
    def psnr(self) -> float:
        return _psnr(self.mse, self.psnr_cap_db)


_IMAGE_NA = ImageSimilarityStats(ssim=math.nan, mse=math.nan, n_pixels=0)


def _psnr(mse: float, cap: float) -> float:
    if math.isnan(mse):
        return math.nan
    if mse <= 0:
        return cap
    return min(cap, 10.0 * math.log10(255.0 ** 2 / mse))


def image_similarity(
    gt_img: np.ndarray, pred_img: np.ndarray, cfg: "VectorMetricConfig | None" = None,
) -> ImageSimilarityStats:
    """`gt_img`/`pred_img`: same-shape uint8 grayscale renders."""
    from skimage.metrics import structural_similarity

    cfg = cfg or VectorMetricConfig()
    if gt_img.shape != pred_img.shape:
        raise ValueError(f"render shapes differ: {gt_img.shape} vs {pred_img.shape}")
    a = gt_img.astype(np.float64)
    b = pred_img.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    ssim = float(structural_similarity(
        a, b, data_range=255.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
    ))
    return ImageSimilarityStats(ssim=ssim, mse=mse, n_pixels=int(a.size), psnr_cap_db=cfg.psnr_cap_db)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
@dataclass
class VectorMetricSuiteResult:
    label_counts: dict[str, int]
    liu_dori: LiuDoriResult
    pixel: PixelStats
    image: ImageSimilarityStats
    length: LengthStats
    direction: DirectionStats


def evaluate_vector_metrics(
    gt_by_type: "dict[str, list[GeometryEntry]]",
    preds: "list[GeometryEntry]",
    label_counts: "dict[str, int]",
    *,
    images: "tuple[np.ndarray, np.ndarray] | None" = None,
    ink: "tuple[np.ndarray, np.ndarray] | None" = None,
    dpi: "float | None" = None,
    text_boxes: "list | None" = None,
    cfg: VectorMetricConfig = VectorMetricConfig(),
) -> VectorMetricSuiteResult:
    """`images`: `(gt_render, pred_render)` from `vector_render.render_pair`
    (SSIM/PSNR); `ink`: `(gt_mask, pred_mask)` from
    `vector_render.render_ink_pair` (D_p/F_p/PRI). Either `None` leaves its
    stats not-applicable. `dpi`: the rasterised PDF's resolution (width
    floor + curve sampling); `None` -> `cfg.default_dpi`. `text_boxes`:
    page-space text-label bboxes for the text / non-text split."""
    gt_entries = gt_entries_union(gt_by_type)
    gt_polys = flatten_entries(gt_entries, cfg)
    pred_polys = flatten_entries(preds, cfg)
    _gt_bars, _pred_bars, _overlaps, ld = liu_dori.evaluate(
        gt_entries, preds, dpi or cfg.default_dpi, cfg.liu_dori, text_boxes=text_boxes,
    )
    image = image_similarity(images[0], images[1], cfg) if images is not None else _IMAGE_NA
    pixel = liu_dori.pixel_stats(ink[0], ink[1], cfg.liu_dori) if ink is not None else liu_dori.PIXEL_NA
    return VectorMetricSuiteResult(
        label_counts={t: int(label_counts.get(t, 0)) for t in VECTOR_TYPES},
        liu_dori=ld,
        pixel=pixel,
        image=image,
        length=length_stats(gt_polys, pred_polys),
        direction=direction_stats(gt_polys, pred_polys, cfg.direction_bins),
    )


def gt_entries_union(gt_by_type: "dict[str, list[GeometryEntry]]") -> "list[GeometryEntry]":
    """Every `VECTOR_TYPES` type's entries, each tagged with its type."""
    return [e if e.vector_type == t else _with_type(e, t) for t in VECTOR_TYPES for e in gt_by_type.get(t, [])]


def _with_type(e: GeometryEntry, t: str) -> GeometryEntry:
    from dataclasses import replace
    return replace(e, vector_type=t)


def aggregate_vector_metrics(results: "list[VectorMetricSuiteResult]") -> "VectorMetricSuiteResult | None":
    """Liu-Dori ratios micro-averaged, pixel counts summed, SSIM
    pixel-weighted, PSNR from pooled MSE, lengths and direction histograms
    summed (intersection recomputed)."""
    if not results:
        return None
    imgs = [r.image for r in results if r.image.n_pixels and not math.isnan(r.image.mse)]
    n_px = sum(i.n_pixels for i in imgs)
    image = ImageSimilarityStats(
        ssim=sum(i.ssim * i.n_pixels for i in imgs) / n_px,
        mse=sum(i.mse * i.n_pixels for i in imgs) / n_px,
        n_pixels=n_px, psnr_cap_db=imgs[0].psnr_cap_db,
    ) if n_px else _IMAGE_NA

    bins = len(results[0].direction.gt_hist)
    gt_h = np.zeros(bins)
    pred_h = np.zeros(bins)
    for r in results:
        gt_h += np.asarray(r.direction.gt_hist)
        pred_h += np.asarray(r.direction.pred_hist)

    return VectorMetricSuiteResult(
        label_counts={t: sum(r.label_counts.get(t, 0) for r in results) for t in VECTOR_TYPES},
        liu_dori=liu_dori.aggregate_liu_dori([r.liu_dori for r in results]),
        pixel=liu_dori.aggregate_pixel([r.pixel for r in results if r.pixel.gt_ink or r.pixel.pred_ink]),
        image=image,
        length=LengthStats(
            gt_length=sum(r.length.gt_length for r in results),
            pred_length=sum(r.length.pred_length for r in results),
        ),
        direction=DirectionStats(gt_hist=tuple(gt_h), pred_hist=tuple(pred_h)),
    )
