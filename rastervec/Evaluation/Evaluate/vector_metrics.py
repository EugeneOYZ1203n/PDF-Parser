"""Vector-geometry metrics for a raster (P2) run: count accuracy, image
similarity (SSIM/PSNR), total length and direction distribution.

2 GT vector types (`VECTOR_TYPES`): `vector_to_raster` (`GeometryAnnotation`
entries with `source=="auto"`), `original_raster` (`source=="manual"`).
Predictions are the vectors of a pipeline run on the page's rasterised PDF
(P2's output -- that page has no native vectors). The page is scored against
the **union** of both GT types (that is what P2 sees); recall is also
reported per type. `original_vector` is a `metrics.TEXT_TYPES` text type
only, never scored here.

Every metric works off one shared prep step, `flatten_entries`: each
`GeometryEntry` becomes points (`"l"` 2, `"c"` `curve_segments + 1`
equal-t samples; `re`/`qu` were already split into 4 lines upstream), and
consecutive touching entries of one group (one source `Vector`) are joined
into one polyline.

- **Count** (`build_strokes` + `pair_strokes`): each polyline is
  RDP-simplified (removes 1 px staircases), split at turns sharper than
  `turn_tol_deg`, then pieces are chained end to end across groups when the
  ends are within `endpoint_tolerance_pt` and continue straight (within
  `turn_tol_deg`) -- at a junction of 3+ ends the straightest pairs are
  joined, so a line split at a crossing is one stroke again. GT and
  predictions go through the identical chain, then strokes pair greedily
  nearest-first when both ends are within tolerance and both end tangents
  agree. Several predicted pieces drawn along one GT line are therefore one
  paired stroke, not one hit + spurious extras.
- **Image** (`image_similarity`): SSIM + PSNR between grayscale renders of GT
  and predictions at their real widths (`vector_render.py` renders; this
  module only compares arrays).
- **Length** (`length_stats`): sum of raw flattened segment lengths, GT vs
  predicted; ratio pred / GT.
- **Direction** (`direction_stats`): length-weighted histogram of raw
  flattened segment angles mod 180 deg (`direction_bins` bins); score =
  intersection of the two normalised histograms (1 = identical).

Length and direction deliberately use the raw (un-simplified) geometry, so
staircase output is penalised there.

Pure -- no pipeline/label_schema/fitz import; callers (`adapters.py`) build
`GeometryEntry` lists via `geometry_entries_from_vector`/
`geometry_entries_from_annotations` (duck-typed).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rastervec.Evaluation.Evaluate.metrics import Ratio

VECTOR_TYPES: tuple[str, ...] = ("vector_to_raster", "original_raster")

Point = tuple[float, float]

_TOUCH_EPS = 1e-6


@dataclass(frozen=True)
class VectorMetricConfig:
    endpoint_tolerance_pt: float = 2.0
    rdp_eps_pt: float = 0.5
    turn_tol_deg: float = 15.0
    curve_segments: int = 10
    direction_bins: int = 36
    render_dpi: float = 150.0
    psnr_cap_db: float = 100.0
    # Width (pt) a fill-only entry (no stroke width) is rendered with.
    default_width_pt: float = 0.5


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
            width=a.width, group=i, vector_type=vector_type,
        )
        for i, a in enumerate(anns)
    ]


def geometry_entries_from_vector(v, *, group: int = -1) -> list[GeometryEntry]:
    """Decomposes one `Vector` (duck-typed) into `GeometryEntry` "l"/"c"
    primitives -- "re"/"qu" expand to their 4 edges, matching
    `label_schema.geometry_annotations_for_vector`'s decomposition."""
    from rastervec.commons.helpers.geometry import item_points

    common = dict(color=v.color, fill=v.fill, width=v.width, group=group)
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
# Strokes (count accuracy)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Stroke:
    p0: Point
    p1: Point
    # Outward unit tangents at p0 / p1 ((0, 0) for a degenerate stroke).
    t0: Point
    t1: Point
    vector_type: str = ""


def _rdp(pts: np.ndarray, eps: float) -> np.ndarray:
    """Iterative Ramer-Douglas-Peucker."""
    n = len(pts)
    if n <= 2:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        a, b = pts[i], pts[j]
        seg = pts[i + 1:j]
        d = b - a
        norm = math.hypot(d[0], d[1])
        if norm == 0.0:
            dist = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            dist = np.abs(d[0] * (seg[:, 1] - a[1]) - d[1] * (seg[:, 0] - a[0])) / norm
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return pts[keep]


def _unit(v: np.ndarray) -> np.ndarray:
    n = math.hypot(v[0], v[1])
    return v / n if n > 0 else np.zeros(2)


def _split_at_turns(pts: np.ndarray, cos_tol: float) -> list[np.ndarray]:
    """Split a polyline at every interior vertex where the direction turns
    by more than the tolerance."""
    if len(pts) <= 2:
        return [pts]
    d = np.diff(pts, axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    ok = n > 0
    pts, d, n = np.concatenate([pts[:1], pts[1:][ok]]), d[ok], n[ok]
    if len(d) <= 1:
        return [pts]
    u = d / n[:, None]
    cos = (u[:-1] * u[1:]).sum(axis=1)
    cuts = np.nonzero(cos < cos_tol)[0] + 1  # vertex indices to split at
    pieces, start = [], 0
    for c in cuts:
        pieces.append(pts[start:c + 1])
        start = c
    pieces.append(pts[start:])
    return pieces


def _piece_ends(piece: np.ndarray):
    """(p0, outward t0, p1, outward t1)."""
    p0, p1 = piece[0], piece[-1]
    if len(piece) < 2:
        z = np.zeros(2)
        return p0, z, p1, z
    return p0, _unit(piece[0] - piece[1]), p1, _unit(piece[-1] - piece[-2])


def build_strokes(polys: list[FlatPolyline], cfg: "VectorMetricConfig | None" = None) -> list[Stroke]:
    """Polylines -> strokes: RDP, split at sharp turns, then chain pieces end
    to end across polylines (ends within tolerance, straight continuation;
    at a junction the straightest pairs join, each end at most once). Every
    polyline should share one `vector_type` -- callers chain per type."""
    from scipy.spatial import cKDTree

    cfg = cfg or VectorMetricConfig()
    cos_tol = math.cos(math.radians(cfg.turn_tol_deg))
    pieces: list[np.ndarray] = []
    vtype = polys[0].vector_type if polys else ""
    for poly in polys:
        pieces.extend(_split_at_turns(_rdp(poly.points, cfg.rdp_eps_pt), cos_tol))
    if not pieces:
        return []

    n = len(pieces)
    ends = np.zeros((2 * n, 2))
    tans = np.zeros((2 * n, 2))
    for i, pc in enumerate(pieces):
        p0, t0, p1, t1 = _piece_ends(pc)
        ends[2 * i], tans[2 * i] = p0, t0
        ends[2 * i + 1], tans[2 * i + 1] = p1, t1

    # Candidate joins: ends of different pieces within tolerance whose
    # outward tangents point at each other (straight continuation).
    cands = []
    for a, b in cKDTree(ends).query_pairs(cfg.endpoint_tolerance_pt):
        if a // 2 == b // 2:
            continue
        if not (tans[a].any() and tans[b].any()):  # a degenerate piece never chains
            continue
        cos = -float(tans[a] @ tans[b])
        if cos >= cos_tol:
            dist = float(np.hypot(*(ends[a] - ends[b])))
            cands.append((-cos, dist, a, b))
    cands.sort()

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    link = [-1] * (2 * n)
    for _c, _d, a, b in cands:
        if link[a] >= 0 or link[b] >= 0:
            continue
        ra, rb = find(a // 2), find(b // 2)
        if ra == rb:  # would close a loop
            continue
        parent[ra] = rb
        link[a], link[b] = b, a

    strokes: list[Stroke] = []
    seen = [False] * n
    for start_end in range(2 * n):
        piece = start_end // 2
        if seen[piece] or link[start_end] >= 0:
            continue
        # Walk from a free end through the chain.
        e = start_end
        first = e
        while True:
            seen[e // 2] = True
            other = e ^ 1
            nxt = link[other]
            if nxt < 0:
                last = other
                break
            e = nxt
        strokes.append(Stroke(
            p0=tuple(ends[first]), p1=tuple(ends[last]),
            t0=tuple(tans[first]), t1=tuple(tans[last]), vector_type=vtype,
        ))
    return strokes


@dataclass(frozen=True)
class StrokePairing:
    paired: "list[tuple[int, int]]"  # (gt_idx, pred_idx)
    unpaired_gt: "list[int]"
    unpaired_pred: "list[int]"


def _tangents_agree(a, b, cos_tol: float) -> bool:
    """Undirected agreement; a degenerate tangent always agrees."""
    if not (any(a) and any(b)):
        return True
    return abs(a[0] * b[0] + a[1] * b[1]) >= cos_tol


def pair_strokes(gt: list[Stroke], pred: list[Stroke], cfg: "VectorMetricConfig | None" = None) -> StrokePairing:
    """Greedy nearest-first: a pair needs both ends within tolerance (in the
    better orientation) and both end tangents agreeing."""
    from scipy.spatial import cKDTree

    cfg = cfg or VectorMetricConfig()
    tol = cfg.endpoint_tolerance_pt
    cos_tol = math.cos(math.radians(cfg.turn_tol_deg))
    cands: list[tuple[float, int, int]] = []
    if gt and pred:
        pred_ends = np.array([s.p0 for s in pred] + [s.p1 for s in pred])
        tree = cKDTree(pred_ends)
        m = len(pred)
        for gi, g in enumerate(gt):
            near = {j % m for j in tree.query_ball_point(g.p0, tol)}
            for pj in near:
                p = pred[pj]
                best = None
                for q0, q1, u0, u1 in ((p.p0, p.p1, p.t0, p.t1), (p.p1, p.p0, p.t1, p.t0)):
                    d0, d1 = math.dist(g.p0, q0), math.dist(g.p1, q1)
                    if d0 > tol or d1 > tol:
                        continue
                    if not (_tangents_agree(g.t0, u0, cos_tol) and _tangents_agree(g.t1, u1, cos_tol)):
                        continue
                    if best is None or d0 + d1 < best:
                        best = d0 + d1
                if best is not None:
                    cands.append((best, gi, pj))
    cands.sort()
    used_g: set[int] = set()
    used_p: set[int] = set()
    paired: list[tuple[int, int]] = []
    for _d, gi, pj in cands:
        if gi in used_g or pj in used_p:
            continue
        used_g.add(gi)
        used_p.add(pj)
        paired.append((gi, pj))
    return StrokePairing(
        paired=paired,
        unpaired_gt=[i for i in range(len(gt)) if i not in used_g],
        unpaired_pred=[j for j in range(len(pred)) if j not in used_p],
    )


@dataclass(frozen=True)
class VectorCountStats:
    n_gt_strokes: int
    n_pred_strokes: int
    n_paired: int
    n_missed: int
    n_spurious: int
    precision: Ratio
    recall: Ratio
    # Per GT type: Ratio(paired GT strokes of that type, GT strokes of that type).
    recall_by_type: "dict[str, Ratio]"
    n_gt_primitives: int = 0
    n_pred_primitives: int = 0


def vector_count_stats(
    gt_strokes: list[Stroke], pred_strokes: list[Stroke], pairing: StrokePairing,
    *, n_gt_primitives: int = 0, n_pred_primitives: int = 0,
) -> VectorCountStats:
    n_paired = len(pairing.paired)
    n_missed = len(pairing.unpaired_gt)
    n_spurious = len(pairing.unpaired_pred)
    paired_gt = {gi for gi, _pj in pairing.paired}
    recall_by_type = {}
    for t in VECTOR_TYPES:
        idx = [i for i, s in enumerate(gt_strokes) if s.vector_type == t]
        recall_by_type[t] = Ratio(float(sum(1 for i in idx if i in paired_gt)), float(len(idx)))
    return VectorCountStats(
        n_gt_strokes=len(gt_strokes), n_pred_strokes=len(pred_strokes),
        n_paired=n_paired, n_missed=n_missed, n_spurious=n_spurious,
        precision=Ratio(float(n_paired), float(n_paired + n_spurious)),
        recall=Ratio(float(n_paired), float(n_paired + n_missed)),
        recall_by_type=recall_by_type,
        n_gt_primitives=n_gt_primitives, n_pred_primitives=n_pred_primitives,
    )


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
    count: VectorCountStats
    image: ImageSimilarityStats
    length: LengthStats
    direction: DirectionStats
    pairing: "StrokePairing | None" = field(default=None, repr=False)  # per page only


def evaluate_vector_metrics(
    gt_by_type: "dict[str, list[GeometryEntry]]",
    preds: "list[GeometryEntry]",
    label_counts: "dict[str, int]",
    *,
    images: "tuple[np.ndarray, np.ndarray] | None" = None,
    cfg: VectorMetricConfig = VectorMetricConfig(),
) -> VectorMetricSuiteResult:
    """`images`: `(gt_render, pred_render)` from `vector_render.render_pair`;
    `None` leaves the image stats not-applicable (nan)."""
    gt_polys_by_type = {
        t: flatten_entries([e if e.vector_type == t else _with_type(e, t) for e in gt_by_type.get(t, [])], cfg)
        for t in VECTOR_TYPES
    }
    gt_polys = [p for t in VECTOR_TYPES for p in gt_polys_by_type[t]]
    pred_polys = flatten_entries(preds, cfg)

    gt_strokes = [s for t in VECTOR_TYPES for s in build_strokes(gt_polys_by_type[t], cfg)]
    pred_strokes = build_strokes(pred_polys, cfg)
    pairing = pair_strokes(gt_strokes, pred_strokes, cfg)
    count = vector_count_stats(
        gt_strokes, pred_strokes, pairing,
        n_gt_primitives=sum(len(gt_by_type.get(t, [])) for t in VECTOR_TYPES),
        n_pred_primitives=len(preds),
    )
    image = image_similarity(images[0], images[1], cfg) if images is not None else _IMAGE_NA
    return VectorMetricSuiteResult(
        label_counts={t: int(label_counts.get(t, 0)) for t in VECTOR_TYPES},
        count=count,
        image=image,
        length=length_stats(gt_polys, pred_polys),
        direction=direction_stats(gt_polys, pred_polys, cfg.direction_bins),
        pairing=pairing,
    )


def _with_type(e: GeometryEntry, t: str) -> GeometryEntry:
    from dataclasses import replace
    return replace(e, vector_type=t)


def aggregate_vector_metrics(results: "list[VectorMetricSuiteResult]") -> "VectorMetricSuiteResult | None":
    """Counts and lengths summed, SSIM pixel-weighted, PSNR from pooled MSE,
    direction histograms summed (intersection recomputed)."""
    if not results:
        return None
    cs = [r.count for r in results]
    n_paired = sum(c.n_paired for c in cs)
    n_missed = sum(c.n_missed for c in cs)
    n_spurious = sum(c.n_spurious for c in cs)
    count = VectorCountStats(
        n_gt_strokes=sum(c.n_gt_strokes for c in cs),
        n_pred_strokes=sum(c.n_pred_strokes for c in cs),
        n_paired=n_paired, n_missed=n_missed, n_spurious=n_spurious,
        precision=Ratio(float(n_paired), float(n_paired + n_spurious)),
        recall=Ratio(float(n_paired), float(n_paired + n_missed)),
        recall_by_type={
            t: Ratio(
                sum(c.recall_by_type[t].numerator for c in cs),
                sum(c.recall_by_type[t].denominator for c in cs),
            )
            for t in VECTOR_TYPES
        },
        n_gt_primitives=sum(c.n_gt_primitives for c in cs),
        n_pred_primitives=sum(c.n_pred_primitives for c in cs),
    )

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
        count=count,
        image=image,
        length=LengthStats(
            gt_length=sum(r.length.gt_length for r in results),
            pred_length=sum(r.length.pred_length for r in results),
        ),
        direction=DirectionStats(gt_hist=tuple(gt_h), pred_hist=tuple(pred_h)),
    )
