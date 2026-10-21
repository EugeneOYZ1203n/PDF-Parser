"""Liu & Dori (1997) vector-level line-detection protocol for P2 output.

Liu Wenyin, Dov Dori, "A protocol for performance evaluation of line
detection algorithms", Machine Vision and Applications 9:240-250 (1997),
https://doi.org/10.1007/s001380050045. Equation numbers below are the
paper's. `docs/RasterToVectorEvaluation.md` is the long-form description,
including every deviation (marked DEVIATION here) and why.

Lines are straight **bars** on both sides (`bars_from_entries`): every `"l"`
item is one bar as-is; a `"c"` item is sampled by arc length
(`curve_sample_px` raster px) and RDP-simplified (eps = `rdp_eps_frac` x
the item's floored width), each resulting piece its own bar. Nothing is
chained, de-duplicated or dropped -- GT = raw annotation items, preds =
raw P2 `Vector` items.

Matching (Sect. 3): a GT bar g and a predicted bar k overlap when at least 2
of their 4 endpoints lie inside the other bar's area (distance to it <=
half its width). The overlap segment c = k n g lies on k, between the
two touching points (or k itself / g's projection onto k). d1, d2 = the
distances of c's ends to g; d_overlap = the Hausdorff distance between c
and its projection onto g, which must stay below W(g)/2.

Quality (Eqs. 12-14, 17-27): Q_pt = exp(-(d1+d2)/W(g)), Q_od =
exp(-2 d_overlap/W(g)), Q_w = exp(-|W(k)-W(g)|/W(g)), Q_v(c) = their
geometric mean; per line Q_b (length-weighted Q_v(c), Eq. 18/23), Q_fr
(sqrt(sum l(c)^2)/sum l(c), Eq. 19/22), Q_v = Q_b Q_fr (Eq. 20/24); D_v =
length-weighted Q_v(g) over GT (Eq. 21), F_v = length-weighted 1-Q_v(k)
over preds (Eqs. 25-26), VRI = beta D_v + (1-beta)(1-F_v) (Eq. 27). Pixel
level (Eqs. 5, 10, 11): D_p, F_p, PRI from ink masks (`pixel_stats`).

DEVIATIONS (production config): widths floored at `floor_px` raster px
(hairlines are width 0 in CAD); an undirected angle gate
(`match_angle_tol_deg`) before the endpoint test; style/shape factors
dropped so Q_v(c) = (Q_pt Q_od Q_w)^(1/3). `paper_mode=True` turns all
three off (5th root with Q_st/Q_sh, plus the paper's even-width rule that
a 1-px offset counts as 0) -- used by the tests to reproduce the paper's
Table 2.

Pure: numpy + scipy only, no fitz / pipeline / label_schema import. Input
is `vector_metrics.GeometryEntry` (duck-typed: kind, points, width,
dashes, group, vector_type).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rastervec.Evaluation.Evaluate.metrics import Ratio

Point = tuple[float, float]
Bbox = tuple[float, float, float, float]

# Buckets of the |D(g)| / |G(k)| histograms: 0, 1, 2, 3-5, 6+.
FRAG_BUCKETS: tuple[str, ...] = ("0", "1", "2", "3-5", "6+")
_EPS = 1e-9


@dataclass(frozen=True)
class LiuDoriConfig:
    # Width floor, in raster pixels of the rasterised PDF (W_eff = max(W, floor_px * 72 / dpi)).
    floor_px: float = 2.0
    # Arc-length sampling step for `"c"` items, in raster pixels.
    curve_sample_px: float = 0.5
    # RDP epsilon for flattened curves, as a fraction of the item's W_eff.
    rdp_eps_frac: float = 0.25
    # Undirected angle gate (deg) applied before the endpoint test.
    match_angle_tol_deg: float = 15.0
    beta: float = 0.5  # VRI weight of D_v (Eq. 27)
    alpha: float = 0.5  # PRI weight of D_p (Eq. 11)
    pixel_render_dpi: float = 300.0
    ink_threshold: int = 128  # gray < threshold = ink
    # Test-only: the paper exactly (no floor, no angle gate, 5th root with
    # Q_st/Q_sh, even-width 1-px rule).
    paper_mode: bool = False


# --------------------------------------------------------------------------
# Bars -- struct of arrays (a dense P2 page has 1e5-1e6 items; one Python
# object per bar / overlap / line score cost seconds and hundreds of MB)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Bar:
    """One bar, as returned by indexing / iterating `Bars`."""
    p0: Point
    p1: Point
    width: float  # raw width (0 when the source had none)
    w_eff: float  # floored width used everywhere
    dashed: bool = False
    is_text: bool = False
    vector_type: str = ""
    # Source: GeometryEntry group (GT annotation / pred Vector) and index of
    # the entry within the flattened entry list.
    group: int = -1
    entry: int = -1

    @property
    def length(self) -> float:
        return math.dist(self.p0, self.p1)


@dataclass
class Bars:
    """N straight bars as parallel arrays (`p0`/`p1` (N, 2) page space)."""
    p0: np.ndarray
    p1: np.ndarray
    width: np.ndarray
    w_eff: np.ndarray
    dashed: np.ndarray
    is_text: np.ndarray
    vector_type: np.ndarray  # str
    group: np.ndarray
    entry: np.ndarray

    @classmethod
    def empty(cls) -> "Bars":
        z = np.zeros(0)
        return cls(np.zeros((0, 2)), np.zeros((0, 2)), z, z.copy(), np.zeros(0, bool), np.zeros(0, bool),
                   np.zeros(0, dtype="<U1"), np.zeros(0, np.int64), np.zeros(0, np.int64))

    def __len__(self) -> int:
        return len(self.width)

    def __getitem__(self, i: int) -> Bar:
        return Bar(
            p0=(float(self.p0[i, 0]), float(self.p0[i, 1])), p1=(float(self.p1[i, 0]), float(self.p1[i, 1])),
            width=float(self.width[i]), w_eff=float(self.w_eff[i]), dashed=bool(self.dashed[i]),
            is_text=bool(self.is_text[i]), vector_type=str(self.vector_type[i]),
            group=int(self.group[i]), entry=int(self.entry[i]),
        )

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    @property
    def length(self) -> np.ndarray:
        return np.hypot(*(self.p1 - self.p0).T) if len(self) else np.zeros(0)


def is_dashed(dashes) -> bool:
    """PyMuPDF dash strings look like `"[6 3] 0"`; `"[] 0"` / None = solid."""
    if not dashes:
        return False
    s = str(dashes).strip()
    if s.startswith("["):
        inner = s[1:s.find("]")] if "]" in s else s[1:]
        return bool(inner.strip())
    return False


def width_floor_pt(dpi: float, cfg: LiuDoriConfig) -> float:
    if cfg.paper_mode:
        return 0.0
    return cfg.floor_px * 72.0 / float(dpi)


def _cubic(pts, t: np.ndarray) -> np.ndarray:
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in pts)
    t = t[:, None]
    u = 1.0 - t
    return u ** 3 * p0 + 3 * u * u * t * p1 + 3 * u * t * t * p2 + t ** 3 * p3


def _arc_length_samples(pts, step: float) -> np.ndarray:
    """Points along a cubic every ~`step` of arc length (ends included)."""
    dense = _cubic(pts, np.linspace(0.0, 1.0, 129))
    seg = np.hypot(*np.diff(dense, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= _EPS:
        return dense[[0, -1]]
    n = max(1, int(math.ceil(total / max(step, _EPS))))
    s = np.linspace(0.0, total, n + 1)
    t = np.interp(s, cum, np.linspace(0.0, 1.0, 129))
    return _cubic(pts, t)


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


def _text_fraction(p0: np.ndarray, p1: np.ndarray, boxes: "list[Bbox]") -> np.ndarray:
    """Per segment, the largest fraction of its length inside any one box
    (Liang-Barsky clip, vectorised over segments)."""
    n = len(p0)
    best = np.zeros(n)
    if not n or not boxes:
        return best
    d = p1 - p0
    for x0, y0, x1, y1 in boxes:
        t0 = np.zeros(n)
        t1 = np.ones(n)
        ok = np.ones(n, dtype=bool)
        for p, q in (
            (-d[:, 0], p0[:, 0] - x0), (d[:, 0], x1 - p0[:, 0]),
            (-d[:, 1], p0[:, 1] - y0), (d[:, 1], y1 - p0[:, 1]),
        ):
            par = np.abs(p) < _EPS
            ok &= ~(par & (q < 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(par, 0.0, q / np.where(par, 1.0, p))
            t0 = np.where(~par & (p < 0), np.maximum(t0, r), t0)
            t1 = np.where(~par & (p > 0), np.minimum(t1, r), t1)
        frac = np.where(ok & (t1 > t0), t1 - t0, 0.0)
        best = np.maximum(best, frac)
    return best


def bars_from_entries(
    entries: list, dpi: float, cfg: LiuDoriConfig = LiuDoriConfig(), *,
    text_boxes: "list[Bbox] | None" = None,
) -> Bars:
    """`entries`: `vector_metrics.GeometryEntry` list (GT or preds). Zero-
    length bars are dropped (they carry no length weight)."""
    floor = width_floor_pt(dpi, cfg)
    step = cfg.curve_sample_px * 72.0 / float(dpi)
    dashed_cache: dict = {}
    # One row per *entry* for the straight items (the bulk), expanded rows
    # for curves; per-bar attributes gathered by entry index.
    a_pts: list = []
    b_pts: list = []
    owner: list[int] = []
    widths = np.empty(len(entries))
    dashed = np.empty(len(entries), dtype=bool)
    for ei, e in enumerate(entries):
        widths[ei] = float(e.width) if e.width is not None else 0.0
        dk = getattr(e, "dashes", None)
        if dk not in dashed_cache:
            dashed_cache[dk] = is_dashed(dk)
        dashed[ei] = dashed_cache[dk]
        pts = e.points
        if e.kind == "c" and len(pts) == 4:
            w_eff = max(widths[ei], floor)
            poly = _arc_length_samples(pts, step)
            if w_eff > 0:
                poly = _rdp(poly, cfg.rdp_eps_frac * w_eff)
            a_pts.extend(poly[:-1].tolist())
            b_pts.extend(poly[1:].tolist())
            owner.extend([ei] * (len(poly) - 1))
        else:
            for a, b in zip(pts[:-1], pts[1:]):
                a_pts.append(a)
                b_pts.append(b)
                owner.append(ei)
    if not owner:
        return Bars.empty()
    p0 = np.asarray(a_pts, dtype=float).reshape(-1, 2)
    p1 = np.asarray(b_pts, dtype=float).reshape(-1, 2)
    idx = np.asarray(owner, dtype=np.int64)
    keep = np.hypot(*(p1 - p0).T) > _EPS
    p0, p1, idx = p0[keep], p1[keep], idx[keep]
    width = widths[idx]
    vtypes = np.array([e.vector_type for e in entries], dtype=str)[idx] if len(idx) else np.zeros(0, "<U1")
    groups = np.array([e.group for e in entries], dtype=np.int64)[idx]
    return Bars(
        p0=p0, p1=p1, width=width, w_eff=np.maximum(width, floor), dashed=dashed[idx],
        is_text=_text_fraction(p0, p1, list(text_boxes or [])) > 0.5,
        vector_type=vtypes, group=groups, entry=idx,
    )


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Overlap:
    """One overlap, as returned by indexing `Overlaps`."""
    gt: int
    pred: int
    length: float  # l(c)
    d1: float
    d2: float
    d_overlap: float
    q_pt: float
    q_od: float
    q_w: float
    q_v: float


_OVERLAP_FIELDS = ("gt", "pred", "length", "d1", "d2", "d_overlap", "q_pt", "q_od", "q_w", "q_v")


@dataclass
class Overlaps:
    """Every accepted (g, k) overlap as parallel arrays."""
    gt: np.ndarray
    pred: np.ndarray
    length: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    d_overlap: np.ndarray
    q_pt: np.ndarray
    q_od: np.ndarray
    q_w: np.ndarray
    q_v: np.ndarray

    @classmethod
    def empty(cls) -> "Overlaps":
        return cls(np.zeros(0, np.int64), np.zeros(0, np.int64), *(np.zeros(0) for _ in range(8)))

    def __len__(self) -> int:
        return len(self.gt)

    def __getitem__(self, i: int) -> Overlap:
        return Overlap(**{f: (int if f in ("gt", "pred") else float)(getattr(self, f)[i]) for f in _OVERLAP_FIELDS})

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def _proj(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Closest points on segments a-b to points p (row-wise) and distances."""
    d = b - a
    dd = (d * d).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dd > 0, ((p - a) * d).sum(axis=1) / np.where(dd > 0, dd, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    q = a + t[:, None] * d
    return q, np.hypot(*(p - q).T)


# Spacing (pt) of the probe points along each bar in the candidate search;
# at least the widest bar, so a probe ball never gets smaller than a line.
_PROBE_STEP_PT = 2.0


def _probes(a0: np.ndarray, a1: np.ndarray, step: float) -> "tuple[np.ndarray, np.ndarray]":
    """Points every <= `step` along each segment (ends included) and the
    index of the segment each belongs to."""
    lengths = np.hypot(*(a1 - a0).T)
    n = np.maximum(1, np.ceil(lengths / step).astype(np.int64))
    owner = np.repeat(np.arange(len(a0)), n + 1)
    start = np.repeat(np.cumsum(n + 1) - (n + 1), n + 1)
    t = (np.arange(len(owner)) - start) / np.repeat(n, n + 1)
    pts = a0[owner] + t[:, None] * (a1 - a0)[owner]
    return pts, owner


def _candidate_pairs(gt: Bars, pred: Bars) -> np.ndarray:
    """Unique (gi, pk) pairs where an endpoint of one bar may lie inside the
    other bar's area -- the necessary condition of the 2-of-4 endpoint
    rule. Each bar is probed every `_PROBE_STEP_PT` along its length; the
    other side's endpoints within (step / 2 + max half-width) of a probe are
    candidates. Local by construction: a long border line only meets the
    endpoints actually lying along it, not every endpoint inside a ball of
    its whole length. Vectorised (`sparse_distance_matrix` + `np.unique`)."""
    from scipy.spatial import cKDTree

    if not len(gt) or not len(pred):
        return np.zeros((0, 2), dtype=np.int64)
    m_pred = len(pred)
    step = max(_PROBE_STEP_PT, float(max(gt.w_eff.max(), pred.w_eff.max())))
    keys = []
    for own, other, flip in ((gt, pred, False), (pred, gt, True)):
        probes, owner = _probes(own.p0, own.p1, step)
        radius = step / 2.0 + float(own.w_eff.max()) / 2.0 + 1e-6
        hits = cKDTree(probes).sparse_distance_matrix(
            cKDTree(np.concatenate([other.p0, other.p1])), radius, output_type="ndarray",
        )
        del probes
        if not len(hits):
            continue
        own_idx = owner[hits["i"]].astype(np.int64)
        other_idx = (hits["j"] % len(other)).astype(np.int64)
        del hits
        gi, pk = (other_idx, own_idx) if flip else (own_idx, other_idx)
        keys.append(np.unique(gi * m_pred + pk))
    if not keys:
        return np.zeros((0, 2), dtype=np.int64)
    k = np.unique(np.concatenate(keys))
    return np.stack([k // m_pred, k % m_pred], axis=1)


# Candidate pairs evaluated per `match` chunk -- the geometry below holds
# ~30 temporaries per pair, so this caps that working set (~50 MB).
_MATCH_CHUNK = 200_000


def match(gt: Bars, pred: Bars, cfg: LiuDoriConfig = LiuDoriConfig()) -> Overlaps:
    pairs = _candidate_pairs(gt, pred)
    if not len(pairs):
        return Overlaps.empty()
    parts = [_match_pairs(gt, pred, pairs[i:i + _MATCH_CHUNK], cfg) for i in range(0, len(pairs), _MATCH_CHUNK)]
    if len(parts) == 1:
        return parts[0]
    return Overlaps(**{f: np.concatenate([getattr(p, f) for p in parts]) for f in _OVERLAP_FIELDS})


def _match_pairs(gt: Bars, pred: Bars, pairs: np.ndarray, cfg: LiuDoriConfig) -> Overlaps:
    gi, pk = pairs[:, 0], pairs[:, 1]
    g0, g1 = gt.p0[gi], gt.p1[gi]
    k0, k1 = pred.p0[pk], pred.p1[pk]
    if cfg.paper_mode:
        wg, wk = gt.width[gi], pred.width[pk]
    else:
        wg, wk = gt.w_eff[gi], pred.w_eff[pk]

    keep = np.ones(len(pairs), dtype=bool)
    if not cfg.paper_mode:
        dg, dk = g1 - g0, k1 - k0
        cos = np.abs((dg * dk).sum(axis=1)) / (np.hypot(*dg.T) * np.hypot(*dk.T))
        keep &= cos >= math.cos(math.radians(cfg.match_angle_tol_deg)) - 1e-12

    _, dist_g0 = _proj(g0, k0, k1)
    _, dist_g1 = _proj(g1, k0, k1)
    _, dist_k0 = _proj(k0, g0, g1)
    _, dist_k1 = _proj(k1, g0, g1)
    tol_k, tol_g = wk / 2.0 + _EPS, wg / 2.0 + _EPS
    g0_in, g1_in = dist_g0 <= tol_k, dist_g1 <= tol_k
    k0_in, k1_in = dist_k0 <= tol_g, dist_k1 <= tol_g
    keep &= (g0_in.astype(int) + g1_in + k0_in + k1_in) >= 2

    # Overlap segment c on k: all of k (both k ends inside g), g's ends
    # projected onto k (both g ends inside k), or partial -- k's inside end
    # to g's inside end projected onto k.
    pg0, _ = _proj(g0, k0, k1)
    pg1, _ = _proj(g1, k0, k1)
    both_k = k0_in & k1_in
    both_g = g0_in & g1_in & ~both_k
    k_end = np.where(k0_in[:, None], k0, k1)
    g_proj = np.where(g0_in[:, None], pg0, pg1)
    c0 = np.where(both_k[:, None], k0, np.where(both_g[:, None], pg0, k_end))
    c1 = np.where(both_k[:, None], k1, np.where(both_g[:, None], pg1, g_proj))
    lc = np.hypot(*(c1 - c0).T)
    keep &= lc > _EPS

    q0, d1 = _proj(c0, g0, g1)
    q1, d2 = _proj(c1, g0, g1)
    # Hausdorff between c and its projection q0-q1 onto g.
    _, e0 = _proj(q0, c0, c1)
    _, e1 = _proj(q1, c0, c1)
    d_ov = np.maximum.reduce([d1, d2, e0, e1])
    if cfg.paper_mode:
        even = (np.abs(wg - np.round(wg)) < 1e-9) & (np.round(wg) % 2 == 0)
        d1 = np.where(even & (d1 <= 1.0 + 1e-9), 0.0, d1)
        d2 = np.where(even & (d2 <= 1.0 + 1e-9), 0.0, d2)
        d_ov = np.where(even & (d_ov <= 1.0 + 1e-9), 0.0, d_ov)
    keep &= d_ov < wg / 2.0 + _EPS

    idx = np.nonzero(keep)[0]
    gi, pk, wg, wk = gi[idx], pk[idx], wg[idx], wk[idx]
    d1, d2, d_ov, lc = d1[idx], d2[idx], d_ov[idx], lc[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        q_pt = np.exp(-(d1 + d2) / wg)
        q_od = np.exp(-2.0 * d_ov / wg)
        q_w = np.exp(-np.abs(wk - wg) / wg)
    if cfg.paper_mode:
        st_g = np.where(gt.dashed[gi], 2.0, 1.0)
        st_k = np.where(pred.dashed[pk], 2.0, 1.0)
        q_st = np.exp(-np.abs(st_k - st_g))
        q_v = (q_pt * q_od * q_w * q_st) ** (1.0 / 5.0)  # Q_sh == 1: every line is a bar
    else:
        q_v = (q_pt * q_od * q_w) ** (1.0 / 3.0)
    return Overlaps(gt=gi, pred=pk, length=lc, d1=d1, d2=d2, d_overlap=d_ov,
                    q_pt=q_pt, q_od=q_od, q_w=q_w, q_v=q_v)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LineScore:
    """One GT or predicted bar: Eqs. 18-20 (GT) / 22-24 (pred). `q_fr` is
    nan and `q_b`/`q_v` 0 when nothing matched."""
    length: float
    n_matched: int  # |D(g)| / |G(k)|
    q_b: float
    q_fr: float
    q_v: float


@dataclass
class LineScores:
    """`LineScore` per bar as parallel arrays."""
    length: np.ndarray
    n_matched: np.ndarray
    q_b: np.ndarray
    q_fr: np.ndarray
    q_v: np.ndarray

    def __len__(self) -> int:
        return len(self.length)

    def __getitem__(self, i: int) -> LineScore:
        return LineScore(float(self.length[i]), int(self.n_matched[i]), float(self.q_b[i]),
                         float(self.q_fr[i]), float(self.q_v[i]))

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def _line_scores(lengths: np.ndarray, idx: np.ndarray, ov: Overlaps) -> LineScores:
    n = len(lengths)
    sum_l = np.bincount(idx, weights=ov.length, minlength=n)
    sum_l2 = np.bincount(idx, weights=ov.length ** 2, minlength=n)
    sum_ql = np.bincount(idx, weights=ov.q_v * ov.length, minlength=n)
    count = np.bincount(idx, minlength=n).astype(np.int64)
    matched = count > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        q_b = np.where(matched, sum_ql / np.maximum(lengths, sum_l), 0.0)
        q_fr = np.where(matched, np.sqrt(sum_l2) / sum_l, np.nan)
    q_v = np.where(matched, q_b * q_fr, 0.0)
    return LineScores(length=lengths, n_matched=count, q_b=q_b, q_fr=q_fr, q_v=q_v)


def _bucket_counts(n_matched: np.ndarray) -> "dict[str, int]":
    edges = {"0": (0, 0), "1": (1, 1), "2": (2, 2), "3-5": (3, 5), "6+": (6, np.iinfo(np.int64).max)}
    return {b: int(((n_matched >= lo) & (n_matched <= hi)).sum()) for b, (lo, hi) in edges.items()}


@dataclass(frozen=True)
class SideStats:
    """Length-weighted sums over one side's lines (GT or pred)."""
    q_b: Ratio  # sum Q_b l / sum l
    q_fr: Ratio  # sum Q_fr l / sum l, matched lines only
    q_v: Ratio  # sum Q_v l / sum l (= D_v for GT, 1 - F_v for preds)
    histogram: "dict[str, int]"  # |D(g)| or |G(k)| bucket -> line count
    matched_count: Ratio  # sum |D| / number of matched lines

    @property
    def n_lines(self) -> int:
        return sum(self.histogram.values())


def _side_stats(s: LineScores) -> SideStats:
    m = s.n_matched > 0
    total_l = float(s.length.sum())
    return SideStats(
        q_b=Ratio(float((s.q_b * s.length).sum()), total_l),
        q_fr=Ratio(float((s.q_fr[m] * s.length[m]).sum()), float(s.length[m].sum())),
        q_v=Ratio(float((s.q_v * s.length).sum()), total_l),
        histogram=_bucket_counts(s.n_matched),
        matched_count=Ratio(float(s.n_matched[m].sum()), float(m.sum())),
    )


def _merge_side(a: SideStats, b: SideStats) -> SideStats:
    return SideStats(
        q_b=_add(a.q_b, b.q_b), q_fr=_add(a.q_fr, b.q_fr), q_v=_add(a.q_v, b.q_v),
        histogram={k: a.histogram.get(k, 0) + b.histogram.get(k, 0) for k in FRAG_BUCKETS},
        matched_count=_add(a.matched_count, b.matched_count),
    )


def _add(a: Ratio, b: Ratio) -> Ratio:
    return Ratio(a.numerator + b.numerator, a.denominator + b.denominator)


@dataclass(frozen=True)
class Detection:
    """D_v / F_v / VRI over one subset of lines (both sides restricted the
    same way where that makes sense)."""
    d_v: Ratio  # sum Q_v(g) l(g) / sum l(g)   (Eq. 21)
    f_v: Ratio  # sum (1 - Q_v(k)) l(k) / sum l(k)   (Eq. 26)
    beta: float = 0.5

    @property
    def vri(self) -> float:  # Eq. 27
        d, f = self.d_v.value, self.f_v.value
        if math.isnan(d) or math.isnan(f):
            return math.nan
        return self.beta * d + (1.0 - self.beta) * (1.0 - f)


_NA = Ratio(0.0, math.nan)


def _d_v(s: LineScores, mask: np.ndarray) -> Ratio:
    if not mask.any():
        return _NA
    return Ratio(float((s.q_v[mask] * s.length[mask]).sum()), float(s.length[mask].sum()))


def _f_v(s: LineScores, mask: np.ndarray) -> Ratio:
    if not mask.any():
        return _NA
    return Ratio(float(((1.0 - s.q_v[mask]) * s.length[mask]).sum()), float(s.length[mask].sum()))


def _merge_detection(a: Detection, b: Detection) -> Detection:
    def add(x: Ratio, y: Ratio) -> Ratio:
        if not x.applicable:
            return y
        if not y.applicable:
            return x
        return _add(x, y)
    return Detection(d_v=add(a.d_v, b.d_v), f_v=add(a.f_v, b.f_v), beta=a.beta)


@dataclass
class LiuDoriResult:
    overall: Detection
    # Length-weighted (by l(c)) means of the per-overlap factors.
    q_pt: Ratio
    q_od: Ratio
    q_w: Ratio
    q_v_overlap: Ratio
    gt: SideStats
    pred: SideStats
    by_text: "dict[str, Detection]"  # "text" / "non-text"
    by_style: "dict[str, Detection]"  # "solid" / "dashed" (GT split; F_v n/a)
    by_type: "dict[str, Detection]"  # GT vector type (F_v n/a)
    n_overlaps: int
    gt_length: float
    pred_length: float
    # Per-bar scores, only with `score(keep_lines=True)` (the viewer
    # overlays); `None` otherwise and after `aggregate_liu_dori`.
    gt_scores: "LineScores | None" = field(default=None, repr=False)
    pred_scores: "LineScores | None" = field(default=None, repr=False)

    @property
    def d_v(self) -> float:
        return self.overall.d_v.value

    @property
    def f_v(self) -> float:
        return self.overall.f_v.value

    @property
    def vri(self) -> float:
        return self.overall.vri


VECTOR_TYPES: tuple[str, ...] = ("vector_to_raster", "original_raster")


def score(
    gt: Bars, pred: Bars, overlaps: Overlaps, cfg: LiuDoriConfig = LiuDoriConfig(), *,
    keep_lines: bool = True,
) -> LiuDoriResult:
    gs = _line_scores(gt.length, overlaps.gt, overlaps)
    ps = _line_scores(pred.length, overlaps.pred, overlaps)
    w = float(overlaps.length.sum())

    def ov(arr: np.ndarray) -> Ratio:
        return Ratio(float((arr * overlaps.length).sum()), w)

    def det(gmask, pmask=None) -> Detection:
        return Detection(d_v=_d_v(gs, gmask), f_v=_f_v(ps, pmask) if pmask is not None else _NA, beta=cfg.beta)

    all_g, all_p = np.ones(len(gt), bool), np.ones(len(pred), bool)
    return LiuDoriResult(
        overall=det(all_g, all_p),
        q_pt=ov(overlaps.q_pt), q_od=ov(overlaps.q_od), q_w=ov(overlaps.q_w), q_v_overlap=ov(overlaps.q_v),
        gt=_side_stats(gs), pred=_side_stats(ps),
        by_text={"text": det(gt.is_text, pred.is_text), "non-text": det(~gt.is_text, ~pred.is_text)},
        by_style={"solid": det(~gt.dashed), "dashed": det(gt.dashed)},
        by_type={t: det(gt.vector_type == t) for t in VECTOR_TYPES},
        n_overlaps=len(overlaps),
        gt_length=float(gs.length.sum()),
        pred_length=float(ps.length.sum()),
        gt_scores=gs if keep_lines else None, pred_scores=ps if keep_lines else None,
    )


def evaluate(
    gt_entries: list, pred_entries: list, dpi: float, cfg: LiuDoriConfig = LiuDoriConfig(), *,
    text_boxes: "list[Bbox] | None" = None, keep_lines: bool = True,
) -> "tuple[Bars, Bars, Overlaps, LiuDoriResult]":
    """Bars -> match -> score, for one page."""
    gt = bars_from_entries(gt_entries, dpi, cfg, text_boxes=text_boxes)
    pred = bars_from_entries(pred_entries, dpi, cfg, text_boxes=text_boxes)
    overlaps = match(gt, pred, cfg)
    return gt, pred, overlaps, score(gt, pred, overlaps, cfg, keep_lines=keep_lines)


def aggregate_liu_dori(results: "list[LiuDoriResult]") -> "LiuDoriResult | None":
    """Micro-average: every Ratio's numerator and denominator summed,
    histograms summed."""
    if not results:
        return None
    acc = results[0]
    out = LiuDoriResult(
        overall=acc.overall, q_pt=acc.q_pt, q_od=acc.q_od, q_w=acc.q_w, q_v_overlap=acc.q_v_overlap,
        gt=acc.gt, pred=acc.pred, by_text=dict(acc.by_text), by_style=dict(acc.by_style),
        by_type=dict(acc.by_type), n_overlaps=acc.n_overlaps, gt_length=acc.gt_length,
        pred_length=acc.pred_length,
    )
    for r in results[1:]:
        out.overall = _merge_detection(out.overall, r.overall)
        out.q_pt, out.q_od, out.q_w = _add(out.q_pt, r.q_pt), _add(out.q_od, r.q_od), _add(out.q_w, r.q_w)
        out.q_v_overlap = _add(out.q_v_overlap, r.q_v_overlap)
        out.gt, out.pred = _merge_side(out.gt, r.gt), _merge_side(out.pred, r.pred)
        for name in ("by_text", "by_style", "by_type"):
            mine, theirs = getattr(out, name), getattr(r, name)
            for k in set(mine) | set(theirs):
                if k in mine and k in theirs:
                    mine[k] = _merge_detection(mine[k], theirs[k])
                elif k in theirs:
                    mine[k] = theirs[k]
        out.n_overlaps += r.n_overlaps
        out.gt_length += r.gt_length
        out.pred_length += r.pred_length
    return out


# --------------------------------------------------------------------------
# Pixel level (Eqs. 5, 10, 11)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PixelStats:
    gt_ink: int
    pred_ink: int
    both_ink: int
    alpha: float = 0.5

    @property
    def d_p(self) -> Ratio:  # Eq. 5
        return Ratio(float(self.both_ink), float(self.gt_ink))

    @property
    def f_p(self) -> Ratio:  # Eq. 10
        return Ratio(float(self.pred_ink - self.both_ink), float(self.pred_ink))

    @property
    def pri(self) -> float:  # Eq. 11
        d, f = self.d_p.value, self.f_p.value
        if math.isnan(d) or math.isnan(f):
            return math.nan
        return self.alpha * d + (1.0 - self.alpha) * (1.0 - f)


PIXEL_NA = PixelStats(gt_ink=0, pred_ink=0, both_ink=0)


def pixel_stats(gt_ink: np.ndarray, pred_ink: np.ndarray, cfg: LiuDoriConfig = LiuDoriConfig()) -> PixelStats:
    """`gt_ink`/`pred_ink`: same-shape boolean ink masks."""
    if gt_ink.shape != pred_ink.shape:
        raise ValueError(f"mask shapes differ: {gt_ink.shape} vs {pred_ink.shape}")
    return PixelStats(
        gt_ink=int(gt_ink.sum()), pred_ink=int(pred_ink.sum()),
        both_ink=int((gt_ink & pred_ink).sum()), alpha=cfg.alpha,
    )


def aggregate_pixel(stats: "list[PixelStats]") -> PixelStats:
    if not stats:
        return PIXEL_NA
    return PixelStats(
        gt_ink=sum(s.gt_ink for s in stats), pred_ink=sum(s.pred_ink for s in stats),
        both_ink=sum(s.both_ink for s in stats), alpha=stats[0].alpha,
    )
