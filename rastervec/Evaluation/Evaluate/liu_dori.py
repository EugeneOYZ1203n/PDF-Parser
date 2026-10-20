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
# Bars
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Bar:
    p0: Point
    p1: Point
    width: float  # raw width (0 when the source had none)
    w_eff: float  # floored width used everywhere
    dashed: bool = False
    is_text: bool = False
    vector_type: str = ""
    # Source: index of the GeometryEntry group (GT annotation / pred Vector)
    # and of the entry within the flattened list.
    group: int = -1
    entry: int = -1

    @property
    def length(self) -> float:
        return math.dist(self.p0, self.p1)


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
) -> list[Bar]:
    """`entries`: `vector_metrics.GeometryEntry` list (GT or preds). Zero-
    length bars are dropped (they carry no length weight)."""
    floor = width_floor_pt(dpi, cfg)
    step = cfg.curve_sample_px * 72.0 / float(dpi)
    raw: list[tuple] = []  # (p0, p1, width, w_eff, dashed, vtype, group, entry)
    for ei, e in enumerate(entries):
        width = float(e.width) if e.width is not None else 0.0
        w_eff = max(width, floor)
        dashed = is_dashed(getattr(e, "dashes", None))
        if e.kind == "c" and len(e.points) == 4:
            pts = _arc_length_samples(e.points, step)
            pts = _rdp(pts, cfg.rdp_eps_frac * w_eff) if w_eff > 0 else pts
            segs = list(zip(pts[:-1], pts[1:]))
        else:
            pts = [np.asarray(p, dtype=float) for p in e.points]
            segs = list(zip(pts[:-1], pts[1:]))
        for a, b in segs:
            if math.dist(a, b) <= _EPS:
                continue
            raw.append((
                (float(a[0]), float(a[1])), (float(b[0]), float(b[1])),
                width, w_eff, dashed, e.vector_type, e.group, ei,
            ))
    if not raw:
        return []
    p0 = np.array([r[0] for r in raw])
    p1 = np.array([r[1] for r in raw])
    text = _text_fraction(p0, p1, list(text_boxes or [])) > 0.5
    return [
        Bar(p0=r[0], p1=r[1], width=r[2], w_eff=r[3], dashed=r[4], is_text=bool(text[i]),
            vector_type=r[5], group=r[6], entry=r[7])
        for i, r in enumerate(raw)
    ]


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Overlap:
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


def _arrays(bars: list[Bar]):
    p0 = np.array([b.p0 for b in bars], dtype=float).reshape(-1, 2)
    p1 = np.array([b.p1 for b in bars], dtype=float).reshape(-1, 2)
    w = np.array([b.w_eff for b in bars], dtype=float)
    return p0, p1, w


def _proj(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """Closest points on segments a-b to points p (row-wise) and distances."""
    d = b - a
    dd = (d * d).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dd > 0, ((p - a) * d).sum(axis=1) / np.where(dd > 0, dd, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    q = a + t[:, None] * d
    return q, np.hypot(*(p - q).T)


def _candidate_pairs(gt: list[Bar], pred: list[Bar]) -> np.ndarray:
    """(gi, pk) pairs where some endpoint of one may lie inside the other:
    for every bar, the other side's endpoints within its bounding ball
    (half length + half width) around its midpoint."""
    from scipy.spatial import cKDTree

    pairs: set[tuple[int, int]] = set()
    for own, other, flip in ((gt, pred, False), (pred, gt, True)):
        if not own or not other:
            continue
        o0, o1, _ = _arrays(other)
        tree = cKDTree(np.concatenate([o0, o1]))
        m = len(other)
        a0, a1, w = _arrays(own)
        mids = (a0 + a1) / 2.0
        radii = np.hypot(*(a1 - a0).T) / 2.0 + w / 2.0 + 1e-6
        for i, hits in enumerate(tree.query_ball_point(mids, radii)):
            for j in hits:
                j %= m
                pairs.add((j, i) if flip else (i, j))
    if not pairs:
        return np.zeros((0, 2), dtype=int)
    return np.array(sorted(pairs), dtype=int)


def match(gt: list[Bar], pred: list[Bar], cfg: LiuDoriConfig = LiuDoriConfig()) -> list[Overlap]:
    pairs = _candidate_pairs(gt, pred)
    if not len(pairs):
        return []
    gi, pk = pairs[:, 0], pairs[:, 1]
    G0, G1, Wg_all = _arrays(gt)
    K0, K1, Wk_all = _arrays(pred)
    g0, g1, wg = G0[gi], G1[gi], Wg_all[gi]
    k0, k1, wk = K0[pk], K1[pk], Wk_all[pk]
    if cfg.paper_mode:
        wg = np.array([gt[i].width for i in gi], dtype=float)
        wk = np.array([pred[j].width for j in pk], dtype=float)

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
    n_in = g0_in.astype(int) + g1_in + k0_in + k1_in
    keep &= n_in >= 2

    # Overlap segment c on k.
    pg0, _ = _proj(g0, k0, k1)
    pg1, _ = _proj(g1, k0, k1)
    both_k = k0_in & k1_in
    both_g = g0_in & g1_in & ~both_k
    partial = ~both_k & ~both_g
    k_end = np.where(k0_in[:, None], k0, k1)
    g_proj = np.where(g0_in[:, None], pg0, pg1)
    c0 = np.where(both_k[:, None], k0, np.where(both_g[:, None], pg0, k_end))
    c1 = np.where(both_k[:, None], k1, np.where(both_g[:, None], pg1, g_proj))
    keep &= ~(partial & ~((k0_in | k1_in) & (g0_in | g1_in)))
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

    with np.errstate(divide="ignore", invalid="ignore"):
        q_pt = np.exp(-(d1 + d2) / wg)
        q_od = np.exp(-2.0 * d_ov / wg)
        q_w = np.exp(-np.abs(wk - wg) / wg)
    if cfg.paper_mode:
        st_g = np.array([2.0 if gt[i].dashed else 1.0 for i in gi])
        st_k = np.array([2.0 if pred[j].dashed else 1.0 for j in pk])
        q_st = np.exp(-np.abs(st_k - st_g))
        q_v = (q_pt * q_od * q_w * q_st) ** (1.0 / 5.0)  # Q_sh == 1: every line is a bar
    else:
        q_v = (q_pt * q_od * q_w) ** (1.0 / 3.0)

    out = []
    for idx in np.nonzero(keep)[0]:
        out.append(Overlap(
            gt=int(gi[idx]), pred=int(pk[idx]), length=float(lc[idx]),
            d1=float(d1[idx]), d2=float(d2[idx]), d_overlap=float(d_ov[idx]),
            q_pt=float(q_pt[idx]), q_od=float(q_od[idx]), q_w=float(q_w[idx]), q_v=float(q_v[idx]),
        ))
    return out


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


def _line_scores(n: int, lengths: np.ndarray, overlaps: list[Overlap], side: str) -> list[LineScore]:
    sum_l = np.zeros(n)
    sum_l2 = np.zeros(n)
    sum_ql = np.zeros(n)
    count = np.zeros(n, dtype=int)
    for o in overlaps:
        i = o.gt if side == "gt" else o.pred
        sum_l[i] += o.length
        sum_l2[i] += o.length ** 2
        sum_ql[i] += o.q_v * o.length
        count[i] += 1
    out = []
    for i in range(n):
        if count[i] == 0:
            out.append(LineScore(float(lengths[i]), 0, 0.0, math.nan, 0.0))
            continue
        q_b = sum_ql[i] / max(lengths[i], sum_l[i])
        q_fr = math.sqrt(sum_l2[i]) / sum_l[i]
        out.append(LineScore(float(lengths[i]), int(count[i]), float(q_b), float(q_fr), float(q_b * q_fr)))
    return out


def _bucket(n: int) -> str:
    if n <= 2:
        return str(n)
    return "3-5" if n <= 5 else "6+"


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


def _side_stats(scores: list[LineScore]) -> SideStats:
    total_l = sum(s.length for s in scores)
    matched = [s for s in scores if s.n_matched]
    hist = {b: 0 for b in FRAG_BUCKETS}
    for s in scores:
        hist[_bucket(s.n_matched)] += 1
    return SideStats(
        q_b=Ratio(sum(s.q_b * s.length for s in scores), total_l),
        q_fr=Ratio(sum(s.q_fr * s.length for s in matched), sum(s.length for s in matched)),
        q_v=Ratio(sum(s.q_v * s.length for s in scores), total_l),
        histogram=hist,
        matched_count=Ratio(float(sum(s.n_matched for s in matched)), float(len(matched))),
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


def _detection(gt_scores, pred_scores, beta: float) -> Detection:
    gl = sum(s.length for s in gt_scores)
    pl = sum(s.length for s in pred_scores)
    return Detection(
        d_v=Ratio(sum(s.q_v * s.length for s in gt_scores), gl) if gt_scores else _NA,
        f_v=Ratio(sum((1.0 - s.q_v) * s.length for s in pred_scores), pl) if pred_scores else _NA,
        beta=beta,
    )


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
    # Per page only (overlays); dropped by `aggregate_liu_dori`.
    gt_scores: "list[LineScore]" = field(default_factory=list, repr=False)
    pred_scores: "list[LineScore]" = field(default_factory=list, repr=False)

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
    gt: list[Bar], pred: list[Bar], overlaps: list[Overlap], cfg: LiuDoriConfig = LiuDoriConfig(),
) -> LiuDoriResult:
    gt_scores = _line_scores(len(gt), np.array([b.length for b in gt]), overlaps, "gt")
    pred_scores = _line_scores(len(pred), np.array([b.length for b in pred]), overlaps, "pred")
    w = sum(o.length for o in overlaps)

    def ov(attr: str) -> Ratio:
        return Ratio(sum(getattr(o, attr) * o.length for o in overlaps), w)

    def sub(gt_pred, pred_pred=None) -> Detection:
        gs = [s for b, s in zip(gt, gt_scores) if gt_pred(b)]
        ps = [s for b, s in zip(pred, pred_scores) if pred_pred(b)] if pred_pred else []
        d = _detection(gs, ps, cfg.beta)
        return d if pred_pred else Detection(d_v=d.d_v, f_v=_NA, beta=cfg.beta)

    return LiuDoriResult(
        overall=_detection(gt_scores, pred_scores, cfg.beta),
        q_pt=ov("q_pt"), q_od=ov("q_od"), q_w=ov("q_w"), q_v_overlap=ov("q_v"),
        gt=_side_stats(gt_scores), pred=_side_stats(pred_scores),
        by_text={
            "text": sub(lambda b: b.is_text, lambda b: b.is_text),
            "non-text": sub(lambda b: not b.is_text, lambda b: not b.is_text),
        },
        by_style={
            "solid": sub(lambda b: not b.dashed),
            "dashed": sub(lambda b: b.dashed),
        },
        by_type={t: sub(lambda b, t=t: b.vector_type == t) for t in VECTOR_TYPES},
        n_overlaps=len(overlaps),
        gt_length=float(sum(s.length for s in gt_scores)),
        pred_length=float(sum(s.length for s in pred_scores)),
        gt_scores=gt_scores, pred_scores=pred_scores,
    )


def evaluate(
    gt_entries: list, pred_entries: list, dpi: float, cfg: LiuDoriConfig = LiuDoriConfig(), *,
    text_boxes: "list[Bbox] | None" = None,
) -> "tuple[list[Bar], list[Bar], list[Overlap], LiuDoriResult]":
    """Bars -> match -> score, for one page."""
    gt = bars_from_entries(gt_entries, dpi, cfg, text_boxes=text_boxes)
    pred = bars_from_entries(pred_entries, dpi, cfg, text_boxes=text_boxes)
    overlaps = match(gt, pred, cfg)
    return gt, pred, overlaps, score(gt, pred, overlaps, cfg)


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
