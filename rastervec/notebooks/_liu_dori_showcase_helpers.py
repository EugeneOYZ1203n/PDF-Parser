"""Helpers behind `liu_dori_showcase.ipynb` -- tiny synthetic GT / prediction
cases that show what every number of the raster-to-vector (P2) evaluation
(`docs/RasterToVectorEvaluation.md`) reacts to and how to read it.

Nothing here re-implements a metric: `run` calls `liu_dori.evaluate`
(bars, overlaps, per-line scores) and `vector_metrics.evaluate_vector_metrics`
on `vector_render`'s renders (pixel level, SSIM / PSNR, length, direction),
exactly as the benchmark does. The one new piece of geometry is
`overlap_segment` -- the *location* of an overlap c, which `liu_dori.match`
only measures (its length, d1, d2, d_overlap); it is drawing-only.

Units: page space in pt at `dpi = 72`, so 1 pt = 1 raster px and the width
floor is 2 pt (as in `test_liu_dori.py`). The canonical GT is the paper's
Table 2 line, (10, 20)-(90, 20) width 8 on a 100 x 40 pt page.

Pictures (matplotlib Agg): GT bars as translucent bands at W_eff, predicted
bars as thin lines, both coloured red -> yellow -> green by their line's
Q_v (the viewer's `gt by Q_v` / `pred by Q_v` scale); GT endpoints purple,
pred endpoints cyan (the viewer's endpoint layers); each overlap c in black
with dotted d1 / d2 legs to g.
"""
from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

from rastervec.commons.models import PageMeta  # noqa: E402
from rastervec.commons.paths import output_dir  # noqa: E402
from rastervec.Evaluation.Evaluate import liu_dori, vector_render  # noqa: E402
from rastervec.Evaluation.Evaluate.liu_dori import FRAG_BUCKETS, LiuDoriConfig  # noqa: E402
from rastervec.Evaluation.Evaluate.vector_metrics import (  # noqa: E402
    VECTOR_TYPES, GeometryEntry, VectorMetricConfig, evaluate_vector_metrics,
)

PROD = LiuDoriConfig()
PAPER = LiuDoriConfig(paper_mode=True)

QV_CMAP = LinearSegmentedColormap.from_list("qv", [(0.85, 0.1, 0.1), (0.95, 0.8, 0.0), (0.1, 0.65, 0.2)])
GT_ENDPOINT = (0.55, 0.0, 0.75)
PRED_ENDPOINT = (0.0, 0.75, 0.85)
UNSCORED = (0.6, 0.6, 0.6)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
_group = iter(range(10 ** 9))


def line(p0, p1, *, width: float = 8.0, dashes=None, vtype: str = "vector_to_raster") -> GeometryEntry:
    """One `"l"` entry, its own group (one source annotation / Vector)."""
    return GeometryEntry(kind="l", points=(tuple(p0), tuple(p1)), width=width, dashes=dashes,
                         group=next(_group), vector_type=vtype)


def curve(p0, c1, c2, p3, *, width: float = 2.0, vtype: str = "vector_to_raster") -> GeometryEntry:
    """One cubic `"c"` entry."""
    return GeometryEntry(kind="c", points=(tuple(p0), tuple(c1), tuple(c2), tuple(p3)), width=width,
                         group=next(_group), vector_type=vtype)


def polyline(points, **kw) -> "list[GeometryEntry]":
    return [line(a, b, **kw) for a, b in zip(points[:-1], points[1:])]


def pieces(p0, p1, n: int, *, gap: float = 0.0, **kw) -> "list[GeometryEntry]":
    """`n` equal collinear pieces of p0-p1, `gap` pt between consecutive ones."""
    a, b = np.asarray(p0, float), np.asarray(p1, float)
    length = float(np.hypot(*(b - a)))
    u = (b - a) / length
    step = (length - gap * (n - 1)) / n
    return [line(a + u * i * (step + gap), a + u * (i * (step + gap) + step), **kw) for i in range(n)]


def rotated(center, length: float, angle_deg: float, **kw) -> GeometryEntry:
    """A `length` line centred on `center` at `angle_deg` (y down)."""
    c = np.asarray(center, float)
    t = math.radians(angle_deg)
    d = np.array([math.cos(t), math.sin(t)]) * length / 2.0
    return line(c - d, c + d, **kw)


def G(**kw) -> GeometryEntry:
    """The paper's Table 2 GT line."""
    return line((10, 20), (90, 20), **kw)


@dataclass
class Scenario:
    name: str
    gt: "list[GeometryEntry]"
    pred: "list[GeometryEntry]"
    page: "tuple[float, float]" = (100.0, 40.0)
    dpi: float = 72.0
    text_boxes: "list | None" = None
    note: str = ""
    cfg: LiuDoriConfig = PROD


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------
@dataclass
class Outcome:
    sc: Scenario
    gt: liu_dori.Bars
    pred: liu_dori.Bars
    overlaps: liu_dori.Overlaps
    ld: liu_dori.LiuDoriResult  # with per-line scores
    suite: object  # vector_metrics.VectorMetricSuiteResult
    ink: "tuple[np.ndarray, np.ndarray]"
    images: "tuple[np.ndarray, np.ndarray]"


def page_meta(sc: Scenario) -> PageMeta:
    w, h = sc.page
    return PageMeta(index=0, number=1, mediabox=(0, 0, w, h), rotation=0, width=w, height=h)


def run(sc: Scenario) -> Outcome:
    gt_bars, pred_bars, overlaps, ld = liu_dori.evaluate(
        sc.gt, sc.pred, sc.dpi, sc.cfg, text_boxes=sc.text_boxes, keep_lines=True,
    )
    gt_by_type = {t: [e for e in sc.gt if e.vector_type == t] for t in VECTOR_TYPES}
    vcfg = VectorMetricConfig(liu_dori=sc.cfg)
    meta = page_meta(sc)
    images = vector_render.render_pair(gt_by_type, sc.pred, meta, vcfg)
    ink = vector_render.render_ink_pair(gt_by_type, sc.pred, meta, sc.dpi, vcfg)
    suite = evaluate_vector_metrics(
        gt_by_type, sc.pred, {t: len(v) for t, v in gt_by_type.items()},
        images=images, ink=ink, dpi=sc.dpi, text_boxes=sc.text_boxes, cfg=vcfg,
    )
    return Outcome(sc, gt_bars, pred_bars, overlaps, ld, suite, ink, images)


def run_all(scenarios: "list[Scenario]") -> "list[Outcome]":
    return [run(s) for s in scenarios]


def _hist(h: "dict[str, int]") -> str:
    return " ".join(f"{k}:{h.get(k, 0)}" for k in FRAG_BUCKETS)


def values(o: Outcome) -> "dict[str, object]":
    """Every number the doc discusses for this case (nan = n/a)."""
    ld, s = o.ld, o.suite
    v: "dict[str, object]" = {
        "GT bars": len(o.gt), "pred bars": len(o.pred), "overlaps": len(o.overlaps),
        "W_eff(g)": float(o.gt.w_eff[0]) if len(o.gt) else math.nan,
        "D_v": ld.d_v, "F_v": ld.f_v, "VRI": ld.vri,
        "Q_pt": ld.q_pt.value, "Q_od": ld.q_od.value, "Q_w": ld.q_w.value, "Q_v(c)": ld.q_v_overlap.value,
        "Q_b(g)": ld.gt.q_b.value, "Q_fr(g)": ld.gt.q_fr.value, "Q_v(g)": ld.gt.q_v.value,
        "Q_b(k)": ld.pred.q_b.value, "Q_fr(k)": ld.pred.q_fr.value, "Q_v(k)": ld.pred.q_v.value,
        "|D(g)|": _hist(ld.gt.histogram), "|G(k)|": _hist(ld.pred.histogram),
        "mean |D(g)|": ld.gt.matched_count.value, "mean |G(k)|": ld.pred.matched_count.value,
        "D_p": s.pixel.d_p.value, "F_p": s.pixel.f_p.value, "PRI": s.pixel.pri,
        "SSIM": s.image.ssim, "PSNR": s.image.psnr,
        "length ratio": s.length.ratio.value, "direction": s.direction.intersection,
    }
    # Splits; F_v of the GT-only splits (solid/dashed, GT type) is always n/a.
    for dets in (ld.by_text, ld.by_style, ld.by_type):
        for name, det in dets.items():
            v[f"D_v {name}"] = det.d_v.value
            v[f"F_v {name}"] = det.f_v.value
    return v


def per_overlap(o: Outcome) -> "list[dict[str, float]]":
    return [dict(gt=c.gt, pred=c.pred, l=c.length, d1=c.d1, d2=c.d2, d_overlap=c.d_overlap,
                 Q_pt=c.q_pt, Q_od=c.q_od, Q_w=c.q_w, Q_v=c.q_v) for c in o.overlaps]


def _fmt(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.3f}"


def summary_table(outcomes: "list[Outcome]", keys: "list[str]") -> str:
    """Markdown table, one row per case."""
    rows = ["| case | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for o in outcomes:
        v = values(o)
        rows.append(f"| {o.sc.name} | " + " | ".join(_fmt(v.get(k)) for k in keys) + " |")
    return "\n".join(rows)


# --------------------------------------------------------------------------
# Geometry for drawing only
# --------------------------------------------------------------------------
def _p(p, a, b):
    q, d = liu_dori._proj(np.asarray([p], float), np.asarray([a], float), np.asarray([b], float))
    return q[0], float(d[0])


def overlap_segment(o: Outcome, i: int) -> "tuple[np.ndarray, np.ndarray]":
    """Ends of overlap i's segment c on k -- the same three cases as
    `liu_dori._match_pairs` (all of k / g's ends projected onto k / partial)."""
    c = o.overlaps[i]
    g0, g1 = o.gt.p0[c.gt], o.gt.p1[c.gt]
    k0, k1 = o.pred.p0[c.pred], o.pred.p1[c.pred]
    if o.sc.cfg.paper_mode:
        wg, wk = o.gt.width[c.gt], o.pred.width[c.pred]
    else:
        wg, wk = o.gt.w_eff[c.gt], o.pred.w_eff[c.pred]
    eps = 1e-9
    pg0, dg0 = _p(g0, k0, k1)
    pg1, dg1 = _p(g1, k0, k1)
    _, dk0 = _p(k0, g0, g1)
    _, dk1 = _p(k1, g0, g1)
    g0_in, g1_in = dg0 <= wk / 2 + eps, dg1 <= wk / 2 + eps
    k0_in, k1_in = dk0 <= wg / 2 + eps, dk1 <= wg / 2 + eps
    if k0_in and k1_in:
        return k0, k1
    if g0_in and g1_in:
        return pg0, pg1
    return (k0 if k0_in else k1), (pg0 if g0_in else pg1)


def _band(p0, p1, w) -> np.ndarray:
    d = np.asarray(p1, float) - np.asarray(p0, float)
    n = np.array([-d[1], d[0]]) / max(float(np.hypot(*d)), 1e-12) * w / 2.0
    return np.array([p0 + n, p1 + n, p1 - n, p0 - n])


# --------------------------------------------------------------------------
# Pictures
# --------------------------------------------------------------------------
def plot(o: Outcome, ax, *, annotate: "bool | None" = None, title: bool = True) -> None:
    """The case: GT bands + pred lines by Q_v, endpoints, overlaps."""
    w, h = o.sc.page
    gs, ps = o.ld.gt_scores, o.ld.pred_scores
    for box in o.sc.text_boxes or []:
        x0, y0, x1, y1 = box
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=1, ec="0.4"))
        ax.text(x0, y0 - 0.5, "text box", fontsize=7, color="0.4", va="bottom")
    for i in range(len(o.gt)):
        col = QV_CMAP(gs.q_v[i])
        band = _band(o.gt.p0[i], o.gt.p1[i], o.gt.w_eff[i])
        ax.add_patch(Polygon(band, closed=True, fc=(*col[:3], 0.3), ec=(*col[:3], 0.8), lw=0.8,
                             ls=":" if o.gt.dashed[i] else "-"))
    for i in range(len(o.pred)):
        ax.plot(*np.stack([o.pred.p0[i], o.pred.p1[i]]).T, color=QV_CMAP(ps.q_v[i]), lw=2.2,
                solid_capstyle="butt", zorder=3)
    if len(o.gt):
        e = np.concatenate([o.gt.p0, o.gt.p1])
        ax.scatter(*e.T, s=22, color=GT_ENDPOINT, zorder=4, lw=0)
    if len(o.pred):
        e = np.concatenate([o.pred.p0, o.pred.p1])
        ax.scatter(*e.T, s=12, color=PRED_ENDPOINT, zorder=5, lw=0)
    annotate = len(o.overlaps) <= 3 if annotate is None else annotate
    for i in range(len(o.overlaps)):
        c0, c1 = overlap_segment(o, i)
        ax.plot(*np.stack([c0, c1]).T, color="k", lw=0.7, alpha=0.8, zorder=6)
        if not annotate:
            continue
        c = o.overlaps[i]
        g0, g1 = o.gt.p0[c.gt], o.gt.p1[c.gt]
        for end, d in ((c0, c.d1), (c1, c.d2)):
            if d > 1e-6:
                q, _ = _p(end, g0, g1)
                ax.plot(*np.stack([end, q]).T, color="k", lw=0.8, ls=":", zorder=6)
        mid = (c0 + c1) / 2.0
        ax.annotate(f"l(c)={c.length:.1f}  d1={c.d1:.2f}  d2={c.d2:.2f}  d_ov={c.d_overlap:.2f}\n"
                    f"Q_pt={c.q_pt:.3f}  Q_od={c.q_od:.3f}  Q_w={c.q_w:.3f}  Q_v(c)={c.q_v:.3f}", mid,
                    xytext=(0, 16), textcoords="offset points", fontsize=6.5, ha="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.5, alpha=0.9))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    if title:
        text = o.sc.name + (f"\n{o.sc.note}" if o.sc.note else "")
        ax.set_title("\n".join(textwrap.fill(t, 95) for t in text.split("\n")), fontsize=8.5, loc="left")


def _value_text(o: Outcome, keys: "list[str]") -> str:
    v = values(o)
    width = max(len(k) for k in keys)
    return "\n".join(f"{k:<{width}}  {_fmt(v.get(k))}" for k in keys)


@dataclass
class Book:
    """One output folder per notebook run; every figure saved as a PNG."""
    root: Path = field(default_factory=lambda: output_dir(
        "liu_dori_showcase", datetime.now().strftime("%Y%m%d_%H%M%S")))
    saved: "list[Path]" = field(default_factory=list)

    def save(self, fig, name: str) -> Path:
        path = self.root / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        self.saved.append(path)
        return path

    def cases(self, name: str, outcomes: "list[Outcome]", keys: "list[str]") -> Path:
        """One row per case: picture | its values."""
        rows = []
        for o in outcomes:
            w, h = o.sc.page
            rows.append(max(1.6, 7.5 * h / w) + 0.6)
        fig = plt.figure(figsize=(12, sum(rows)))
        grid = fig.add_gridspec(len(outcomes), 2, width_ratios=[3, 1.35], height_ratios=rows, hspace=0.55)
        for r, o in enumerate(outcomes):
            plot(o, fig.add_subplot(grid[r, 0]))
            ax = fig.add_subplot(grid[r, 1])
            ax.axis("off")
            ax.text(0, 1, _value_text(o, keys), family="monospace", fontsize=8, va="top")
        return self.save(fig, name)

    def sweep(self, name: str, make, params, keys: "list[str]", *, xlabel: str,
              refs: "dict[str, object] | None" = None, n_examples: int = 3) -> Path:
        """`make(param) -> Scenario`; metrics vs param (+ closed-form `refs`:
        label -> f(param), dashed), with a few example cases above."""
        params = list(params)
        outs = [run(make(p)) for p in params]
        picks = sorted({0, len(outs) // 2, len(outs) - 1})[:n_examples]
        fig = plt.figure(figsize=(12, 6.2))
        grid = fig.add_gridspec(2, len(picks), height_ratios=[1, 1.7], hspace=0.45)
        for c, i in enumerate(picks):
            ax = fig.add_subplot(grid[0, c])
            plot(outs[i], ax, annotate=False, title=False)
            ax.set_title(f"{xlabel} = {_fmt(params[i])}", fontsize=8)
        ax = fig.add_subplot(grid[1, :])
        xs = np.asarray(params, float)
        for k in keys:
            ax.plot(xs, [values(o)[k] for o in outs], marker="o", ms=3, label=k)
        for label, f in (refs or {}).items():
            fine = np.linspace(xs.min(), xs.max(), 200)
            ax.plot(fine, [f(x) for x in fine], ls="--", color="0.3", lw=1, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        return self.save(fig, name)

    def pixels(self, name: str, outcomes: "list[Outcome]", keys: "list[str]") -> Path:
        """Per case: the vector picture, the ink overlay (D_p / F_p input)
        and the two grayscale renders (SSIM / PSNR input)."""
        fig, axes = plt.subplots(len(outcomes), 4, figsize=(15, 2.6 * len(outcomes)), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1.3, 1.3, 1.3, 1.0]))
        for r, o in enumerate(outcomes):
            plot(o, axes[r, 0], annotate=False)
            axes[r, 1].imshow(ink_overlay(o))
            axes[r, 1].set_title("ink: red GT only / blue pred only / black both", fontsize=7.5)
            pair = np.concatenate([o.images[0], np.full((o.images[0].shape[0], 4), 128, np.uint8),
                                   o.images[1]], axis=1)
            axes[r, 2].imshow(pair, cmap="gray", vmin=0, vmax=255)
            axes[r, 2].set_title("SSIM renders: GT | pred", fontsize=7.5)
            for ax in axes[r, 1:3]:
                ax.set_xticks([])
                ax.set_yticks([])
            axes[r, 3].axis("off")
            axes[r, 3].text(0, 1, _value_text(o, keys), family="monospace", fontsize=8, va="top")
        fig.tight_layout()
        return self.save(fig, name)

    def histograms(self, name: str, o: Outcome) -> Path:
        """The case + its |D(g)| / |G(k)| bucket charts."""
        fig = plt.figure(figsize=(13, 5))
        grid = fig.add_gridspec(1, 3, width_ratios=[2.2, 1, 1])
        plot(o, fig.add_subplot(grid[0]), annotate=False)
        for c, (side, label) in enumerate(((o.ld.gt, "|D(g)| per GT line"), (o.ld.pred, "|G(k)| per pred line"))):
            ax = fig.add_subplot(grid[c + 1])
            counts = [side.histogram.get(b, 0) for b in FRAG_BUCKETS]
            ax.bar(FRAG_BUCKETS, counts, color=["#d62728", "#2ca02c", "#ff7f0e", "#ff7f0e", "#ff7f0e"])
            ax.set_title(f"{label}\nmean (matched) = {_fmt(side.matched_count.value)}", fontsize=9)
            ax.set_ylabel("lines")
        return self.save(fig, name)

    def direction(self, name: str, outcomes: "list[Outcome]") -> Path:
        """GT (grey bars) vs pred (step line) length-weighted angle histograms."""
        fig, axes = plt.subplots(len(outcomes), 2, figsize=(12, 2.8 * len(outcomes)), squeeze=False,
                                 gridspec_kw=dict(width_ratios=[1.3, 1]))
        for r, o in enumerate(outcomes):
            plot(o, axes[r, 0], annotate=False)
            g = np.asarray(o.suite.direction.gt_hist)
            p = np.asarray(o.suite.direction.pred_hist)
            bins = len(g)
            x = np.arange(bins) * 180.0 / bins
            ax = axes[r, 1]
            ax.bar(x, g / max(g.sum(), 1e-12), width=180.0 / bins, align="edge", color="0.75", label="GT")
            ax.step(np.append(x, 180.0), np.append(p, p[-1:]) / max(p.sum(), 1e-12), where="post",
                    color="#1f77b4", label="pred")
            ax.set_xlim(0, 180)
            ax.set_xlabel("angle (deg, mod 180)", fontsize=8)
            ax.set_title(f"direction intersection = {_fmt(o.suite.direction.intersection)}   "
                         f"length ratio = {_fmt(o.suite.length.ratio.value)}", fontsize=8.5)
            ax.legend(fontsize=7)
        fig.tight_layout()
        return self.save(fig, name)


def ink_overlay(o: Outcome) -> np.ndarray:
    """RGB uint8: white paper, GT-only ink red, pred-only blue, both black."""
    g, p = o.ink
    img = np.full((*g.shape, 3), 255, np.uint8)
    img[g & ~p] = (220, 40, 40)
    img[p & ~g] = (40, 90, 230)
    img[g & p] = (0, 0, 0)
    return img


# --------------------------------------------------------------------------
# The notebook's cases, one builder per section
# --------------------------------------------------------------------------
def bars_cases() -> "list[Scenario]":
    arc = ((10, 32), (25, 2), (75, 2), (90, 32))
    hair = [line((10, 20), (90, 20), width=0.0)]
    return [
        Scenario("straight 'l' item = one bar", [G()], [G()]),
        Scenario("curve, W 2 -> sampled + RDP (eps = W_eff/4) -> several bars",
                 [curve(*arc, width=2.0)], [curve(*arc, width=2.0)]),
        Scenario("same curve, W 12 -> eps 3, fewer bars", [curve(*arc, width=12.0)], [curve(*arc, width=12.0)]),
        Scenario("same curve, W 0 at 300 dpi -> W_eff 0.48, eps 0.12, more bars",
                 [curve(*arc, width=0.0)], [curve(*arc, width=0.0)], dpi=300.0),
        *[Scenario(f"W=0 hairline, pred 0.6 pt off, dpi {d}",
                   hair, [line((10, 20.6), (90, 20.6), width=0.0)], dpi=float(d),
                   note=f"W_eff = 2 px * 72/{d} = {2 * 72 / d:.2f} pt")
          for d in (72, 150, 300)],
    ]


def matching_cases() -> "list[Scenario]":
    return [
        Scenario("k inside g: both k ends in g's area -> c = all of k", [G()], [line((30, 20), (70, 20))]),
        Scenario("g inside k: both g ends in k's area -> c = g projected onto k", [G()], [line((4, 20), (96, 20))]),
        Scenario("partial: one end of each inside -> c from k's end to g's end", [G()], [line((50, 20), (98, 20))]),
        Scenario("offset 3.9 < W/2 = 4: matched (but Q_od low)", [G()], [line((10, 23.9), (90, 23.9))]),
        Scenario("offset 4.5 > W/2: no endpoint inside -> no overlap", [G()], [line((10, 24.5), (90, 24.5))]),
        Scenario("angle 10 deg: passes the 15 deg gate", [G()], [rotated((50, 20), 20, 10)]),
        Scenario("angle 20 deg: rejected by the gate (ends are still inside g)", [G()], [rotated((50, 20), 20, 20)]),
        Scenario("angle 20 deg, paper_mode (no gate): matched", [G()], [rotated((50, 20), 20, 20)], cfg=PAPER),
    ]


def factor_cases() -> "list[Scenario]":
    return [
        Scenario("exact", [G()], [G()]),
        Scenario("parallel offset 2: d1 = d2 = d_ov = 2", [G()], [line((10, 22), (90, 22))]),
        Scenario("one end 3 off (skew): d1 0, d2 3, d_ov 3", [G()], [line((10, 20), (90, 23))]),
        Scenario("width 4 vs 8: only Q_w", [G()], [line((10, 20), (90, 20), width=4.0)]),
        Scenario("short (40 of 80): every Q_* = 1, yet D_v 0.5", [G()], [line((30, 20), (70, 20))],
                 note="per-overlap factors ignore uncovered length"),
        Scenario("overshoot (90 for 80): Q_* = 1, F_v > 0", [G()], [line((5, 20), (95, 20))],
                 note="extent errors show in Q_b, not Q_pt"),
    ]


def offset_case(off: float) -> Scenario:
    return Scenario(f"offset {off}", [G()], [line((10, 20 + off), (90, 20 + off))])


def skew_case(end_off: float) -> Scenario:
    return Scenario(f"end offset {end_off}", [G()], [line((10, 20), (90, 20 + end_off))])


def width_case(ratio: float) -> Scenario:
    return Scenario(f"W(k)/W(g) {ratio}", [G()], [line((10, 20), (90, 20), width=8.0 * ratio)], page=(100, 50))


def coverage_case(frac: float) -> Scenario:
    return Scenario(f"covered {frac}", [G()], [line((10, 20), (10 + 80 * max(frac, 0.01), 20))])


def fragment_case(n: int) -> Scenario:
    return Scenario(f"{n} pieces", [G()], pieces((10, 20), (90, 20), int(n)))


def consolidation_case(n: int) -> Scenario:
    return Scenario(f"GT in {n} items", pieces((10, 20), (90, 20), int(n)), [G()])


def line_cases() -> "list[Scenario]":
    return [
        Scenario("partial coverage: Q_b(g) = covered fraction", [G()], [line((10, 20), (60, 20))]),
        Scenario("2 equal pieces: Q_b 1, Q_fr(g) = 1/sqrt 2", [G()], pieces((10, 20), (90, 20), 2)),
        Scenario("4 pieces with 2 pt gaps: Q_b < 1 and Q_fr ~ 1/2", [G()], pieces((10, 20), (90, 20), 4, gap=2.0)),
        Scenario("one pred across 3 GT items: Q_fr(k) = 1/sqrt 3", pieces((10, 20), (90, 20), 3), [G()],
                 note="GT side all green, the prediction yellow"),
        Scenario("overshooting pred: Q_b(k) = 80/90", [G()], [line((5, 20), (95, 20))]),
    ]


def _three(**kw) -> "list[GeometryEntry]":
    return [line((10, y), (90, y), **kw) for y in (12, 28, 44)]


def page_cases() -> "list[Scenario]":
    p = (100.0, 56.0)
    gt = _three()
    return [
        Scenario("perfect", gt, _three(), page=p),
        Scenario("one of three lines missed", gt, _three()[:2], page=p),
        Scenario("all found + one 80 pt false alarm", gt, _three() + [line((10, 52), (90, 4), width=2.0)], page=p),
        Scenario("every line traced twice (exact duplicates)", gt, _three() + _three(), page=p,
                 note="F_v blind to it; D_v drops via Q_fr(g), length ratio 2"),
        Scenario("one pred per pair of GT pieces (merged)", [e for y in (12, 28, 44) for e in
                 pieces((10, y), (90, y), 2, gap=10.0)], _three(), page=p),
        Scenario("A: finds all, as much junk -> VRI 0.75",
                 gt[:2], _three()[:2] + [line((10, 50), (90, 50)), line((10, 36), (90, 36))], page=p),
        Scenario("B: finds half, no junk -> VRI 0.75", gt[:2], _three()[:1], page=p),
        Scenario("predicts nothing: F_v and VRI n/a (not 0.5)", gt, [], page=p),
    ]


def drawing_case() -> Scenario:
    w = 2.0
    gt = [
        *pieces((10, 10), (110, 10), 2, width=w),         # top edge, stored as 2 CAD items
        line((110, 10), (110, 80), width=w),               # right
        line((110, 80), (10, 80), width=w),                # bottom
        line((10, 80), (10, 10), width=w),                 # left
        *[line((x, 10), (x, 80), width=w) for x in (30, 50, 70, 90)],  # hatch
        line((15, 75), (105, 15), width=w),                # diagonal
    ]
    pred = [
        line((10, 10), (110, 10), width=1.0),              # one pred across both top items -> |G(k)| 2
        *pieces((110, 10), (110, 80), 3, gap=1.0, width=1.0),  # right fragmented -> |D(g)| 3
        # bottom missed -> |D(g)| 0
        line((10, 80.4), (10, 10.4), width=1.0),           # left, slight offset
        line((30, 10), (30, 80), width=1.0), line((50, 10), (50, 80), width=1.0),
        *pieces((70, 10), (70, 80), 2, width=1.0),         # hatch 70 in two
        # hatch 90 missed
        line((15, 75), (105, 15), width=1.0), line((15.3, 75.3), (105.3, 15.3), width=1.0),  # diagonal twice
        line((80, 50), (95, 58), width=1.0),               # noise -> |G(k)| 0
    ]
    return Scenario("small drawing", gt, pred, page=(120.0, 90.0),
                    note="top: 2 GT items / 1 pred; right: 3 pieces; bottom + hatch 90 missed; "
                         "hatch 70: 2 pieces; diagonal traced twice; 1 noise stroke")


def split_cases() -> "list[Scenario]":
    p = (120.0, 50.0)
    box = [(70.0, 8.0, 110.0, 42.0)]
    glyph = [line((80, 12), (80, 38), width=1.0), line((80, 12), (100, 12), width=1.0),
             line((80, 25), (95, 25), width=1.0), line((80, 38), (100, 38), width=1.0)]  # an "E"
    wall = [line((10, 10), (10, 40), width=2.0), line((10, 40), (60, 40), width=2.0)]
    wobbly = [rotated((82, 25), 26, 80, width=1.0), line((80, 13), (100, 13), width=1.0),
              line((84, 30), (96, 20), width=1.0)]
    dashed = [line((10, 20), (100, 20), width=2.0, dashes="[6 3] 0")]
    dash_pred = [line((10 + 9 * i, 20), (16 + 9 * i, 20), width=2.0) for i in range(10)]
    typed = [line((10, 15), (110, 15), width=2.0), line((10, 35), (110, 35), width=2.0, vtype="original_raster")]
    return [
        Scenario("text erased before tracing (RasterText): text D_v 0, non-text 1", wall + glyph, list(wall),
                 page=p, text_boxes=box),
        Scenario("text not erased, traced badly: text F_v up", wall + glyph, list(wall) + wobbly,
                 page=p, text_boxes=box),
        Scenario("dashed GT [6 3], traced as its solid dashes", dashed, dash_pred, page=(110.0, 40.0),
                 note="Q_b <= 6/9 and Q_fr ~ 1/sqrt(10): dashed D_v is low by construction"),
        Scenario("GT types: vector_to_raster found, original_raster missed", typed, typed[:1], page=p),
    ]


def pixel_cases(seed: int = 0) -> "list[Scenario]":
    rng = np.random.default_rng(seed)
    stairs = [(10.0, 10.0)]
    while stairs[-1][0] < 90:
        x, y = stairs[-1]
        stairs += [(x + 2, y), (x + 2, y + 1)]
    diag = [line((10, 10), (90, 50), width=2.0)]
    noise = [rotated(rng.uniform((5, 5), (95, 35)), rng.uniform(3, 8), rng.uniform(0, 180), width=1.0)
             for _ in range(12)]
    return [
        Scenario("8 pieces, 0.5 pt gaps: D_p ~ 1, D_v ~ 1/sqrt 8", [G()], pieces((10, 20), (90, 20), 8, gap=0.5)),
        Scenario("pixel staircase on a 26.6 deg line: every step fails the angle gate",
                 diag, polyline(stairs, width=1.0), page=(100.0, 60.0), note="D_v 0 / F_v 1 though the ink sits on the line"),
        Scenario("exact + 12 noise strokes: F_p and F_v rise together", [G()], [G()] + noise),
    ]


def ssim_cases() -> "list[Scenario]":
    sparse = [line((40, 60), (360, 60), width=1.0), line((40, 150), (360, 150), width=1.0),
              line((40, 240), (360, 240), width=1.0)]
    dense = [line((10, y), (90, y), width=1.0) for y in range(4, 40, 4)]
    return [
        Scenario("identical: SSIM 1, PSNR capped at 100", [G()], [G()]),
        Scenario("sparse 400x300 page, 1 of 3 lines missed", sparse, sparse[:2], page=(400.0, 300.0),
                 note="D_v 0.67 but SSIM stays high -- blank paper dominates"),
        Scenario("dense page, 1 of 9 lines missed", dense, dense[:-1]),
        Scenario("dense page, all lines 1 pt off", dense, [line((10, y + 1), (90, y + 1), width=1.0)
                                                            for y in range(4, 40, 4)]),
    ]


def sanity_cases() -> "list[Scenario]":
    p = (100.0, 100.0)
    grid = [line((10, y), (90, y), width=2.0) for y in (20, 40, 60, 80)] + [line((50, 10), (50, 90), width=2.0)]

    def transpose(es):
        return [replace(e, points=tuple((y, x) for x, y in e.points)) for e in es]

    return [
        Scenario("thick stroke traced on both edges", [line((10, 50), (90, 50), width=6.0)],
                 [line((10, 47.5), (90, 47.5), width=1.0), line((10, 52.5), (90, 52.5), width=1.0)], page=p,
                 note="length ratio 2"),
        Scenario("half the grid missed", grid, grid[:2] + grid[4:], page=p),
        Scenario("transposed coordinates (old rotated rasterised.pdf)", grid, transpose(grid), page=p,
                 note="direction histogram peaks swap 0 <-> 90 deg"),
        Scenario("prediction rotated 10 deg", grid,
                 [rotated(((e.points[0][0] + e.points[1][0]) / 2, (e.points[0][1] + e.points[1][1]) / 2),
                          math.dist(*e.points), math.degrees(math.atan2(e.points[1][1] - e.points[0][1],
                                                                        e.points[1][0] - e.points[0][0])) + 10,
                          width=2.0) for e in grid], page=p),
    ]


def bias_cases() -> "list[Scenario]":
    arc = ((10, 32), (25, 2), (75, 2), (90, 32))
    half = _split_cubic(arc)
    return [
        Scenario("duplicate GT (CAD drew it twice), one perfect pred", [G(), G()], [G()],
                 note="Q_fr(k) = 0.71 -> F_v 0.29"),
        Scenario("CAD-split GT (3 abutting items), one perfect pred", pieces((10, 20), (90, 20), 3), [G()]),
        Scenario("curve traced exactly, but as 2 cubic halves", [curve(*arc)], [curve(*h) for h in half],
                 note="bars break at different points -> |D(g)| / |G(k)| > 1"),
        Scenario("abutting items: pred 2 pokes 2 pt into GT 1", [line((10, 20), (50, 20)), line((50, 20), (90, 20))],
                 [line((10, 20), (50, 20)), line((48, 20), (90, 20))], note="GT 1 gets |D(g)| = 2"),
    ]


def _split_cubic(pts) -> "list[tuple]":
    p0, p1, p2, p3 = (np.asarray(p, float) for p in pts)
    a, b, c = (p0 + p1) / 2, (p1 + p2) / 2, (p2 + p3) / 2
    d, e = (a + b) / 2, (b + c) / 2
    m = (d + e) / 2
    return [tuple(map(tuple, (p0, a, d, m))), tuple(map(tuple, (m, e, c, p3)))]


def diagnostic_cases() -> "list[Scenario]":
    """One case per row of the doc's 'Diagnostic patterns' table."""
    pc, pg, sc, fc, bc = pixel_cases(), page_cases(), sanity_cases(), factor_cases(), bias_cases()
    return [
        replace(pc[0], name="D_v low, D_p high (fragmented)"),
        replace(pg[1], name="D_v low, D_p low (missed)"),
        replace(pc[1], name="F_v high, F_p low (staircase)"),
        replace(pg[3], name="duplicates: Q_fr(g) low, F_v fine, length > 1"),
        replace(pc[2], name="F_v high, F_p high (noise)"),
        replace(fc[2], name="Q_od low (skew)"),
        replace(bc[1], name="|G(k)| 2+ with Q_v(c) high (CAD-split GT)"),
        replace(sc[2], name="direction low (transposed)"),
    ]
