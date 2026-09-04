"""Matplotlib overlays for every pipeline stage."""
from __future__ import annotations

import numpy as np

from .types_ import GroundTruth, PipelineResult, StaircaseRegion, SymbolInstance


def _ax(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def show_stages(result: PipelineResult, figsize=(15, 10)):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=figsize)
    a = axes.ravel()

    a[0].imshow(result.gray, cmap="gray")
    _ax(a[0], "input")

    a[1].imshow(result.ink, cmap="gray")
    _ax(a[1], "ink (binarised)")

    tg = np.zeros((*result.ink.shape, 3), np.uint8)
    tg[result.graphics_mask] = (60, 60, 60)
    tg[result.text_mask] = (220, 60, 60)
    a[2].imshow(tg)
    _ax(a[2], "text (red) / graphics")

    tt = np.zeros((*result.ink.shape, 3), np.uint8)
    tt[result.thin_mask] = (80, 80, 80)
    tt[result.thick_mask] = (40, 120, 220)
    a[3].imshow(tt)
    _ax(a[3], "thick (blue) / thin")

    a[4].imshow(result.dist_map, cmap="magma")
    a[4].imshow(np.ma.masked_where(~result.skeleton, result.skeleton), cmap="spring")
    _ax(a[4], "skeleton + distance transform")

    a[5].imshow(np.ones_like(result.gray) * 255, cmap="gray", vmin=0, vmax=255)
    for ch in result.graph.chains:
        p = np.asarray(ch)
        a[5].plot(p[:, 0], p[:, 1], lw=0.6)
    nx = np.asarray(result.graph.nodes) if result.graph.nodes else np.empty((0, 2))
    if len(nx):
        a[5].scatter(nx[:, 0], nx[:, 1], s=6, c="k")
    a[5].set_ylim(result.gray.shape[0], 0)
    _ax(a[5], f"graph: {len(result.graph.chains)} chains, {len(result.graph.nodes)} nodes")

    _draw_vectors(a[6], result, "vectorised (straight=blue, arc=green)")

    a[7].imshow(result.remainder, cmap="gray")
    _ax(a[7], "remainder (not vectorised)")

    _draw_vectors(a[8], result, "final + junctions", show_junctions=True, base=result.gray)

    _draw_vectors(a[9], result, "staircases + symbols", base=result.gray)
    _draw_staircases(a[9], result.staircases)
    _draw_symbols(a[9], result.symbols)

    a[10].axis("off")
    a[11].axis("off")

    fig.tight_layout()
    return fig


def _draw_staircases(ax, staircases: list[StaircaseRegion]):
    for r in staircases:
        poly = np.asarray(r.polygon + [r.polygon[0]])
        ax.plot(poly[:, 0], poly[:, 1], c="tab:purple", ls=":", lw=1.2)
        for t in r.treads:
            ax.plot([t.p0[0], t.p1[0]], [t.p0[1], t.p1[1]], c="tab:cyan", lw=1.0)
        cx = (r.axis[0][0] + r.axis[1][0]) / 2
        cy = (r.axis[0][1] + r.axis[1][1]) / 2
        ax.annotate(f"n={r.n_treads}", (cx, cy), color="tab:purple", fontsize=7)


def _draw_symbols(ax, symbols: list[SymbolInstance]):
    colors = {"door": "tab:pink", "window": "teal"}
    for s in symbols:
        c = colors.get(s.family, "black")
        x0, y0, x1, y1 = s.bbox
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], c=c, lw=1.0)
        ax.annotate(s.family, (x0, y0), color=c, fontsize=7)


def _draw_vectors(ax, result: PipelineResult, title, show_junctions=False, base=None):
    if base is not None:
        ax.imshow(base, cmap="gray", alpha=0.35)
    else:
        ax.imshow(np.ones_like(result.gray) * 255, cmap="gray", vmin=0, vmax=255)
    for s in result.segments:
        c = "tab:red" if s.dashed else ("tab:orange" if s.thick else "tab:blue")
        ls = "--" if s.dashed else "-"
        ax.plot([s.p0[0], s.p1[0]], [s.p0[1], s.p1[1]], c=c, ls=ls,
                lw=1.2 + (0.8 if s.thick else 0))
    for arc in result.arcs:
        p = np.asarray(arc.polyline)
        ax.plot(p[:, 0], p[:, 1], c="tab:green", lw=1.4)
    if show_junctions:
        j = np.asarray([jj.xy for jj in result.junctions]) if result.junctions else np.empty((0, 2))
        if len(j):
            ax.scatter(j[:, 0], j[:, 1], s=28, facecolors="none", edgecolors="magenta", lw=1.5)
    ax.set_xlim(0, result.gray.shape[1])
    ax.set_ylim(result.gray.shape[0], 0)
    _ax(ax, title)


def compare_methods(results: dict, base=None, ncols: int = 3, figsize=None,
                    show_junctions: bool = True):
    """One panel per pipeline variant, extracted vectors drawn over `base`
    (falls back to each result's own gray). `results` maps label -> PipelineResult
    (or None / an Exception for a variant that failed)."""
    import matplotlib.pyplot as plt

    items = list(results.items())
    n = len(items)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    figsize = figsize or (5 * ncols, 5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axa = axes.ravel()
    for ax, (label, res) in zip(axa, items):
        if not isinstance(res, PipelineResult):
            ax.text(0.5, 0.5, f"{label}\n{res}", ha="center", va="center",
                    fontsize=8, color="tab:red", transform=ax.transAxes)
            ax.axis("off")
            continue
        b = base if base is not None else res.gray
        n_dash = sum(s.dashed for s in res.segments)
        _draw_vectors(
            ax, res,
            f"{label}: {len(res.segments)} seg ({n_dash} dash), "
            f"{len(res.arcs)} arc, {len(res.junctions)} junc",
            show_junctions=show_junctions, base=b,
        )
    for ax in axa[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def show_vs_ground_truth(result: PipelineResult, gt: GroundTruth, figsize=(12, 6)):
    import matplotlib.pyplot as plt

    fig, (a0, a1) = plt.subplots(1, 2, figsize=figsize)
    for ax, (title, segs, arcs, juncs, stairs, syms) in (
        (a0, ("ground truth", gt.segments, gt.arcs, gt.junctions, gt.staircases, gt.symbols)),
        (a1, ("prediction", result.segments, result.arcs, result.junctions,
              result.staircases, result.symbols)),
    ):
        ax.imshow(np.ones_like(result.gray) * 255, cmap="gray", vmin=0, vmax=255)
        for s in segs:
            c = "tab:red" if s.dashed else ("tab:orange" if s.thick else "tab:blue")
            ax.plot([s.p0[0], s.p1[0]], [s.p0[1], s.p1[1]], c=c,
                    ls="--" if s.dashed else "-", lw=1.3)
        for arc in arcs:
            p = np.asarray(arc.polyline)
            ax.plot(p[:, 0], p[:, 1], c="tab:green", lw=1.5)
        jj = np.asarray([j.xy for j in juncs]) if juncs else np.empty((0, 2))
        if len(jj):
            ax.scatter(jj[:, 0], jj[:, 1], s=26, facecolors="none",
                       edgecolors="magenta", lw=1.4)
        _draw_staircases(ax, stairs)
        _draw_symbols(ax, syms)
        ax.set_xlim(0, result.gray.shape[1])
        ax.set_ylim(result.gray.shape[0], 0)
        _ax(ax, title)
    fig.tight_layout()
    return fig
