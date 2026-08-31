"""Matplotlib overlays for every pipeline stage."""
from __future__ import annotations

import numpy as np

from .types_ import GroundTruth, PipelineResult


def _ax(ax, title):
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def show_stages(result: PipelineResult, figsize=(15, 10)):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=figsize)
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

    fig.tight_layout()
    return fig


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


def show_vs_ground_truth(result: PipelineResult, gt: GroundTruth, figsize=(12, 6)):
    import matplotlib.pyplot as plt

    fig, (a0, a1) = plt.subplots(1, 2, figsize=figsize)
    for ax, (title, segs, arcs, juncs) in (
        (a0, ("ground truth", gt.segments, gt.arcs, gt.junctions)),
        (a1, ("prediction", result.segments, result.arcs, result.junctions)),
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
        ax.set_xlim(0, result.gray.shape[1])
        ax.set_ylim(result.gray.shape[0], 0)
        _ax(ax, title)
    fig.tight_layout()
    return fig
