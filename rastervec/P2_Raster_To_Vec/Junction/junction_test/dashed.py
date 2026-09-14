"""Dashed-line detection (Dov Dori 1998, as adapted in Dosch 2000 Sec 3.1).

Approach: every short segment is a candidate dash. Seed a 'virtual line' from
each pair of short segments; collect all short segments whose midpoint lies on
that line (small lateral distance) with a compatible orientation; if the run has
>= DASH_MIN_COUNT members and roughly regular spacing, collapse it to one
'dashed' Segment. Non-dash segments pass through unchanged.

`_find_regular_runs` is the reusable core (candidate seeding + run growth +
regularity check) shared with `staircase.py`, which applies the same
seed/grow/regularity-check machinery to stair treads (perpendicular to their
walking axis, instead of dashes lying end-to-end along their own line).
"""
from __future__ import annotations

import numpy as np

from .geom import dist, heading_deg, point_to_segment_dist
from .types_ import Segment


def _free_endpoints(seg: Segment, others: list[Segment], touch_tol: float) -> int:
    free = 0
    for ep in (seg.p0, seg.p1):
        if not any(
            o is not seg and point_to_segment_dist(ep, o.p0, o.p1) <= touch_tol
            for o in others
        ):
            free += 1
    return free


def _find_regular_runs(
    mids: list[tuple[float, float]],
    axis_headings: list[float],
    lengths: list[float],
    is_key: list[bool],
    *,
    max_len: float,
    max_gap: float,
    min_count: int,
    max_count: int | None,
    angle_tol_deg: float,
    lateral_tol: float,
    require_key_fraction: float,
    max_candidates: int,
) -> list[list[int]]:
    """Seed virtual lines from pairs of 'key' candidates, grow a run of
    regularly-spaced, axis-aligned members along each seeded line.

    `axis_headings[i]` is the heading (0..180) tested for alignment against
    each candidate seed-pair's line direction -- callers control what
    "aligned" means for their feature (e.g. `staircase.py` passes each
    segment's heading rotated +90 deg, since tread segments run perpendicular
    to their walking axis rather than along it, unlike dashes).

    Returns a list of member-index runs (no SVD-collapse or Segment
    construction here -- that stays caller-specific).
    """
    from scipy.spatial import cKDTree

    n = len(mids)
    cand = [i for i in range(n) if lengths[i] <= max_len]
    if len(cand) < min_count or len(cand) > max_candidates:
        return []

    step = max_len + max_gap
    reach = 3 * step
    seed_reach = 10 * step
    tree = cKDTree(np.array([mids[i] for i in cand], float))

    best_runs: list[list[int]] = []
    used_in_run: set[int] = set()

    for i in cand:
        if i in used_in_run or not is_key[i]:
            continue
        neigh = sorted(tree.query_ball_point(mids[i], seed_reach),
                       key=lambda l: dist(mids[i], mids[cand[l]]))[:40]
        for local_j in neigh:
            if i in used_in_run:
                break
            j = cand[local_j]
            if j <= i or j in used_in_run or not is_key[j]:
                continue
            p, q = np.array(mids[i]), np.array(mids[j])
            pair_d = dist(mids[i], mids[j])
            if not (6.0 < pair_d <= seed_reach):
                continue
            axis = q - p
            axis = axis / (np.hypot(*axis) + 1e-9)
            line_deg = np.degrees(np.arctan2(axis[1], axis[0])) % 180.0

            members = []
            centre = (p + q) / 2
            member_radius = pair_d / 2 + reach
            nearby = {cand[l] for l in tree.query_ball_point(centre, member_radius)}
            for k in nearby:
                m = np.array(mids[k])
                along = (m - p) @ axis
                lateral = np.hypot(*((m - p) - along * axis))
                if lateral > lateral_tol:
                    continue
                da = abs(axis_headings[k] - line_deg) % 180.0
                if min(da, 180.0 - da) > angle_tol_deg and lengths[k] > 3:
                    continue
                members.append((along, k))
            members.sort()
            if len(members) < min_count or (max_count is not None and len(members) > max_count):
                continue
            positions = [m[0] for m in members]
            gaps = np.diff(positions)
            if len(gaps) and (gaps.max() > max_len + max_gap):
                continue
            # regular spacing (hallmark of a dashed line/tread run, not a random cluster)
            if len(gaps) >= 2 and gaps.std() > 0.6 * gaps.mean() + 3.0:
                continue
            run = [k for _, k in members]
            if require_key_fraction > 0.0 and np.mean([is_key[k] for k in run]) < require_key_fraction:
                continue
            if len(run) >= min_count and not (set(run) & used_in_run):
                best_runs.append(run)
                used_in_run.update(run)

    return best_runs


def detect(
    segments: list[Segment],
    *,
    dash_max_len: float = 22.0,
    dash_max_gap: float = 26.0,
    dash_min_count: int = 3,
    angle_tol_deg: float = 14.0,
    lateral_tol: float = 4.0,
    touch_tol: float = 2.5,
    max_candidates: int = 400,
) -> list[Segment]:
    segs = list(segments)
    lengths = [dist(s.p0, s.p1) for s in segs]
    mids = [((s.p0[0] + s.p1[0]) / 2, (s.p0[1] + s.p1[1]) / 2) for s in segs]
    axis_headings = [heading_deg(s.p0, s.p1) % 180.0 for s in segs]

    # precompute "is this a Dori key" (short + >=1 free endpoint) for all segments
    ep_tree_pts = np.array([ep for s in segs for ep in (s.p0, s.p1)], float)
    is_key = [False] * len(segs)
    if len(ep_tree_pts):
        from scipy.spatial import cKDTree
        ep_tree = cKDTree(ep_tree_pts)
        for idx, s in enumerate(segs):
            if lengths[idx] > dash_max_len:
                continue
            free = 0
            for ep in (s.p0, s.p1):
                near = ep_tree.query_ball_point(ep, touch_tol)
                if len({n // 2 for n in near} - {idx}) == 0:
                    free += 1
            is_key[idx] = free >= 1

    best_runs = _find_regular_runs(
        mids, axis_headings, lengths, is_key,
        max_len=dash_max_len, max_gap=dash_max_gap, min_count=dash_min_count,
        max_count=None, angle_tol_deg=angle_tol_deg, lateral_tol=lateral_tol,
        require_key_fraction=0.5, max_candidates=max_candidates,
    )

    result: list[Segment] = []
    consumed: set[int] = set()
    for run in best_runs:
        pts = np.array([mids[k] for k in run], float)
        c = pts.mean(0)
        d = pts - c
        axis = np.linalg.svd(d, full_matrices=False)[2][0]
        t = d @ axis
        p_lo = tuple((c + axis * t.min()).tolist())
        p_hi = tuple((c + axis * t.max()).tolist())
        w = float(np.median([segs[k].width for k in run]))
        result.append(Segment(p0=p_lo, p1=p_hi, width=w, dashed=True,
                              thick=any(segs[k].thick for k in run)))
        consumed.update(run)

    for i in range(len(segs)):
        if i not in consumed:
            result.append(segs[i])
    return result
