"""Dashed-line detection (Dov Dori 1998, as adapted in Dosch 2000 Sec 3.1).

Approach: every short segment is a candidate dash. Seed a 'virtual line' from
each pair of short segments; collect all short segments whose midpoint lies on
that line (small lateral distance) with a compatible orientation; if the run has
>= DASH_MIN_COUNT members and roughly regular spacing, collapse it to one
'dashed' Segment. Non-dash segments pass through unchanged.
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
    from scipy.spatial import cKDTree

    segs = list(segments)
    lengths = [dist(s.p0, s.p1) for s in segs]
    mids = [((s.p0[0] + s.p1[0]) / 2, (s.p0[1] + s.p1[1]) / 2) for s in segs]
    cand = [i for i in range(len(segs)) if lengths[i] <= dash_max_len]

    best_runs: list[list[int]] = []
    used_in_run: set[int] = set()
    if len(cand) < dash_min_count or len(cand) > max_candidates:
        # too few, or too dense (a text/hatching-heavy page) to trust -- skip,
        # exactly the case Dosch 2000 hands to interactive correction
        return list(segs)

    step = dash_max_len + dash_max_gap
    reach = 3 * step            # adjacent-dash search radius when growing a run
    seed_reach = 10 * step      # how far apart a seed pair may be
    tree = cKDTree(np.array([mids[i] for i in cand], float))

    # precompute "is this a Dori key" (short + >=1 free endpoint) for all segments
    ep_tree = cKDTree(np.array([ep for s in segs for ep in (s.p0, s.p1)], float))
    is_key = [False] * len(segs)
    for idx, s in enumerate(segs):
        if lengths[idx] > dash_max_len:
            continue
        free = 0
        for ep in (s.p0, s.p1):
            # endpoint is 'free' if no OTHER segment endpoint is within touch_tol
            near = ep_tree.query_ball_point(ep, touch_tol)
            if len({n // 2 for n in near} - {idx}) == 0:
                free += 1
        is_key[idx] = free >= 1

    # seed a virtual line from each key + a nearby key
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
                seg_deg = heading_deg(segs[k].p0, segs[k].p1) % 180.0
                da = abs(seg_deg - line_deg) % 180.0
                if min(da, 180.0 - da) > angle_tol_deg and lengths[k] > 3:
                    continue
                members.append((along, k))
            members.sort()
            if len(members) < dash_min_count:
                continue
            positions = [m[0] for m in members]
            gaps = np.diff(positions)
            if len(gaps) and (gaps.max() > dash_max_len + dash_max_gap):
                continue
            # regular spacing (hallmark of a dashed line, not a random cluster)
            if len(gaps) >= 2 and gaps.std() > 0.6 * gaps.mean() + 3.0:
                continue
            run = [k for _, k in members]
            # at least half the run members must be short + free-ended keys
            if np.mean([is_key[k] for k in run]) < 0.5:
                continue
            if len(run) > len(_longest(best_runs)):
                pass
            if len(run) >= dash_min_count and not (set(run) & used_in_run):
                best_runs.append(run)
                used_in_run.update(run)

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


def _longest(runs):
    return max(runs, key=len) if runs else []
