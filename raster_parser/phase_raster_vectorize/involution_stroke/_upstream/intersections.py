# Vendored from https://github.com/verven/InvolutionStrokeVectorization
# file main/intersections.py @ 178eede602bfb488b46cfc38ea2176ff902a4da2
# (no LICENSE upstream -- see ../README.md). Edits, all mechanical:
#   - `np.concat` -> `np.concatenate` (numpy 1.x)
#   - `print` rebound to a debug logger
#   - only the inference helpers kept: compute_matrix_for_rescaling, densify_polyline_maxdist, involutive_assignment2, split_in_half, resample_polyline, get_line_tensor, reconstruct_ordered_paths, reorient_sublines, get_matched_lines
#   - (training/noise/visualisation/model-loading code and unused imports dropped)
# Otherwise verbatim: do not reformat, so a re-vendor diffs cleanly.

import networkx as nx
import numpy as np
import torch
from scipy.interpolate import interp1d

import logging as _logging
_LOG = _logging.getLogger("raster_parser.involution_stroke.upstream")


def print(*args, **_kwargs):  # noqa: A001 -- upstream's progress prints -> debug log
    _LOG.debug(" ".join(str(a) for a in args))


def compute_matrix_for_rescaling(in_w: float, in_h: float, target: float, align: str = "center") -> np.ndarray:
    """
    Compute matrix (a, b, c, d, e, f) for uniform scaling to fit (no stretching)
    into a square of size `target`. `align` can be "center" or "top-left".
    - a = scale_x = scale
    - d = scale_y = scale
    - b = c = 0
    - e = tx, f = ty (translation)
    """
    if in_w <= 0 or in_h <= 0:
        raise ValueError("input width and height must be positive numbers")

    s = min(target / in_w, target / in_h)
    scaled_w = in_w * s
    scaled_h = in_h * s

    if align == "center":
        tx = (target - scaled_w) / 2.0
        ty = (target - scaled_h) / 2.0
    elif align == "top-left":
        tx = 0.0
        ty = 0.0
    else:
        raise ValueError("invalid align")

    # matrix(a b c d e f) with no skew/rotation
    return np.array([[s, 0, tx], [0, s, ty], [0, 0, 1]])


def densify_polyline_maxdist(polyline: np.ndarray, max_dist: float) -> np.ndarray:
    """
    Returns a densified polyline where every segment has length at most max_dist.
    Segments shorter than max_dist are left unchanged.

    Args:
        polyline: np.ndarray of shape (N, 2)
        max_dist: float, maximum allowed segment length

    Returns:
        np.ndarray of shape (M, 2) where M >= N
    """
    # 1. Handle edge cases
    if len(polyline) < 2:
        return polyline.copy()
    if max_dist <= 0:
        raise ValueError("max_dist must be positive")

    # 2. Calculate lengths of existing segments
    # shape: (N-1, 2)
    diffs = polyline[1:] - polyline[:-1]
    # shape: (N-1,)
    lengths = np.linalg.norm(diffs, axis=1)

    # 3. Determine how many segments each existing segment needs to be split into
    # We use ceil to ensure the new sub-segments are <= max_dist.
    # We enforce at least 1 segment (max with 1) to handle 0-length segments gracefully.
    num_segments = np.ceil(lengths / max_dist).astype(int)
    num_segments = np.maximum(num_segments, 1)

    # 4. Generate the new points
    new_points = []

    for i in range(len(num_segments)):
        p_start = polyline[i]
        p_end = polyline[i+1]
        n = num_segments[i]

        if n == 1:
            # If the segment is already short enough, keep the start point.
            # We treat the list as [start, end), appending the final end point later.
            new_points.append(p_start.reshape(1, 2))
        else:
            # Generate n+1 points (including start and end), then exclude the last one
            # so we don't duplicate points when moving to the next segment.
            # shape: (n, 2)
            segment_points = np.linspace(p_start, p_end, n + 1)[:-1]
            new_points.append(segment_points)

    # 5. Add the very last point of the original polyline
    new_points.append(polyline[-1].reshape(1, 2))

    # 6. Concatenate all parts
    return np.vstack(new_points)


def involutive_assignment2(C: np.ndarray, maximize: bool = True):
    n = C.shape[0]
    diag = np.diag(C).copy()

    # Build n-node graph where matching = swaps, unmatched = fixed points
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if C[i, j] == 0.0 or not np.isfinite(C[i, j]):
                continue
            # Gain of swapping (i,j) vs fixing both
            gain = 2 * C[i, j] - diag[i] - diag[j]
            if maximize and gain > 0:
                G.add_edge(i, j, weight=gain)
            elif not maximize and gain < 0:
                G.add_edge(i, j, weight=-gain)

    matching = nx.max_weight_matching(G, maxcardinality=False)

    perm = np.arange(n, dtype=int)
    for u, v in matching:
        perm[u] = v
        perm[v] = u

    total_cost = sum(C[i, perm[i]] for i in range(n))
    return perm, total_cost


def split_in_half(polylines: list[np.ndarray], 
                                    connections: list[tuple[int, int]],):
    
    res = {}
    half_edge_to_node = {}
    c = 1
    for (l_node_idx, r_node_idx), pl in zip(connections, polylines):
        #special case
        if len(pl) ==2:
            mid = (pl[0]+pl[1])/2
            left_points = np.array([pl[0], mid])
            right_points = np.array([mid, pl[1]])
        else:
            line = densify_polyline_maxdist(pl, 0.5)
            k = len(line)//2
            # we want the connection point to overlap, so k+1
            left_points = line[:k+1]
            right_points =line[k:][::-1]
        half_edge_to_node[c] = (l_node_idx, left_points)
        half_edge_to_node[c+1] = (r_node_idx, right_points)
        res.setdefault(l_node_idx, []).append(c)
        res.setdefault(r_node_idx, []).append(c+1)
        c+=2
    
    assert len(half_edge_to_node) == 2*len(polylines)
    return res, half_edge_to_node


def resample_polyline(points, N):
    """resamples a single polyline of shape (V, 2) to (N, 2). """
    points = np.asarray(points)
    if len(points) == 1:
        return np.repeat(points, N, axis=0)

    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    
    total_length = cum_dists[-1]
    if total_length == 0:
        return np.repeat(points[:1], N, axis=0)
    
    cum_dists_norm = cum_dists / total_length
    interpolator = interp1d(cum_dists_norm, points, axis=0, kind='linear')
    t = np.linspace(0, 1, N)
    resampled = interpolator(t)
    
    resampled[0] = points[0]
    resampled[-1] = points[-1]
    return resampled


def get_line_tensor(half_edges: dict):
    mat =  compute_matrix_for_rescaling(768, 2024, 512)
    res = torch.zeros(len(half_edges)+1, 100, 2)
    for k, v in half_edges.items():
        res[k] = torch.tensor(resample_polyline(v[1], 100))
    return res


def reconstruct_ordered_paths(nodes, m1, m2):
    visited = set()
    paths = []

    def next_node(curr, prev):
        """Return the next node from curr that is not prev (if possible)."""
        a, b = m1[curr], m2[curr]

        # Choose endpoint that is not prev
        if a != prev and a != curr:
            return a
        if b != prev and b != curr:
            return b
        return None

    for start in nodes:
        if start in visited:
            continue

        # Build path starting from 'start'
        path = [start]
        visited.add(start)

        prev = None
        curr = start
        while True:
            nxt = next_node(curr, prev)
            if nxt is None or nxt in visited:
                break
            path.append(nxt)
            visited.add(nxt)
            prev, curr = curr, nxt

        prev = None
        curr = start
        while True:
            nxt = next_node(curr, prev)
            if nxt is None or nxt in visited:
                break
            path.insert(0, nxt)
            visited.add(nxt)
            prev, curr = curr, nxt

        paths.append(path)

    return paths


def reorient_sublines(sublines):
    """
    Ensure all sub-polylines are consistently oriented.

    Parameters
    ----------
    sublines : list of np.ndarray
        Each array is shape (N, D) representing a polyline.

    Returns
    -------
    list of np.ndarray
        Reoriented sublines.
    """

    if not sublines:
        return []

    oriented = [sublines[0]]

    for line in sublines[1:]:
        prev = oriented[-1]

        prev_end = prev[-1]
        curr_start = line[0]
        curr_end = line[-1]

        if np.array_equal(prev_end, curr_start):
            # already correctly oriented
            oriented.append(line)

        elif np.array_equal(prev_end, curr_end):
            # needs reversing
            oriented.append(line[::-1])

        else:
            raise ValueError("Sublines do not connect exactly.")

    return oriented


def get_matched_lines(rawmat: np.ndarray, half_edges: dict):
    print(rawmat.shape)
    N = len(half_edges)
    
    #print({k: v[0] for k, v in half_edges.items()})
    assert rawmat.ndim == 2
    assert rawmat.shape[0] == len(half_edges)+1
    # with utils.timer("computing assignment"):
    perm, _ = involutive_assignment2(rawmat)
    # with utils.timer("computing assignment slow"):
    #     perm2, _ = involutive_assignment(rawmat)
    # print("result equal: ", np.array_equal(perm, perm2))
        
    print("perm:", perm.shape)
    pairs = np.array([
        (x+1 if x%2 else x-1) for x in range(0, len(half_edges)+1)])
    assert len(pairs) == len(half_edges) +1
    nodes = np.arange(1, N+1)
    components = reconstruct_ordered_paths(nodes, pairs, perm)
    paths = []
    for idx_set in components:
        paths.append(
            reorient_sublines([half_edges[i][1] for i in idx_set])
        )
    
    return paths
