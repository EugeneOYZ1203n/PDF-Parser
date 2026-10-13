# Vendored from https://github.com/verven/InvolutionStrokeVectorization
# file main/graph_extraction/shapeutils.py @ 178eede602bfb488b46cfc38ea2176ff902a4da2
# (no LICENSE upstream -- see ../README.md). Edits, all mechanical:
#   - `np.concat` -> `np.concatenate` (numpy 1.x)
#   - `print` rebound to a debug logger
# Otherwise verbatim: do not reformat, so a re-vendor diffs cleanly.

import numpy as np
from matplotlib.path import Path
import cv2

from shapely.geometry import Polygon
from shapely import maximum_inscribed_circle
from scipy.spatial import Voronoi, cKDTree
from skimage.draw import line

from shapely.geometry import Polygon, Point, LineString, MultiPolygon

import logging as _logging
_LOG = _logging.getLogger("rastervec.P2.InvolutionStroke.upstream")


def print(*args, **_kwargs):  # noqa: A001 -- upstream's progress prints -> debug log
    _LOG.debug(" ".join(str(a) for a in args))



def compute_centroid(points):
    """
    Compute the centroid of a list of points or a (N,2) ndarray.
    Returns a tuple (x, y).
    """
    points = np.array(points)
    return tuple(points.mean(axis=0))

def merge_points(points):
    # Convert to a list of tuples and sort them (y first, then x) 
    # for deterministic behavior
    pts_list = sorted([tuple(p) for p in points])
    pts_set = set(pts_list)
    
    used = set()
    merged_points = []

    # --- Pass 1: Horizontal Merges (dx=1, dy=0) ---
    for x, y in pts_list:
        if (x, y) in used:
            continue
        
        neighbor = (x + 1, y)
        if neighbor in pts_set and neighbor not in used:
            # Create midpoint
            merged_points.append([(x + neighbor[0]) / 2.0, y])
            used.add((x, y))
            used.add(neighbor)

    # --- Pass 2: Vertical Merges (dx=0, dy=1) ---
    for x, y in pts_list:
        if (x, y) in used:
            continue
            
        neighbor = (x, y + 1)
        if neighbor in pts_set and neighbor not in used:
            # Create midpoint
            merged_points.append([x, (y + neighbor[1]) / 2.0])
            used.add((x, y))
            used.add(neighbor)

    # --- Pass 3: Collect unmerged points ---
    for p in pts_list:
        if p not in used:
            merged_points.append(list(p))

    return np.array(merged_points)

def get_closed_polygon_distances(points):
    """
    Calculates successive distances for a closed polygon.
    Returns N distances for N input points.
    """
    # Shift points by -1 to align p[i] with p[i+1] 
    # and p[last] with p[first]
    shifted_points = np.roll(points, -1, axis=0)
    
    # Calculate Euclidean distance between the original and shifted points
    distances = np.linalg.norm(points - shifted_points, axis=1)
    
    return distances

def merge_points_with_context(points):
    # Convert to set of tuples for O(1) lookups
    pts_list = [tuple(p) for p in points]
    pts_set = set(pts_list)
    
    used = set()
    merged_results = []

    def has_third_neighbor(pt, partner, original_set):
        """
        Checks if 'pt' has another point in 'original_set' (excluding itself and its 
        potential merge partner) within dx <= 2 and dy <= 2.
        """
        px, py = pt
        # Search the 5x5 grid around the point
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor = (px + dx, py + dy)
                if neighbor in original_set:
                    # Condition: must be a different point than the pair being merged
                    if neighbor != pt and neighbor != partner:
                        return True
        return False

    def can_merge(p1, p2, original_set):
        """Both points must satisfy the third-neighbor condition."""
        return has_third_neighbor(p1, p2, original_set) and \
               has_third_neighbor(p2, p1, original_set)

    # Sort list for deterministic behavior (y then x)
    sorted_pts = sorted(pts_list, key=lambda p: (p[1], p[0]))

    # --- Pass 1: Horizontal Merges (priority) ---
    for x, y in sorted_pts:
        if (x, y) in used:
            continue
        
        neighbor = (x + 1, y)
        if neighbor in pts_set and neighbor not in used:
            if can_merge((x, y), neighbor, pts_set):
                merged_results.append(((x + neighbor[0]) / 2.0, y))
                used.add((x, y))
                used.add(neighbor)

    # --- Pass 2: Vertical Merges ---
    for x, y in sorted_pts:
        if (x, y) in used:
            continue
            
        neighbor = (x, y + 1)
        if neighbor in pts_set and neighbor not in used:
            if can_merge((x, y), neighbor, pts_set):
                merged_results.append((x, (y + neighbor[1]) / 2.0))
                used.add((x, y))
                used.add(neighbor)

    # --- Pass 3: Collect unmerged points ---
    for p in sorted_pts:
        if p not in used:
            merged_results.append(tuple(p))

    return merged_results

def segment_intersection(p1, p2, q1, q2):
    """
    Compute the intersection point between two line segments p1–p2 and q1–q2.
    Only considers the general case (non-parallel, single intersection).
    Returns:
        intersection_point (np.ndarray of shape (2,)) if they intersect,
        otherwise None.
    """
    p1, p2, q1, q2 = map(np.asarray, (p1, p2, q1, q2))

    r = p2 - p1
    s = q2 - q1
    r_cross_s = r[0] * s[1] - r[1] * s[0]

    if np.isclose(r_cross_s, 0.0):
        # Parallel or collinear → skip (not general case)
        return None, None, None

    q_p = q1 - p1
    t = (q_p[0] * s[1] - q_p[1] * s[0]) / r_cross_s
    u = (q_p[0] * r[1] - q_p[1] * r[0]) / r_cross_s

    if 0 <= t <= 1 and 0 <= u <= 1:
        if t < 1e-3 or u < 1e-3:
            print("warning: small values", t, u)
            print(p1, p2, q1, q2)
        # Segments intersect within bounds
        intersection = p1 + t * r
        return (intersection, t, u)
    else:
        # Intersection point is outside one or both segments
        return None, None, None

def find_closest_furthest_point(points, target, closest=True):
    
    """
    Given a list of points and a target point, 
    returns the point in the list closest to the target.
    
    Parameters
    ----------
    points : np.ndarray of shape (n, d)
        List of n points in d-dimensional space.
    target : np.ndarray of shape (d,)
        The target point.

    Returns
    -------
    closest_point : np.ndarray of shape (d,)
        The point in `points` closest to `target`.
    closest_index : int
        Index of `closest_point` in `points`.
    distance : float
        Euclidean distance between the target and the closest point.
    """
    # Compute distances from target to each point
    t =  np.array(target)
    p = np.array(points)
    assert len(p.shape) > 1
    assert t.shape[0] == 2
    
    dists = np.linalg.norm(np.array(points) - t, axis=1)
    
    # Find index of minimum distance
    index = np.argmin(dists) if closest else np.argmax(dists)
    
    # Extract results
    closest_point = points[index]
    distance = dists[index]
    
    return closest_point, index, distance

def segment_pixels(p0, p1, approx = True):
    """
    Return all pixel coordinates (i, j) whose unit squares intersect
    the segment from p0 to p1.
    p0, p1: numpy arrays or tuples (x, y), with float coordinates.
    """
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    x0, y0 = p0
    x1, y1 = p1

    dx = x1 - x0
    dy = y1 - y0
    length = np.hypot(dx, dy)
    if length == 0:
        return np.array([(int(np.floor(x0)), int(np.floor(y0)))])

    # Direction steps
    step_x = np.sign(dx)
    step_y = np.sign(dy)
    t_delta_x = abs(1 / dx) if dx != 0 else np.inf
    t_delta_y = abs(1 / dy) if dy != 0 else np.inf

    # Start pixel
    ix = int(np.floor(x0))
    iy = int(np.floor(y0))

    # Initialize t_max_x, t_max_y (when line crosses first vertical/horizontal gridline)
    if dx > 0:
        t_max_x = t_delta_x * (1 - (x0 - ix))
    elif dx < 0:
        t_max_x = t_delta_x * (x0 - ix)
    else:
        t_max_x = np.inf

    if dy > 0:
        t_max_y = t_delta_y * (1 - (y0 - iy))
    elif dy < 0:
        t_max_y = t_delta_y * (y0 - iy)
    else:
        t_max_y = np.inf

    pixels = [(ix, iy)]
    t = 0.0

    while t <= 1:
        if t_max_x < t_max_y:
            ix += int(step_x)
            t = t_max_x
            t_max_x += t_delta_x
        else:
            iy += int(step_y)
            t = t_max_y
            t_max_y += t_delta_y

        if 0 <= t <= 1:
            p = (1-t)*p0 + t*p1
            if not approx or np.abs(p-np.round(p)).sum() > 1e-4:   
                pixels.append((ix, iy))
        else:
            break

    return np.array(pixels)


def segment_cover(p0: tuple, p1: tuple, mask: np.ndarray, approx=True):
    
    pixel_coords = segment_pixels(p0, p1, approx=approx)
    return np.sum(mask[pixel_coords[:, 1], pixel_coords[:, 0]] == 0)

def segment_blocked(p0, p1, blocked):
    r0, c0 = int(np.floor(p0[1])), int(np.floor(p0[0]))
    r1, c1 = int(np.floor(p1[1])), int(np.floor(p1[0]))
    rr, cc = line(r0, c0, r1, c1)
    return any((r, c) in blocked for r, c in zip(rr, cc))

def lerp(p0, p1, t):
    return p0*(1-t) + p1*t

def points_in_polygon(polygon: np.ndarray, points: np.ndarray):
    
    path = Path(polygon)

    # Vectorized point-in-polygon test
    inside = path.contains_points(points)

    return inside

def segment_polygon_intersections(polygon, p1, p2, tol=1e-10):
    """
    Computes intersections between a segment and a polygon.
    
    Returns: (starts_inside, t_values)
    - starts_inside: bool, True if the first sub-segment is inside the polygon.
    - t_values: list of floats, sorted intersection parameters in range [0, 1].
    """
    p1 = np.array(p1, dtype=np.float64)
    p2 = np.array(p2, dtype=np.float64)
    poly = np.array(polygon, dtype=np.float64)
    
    # 1. Setup edge vectors
    v1 = poly
    v2 = np.roll(poly, -1, axis=0) # Closed loop: (v0,v1), (v1,v2) ... (vN,v0)
    
    d_seg = p2 - p1
    d_edge = v2 - v1
    
    # 2. Solve linear system using 2D cross products (Cramer's Rule)
    # p1 + t*d_seg = v1 + u*d_edge
    # det is the denominator
    det = -d_seg[0] * d_edge[:, 1] + d_edge[:, 0] * d_seg[1]
    det += 1e-9
    
    # Filter parallel edges
    non_parallel = np.abs(det) > 1e-14
    
    dx = v1[:, 0] - p1[0]
    dy = v1[:, 1] - p1[1]
    
    # t = det([v1-p1, -d_edge]) / det
    t_all = (dx * (-d_edge[:, 1]) - (-d_edge[:, 0]) * dy) / det
    # u = det([d_seg, v1-p1]) / det
    u_all = (d_seg[0] * dy - dx * d_seg[1]) / det
    
    # Intersection must be within both segment bounds [0, 1]
    mask = non_parallel & (t_all >= -1e-12) & (t_all <= 1.0 + 1e-12) & \
                        (u_all >= -1e-12) & (u_all <= 1.0 + 1e-12)
    
    raw_t_values = t_all[mask]
    
    # 3. Handle exclusion rule
    valid_t = []
    for t_val in raw_t_values:
        # Check if t refers to the start or endpoint of the segment
        is_p1 = abs(t_val) < 1e-12
        is_p2 = abs(t_val - 1.0) < 1e-12
        
        if is_p1:
            # Check if p1 is exactly one of the polygon vertices
            if np.any(np.all(poly == p1, axis=1)):
                continue
        elif is_p2:
            # Check if p2 is exactly one of the polygon vertices
            if np.any(np.all(poly == p2, axis=1)):
                continue
        
        valid_t.append(float(np.clip(t_val, 0, 1)))
    
    # Sort and remove duplicates (e.g., hitting a vertex results in two edges returning same t)
    sorted_t = sorted(list(set(np.round(valid_t, decimals=10))))
    
    # 4. Determine if the segment starts inside
    # Test midpoint between t=0 and the first intersection (or t=1 if none)
    first_t = sorted_t[0] if sorted_t else 1.0
    mid_t = first_t / 2.0
    midpoint = p1 + mid_t * d_seg
    
    inside_flags = points_in_polygon(poly, midpoint.reshape(1, 2))
    starts_inside = bool(inside_flags[0])
    
    return starts_inside, sorted_t


def orthogonal_least_squares(points: np.ndarray):
    """
    Perform orthogonal (total) least squares fitting for a set of 2D points.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 2), where each row is a 2D point [x, y].

    Returns
    -------
    point : np.ndarray
        A point [x0, y0] lying on the best-fit line (the centroid of the data).
    normal : np.ndarray
        A unit normal vector [nx, ny] perpendicular to the best-fit line.
    """
    # Ensure correct shape
    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[1] == 2, "Input must be (N, 2)"

    # Compute the centroid
    centroid = points.mean(axis=0)

    # Center the data
    centered = points - centroid

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Singular Value Decomposition
    _, _, vh = np.linalg.svd(cov)

    # Normal vector = eigenvector corresponding to smallest singular value
    normal = vh[-1]

    # Normalize the normal vector
    normal /= np.linalg.norm(normal)

    return centroid, normal


def compute_aabb(polygon: np.ndarray) -> np.ndarray:
    """
    Compute the axis-aligned bounding box (AABB) of a 2D polygon.

    Parameters
    ----------
    polygon : np.ndarray
        Array of shape (n, 2) representing the polygon vertices.

    Returns
    -------
    aabb : np.ndarray
        Array of shape (2, 2), where:
        aabb[0] = [xmin, ymin]
        aabb[1] = [xmax, ymax]
    """
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("Polygon must be an array of shape (n, 2).")
    
    xmin, ymin = polygon.min(axis=0)
    xmax, ymax = polygon.max(axis=0)
    
    return np.array([[xmin, ymin], [xmax, ymax]])


def generate_grid_in_aabb(aabb: np.ndarray, step: float) -> np.ndarray:
    """
    Generate a grid of points within a given AABB, spaced by the specified step size.

    Parameters
    ----------
    aabb : np.ndarray
        Array of shape (2, 2), where:
        aabb[0] = [xmin, ymin]
        aabb[1] = [xmax, ymax]
    step : float
        Step size between grid points.

    Returns
    -------
    grid_points : np.ndarray
        Array of shape (m, 2) containing the sampled grid points.
    """
    if aabb.shape != (2, 2):
        raise ValueError("AABB must be an array of shape (2, 2).")
    if step <= 0:
        raise ValueError("Step size must be positive.")

    xmin, ymin = aabb[0]
    xmax, ymax = aabb[1]

    xs = np.arange(xmin, xmax + step, step)
    ys = np.arange(ymin, ymax + step, step)
    xv, yv = np.meshgrid(xs, ys)
    
    grid_points = np.column_stack((xv.ravel(), yv.ravel()))
    return grid_points

def angle_metric(p, out_points, out_vectors_norm):
    assert p.shape == (2,)
    assert out_vectors_norm.shape[1] == 2
    assert out_points.shape[1] == 2
    # Vector from out_points to p
    diffs = p - out_points  # shape (M, 2)
    diffs_norm = np.linalg.norm(diffs, axis=1, keepdims=True)

        # Initialize normalized diffs
    diffs_normed = np.empty_like(diffs)

    # Normal case: normalize normally
    nonzero_mask = diffs_norm.squeeze() > 0
    diffs_normed[nonzero_mask] = diffs[nonzero_mask] / diffs_norm[nonzero_mask]

    # Degenerate case: set opposite direction of out_vectors_norm (so that cosine = -1)
    diffs_normed[~nonzero_mask] = -out_vectors_norm[~nonzero_mask]

    # Compute neg cosine similarities for all out_vectors
    cosines = -np.sum(diffs_normed * out_vectors_norm, axis=1)

    # Sum up total neg cosine similarity for this point
    score = np.sum(cosines)
    return score

def best_point_angle_optimized(points: np.ndarray, out_points: np.ndarray, out_vectors: np.ndarray):
    assert points.shape[1] == 2
    assert out_points.shape[1] == 2
    assert out_vectors.shape[1] == 2
    assert out_points.shape == out_vectors.shape

    # Normalize out_vectors once
    out_vectors_norm = out_vectors / np.linalg.norm(out_vectors, axis=1, keepdims=True)

    best_score = -np.inf
    best_point = None

    for p in points:
        # Vector from out_points to p
        diffs = p - out_points  # shape (M, 2)
        diffs_norm = np.linalg.norm(diffs, axis=1, keepdims=True)

         # Initialize normalized diffs
        diffs_normed = np.empty_like(diffs)

        # Normal case: normalize normally
        nonzero_mask = diffs_norm.squeeze() > 0
        diffs_normed[nonzero_mask] = diffs[nonzero_mask] / diffs_norm[nonzero_mask]

        # Degenerate case: set opposite direction of out_vectors_norm (so that cosine = -1)
        diffs_normed[~nonzero_mask] = -out_vectors_norm[~nonzero_mask]

        # Compute neg cosine similarities for all out_vectors
        cosines = -np.sum(diffs_normed * out_vectors_norm, axis=1)

        # Sum up total neg cosine similarity for this point
        score = np.sum(cosines)
        # print("point, cosine score", p, score)

        # Track the best
        if score > best_score:
            best_score = score
            best_point = p

    return best_point, best_score


def grid_sample(img, points, mode="bilinear", eps=1e-7):
    """
    Fast grid sampling for a greyscale OpenCV image.
    
    Args:
        img: (H, W) numpy array (greyscale).
        points: (N, 2) numpy array of (x, y) float coordinates.
        mode: "bilinear" or "max".
        eps: epsilon for integer coordinate detection in "max" mode.
        
    Returns:
        (N,) numpy array of sampled values.
    """
    assert mode in ["bilinear", "bilinear shifted", "max", "max shifted", "min", "min shifted"]
    H, W = img.shape
    x = points[:, 0]
    y = points[:, 1]
    
    if "shifted" in mode:
        x = x-0.5
        y = y-0.5

    # Clip coordinates to stay within image boundaries
    # We use -1.001 to ensure that even after floor/ceil, we stay in bounds
    x = np.clip(x, 0, W - 1.0001)
    y = np.clip(y, 0, H - 1.0001)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    
        
    x1 = x0 + 1
    y1 = y0 + 1

    # Weights
    wx = x - x0
    wy = y - y0

    # Sample the 4 surrounding pixels
    v00 = img[y0, x0]
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]


    if "bilinear" in mode:

        # Bilinear interpolation formula
        return (1 - wx) * (1 - wy) * v00 + \
               wx * (1 - wy) * v01 + \
               (1 - wx) * wy * v10 + \
               wx * wy * v11
    elif "max" in mode:
        return max(v00, v01, v10, v11)
    else:
        return min(v00, v01, v10, v11)



def furthest_point_signed(L, mode=0):
    """
    Finds the index and signed distance of the point in polyline L furthest 
    from the line connecting L[0] and L[-1], with filtering options.

    Args:
        L (np.ndarray): Nx2 array representing the polyline.
        mode (int): 
             0 -> Consider BOTH sides (furthest absolute distance).
             1 -> Consider POSITIVE side only (furthest to the left).
            -1 -> Consider NEGATIVE side only (furthest to the right).

    Returns:
        tuple: (index (int), signed_distance (float))
    """
    p1 = L[0]
    p2 = L[-1]
    
    # 1. Setup vectors
    line_vec = p2 - p1
    point_vecs = L - p1
    
    # 2. Handle edge case: Start and end are the same (Line length is 0)
    # Signed distance is undefined without a line direction. 
    # We fallback to standard Euclidean distance (always positive).
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq == 0:
        dists_sq = np.sum(point_vecs**2, axis=1)
        idx = np.argmax(dists_sq)
        return idx, np.sqrt(dists_sq[idx])

    # 3. Compute 2D Cross Product (Signed Area)
    # Result is positive if point is 'left' of line, negative if 'right'
    cross_products = np.cross(line_vec, point_vecs)
    
    # 4. Determine Index based on Mode
    if mode == 0: 
        # Furthest distance regardless of side (Max Absolute Value)
        # We find the index of the max absolute, but we retrieve the 
        # original signed value to return the signed distance.
        idx = np.argmax(np.abs(cross_products))
        val = cross_products[idx]
        
    elif mode == 1:
        # Furthest on Positive side (Max Algebraic Value)
        # Note: If all points are on the negative side, this will return 
        # the point closest to the line (usually the start/end point with dist 0).
        idx = np.argmax(cross_products)
        val = cross_products[idx]
        
    elif mode == -1:
        # Furthest on Negative side (Min Algebraic Value)
        idx = np.argmin(cross_products)
        val = cross_products[idx]
        
    else:
        raise ValueError("Mode must be 0 (both), 1 (positive), or -1 (negative)")

    # 5. Convert Area to Distance
    # Signed Distance = Signed Area / Base Length
    line_len = np.sqrt(line_len_sq)
    signed_dist = val / line_len
    
    return idx, signed_dist


def normal_towards_point(segment, point):
    """
    Computes the normalized 2D vector normal to the segment 
    that points towards the specific point.

    Args:
        segment (np.ndarray): 2x2 array [[x1, y1], [x2, y2]]
        point (np.ndarray): 1D array [x, y]

    Returns:
        np.ndarray: Normalized vector [nx, ny]
    """
    p1 = segment[0]
    p2 = segment[1]
    
    # 1. Compute the line vector
    dx, dy = p2 - p1
    
    # 2. Create a candidate normal vector (rotate 90 degrees CCW)
    # The normal to (dx, dy) is (-dy, dx)
    normal = np.array([-dy, dx])
    
    # 3. Check alignment using Dot Product
    # Vector from start of segment to the target point
    vec_to_point = point - p1
    
    # If dot product is negative, the normal points away; flip it.
    if np.dot(normal, vec_to_point) < 0:
        normal = -normal
        
    # 4. Normalize (Handle zero-length segment edge case)
    norm = np.linalg.norm(normal)
    if norm == 0:
        # Segment is a point, or inputs are NaN. Return zero vector or handle error.
        return np.array([0.0, 0.0])
        
    return normal / norm



def get_polygon_voronoi(poly_array, densification_distance=0.1, connection_tol = 0.105):
    """
    Computes the Voronoi diagram inside a polygon.
    
    Returns:
    - vertices: (M, 2) numpy array of unique coordinates.
    - edge_indices: (E, 2) numpy array of indices into the vertices array.
    """
    poly = Polygon(poly_array)
    if not poly.is_valid:
        poly = poly.buffer(0)
        
    # 1. Densify the boundary to approximate the Segment Voronoi
    boundary = poly.exterior
    points = []
    distances = np.arange(0, boundary.length, densification_distance)
    for d in distances:
        pt = boundary.interpolate(d)
        points.append((pt.x, pt.y))
    points.extend(poly_array)
    points = np.unique(np.array(points), axis=0)
    
    # 2. Compute Point Voronoi
    vor = Voronoi(points)
    
    # 3. Filter ridges (edges) that are inside the polygon
    valid_ridges = []
    for ridge in vor.ridge_vertices:
        # Ignore infinite ridges
        if -1 in ridge:
            continue
            
        p1 = vor.vertices[ridge[0]]
        p2 = vor.vertices[ridge[1]]
        
        # Check if the midpoint of the edge is inside the polygon
        
        if (poly.contains(Point(p1))) and (
            poly.contains(Point(p2))):
            valid_ridges.append(ridge)
            
    # 4. Extract unique vertices used by valid ridges and re-index
    used_vertex_indices = sorted(list(set(idx for ridge in valid_ridges for idx in ridge)))
    
    # Create a mapping: {old_index: new_index}
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertex_indices)}
    
    # Final points array
    out_vertices = vor.vertices[used_vertex_indices]
    
    # Final edge indices array (mapped to the new points indices)
    out_edges = np.array([[index_map[v1], index_map[v2]] for v1, v2 in valid_ridges])
    
    return out_vertices, out_edges
    

def get_multipolygon_voronoi(poly_array: np.ndarray, holes_array: list[np.ndarray], densification_distance=0.1, connection_tol = 0.105):
    """
    Computes the Voronoi diagram inside a polygon.
    
    Returns:
    - vertices: (M, 2) numpy array of unique coordinates.
    - edge_indices: (E, 2) numpy array of indices into the vertices array.
    """
    # Create polygon with holes
    if holes_array is None:
        poly = Polygon(poly_array)
    else:
        poly = Polygon(poly_array, holes_array)
    
    # Validate and fix if necessary
    if isinstance(poly, MultiPolygon):
        # Use the largest polygon (or handle as needed)
        poly = max(poly.geoms, key=lambda p: p.area)
    
    # 1. Densify all boundaries (exterior + holes)
    points = []
    
    # Process exterior boundary
    exterior_boundary = poly.exterior
    distances = np.arange(0, exterior_boundary.length, densification_distance)
    for d in distances:
        pt = exterior_boundary.interpolate(d)
        points.append((pt.x, pt.y))
    
    # Process interior boundaries (holes)
    for interior_ring in poly.interiors:
        distances = np.arange(0, interior_ring.length, densification_distance)
        for d in distances:
            pt = interior_ring.interpolate(d)
            points.append((pt.x, pt.y))
    
    # Add original vertices (both exterior and hole vertices)
    points.extend(poly_array)
    if holes_array is not None:
        for hole in holes_array:
            points.extend(hole)
    
    # Remove duplicates
    points = np.unique(np.array(points), axis=0)
    
    # 2. Compute Point Voronoi
    vor = Voronoi(points)
    print("voronoi done ")
    
    # 3. Filter ridges (edges) that are inside the polygon
    valid_ridges = []
    for ridge in vor.ridge_vertices:
        # Ignore infinite ridges
        if -1 in ridge:
            continue
            
        p1 = vor.vertices[ridge[0]]
        p2 = vor.vertices[ridge[1]]
        
        # Check if the midpoint of the edge is inside the polygon
        
        if (poly.contains(Point(p1))) and (
            poly.contains(Point(p2))):
            valid_ridges.append(ridge)
            
    # 4. Extract unique vertices used by valid ridges and re-index
    used_vertex_indices = sorted(list(set(idx for ridge in valid_ridges for idx in ridge)))
    
    # Create a mapping: {old_index: new_index}
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertex_indices)}
    
    # Final points array
    out_vertices = vor.vertices[used_vertex_indices]
    
    # Final edge indices array (mapped to the new points indices)
    out_edges = np.array([[index_map[v1], index_map[v2]] for v1, v2 in valid_ridges])
    
    return out_vertices, out_edges
    


def get_max_inscribed_circle(poly_array, tolerance=None):
    """
    Finds the maximum inscribed circle of a polygon.
    
    Parameters:
    - poly_array: N, 2 numpy array of vertex coordinates.
    - tolerance: (Optional) The algorithm stops when the search area is 
                 smaller than this. Defaults to max(width, height) / 1000.
                 
    Returns:
    - center: numpy array [x, y]
    - radius: float
    """
    # 1. Create the Shapely polygon
    poly = Polygon(poly_array)
    if not poly.is_valid:
        poly = poly.buffer(0)
        
    # 2. Compute the maximum inscribed circle
    # Returns a LineString(center_point, boundary_point)
    result_line = maximum_inscribed_circle(poly, tolerance=tolerance)
    
    # 3. Extract points and calculate radius
    center = np.array(result_line.coords[0])
    
    radius = result_line.length # Length of the line connecting center to boundary
    
    return center, radius


def create_mask_from_points(points, mask):
    """
    Creates a new blank image the same size as the given mask,
    and sets specified pixel coordinates to 1.

    Args:
        points (list of tuple): List of (x, y) pixel coordinates.
        mask (np.ndarray): A cv2 mask (used to get image size).

    Returns:
        np.ndarray: New single-channel image with points set to 1.
    """
    # Ensure mask is valid
    if mask is None or not isinstance(mask, np.ndarray):
        raise ValueError("mask must be a valid numpy array")

    # Create a blank image (same height and width as mask)
    new_mask = np.zeros_like(mask, dtype=np.uint8)

    # Handle color masks by converting to single channel
    if new_mask.ndim == 3:
        new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2GRAY)

    # Set specified coordinates to 1
    for x, y in points:
        if 0 <= y < new_mask.shape[0] and 0 <= x < new_mask.shape[1]:
            new_mask[y, x] = 255  # note: OpenCV uses (y, x) order

    return new_mask



def smooth_polyline_constrained(points, d, alpha=0.5, iterations=50):
    """
    Smooths a 2D polyline using iterative Laplacian smoothing while strictly 
    constraining points to move at most distance `d` from their origin.
    
    Parameters:
        points (array-like): Nx2 array of (x, y) coordinates.
        d (float): Maximum allowed displacement radius for any point.
        alpha (float): Smoothing rate [0.0 to 1.0]. 0.5 is recommended for stability.
        iterations (int): Number of smoothing passes.
        
    Returns:
        np.ndarray: The smoothed Nx2 polyline.
    """
    # Ensure input is a float array and make a copy for the working set
    P = np.array(points, dtype=float)
    P0 = P.copy()  # Keep original points to enforce the distance constraint
    
    N = len(P)
    if N < 3:
        return P  # A line with < 3 points cannot be smoothed
        
    for _ in range(iterations):
        # --- 1. Smoothing Step ---
        # Calculate midpoints between (i-1) and (i+1) for all internal points
        midpoints = (P[:-2] + P[2:]) / 2.0
        
        # Move internal points towards their neighbor's midpoints
        P[1:-1] = (1.0 - alpha) * P[1:-1] + alpha * midpoints
        
        # --- 2. Constraint Step ---
        # Calculate displacement vectors from original positions
        diff = P[1:-1] - P0[1:-1]
        
        # Calculate distance of each displacement (L2 norm)
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        
        # Create a boolean mask of points that moved further than 'd'
        exceeded = (dist > d).flatten()
        
        if np.any(exceeded):
            # For points that exceeded d, scale their displacement back to exactly d
            # Note: dist[exceeded] is guaranteed to be > d, so no division-by-zero risk
            normalized_diff = diff[exceeded] / dist[exceeded]
            P[1:-1][exceeded] = P0[1:-1][exceeded] + normalized_diff * d
            
    return P


import numpy as np
from scipy.spatial.distance import cdist



def find_path(tip: np.ndarray, left: np.ndarray, right: np.ndarray, p: np.ndarray,
              d: float = 10.0, k: int = 50) -> list | None:
    dists = cdist(p, np.array([tip, left, right]))
    tip_idx   = int(np.argmin(dists[:, 0]))
    left_idx  = int(np.argmin(dists[:, 1]))
    right_idx = int(np.argmin(dists[:, 2]))

    print(f"[find_path] tip_idx={tip_idx} (closest poly pt: {p[tip_idx]})")
    print(f"[find_path] left_idx={left_idx} (closest poly pt: {p[left_idx]})")
    print(f"[find_path] right_idx={right_idx} (closest poly pt: {p[right_idx]})")
    print(f"[find_path] d={d}, k={k}")

    n = len(p)

    def walk_to_target(start_idx: int, direction: int, targets: dict) -> tuple[str, list] | None:
        dir_label = "forward" if direction == 1 else "backward"
        print(f"\n  [walk] Starting {dir_label} from idx={start_idx}, targets={list(targets.keys())}")

        for step in range(1, k + 1):
            idx = (start_idx + direction * step) % n
            curr_dists = {name: np.linalg.norm(p[idx] - pt) for name, pt in targets.items()}

            closest_name = min(curr_dists, key=curr_dists.__getitem__)
            closest_dist = curr_dists[closest_name]

            print(f"  [walk]   step={step} idx={idx} pt={p[idx]} | dists={ {k: f'{v:.3f}' for k, v in curr_dists.items()} } | closest={closest_name} ({closest_dist:.3f})")

            if closest_dist < d:
                indices = [(start_idx + direction * i) % n for i in range(1, step + 1)]
                print(f"  [walk] Reached '{closest_name}' within d={d} at step={step}, indices={indices}")
                return closest_name, indices

        print(f"  [walk] FAILED to reach any target within d={d} in k={k} steps, returning None")
        return None

    targets = {"left": left, "right": right}

    print(f"\n[find_path] Walking FORWARD from tip...")
    fwd_result = walk_to_target(tip_idx, +1, targets)
    if fwd_result is None:
        print("[find_path] Forward walk failed, returning None")
        return None
    fwd_name, fwd_indices = fwd_result

    remaining = {name: pt for name, pt in targets.items() if name != fwd_name}
    print(f"\n[find_path] Walking BACKWARD from tip (remaining target: {list(remaining.keys())})...")
    bwd_result = walk_to_target(tip_idx, -1, remaining)
    if bwd_result is None:
        print("[find_path] Backward walk failed, returning None")
        return None
    bwd_name, bwd_indices = bwd_result

    fwd_segment = [p[i] for i in fwd_indices]
    bwd_segment = [p[i] for i in bwd_indices]

    if fwd_name == "left":
        path = [left] + list(reversed(fwd_segment)) + [tip] + bwd_segment + [right]
    else:
        path = [left] + list(reversed(bwd_segment)) + [tip] + fwd_segment + [right]

    print(f"\n[find_path] Final path ({len(path)} points):")
    for i, pt in enumerate(path):
        label = ""
        if np.allclose(pt, tip):   label = " ← tip"
        if np.allclose(pt, left):  label = " ← left"
        if np.allclose(pt, right): label = " ← right"
        print(f"  [{i}] {pt}{label}")

    return path


def offset_polyline(points: np.ndarray, delta: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a 2D polyline, return two offset polylines displaced by delta
    in the normal direction (one for each side).
    
    Args:
        points: (N, 2) array of polyline vertices
        delta: scalar or (N,) array of offset distances
        
    Returns:
        left_poly, right_poly: two (N, 2) offset polyline arrays
    """
    n = len(points)
    delta = np.full(n, delta) if np.isscalar(delta) else np.asarray(delta)
    normals = np.zeros((n, 2))

    # Edge tangents
    edges = np.diff(points, axis=0)  # (N-1, 2)
    lengths = np.linalg.norm(edges, axis=1, keepdims=True)
    tangents = edges / lengths  # unit tangents

    # Interior points: bisector normal from consecutive tangents
    for i in range(1, n - 1):
        t0 = tangents[i - 1]
        t1 = tangents[i]
        bisector = t0 + t1
        b_len = np.linalg.norm(bisector)
        if b_len < 1e-10:
            # Anti-parallel tangents: use perpendicular of either tangent
            bisector = np.array([-t0[1], t0[0]])
        else:
            bisector /= b_len

        # Rotate bisector 90° to get candidate normal
        normal = np.array([-bisector[1], bisector[0]])

        # Ensure consistent orientation using determinant rule:
        # normal should point to the LEFT of t0 (det(t0, normal) > 0)
        if np.cross(t0, normal) < 0:
            normal = -normal

        normals[i] = normal

    # Endpoints: simple perpendicular of adjacent edge
    def perp_left(t):
        n = np.array([-t[1], t[0]])
        return n if np.cross(t, n) > 0 else -n

    normals[0] = perp_left(tangents[0])
    normals[-1] = perp_left(tangents[-1])

    left_poly  = points + normals * delta[:, None]
    right_poly = points - normals * delta[:, None]

    return left_poly, right_poly


def total_unsigned_turning(polyline: np.ndarray) -> float:
    """Total unsigned turning distance of a 2D polyline (Nx2 array)."""
    d = np.diff(polyline, axis=0)          # segment vectors, shape (N-1, 2)
    # cross and dot between consecutive segments
    cross = d[:-1, 0] * d[1:, 1] - d[:-1, 1] * d[1:, 0]
    dot   = d[:-1, 0] * d[1:, 0] + d[:-1, 1] * d[1:, 1]
    angles = np.arctan2(cross, dot)        # signed turning angles
    return np.sum(angles)