"""
Advanced Synthetic Blueprint Dataset Generator (v10.1 - Loss-Aligned Geometry Targets)

Key Improvements for Loss Compatibility:
- Discretized Anti-Aliased Direction Bins: Adjacent bins receive soft angular weight 
  to prevent step-aliasing across 5-degree bin boundaries.
- Fixed Empty Junction Handling: Junctions without edges yield ALL-ZERO direction targets 
  rather than activating all 72 channels.
- Guaranteed Shape Alignment: Outputs strictly [1, H, W] for junction targets and 
  [DIRECTION_BINS, H, W] float32 tensors.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================
# Configuration Parameters
# ============================================================

DIRECTION_BINS = 72          # 360 degrees / 5-degree resolution
JUNCTION_SNAP = 4.0          # Spatial clustering threshold (pixels)
MIN_EDGE_LENGTH = 4.0        # Filter out negligible edges

BASE_ANGLES_HIERARCHY = {
    "90_deg":    ([0.0, 90.0, 180.0, 270.0], 0.40),
    "45_deg":    ([45.0, 135.0, 225.0, 315.0], 0.25),
    "22_5_deg":  ([22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5], 0.15),
    "12_5_deg":  ([12.5, 37.5, 62.5, 87.5, 102.5, 127.5, 152.5, 177.5, 
                   192.5, 217.5, 242.5, 267.5, 282.5, 307.5, 332.5, 357.5], 0.10),
    "6_25_deg":  ([6.25, 18.75, 31.25, 43.75, 56.25, 68.75, 81.25, 93.75,
                   106.25, 118.75, 131.25, 143.75, 156.25, 168.75, 181.25, 193.75,
                   206.25, 218.75, 231.25, 243.75, 256.25, 268.75, 281.25, 293.75,
                   306.25, 318.75, 331.25, 343.75, 356.25], 0.06),
    "3_125_deg": ([3.125, 15.625, 28.125, 40.625, 53.125, 65.625, 78.125, 90.625,
                   103.125, 115.625, 128.125, 140.625, 153.125, 165.625, 178.125, 190.625,
                   203.125, 215.625, 228.125, 240.625, 253.125, 265.625, 278.125, 290.625,
                   303.125, 315.625, 328.125, 340.625, 353.125], 0.04),
}


class WallStyle(Enum):
    SOLID = "solid"
    DASHED = "dashed"
    HOLLOW = "hollow"


def sample_hierarchical_angle() -> float:
    categories = list(BASE_ANGLES_HIERARCHY.keys())
    weights = [BASE_ANGLES_HIERARCHY[cat][1] for cat in categories]
    chosen_cat = random.choices(categories, weights=weights, k=1)[0]
    
    base_angle = random.choice(BASE_ANGLES_HIERARCHY[chosen_cat][0])
    jitter = random.uniform(-2.5, 2.5)
    return math.radians((base_angle + jitter) % 360.0)


# ============================================================
# Geometry & Helper Utilities
# ============================================================

@dataclass(slots=True)
class Point:
    x: float
    y: float

    def dist(self, other: Point) -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass(slots=True)
class Segment:
    p1: Point
    p2: Point
    thickness: int = 2
    intensity: int = 70
    style: WallStyle = WallStyle.SOLID


def dist_to_segment(pt: Point, p1: Point, p2: Point) -> float:
    dx, dy = p2.x - p1.x, p2.y - p1.y
    if dx == 0 and dy == 0:
        return pt.dist(p1)
    
    t = ((pt.x - p1.x) * dx + (pt.y - p1.y) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj = Point(p1.x + t * dx, p1.y + t * dy)
    return pt.dist(proj)


def dedup_points(points: List[Point], thresh: float = 3.0) -> List[Point]:
    unique: List[Point] = []
    for p in points:
        if not any(p.dist(u) <= thresh for u in unique):
            unique.append(p)
    return unique


class VectorScene:
    def __init__(self, max_segments: int):
        self.segments: List[Segment] = []
        self.erasure_rects: List[Tuple[float, float, float, float]] = []
        self.max_segments = max_segments

    def add_segment(self, p1: Point, p2: Point, thickness: int = 2, intensity: int = 70, style: WallStyle = WallStyle.SOLID) -> bool:
        if len(self.segments) >= self.max_segments:
            return False

        if p1.dist(p2) >= MIN_EDGE_LENGTH:
            self.segments.append(Segment(p1, p2, thickness, intensity, style))
            return True
        return False

    def add_line_erasure_mask(self, x: float, y: float, max_size: float):
        self.erasure_rects.append((x, y, x + max_size, y + max_size))


# ============================================================
# Spatial Partitioning & Ray Casting
# ============================================================

def segment_intersection_coords(p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Tuple[float, float]]:
    rx, ry = p2.x - p1.x, p2.y - p1.y
    sx, sy = p4.x - p3.x, p4.y - p3.y
    denom = rx * sy - ry * sx

    if abs(denom) < 1e-6:
        return None

    qx, qy = p3.x - p1.x, p3.y - p1.y
    t = (qx * sy - qy * sx) / denom
    u = (qx * ry - qy * rx) / denom

    if -1e-4 <= t <= 1.0 + 1e-4 and -1e-4 <= u <= 1.0 + 1e-4:
        return (p1.x + t * rx, p1.y + t * ry)
    return None


def ray_cast_perimeter_intersection(start: Point, angle_rad: float, poly: List[Point]) -> Optional[Point]:
    dx, dy = math.cos(angle_rad), math.sin(angle_rad)
    far_p = Point(start.x + 3000.0 * dx, start.y + 3000.0 * dy)
    
    closest_dist = float('inf')
    closest_pt = None

    n = len(poly)
    for i in range(n):
        edge_p1, edge_p2 = poly[i], poly[(i + 1) % n]
        res = segment_intersection_coords(start, far_p, edge_p1, edge_p2)
        if res:
            ipt = Point(res[0], res[1])
            d = start.dist(ipt)
            if d > 1.0 and d < closest_dist:
                closest_dist = d
                closest_pt = ipt

    return closest_pt


def generate_outer_boundary(target_nodes: int, canvas_w: float, canvas_h: float) -> List[Point]:
    cx, cy = canvas_w / 2.0, canvas_h / 2.0
    max_radius = min(canvas_w, canvas_h) / 2.0 - 24.0

    angles = sorted([sample_hierarchical_angle() for _ in range(target_nodes)])
    
    pts = []
    for a in angles:
        r_factor = random.uniform(0.75, 0.95)
        r = max_radius * r_factor
        px = np.clip(cx + r * math.cos(a), 20.0, canvas_w - 20.0)
        py = np.clip(cy + r * math.sin(a), 20.0, canvas_h - 20.0)
        pts.append(Point(px, py))

    return dedup_points(pts, thresh=15.0)


def partition_interior(boundary_poly: List[Point], interior_budget: int, scene: VectorScene):
    if len(boundary_poly) < 3:
        return

    min_x = min(p.x for p in boundary_poly) + 15
    max_x = max(p.x for p in boundary_poly) - 15
    min_y = min(p.y for p in boundary_poly) + 15
    max_y = max(p.y for p in boundary_poly) - 15

    interior_nodes: List[Point] = []
    spatial_dist = 30.0

    for _ in range(500):
        if len(interior_nodes) >= interior_budget:
            break
        
        pt = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if all(pt.dist(other) > spatial_dist for other in interior_nodes):
            interior_nodes.append(pt)
        
        spatial_dist = max(10.0, spatial_dist * 0.995)

    wall_styles = [WallStyle.SOLID, WallStyle.DASHED, WallStyle.HOLLOW]

    for node in interior_nodes:
        angles_to_try = [sample_hierarchical_angle() for _ in range(3)]

        for angle_rad in angles_to_try:
            style = random.choice(wall_styles)
            thick = random.choice([2, 3, 4])

            hit = ray_cast_perimeter_intersection(node, angle_rad, boundary_poly)
            if hit:
                scene.add_segment(node, hit, thickness=thick, style=style)


# ============================================================
# Hollow Wall Expansion & Physical Segment Extraction
# ============================================================

def expand_hollow_walls(scene: VectorScene) -> VectorScene:
    expanded_scene = VectorScene(max_segments=scene.max_segments * 2)
    expanded_scene.erasure_rects = scene.erasure_rects

    for seg in scene.segments:
        if seg.style == WallStyle.HOLLOW and seg.thickness >= 2:
            dx, dy = seg.p2.x - seg.p1.x, seg.p2.y - seg.p1.y
            length = math.hypot(dx, dy)
            if length > 0:
                nx = -dy / length * (seg.thickness / 2.0)
                ny =  dx / length * (seg.thickness / 2.0)

                p1_a = Point(seg.p1.x + nx, seg.p1.y + ny)
                p2_a = Point(seg.p2.x + nx, seg.p2.y + ny)
                p1_b = Point(seg.p1.x - nx, seg.p1.y - ny)
                p2_b = Point(seg.p2.x - nx, seg.p2.y - ny)

                expanded_scene.add_segment(p1_a, p2_a, thickness=1, intensity=seg.intensity, style=WallStyle.SOLID)
                expanded_scene.add_segment(p1_b, p2_b, thickness=1, intensity=seg.intensity, style=WallStyle.SOLID)
            else:
                expanded_scene.add_segment(seg.p1, seg.p2, seg.thickness, seg.intensity, seg.style)
        else:
            expanded_scene.add_segment(seg.p1, seg.p2, seg.thickness, seg.intensity, seg.style)

    return expanded_scene


def clip_segment_to_canvas_border(p1: Point, p2: Point, w: float, h: float) -> Tuple[Optional[Point], Optional[Point], List[Point]]:
    rect_lines = [
        (Point(0, 0), Point(w - 1, 0)),
        (Point(w - 1, 0), Point(w - 1, h - 1)),
        (Point(w - 1, h - 1), Point(0, h - 1)),
        (Point(0, h - 1), Point(0, 0))
    ]

    p1_in = (0 <= p1.x < w) and (0 <= p1.y < h)
    p2_in = (0 <= p2.x < w) and (0 <= p2.y < h)

    intersections: List[Point] = []
    for bp1, bp2 in rect_lines:
        res = segment_intersection_coords(p1, p2, bp1, bp2)
        if res:
            intersections.append(Point(res[0], res[1]))

    intersections = dedup_points(intersections, thresh=2.0)

    if p1_in and p2_in:
        return p1, p2, []

    candidates = []
    if p1_in: candidates.append(p1)
    if p2_in: candidates.append(p2)
    candidates.extend(intersections)

    if len(candidates) < 2:
        return None, None, []

    candidates.sort(key=lambda pt: (pt.x - p1.x)**2 + (pt.y - p1.y)**2)
    return candidates[0], candidates[-1], intersections


def apply_bounded_vector_augmentations(scene: VectorScene, canvas_w: float, canvas_h: float):
    angle = random.choice([0.0, 90.0, 180.0, 270.0]) if random.random() < 0.5 else random.uniform(-10.0, 10.0)
    scale = random.uniform(0.90, 1.05)
    tx = random.uniform(-10.0, 10.0)
    ty = random.uniform(-10.0, 10.0)

    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    cx, cy = canvas_w / 2.0, canvas_h / 2.0

    def transform_pt(p: Point) -> Point:
        px, py = (p.x - cx) * scale, (p.y - cy) * scale
        rx = px * cos_a - py * sin_a + cx + tx
        ry = px * sin_a + py * cos_a + cy + ty
        return Point(rx, ry)

    new_segments: List[Segment] = []
    for seg in scene.segments:
        tp1, tp2 = transform_pt(seg.p1), transform_pt(seg.p2)
        new_segments.append(Segment(tp1, tp2, seg.thickness, seg.intensity, seg.style))

    scene.segments = new_segments


@dataclass
class GraphJunction:
    position: Point
    connected_angles: List[float] = field(default_factory=list)


def extract_junction_graph(scene: VectorScene, canvas_w: float, canvas_h: float) -> Tuple[VectorScene, List[GraphJunction]]:
    scene = expand_hollow_walls(scene)

    clipped_segments: List[Segment] = []
    border_cut_junctions: List[Point] = []

    for seg in scene.segments:
        cp1, cp2, border_pts = clip_segment_to_canvas_border(seg.p1, seg.p2, canvas_w, canvas_h)
        if cp1 and cp2 and cp1.dist(cp2) >= MIN_EDGE_LENGTH:
            clipped_segments.append(Segment(cp1, cp2, seg.thickness, seg.intensity, seg.style))
            border_cut_junctions.extend(border_pts)

    scene.segments = clipped_segments

    raw_points: List[Point] = list(border_cut_junctions)
    segs = scene.segments

    for seg in segs:
        raw_points.append(seg.p1)
        raw_points.append(seg.p2)

    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            res = segment_intersection_coords(segs[i].p1, segs[i].p2, segs[j].p1, segs[j].p2)
            if res:
                ix, iy = res
                if 0 <= ix < canvas_w and 0 <= iy < canvas_h:
                    raw_points.append(Point(ix, iy))

    clustered_pts = dedup_points(raw_points, thresh=JUNCTION_SNAP)
    junctions: List[GraphJunction] = [GraphJunction(position=pt) for pt in clustered_pts]

    for junc in junctions:
        jp = junc.position
        angles: List[float] = []
        for seg in segs:
            if dist_to_segment(jp, seg.p1, seg.p2) <= JUNCTION_SNAP:
                if jp.dist(seg.p1) > 1.5:
                    angles.append(math.degrees(math.atan2(seg.p1.y - jp.y, seg.p1.x - jp.x)) % 360.0)
                if jp.dist(seg.p2) > 1.5:
                    angles.append(math.degrees(math.atan2(seg.p2.y - jp.y, seg.p2.x - jp.x)) % 360.0)

        junc.connected_angles = angles

    return scene, junctions


# ============================================================
# Rendering & Target Map Generation
# ============================================================

def render_scene(scene: VectorScene, w: int, h: int) -> np.ndarray:
    img = np.full((h, w), 255, dtype=np.uint8)

    for seg in scene.segments:
        p1 = (int(np.clip(round(seg.p1.x), 0, w - 1)), int(np.clip(round(seg.p1.y), 0, h - 1)))
        p2 = (int(np.clip(round(seg.p2.x), 0, w - 1)), int(np.clip(round(seg.p2.y), 0, h - 1)))

        if seg.style == WallStyle.DASHED:
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            num_dashes = int(dist / 8.0)
            for d in range(num_dashes):
                if d % 2 == 0:
                    t1, t2 = d / max(1, num_dashes), (d + 1) / max(1, num_dashes)
                    sp = (int(p1[0] + (p2[0] - p1[0]) * t1), int(p1[1] + (p2[1] - p1[1]) * t1))
                    ep = (int(p1[0] + (p2[0] - p1[0]) * t2), int(p1[1] + (p2[1] - p1[1]) * t2))
                    cv2.line(img, sp, ep, seg.intensity, seg.thickness)
        else:
            cv2.line(img, p1, p2, seg.intensity, seg.thickness, cv2.LINE_AA)

    for x1, y1, x2, y2 in scene.erasure_rects:
        p1 = (int(np.clip(x1, 0, w - 1)), int(np.clip(y1, 0, h - 1)))
        p2 = (int(np.clip(x2, 0, w - 1)), int(np.clip(y2, 0, h - 1)))
        cv2.rectangle(img, p1, p2, 255, -1)

    return img


def generate_targets(
    junctions: List[GraphJunction], 
    w: int, 
    h: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates target tensors strictly aligned for junction_loss and direction_loss.
    Returns:
        heatmap: [1, H, W] float32 tensor
        direction_bins: [DIRECTION_BINS, H, W] float32 tensor
    """
    heatmap = np.zeros((1, h, w), dtype=np.float32)
    direction_bins = np.zeros((DIRECTION_BINS, h, w), dtype=np.float32)

    sigma = 2.0
    rad = int(3 * sigma)
    kernel_size = 2 * rad + 1

    y_k, x_k = np.ogrid[-rad:rad+1, -rad:rad+1]
    g_kernel = np.exp(-(x_k**2 + y_k**2) / (2 * sigma**2)).astype(np.float32)
    bin_size = 360.0 / DIRECTION_BINS

    for junc in junctions:
        jx, jy = round(junc.position.x), round(junc.position.y)
        if not (0 <= jx < w and 0 <= jy < h):
            continue

        x0, x1 = max(0, jx - rad), min(w, jx + rad + 1)
        y0, y1 = max(0, jy - rad), min(h, jy + rad + 1)

        kx0, kx1 = x0 - (jx - rad), kernel_size - ((jx + rad + 1) - x1)
        ky0, ky1 = y0 - (jy - rad), kernel_size - ((jy + rad + 1) - y1)

        patch = g_kernel[ky0:ky1, kx0:kx1]
        heatmap[0, y0:y1, x0:x1] = np.maximum(heatmap[0, y0:y1, x0:x1], patch)

        active_mask = (patch > 0.1).astype(np.float32)

        # FIXED: Only populate direction targets if connected angles exist
        if len(junc.connected_angles) > 0:
            for angle in junc.connected_angles:
                norm_angle = (angle % 360.0) / bin_size
                bin_float = norm_angle % DIRECTION_BINS
                primary_bin = int(bin_float) % DIRECTION_BINS
                
                # Activate exact discretized bin
                direction_bins[primary_bin, y0:y1, x0:x1] = np.maximum(
                    direction_bins[primary_bin, y0:y1, x0:x1], active_mask
                )
                
                # Soft discretization step to prevent aliasing noise in direction_loss
                frac = bin_float - int(bin_float)
                if frac > 0.6:
                    next_bin = (primary_bin + 1) % DIRECTION_BINS
                    direction_bins[next_bin, y0:y1, x0:x1] = np.maximum(
                        direction_bins[next_bin, y0:y1, x0:x1], active_mask * 0.5
                    )
                elif frac < 0.4:
                    prev_bin = (primary_bin - 1) % DIRECTION_BINS
                    direction_bins[prev_bin, y0:y1, x0:x1] = np.maximum(
                        direction_bins[prev_bin, y0:y1, x0:x1], active_mask * 0.5
                    )

    return torch.from_numpy(heatmap), torch.from_numpy(direction_bins)


# ============================================================
# PyTorch Dataset Class
# ============================================================

class SyntheticDataset(Dataset):
    def __init__(self, length: int, width: int = 512, height: int = 512,
                 num_junctions: int = 40, augment: bool = True):
        self.length = length
        self.width = width
        self.height = height
        self.num_junctions = num_junctions
        self.augment = augment

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        scene = VectorScene(max_segments=self.num_junctions * 6)

        outer_budget = max(4, int(self.num_junctions * 0.2))
        interior_budget = max(4, self.num_junctions - outer_budget)

        boundary_poly = generate_outer_boundary(outer_budget, self.width, self.height)

        n_hull = len(boundary_poly)
        for i in range(n_hull):
            p1, p2 = boundary_poly[i], boundary_poly[(i + 1) % n_hull]
            scene.add_segment(p1, p2, thickness=3, style=random.choice([WallStyle.SOLID, WallStyle.HOLLOW]))

        partition_interior(boundary_poly, interior_budget, scene)

        erasure_count = random.randint(self.num_junctions, 2 * self.num_junctions)
        max_size = min(self.width, self.height) * 0.05
        for _ in range(erasure_count):
            size = random.uniform(4.0, max_size)
            x = random.uniform(10, self.width - 10 - size)
            y = random.uniform(10, self.height - 10 - size)
            scene.add_line_erasure_mask(x, y, size)

        if self.augment:
            apply_bounded_vector_augmentations(scene, self.width, self.height)

        scene, junctions = extract_junction_graph(scene, self.width, self.height)

        image_np = render_scene(scene, self.width, self.height)
        heatmap_tensor, direction_tensor = generate_targets(junctions, self.width, self.height)

        image_tensor = 1.0 - (torch.from_numpy(image_np).float() / 255.0)

        return {
            "image": image_tensor.unsqueeze(0),   # Shape: [1, H, W]
            "junction": heatmap_tensor,           # Shape: [1, H, W]
            "direction": direction_tensor,         # Shape: [72, H, W]
        }