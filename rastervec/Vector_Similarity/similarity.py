"""Vector-level similarity grouping.

Groups raw `Vector`s (not clusters) by shape, independent of position, size,
or rotation, via a fixed 5-step test between a candidate `Vector` and a
group's own representative:

    1. item-type-count signature (a `Counter` of `Vector.items[i][0]`, e.g.
       `{"l": 4}` vs `{"c": 1, "l": 2}`) -- a mismatch means "not similar",
       skip immediately, no further geometry work.
    2. PCA over every item point (`helpers.geometry.item_points`) -> the
       principal axis -> rotate the point cloud so that axis points up.
    3. Scale the rotated cloud (uniformly, preserving aspect ratio) so its
       own bbox's longer side fits a 1x1 box.
    4. Translate so the bbox's own min corner sits at (0, 0).
    5. Mean squared error between corresponding points of the two
       normalized clouds (sorted into a canonical order first, since a
       Vector's own item/point order carries no meaning across instances).

This is a prototype: `notebooks/vector_similarity_lab.ipynb` exercises it
directly (extract -> group -> render, no clustering/classification), and
`reclassify_by_similarity` (`pipelines/_steps.py`) uses `vector_similarity_group`
to fold per-vector FAST pass/fail decisions up to a group consensus.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from rastervec.config import SIMILARITY_MSE_THRESHOLD
from rastervec.commons.helpers.geometry import item_points
from rastervec.commons.models import Vector

_ItemSignature = "tuple[tuple[str, int], ...]"


def item_type_signature(v: Vector) -> _ItemSignature:
    """A `Vector`'s item-kind counts (`"l"`, `"c"`, `"re"`, `"qu"`), sorted
    for a stable, hashable key. Two Vectors with a different signature are
    never considered similar (step 1)."""
    return tuple(sorted(Counter(item[0] for item in v.items).items()))


def _vector_points(v: Vector) -> np.ndarray:
    pts = [p for item in v.items for p in item_points(item)]
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))


def normalize_vector(v: Vector) -> np.ndarray:
    """Steps 2-4: PCA-align (principal axis pointing up), uniform-scale to
    fit a 1x1 box, translate the bbox's own min corner to (0, 0). Returns an
    `(N, 2)` array of normalized points, sorted into a canonical
    (lexicographic) order so two shape-identical Vectors normalize to the
    same point sequence regardless of their own items'/points' original
    order. A degenerate Vector (fewer than 2 distinct points) normalizes to
    a single point at the origin."""
    pts = _vector_points(v)
    if len(pts) < 2:
        return np.zeros((1, 2))

    centre = pts.mean(axis=0)
    centered = pts - centre
    cov = np.cov(centered.T)
    if not np.all(np.isfinite(cov)):
        return np.zeros((1, 2))
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]

    # Rotate so the principal axis points along +y ("up").
    theta = np.arctan2(principal[1], principal[0])
    rot = (np.pi / 2.0) - theta
    c, s = np.cos(rot), np.sin(rot)
    rot_matrix = np.array([[c, -s], [s, c]])
    rotated = centered @ rot_matrix.T

    mins = rotated.min(axis=0)
    maxs = rotated.max(axis=0)
    extent = np.maximum(maxs - mins, 1e-9)
    scale = 1.0 / float(extent.max())
    scaled = rotated * scale

    translated = scaled - scaled.min(axis=0)
    order = np.lexsort((translated[:, 1], translated[:, 0]))
    return translated[order]


def point_cloud_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared (Euclidean) distance between corresponding points of
    two already-normalized, canonically-sorted point clouds (step 5). A
    point-count mismatch pairs only the shorter length's worth (the
    mismatch itself is a similarity signal the item-type-signature gate,
    step 1, is meant to have already screened out)."""
    n = min(len(a), len(b))
    if n == 0:
        return float("inf")
    diff = a[:n] - b[:n]
    return float(np.mean(np.sum(diff * diff, axis=1)))


@dataclass
class SimilarityGroup:
    """One shape-similarity group: `members` in the order they were added
    (`members[0]` is the representative every later member was compared
    against)."""

    signature: _ItemSignature
    representative_normalized: np.ndarray
    members: list[Vector] = field(default_factory=list)

    @property
    def representative(self) -> Vector:
        return self.members[0]


def vector_similarity_group(
    vectors: list[Vector], *, mse_threshold: float = SIMILARITY_MSE_THRESHOLD,
) -> list[SimilarityGroup]:
    """Groups `vectors` by shape (position/size/rotation independent) via
    the 5-step test above. A new Vector is only ever compared against
    existing groups sharing its own `item_type_signature` (step 1 already
    requires an exact match, so groups of every other signature can never
    match it) -- `groups_by_signature` makes that a dict lookup instead of
    a full scan of every group on the page, which matters once a page has
    thousands of Vectors and hundreds of distinct shapes (a dense
    engineering drawing, say). Within one signature bucket it's still
    O(members) per Vector (compared against every group's representative
    until one matches `point_cloud_mse` under `mse_threshold`, else a new
    group starts) -- a prototype, not tuned for a signature bucket that is
    itself huge with high shape variety."""
    groups: list[SimilarityGroup] = []
    groups_by_signature: dict[_ItemSignature, list[SimilarityGroup]] = {}
    for v in vectors:
        sig = item_type_signature(v)
        norm = normalize_vector(v)
        bucket = groups_by_signature.get(sig)
        matched = None
        if bucket is not None:
            for g in bucket:
                if point_cloud_mse(norm, g.representative_normalized) <= mse_threshold:
                    matched = g
                    break
        if matched is not None:
            matched.members.append(v)
        else:
            g = SimilarityGroup(signature=sig, representative_normalized=norm, members=[v])
            groups.append(g)
            groups_by_signature.setdefault(sig, []).append(g)
    return groups
