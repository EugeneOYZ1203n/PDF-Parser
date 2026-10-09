from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def load_color(
    input_path,
) -> np.ndarray:
    """Load an image as BGR."""

    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    return image


def convert_to_lab(
    image: np.ndarray,
) -> np.ndarray:
    """Convert BGR image to Lab."""

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB,
    )


def cluster_colors(
    lab: np.ndarray,
    eps: float = 5.0,
    min_samples: int = 50,
    max_colors: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Discover color clusters using DBSCAN.

    DBSCAN is performed on frequent unique colors rather than
    individual pixels.

    Returns:
        labels:
            Cluster label for every pixel.

        centroids:
            Lab centroid for every cluster.
    """

    pixels = lab.reshape(-1, 3).astype(np.uint8)

    packed = (
        pixels[:, 0].astype(np.uint32) << 16
        | pixels[:, 1].astype(np.uint32) << 8
        | pixels[:, 2].astype(np.uint32)
    )

    unique_packed, counts = np.unique(
        packed,
        return_counts=True,
    )

    print(
        f"Found {len(unique_packed):,} unique colors"
    )

    if len(unique_packed) > max_colors:
        indices = np.argpartition(
            counts,
            -max_colors,
        )[-max_colors:]

        unique_packed = unique_packed[indices]
        counts = counts[indices]

    colors = np.column_stack(
        (
            (unique_packed >> 16) & 255,
            (unique_packed >> 8) & 255,
            unique_packed & 255,
        )
    ).astype(np.float32)

    print(
        f"Clustering {len(colors):,} unique colors"
    )

    effective_min_samples = min(
        min_samples,
        max(2, len(colors) // 1000),
    )

    clustering = DBSCAN(
        eps=eps,
        min_samples=effective_min_samples,
        metric="euclidean",
        n_jobs=-1,
    )

    color_labels = clustering.fit_predict(
        colors,
    )

    valid_labels = sorted(
        label
        for label in np.unique(color_labels)
        if label != -1
    )

    if not valid_labels:
        raise RuntimeError(
            "DBSCAN found no color clusters. "
            "Try increasing eps."
        )

    centroids = np.asarray(
        [
            np.average(
                colors[color_labels == label],
                axis=0,
                weights=counts[color_labels == label],
            )
            for label in valid_labels
        ],
        dtype=np.float32,
    )

    print(
        f"Discovered {len(centroids)} color clusters"
    )

    # Assign every pixel to its nearest centroid.
    pixels_float = pixels.astype(np.float32)

    labels = np.empty(
        len(pixels_float),
        dtype=np.int32,
    )

    centroid_norms = np.sum(
        centroids ** 2,
        axis=1,
    )

    chunk_size = 100_000

    for start in range(
        0,
        len(pixels_float),
        chunk_size,
    ):
        end = min(
            start + chunk_size,
            len(pixels_float),
        )

        chunk = pixels_float[start:end]

        distances = (
            np.sum(
                chunk ** 2,
                axis=1,
            )[:, None]
            - 2 * chunk @ centroids.T
            + centroid_norms
        )

        labels[start:end] = np.argmin(
            distances,
            axis=1,
        )

    labels = labels.reshape(
        lab.shape[:2],
    )

    return labels, centroids


def filter_clusters(
    labels: np.ndarray,
    centroids: np.ndarray,
    min_pixels: int = 100,
    max_pixels: int = 500_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove color clusters whose pixel count is outside
    [min_pixels, max_pixels].

    Returns:
        filtered_labels:
            Labels where removed clusters are -1.

        filtered_centroids:
            Centroids corresponding to remaining labels.
    """

    counts = np.bincount(
        labels.ravel(),
        minlength=len(centroids),
    )

    keep = (
        (counts >= min_pixels)
        & (counts <= max_pixels)
    )

    print(
        f"Keeping {np.count_nonzero(keep)} / "
        f"{len(centroids)} color clusters"
    )

    for index, count in enumerate(counts):
        if keep[index]:
            print(
                f"  KEEP   cluster {index:03d}: "
                f"{count:,} pixels"
            )
        else:
            print(
                f"  REMOVE cluster {index:03d}: "
                f"{count:,} pixels"
            )

    filtered_labels = labels.copy()

    filtered_labels[
        ~keep[filtered_labels]
    ] = -1

    kept_indices = np.flatnonzero(keep)

    # Renumber remaining clusters from 0.
    remap = np.full(
        len(centroids),
        -1,
        dtype=np.int32,
    )

    remap[kept_indices] = np.arange(
        len(kept_indices),
    )

    positive = filtered_labels >= 0

    filtered_labels[positive] = remap[
        filtered_labels[positive]
    ]

    filtered_centroids = centroids[
        kept_indices
    ]

    return (
        filtered_labels,
        filtered_centroids,
    )


def cluster_mask(
    labels: np.ndarray,
    cluster_index: int,
) -> np.ndarray:
    """Create a binary mask for one color cluster."""

    return np.where(
        labels == cluster_index,
        255,
        0,
    ).astype(np.uint8)


def lab_to_bgr(
    center: np.ndarray,
) -> np.ndarray:
    """Convert a Lab centroid to BGR."""

    lab = np.array(
        [[center]],
        dtype=np.float32,
    )

    lab = np.clip(
        lab,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR,
    )[0, 0]