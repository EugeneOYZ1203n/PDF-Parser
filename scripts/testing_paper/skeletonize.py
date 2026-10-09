from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Neighbourhood definitions
# ---------------------------------------------------------------------------

NEIGHBOURS = (
    (-1, -1),  # 0
    (0, -1),   # 1
    (1, -1),   # 2
    (1, 0),    # 3
    (1, 1),    # 4
    (0, 1),    # 5
    (-1, 1),   # 6
    (-1, 0),   # 7
)

WEIGHTS = np.array(
    [
        4,
        3,
        4,
        3,
        4,
        3,
        4,
        3,
    ],
    dtype=np.int32,
)


# ---------------------------------------------------------------------------
# Neighbourhood utilities
# ---------------------------------------------------------------------------

def get_neighbour_values(
    image: np.ndarray,
    x: int,
    y: int,
) -> np.ndarray:
    """
    Return the 8-neighbour values around (x, y).

    Values outside the image are treated as zero.
    """

    values = np.zeros(
        8,
        dtype=image.dtype,
    )

    height, width = image.shape

    for i, (dx, dy) in enumerate(NEIGHBOURS):
        nx = x + dx
        ny = y + dy

        if (
            0 <= nx < width
            and 0 <= ny < height
        ):
            values[i] = image[ny, nx]

    return values


# ---------------------------------------------------------------------------
# (3,4) Chamfer distance transform
# ---------------------------------------------------------------------------

def chamfer_distance_transform(
    binary: np.ndarray,
) -> np.ndarray:
    """
    Compute the (3,4)-distance transform.

    Horizontal/vertical moves cost 3.
    Diagonal moves cost 4.

    Uses two raster passes.
    """

    foreground = binary > 0

    inf = np.iinfo(np.int32).max // 4

    dt = np.where(
        foreground,
        inf,
        0,
    ).astype(np.int32)

    height, width = dt.shape

    # Forward pass.
    for y in range(height):
        for x in range(width):
            if dt[y, x] == 0:
                continue

            best = dt[y, x]

            if x > 0:
                best = min(
                    best,
                    dt[y, x - 1] + 3,
                )

            if y > 0:
                best = min(
                    best,
                    dt[y - 1, x] + 3,
                )

                if x > 0:
                    best = min(
                        best,
                        dt[y - 1, x - 1] + 4,
                    )

                if x + 1 < width:
                    best = min(
                        best,
                        dt[y - 1, x + 1] + 4,
                    )

            dt[y, x] = best

    # Backward pass.
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            if dt[y, x] == 0:
                continue

            best = dt[y, x]

            if x + 1 < width:
                best = min(
                    best,
                    dt[y, x + 1] + 3,
                )

            if y + 1 < height:
                best = min(
                    best,
                    dt[y + 1, x] + 3,
                )

                if x > 0:
                    best = min(
                        best,
                        dt[y + 1, x - 1] + 4,
                    )

                if x + 1 < width:
                    best = min(
                        best,
                        dt[y + 1, x + 1] + 4,
                    )

            dt[y, x] = best

    return dt


# ---------------------------------------------------------------------------
# Maximal centres
# ---------------------------------------------------------------------------

def find_maximal_centers(
    dt: np.ndarray,
) -> np.ndarray:
    """
    Detect maximal centres according to the (3,4)-DT criterion.

    Labels 3 and 6 are interpreted as 1 and 5 when testing maximality.
    """

    height, width = dt.shape

    result = np.zeros_like(
        dt,
        dtype=bool,
    )

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            p = int(dt[y, x])

            if p <= 0:
                continue

            neighbours = get_neighbour_values(
                dt,
                x,
                y,
            ).astype(np.int32)

            corrected = neighbours.copy()

            corrected[corrected == 3] = 1
            corrected[corrected == 6] = 5

            corrected_p = p

            if corrected_p == 3:
                corrected_p = 1
            elif corrected_p == 6:
                corrected_p = 5

            if np.all(
                corrected < corrected_p + WEIGHTS
            ):
                result[y, x] = True

    return result


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

def count_components_8(
    values: np.ndarray,
) -> int:
    """Count 8-connected components."""

    binary = values.astype(bool)

    if not binary.any():
        return 0

    components = 0

    for i in range(8):
        previous = binary[(i - 1) % 8]

        if binary[i] and not previous:
            components += 1

    return components


def count_components_4(
    values: np.ndarray,
) -> int:
    """Count 4-connected components."""

    cardinal = np.array(
        [
            values[1],
            values[3],
            values[5],
            values[7],
        ],
        dtype=bool,
    )

    if not cardinal.any():
        return 0

    components = 0

    for i in range(4):
        previous = cardinal[(i - 1) % 4]

        if cardinal[i] and not previous:
            components += 1

    return components


# ---------------------------------------------------------------------------
# Saddle detection
# ---------------------------------------------------------------------------

def find_saddle_pixels(
    dt: np.ndarray,
    maximal_centers: np.ndarray,
) -> np.ndarray:
    """
    Detect saddle pixels according to the local DT definition.
    """

    height, width = dt.shape

    saddle = np.zeros_like(
        dt,
        dtype=bool,
    )

    for y in range(1, height - 1):
        for x in range(1, width - 1):

            p = int(dt[y, x])

            if p <= 0:
                continue

            if maximal_centers[y, x]:
                continue

            neighbours = get_neighbour_values(
                dt,
                x,
                y,
            ).astype(np.int32)

            higher = neighbours > p

            lower = (
                (neighbours > 0)
                & (neighbours < p)
            )

            higher_components = count_components_8(
                higher,
            )

            lower_components = count_components_4(
                lower,
            )

            condition_1 = higher_components > 1
            condition_2 = lower_components > 1

            condition_3 = False

            if p == 3:
                for i in range(0, 8, 2):
                    a = neighbours[i]
                    b = neighbours[(i + 1) % 8]
                    c = neighbours[(i + 2) % 8]

                    if (
                        a == 3
                        and b == 3
                        and c == 3
                    ):
                        condition_3 = True
                        break

            if (
                condition_1
                or condition_2
                or condition_3
            ):
                saddle[y, x] = True

    return saddle


# ---------------------------------------------------------------------------
# Gradient following
# ---------------------------------------------------------------------------

def best_higher_neighbour(
    dt: np.ndarray,
    x: int,
    y: int,
) -> tuple[int, int] | None:
    """
    Find the neighbouring pixel giving the largest positive DT gradient.
    """

    p = int(dt[y, x])

    best = None
    best_gradient = 0.0

    height, width = dt.shape

    for i, (dx, dy) in enumerate(NEIGHBOURS):
        nx = x + dx
        ny = y + dy

        if not (
            0 <= nx < width
            and 0 <= ny < height
        ):
            continue

        value = int(dt[ny, nx])

        if value <= p:
            continue

        gradient = (
            value - p
        ) / float(WEIGHTS[i])

        if gradient > best_gradient:
            best_gradient = gradient
            best = (nx, ny)

    return best


def grow_gradient_path(
    dt: np.ndarray,
    skeleton: np.ndarray,
    start: tuple[int, int],
) -> None:
    """Grow a skeleton path toward increasing DT values."""

    x, y = start

    visited: set[tuple[int, int]] = set()

    while True:
        if (x, y) in visited:
            break

        visited.add(
            (x, y)
        )

        skeleton[y, x] = True

        next_pixel = best_higher_neighbour(
            dt,
            x,
            y,
        )

        if next_pixel is None:
            break

        nx, ny = next_pixel

        if skeleton[ny, nx]:
            break

        x, y = nx, ny


# ---------------------------------------------------------------------------
# Skeleton construction
# ---------------------------------------------------------------------------

def build_skeleton(
    binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the raw distance-transform skeleton.

    Returns:
        skeleton:
            Boolean skeleton.

        dt:
            Chamfer distance transform.
    """

    dt = chamfer_distance_transform(
        binary,
    )

    maximal_centers = find_maximal_centers(
        dt,
    )

    saddle_pixels = find_saddle_pixels(
        dt,
        maximal_centers,
    )

    skeleton = (
        maximal_centers
        | saddle_pixels
    )

    starts = np.argwhere(
        skeleton,
    )

    for y, x in starts:
        grow_gradient_path(
            dt,
            skeleton,
            (
                int(x),
                int(y),
            ),
        )

    return skeleton, dt


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def skeletonize(
    binary: np.ndarray,
) -> np.ndarray:
    """
    Skeletonize a binary foreground mask using the
    (3,4)-distance-transform method.

    Args:
        binary:
            Binary foreground mask. Non-zero pixels are foreground.

    Returns:
        Boolean skeleton image.
    """

    skeleton, _ = build_skeleton(
        binary,
    )

    return skeleton