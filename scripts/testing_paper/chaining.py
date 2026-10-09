from __future__ import annotations

import cv2
import numpy as np


NEIGHBOR_OFFSETS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def get_neighbors(
    skeleton: np.ndarray,
    y: int,
    x: int,
) -> list[tuple[int, int]]:
    """Return 8-connected skeleton neighbours."""

    height, width = skeleton.shape

    neighbors = []

    for dy, dx in NEIGHBOR_OFFSETS_8:
        ny = y + dy
        nx = x + dx

        if (
            0 <= ny < height
            and 0 <= nx < width
            and skeleton[ny, nx]
        ):
            neighbors.append(
                (ny, nx)
            )

    return neighbors


def find_nodes(
    skeleton: np.ndarray,
) -> set[tuple[int, int]]:
    """
    Find endpoints and junctions.

    degree == 1:
        endpoint

    degree >= 3:
        junction
    """

    nodes: set[tuple[int, int]] = set()

    ys, xs = np.nonzero(
        skeleton,
    )

    for y, x in zip(ys, xs):
        point = (
            int(y),
            int(x),
        )

        degree = len(
            get_neighbors(
                skeleton,
                *point,
            )
        )

        if degree == 1 or degree >= 3:
            nodes.add(point)

    return nodes


def trace_chain(
    skeleton: np.ndarray,
    start: tuple[int, int],
    first_neighbor: tuple[int, int],
    nodes: set[tuple[int, int]],
    visited_edges: set[
        tuple[
            tuple[int, int],
            tuple[int, int],
        ]
    ],
) -> list[tuple[int, int]]:
    """
    Trace a skeleton chain from one node to another.
    """

    chain = [start]

    previous = start
    current = first_neighbor

    while True:
        chain.append(current)

        edge = tuple(
            sorted(
                (previous, current)
            )
        )

        visited_edges.add(edge)

        if (
            current in nodes
            and current != start
        ):
            break

        neighbors = get_neighbors(
            skeleton,
            *current,
        )

        next_candidates = [
            neighbor
            for neighbor in neighbors
            if neighbor != previous
        ]

        if not next_candidates:
            break

        next_pixel = next_candidates[0]

        next_edge = tuple(
            sorted(
                (current, next_pixel)
            )
        )

        if next_edge in visited_edges:
            break

        previous = current
        current = next_pixel

    return chain


def chain_skeleton(
    skeleton: np.ndarray,
) -> tuple[
    set[tuple[int, int]],
    list[list[tuple[int, int]]],
]:
    """
    Convert a skeleton into nodes and ordered pixel chains.

    Returns:
        nodes:
            Endpoints and junctions.

        chains:
            Ordered sequences of skeleton pixels connecting nodes.
    """

    nodes = find_nodes(
        skeleton,
    )

    visited_edges: set[
        tuple[
            tuple[int, int],
            tuple[int, int],
        ]
    ] = set()

    chains: list[
        list[tuple[int, int]]
    ] = []

    # Trace chains starting at nodes.
    for node in nodes:
        for neighbor in get_neighbors(
            skeleton,
            *node,
        ):
            edge = tuple(
                sorted(
                    (node, neighbor)
                )
            )

            if edge in visited_edges:
                continue

            chain = trace_chain(
                skeleton,
                node,
                neighbor,
                nodes,
                visited_edges,
            )

            if len(chain) >= 2:
                chains.append(chain)

    # Handle closed loops.
    skeleton_pixels = set(
        zip(
            *np.nonzero(skeleton)
        )
    )

    visited_pixels = {
        pixel
        for edge in visited_edges
        for pixel in edge
    }

    remaining = (
        skeleton_pixels
        - visited_pixels
    )

    while remaining:
        start = next(iter(remaining))

        loop = [start]

        previous = None
        current = start

        while True:
            neighbors = get_neighbors(
                skeleton,
                *current,
            )

            candidates = [
                neighbor
                for neighbor in neighbors
                if neighbor != previous
            ]

            if not candidates:
                break

            next_pixel = candidates[0]

            if next_pixel == start:
                loop.append(start)
                break

            if next_pixel in loop:
                break

            loop.append(next_pixel)

            previous = current
            current = next_pixel

        if len(loop) >= 3:
            chains.append(loop)

        remaining -= set(loop)

    return nodes, chains


def draw_chains(
    skeleton: np.ndarray,
    nodes: set[tuple[int, int]],
    chains: list[list[tuple[int, int]]],
) -> np.ndarray:
    """
    Create a visualization of extracted chains.
    """

    output = np.zeros(
        (*skeleton.shape, 3),
        dtype=np.uint8,
    )

    for chain_index, chain in enumerate(chains):
        color = (
            int((37 * chain_index) % 256),
            int((97 * chain_index) % 256),
            int((173 * chain_index) % 256),
        )

        for y, x in chain:
            output[y, x] = color

    # Nodes are white.
    for y, x in nodes:
        cv2.circle(
            output,
            (x, y),
            radius=2,
            color=(255, 255, 255),
            thickness=-1,
        )

    return output