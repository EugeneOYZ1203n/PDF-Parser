from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from chaining import (
    chain_skeleton,
    draw_chains,
)
from color_clustering import (
    cluster_colors,
    cluster_mask,
    convert_to_lab,
    filter_clusters,
    lab_to_bgr,
    load_color,
)
from dilation import dilate
from skeletonize import skeletonize


def save_image(
    image: np.ndarray,
    output_path: Path,
) -> None:
    """Save an image."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        image,
    ):
        raise RuntimeError(
            f"Could not write image: {output_path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Color cluster, dilate, skeletonize, "
            "and chain an image."
        )
    )

    parser.add_argument(
        "input",
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("skeleton_output"),
    )

    # Color clustering.
    parser.add_argument(
        "--eps",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-colors",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-pixels",
        type=int,
        default=500_000,
    )

    # Dilation.
    parser.add_argument(
        "--dilation-kernel",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--dilation-iterations",
        type=int,
        default=0,
    )

    # Output.
    parser.add_argument(
        "--save-masks",
        action="store_true",
    )

    parser.add_argument(
        "--save-dilated",
        action="store_true",
    )

    parser.add_argument(
        "--save-chains",
        action="store_true",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load image
    # ---------------------------------------------------------

    image = load_color(
        args.input,
    )

    # ---------------------------------------------------------
    # Color clustering
    # ---------------------------------------------------------

    lab = convert_to_lab(
        image,
    )

    labels, centroids = cluster_colors(
        lab,
        eps=args.eps,
        min_samples=args.min_samples,
        max_colors=args.max_colors,
    )

    labels, centroids = filter_clusters(
        labels,
        centroids,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Process each cluster
    # ---------------------------------------------------------

    for cluster_index, centroid in enumerate(
        centroids
    ):
        mask = cluster_mask(
            labels,
            cluster_index,
        )

        pixel_count = int(
            np.count_nonzero(mask)
        )

        bgr = lab_to_bgr(
            centroid,
        )

        print(
            f"\nCluster {cluster_index:03d}: "
            f"{pixel_count:,} pixels, "
            f"BGR={bgr.tolist()}, "
            f"Lab={centroid.tolist()}"
        )

        cluster_dir = (
            args.output_dir
            / f"cluster_{cluster_index:03d}"
        )

        cluster_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -----------------------------------------------------
        # Color mask
        # -----------------------------------------------------

        if args.save_masks:
            save_image(
                mask,
                cluster_dir / "mask.png",
            )

        # -----------------------------------------------------
        # Dilation
        # -----------------------------------------------------

        dilated = dilate(
            mask,
            kernel_size=args.dilation_kernel,
            iterations=args.dilation_iterations,
        )

        if args.save_dilated:
            save_image(
                dilated,
                cluster_dir / "dilated.png",
            )

        # -----------------------------------------------------
        # Skeletonization
        # -----------------------------------------------------

        skeleton = skeletonize(
            dilated,
        )

        save_image(
            (skeleton * 255).astype(np.uint8),
            cluster_dir / "skeleton.png",
        )

        # -----------------------------------------------------
        # Chaining
        # -----------------------------------------------------

        nodes, chains = chain_skeleton(
            skeleton,
        )

        print(
            f"  Nodes:  {len(nodes):,}"
        )

        print(
            f"  Chains: {len(chains):,}"
        )

        if args.save_chains:
            chain_image = draw_chains(
                skeleton,
                nodes,
                chains,
            )

            save_image(
                chain_image,
                cluster_dir / "chains.png",
            )

    print(
        f"\nCreated {len(centroids)} "
        f"processed color clusters."
    )


if __name__ == "__main__":
    main()