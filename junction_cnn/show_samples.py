import random
import cv2
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from junction_cnn.synthetic import SyntheticDataset

# Settings
NUM_SAMPLES = 6
WIDTH = 512
HEIGHT = 512
NUM_JUNCTIONS = 40
DIRECTION_BINS = 72


def render_direction_spokes(
    image: np.ndarray,
    junction_tensor: np.ndarray,
    direction_tensor: np.ndarray,
    junction_thresh: float = 0.3,
    direction_thresh: float = 0.3,
    spoke_len: float = 14.0,
) -> np.ndarray:
    """
    Overlays colored polar spokes at junction centers pointing along all active directional angles.
    """
    h, w = image.shape
    # Convert grayscale input image (0-1 float) to 3-channel RGB uint8 image for overlay
    base_img = (image * 255.0).clip(0, 255).astype(np.uint8)
    vis_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2RGB)

    bin_size_deg = 360.0 / DIRECTION_BINS

    # 1. Find local junction centers (thresholding + local maximum peak extraction)
    junc_map = junction_tensor[0] if junction_tensor.ndim == 3 else junction_tensor
    active_y, active_x = np.where(junc_map > junction_thresh)

    if len(active_y) == 0:
        return vis_img

    # Group pixels to find distinct peak coordinates
    peaks = []
    visited = np.zeros_like(junc_map, dtype=bool)

    for y, x in zip(active_y, active_x):
        if visited[y, x]:
            continue
        # Extract a 5x5 spatial patch around the candidate pixel to find local peak
        y0, y1 = max(0, y - 2), min(h, y + 3)
        x0, x1 = max(0, x - 2), min(w, x + 3)

        patch = junc_map[y0:y1, x0:x1]
        py, px = np.unravel_index(np.argmax(patch), patch.shape)
        peak_y, peak_x = y0 + py, x0 + px

        visited[y0:y1, x0:x1] = True
        peaks.append((peak_y, peak_x))

    # 2. Draw directional spokes for each extracted junction peak
    for y, x in peaks:
        # Extract all active direction bins at this junction center
        active_bins = np.where(direction_tensor[:, y, x] > direction_thresh)[0]

        for b in active_bins:
            # Map bin index to exact angle in radians
            angle_rad = np.radians(b * bin_size_deg)

            dx = spoke_len * np.cos(angle_rad)
            dy = spoke_len * np.sin(angle_rad)

            end_x = int(round(x + dx))
            end_y = int(round(y + dy))

            # Map bin index to HSV color wheel for rendering
            hue = b / DIRECTION_BINS
            rgb_float = mcolors.hsv_to_rgb((hue, 1.0, 1.0))
            color_rgb = (
                int(rgb_float[0] * 255),
                int(rgb_float[1] * 255),
                int(rgb_float[2] * 255),
            )

            # Draw line spoke from junction center along the angle direction
            cv2.line(
                vis_img,
                (x, y),
                (end_x, end_y),
                color_rgb,
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        # Draw a small center dot for the junction location
        cv2.circle(vis_img, (x, y), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    return vis_img


def show_samples():
    dataset = SyntheticDataset(
        length=NUM_SAMPLES,
        width=WIDTH,
        height=HEIGHT,
        num_junctions=NUM_JUNCTIONS,
    )

    indices = random.sample(
        range(len(dataset)),
        NUM_SAMPLES,
    )

    fig, axes = plt.subplots(
        NUM_SAMPLES,
        3,
        figsize=(12, 4 * NUM_SAMPLES),
    )

    if NUM_SAMPLES == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, index in enumerate(indices):
        sample = dataset[index]

        image = sample["image"].squeeze(0).numpy()
        junction = sample["junction"].squeeze(0).numpy()
        direction = sample["direction"].numpy()

        # Show input
        axes[row, 0].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[row, 0].set_title("Input Blueprint")

        # Show junction target heatmap
        axes[row, 1].imshow(
            junction,
            cmap="hot",
            vmin=0,
            vmax=1,
        )
        axes[row, 1].set_title("Junction Target")

        # Show polar spoke direction overlay
        spoke_overlay = render_direction_spokes(
            image=image,
            junction_tensor=junction,
            direction_tensor=direction,
            junction_thresh=0.3,
            direction_thresh=0.3,
            spoke_len=14.0,
        )

        axes[row, 2].imshow(spoke_overlay)
        axes[row, 2].set_title("Direction Target (Polar Spokes)")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_samples()