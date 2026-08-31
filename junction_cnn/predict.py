from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from junction_cnn.constants import (
    DIRECTION_THRESHOLD,
    JUNCTION_THRESHOLD,
    OVERLAY_FADE,
    OVERLAY_JUNCTION_RADIUS,
    OVERLAY_LINE_LENGTH,
    OVERLAY_LINE_WIDTH,
)
from junction_cnn.decode import decode_directions, decode_junctions
from junction_cnn.model import JunctionCNN

DIRECTION_COLORS = [
    (255, 0, 0),    # Blue
    (0, 255, 0),    # Green
    (0, 165, 255),  # Orange
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (0, 128, 255),  # Light orange
    (255, 128, 0),  # Light blue
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run junction/direction prediction on a PNG.")

    parser.add_argument("--input", type=Path, required=True, help="Input PNG image.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to trained model weights.")
    parser.add_argument("--output", type=Path, required=True, help="Output overlay PNG.")
    parser.add_argument("--junction-threshold", type=float, default=JUNCTION_THRESHOLD,
                        help="Minimum junction heatmap confidence.")
    parser.add_argument("--direction-threshold", type=float, default=DIRECTION_THRESHOLD,
                        help="Minimum direction confidence.")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda, cpu, or leave unset for automatic selection.")
    parser.add_argument("--show", action="store_true", help="Display the overlay in a window.")
    parser.add_argument("--save-raw", action="store_true", help="Save the raw prediction heatmap.")

    return parser.parse_args()


def load_model(weights_path: Path, device: torch.device) -> JunctionCNN:
    """Load model from checkpoint or raw state dict."""
    model = JunctionCNN().to(device)
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[7:]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model


def draw_arrowhead(
    img: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Draw arrowhead at the end of a line."""
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_length = max(5, thickness * 3)
    arrow_angle = math.radians(25)

    for sign in [-1, 1]:
        dx = arrow_length * math.cos(angle + sign * arrow_angle)
        dy = arrow_length * math.sin(angle + sign * arrow_angle)
        cv2.line(img, end, (round(end[0] - dx), round(end[1] - dy)), color, thickness, cv2.LINE_AA)


def create_overlay(
    original: np.ndarray,
    junctions: list[tuple[int, int, float]],
    directions: dict[tuple[int, int], list[tuple[float, float]]],
) -> np.ndarray:
    """Create visualization overlay with junctions and directions."""
    if original.ndim == 2:
        base = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        base = original.copy()

    overlay = (base.astype(np.float32) * OVERLAY_FADE).astype(np.uint8)

    for x, y, junction_score in tqdm(junctions, desc="Rendering overlay", leave=False):
        junction_directions = directions.get((x, y), [])

        for idx, (angle, confidence) in enumerate(junction_directions):
            theta = math.radians(angle)
            end = (
                round(x + math.cos(theta) * OVERLAY_LINE_LENGTH),
                round(y + math.sin(theta) * OVERLAY_LINE_LENGTH),
            )

            color = DIRECTION_COLORS[idx % len(DIRECTION_COLORS)]
            intensity = 0.5 + 0.5 * confidence
            color = tuple(int(c * intensity) for c in color)

            cv2.line(overlay, (x, y), end, color, OVERLAY_LINE_WIDTH, cv2.LINE_AA)
            draw_arrowhead(overlay, (x, y), end, color, OVERLAY_LINE_WIDTH)

        radius = max(2, round(OVERLAY_JUNCTION_RADIUS * (0.7 + 0.6 * junction_score)))
        cv2.circle(overlay, (x, y), radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 1, (255, 255, 255), -1)

    return overlay


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input image does not exist: {args.input}")
    if not args.weights.exists():
        raise FileNotFoundError(f"Weights do not exist: {args.weights}")

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    image = cv2.imread(str(args.input), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.input}")

    height, width = image.shape
    print(f"Input image: {width} x {height}")

    image_tensor = 1.0 - (torch.from_numpy(image).float() / 255.0)
    image_tensor = image_tensor[None, None].to(device)

    model = load_model(args.weights, device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        junction_heatmap = torch.sigmoid(outputs["junction_logits"])[0, 0]
        direction_logits = outputs["direction_logits"][0]

    junctions = decode_junctions(
        junction_heatmap,
        threshold=args.junction_threshold,
    )
    print(f"Detected {len(junctions)} raw junctions")

    directions = decode_directions(
        direction_logits,
        junctions,
        threshold=args.direction_threshold,
    )

    total_directions = sum(len(dirs) for dirs in directions.values())
    print(f"Decoded {total_directions} directions")

    output = create_overlay(image, junctions, directions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise RuntimeError(f"Could not write output: {args.output}")
    print(f"Saved prediction overlay to {args.output}")

    if args.save_raw:
        raw_output = args.output.with_suffix(".raw.png")
        raw_vis = np.zeros_like(output)
        raw_vis[:, :, 2] = (junction_heatmap.cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(raw_output), raw_vis)
        print(f"Saved raw prediction to {raw_output}")

    if args.show:
        cv2.imshow("Junction Prediction", output)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()