"""
Script to generate and save only the raw blueprint graph image to the script's directory.
"""

from pathlib import Path
import cv2

from junction_cnn.synthetic import SyntheticDataset

# Configuration
WIDTH = 512
HEIGHT = 512
NUM_JUNCTIONS = 500

# Save image in the same folder as this script
SCRIPT_DIR = Path(__file__).parent.resolve()
OUTPUT_PATH = SCRIPT_DIR / "blueprint_graph.png"


def save_graph_image(output_file: Path = OUTPUT_PATH) -> None:
    # Initialize dataset and fetch a single sample
    dataset = SyntheticDataset(
        length=1,
        width=WIDTH,
        height=HEIGHT,
        num_junctions=NUM_JUNCTIONS,
    )
    sample = dataset[0]

    # Convert tensor back to 8-bit image array (0 to 255)
    # Note: SyntheticDataset converts image via (1.0 - img/255.0), so we invert it back
    img_tensor = sample["image"].squeeze(0).numpy()
    img_uint8 = ((1.0 - img_tensor) * 255.0).astype("uint8")

    # Save image using OpenCV
    cv2.imwrite(str(output_file), img_uint8)

    print(f"Graph image successfully saved to: {output_file}")


if __name__ == "__main__":
    save_graph_image()