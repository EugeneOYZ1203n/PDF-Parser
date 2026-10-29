from __future__ import annotations

from pathlib import Path

import cv2
from autotrace import Bitmap, VectorFormat

# Defaults tuned to keep small / thin linework (autotrace's own defaults are aggressive:
# noise-removal=0.99 and filter-iterations=4 erase fine AEC detail).
_DEFAULT_ERROR_THRESHOLD = 0.5
_DEFAULT_NOISE_REMOVAL = 0.2
_DEFAULT_FILTER_ITERATIONS = 1


def png_to_svg(
    inp: Path,
    out: Path,
    centerline: bool = False,
    color_count: int = 16,
    error_threshold: float = _DEFAULT_ERROR_THRESHOLD,
    corner_always_threshold: float = 80.0,
    corner_surround: int = 4,
    noise_removal: float = _DEFAULT_NOISE_REMOVAL,
    filter_iterations: int = _DEFAULT_FILTER_ITERATIONS,
    despeckle_level: int = 0,
    preserve_width: bool = True,
) -> None:
    """Vectorize ``inp`` PNG to ``out`` SVG via pyautotrace (``Bitmap.trace``)."""
    inp = Path(inp)
    out = Path(out)
    if not inp.is_file():
        raise FileNotFoundError(f"image not found: {inp}")
    out.parent.mkdir(parents=True, exist_ok=True)

    # pyautotrace Bitmap expects HxWx3 RGB uint8.
    bgr = cv2.imread(str(inp), cv2.IMREAD_COLOR)
    if bgr is None:
        gray = cv2.imread(str(inp), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"could not read image: {inp}")
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    try:
        traced = Bitmap(rgb).trace(
            color_count=color_count,
            error_threshold=error_threshold,
            corner_always_threshold=corner_always_threshold,
            corner_surround=corner_surround,
            noise_removal=noise_removal,
            filter_iterations=filter_iterations,
            despeckle_level=despeckle_level,
            centerline=centerline,
            preserve_width=preserve_width if centerline else False,
        )
        traced.save(out, format=VectorFormat.SVG)
    except Exception as e:
        raise RuntimeError(f"pyautotrace failed on {inp}: {e}") from e
