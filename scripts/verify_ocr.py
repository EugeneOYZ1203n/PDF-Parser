"""Run one OCR tool (FAST / PaddleOCR detect / PaddleOCR recognize / PaddleOCR
full) against a PNG or a folder of PNGs, and report results to stdout and a
text file."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch  # noqa: F401 -- must precede any paddle/paddleocr import (Windows DLL clash)
from PIL import Image

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rastervec.commons.paths import output_dir

TOOLS = ["fast", "paddle_detect", "paddle_recog", "paddle_full"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="PNG file or folder of PNGs to run OCR on.")
    parser.add_argument("--tool", required=True, choices=TOOLS, help="Which OCR tool to run.")
    parser.add_argument("--out", default=None, help="Output directory (default: outputs/verify_ocr/).")
    parser.add_argument(
        "--fast-threshold",
        type=float,
        default=0.5,
        help="FAST score threshold for the flagged-area stat (default: 0.5).",
    )
    return parser


def _collect_images(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.png"))
    return [path]


def _build_detect_engine():
    from paddleocr import PaddleOCR

    from rastervec.P3_Vector_Parsing.FastIntoPaddle.config import OCR_LANG, OCR_VERSION

    return PaddleOCR(
        ocr_version=OCR_VERSION,
        lang=OCR_LANG,
        use_angle_cls=True,
        show_log=False,
        det_limit_side_len=4000,
    )


def _run_fast(image: Image.Image, threshold: float, out_dir: Path, stem: str) -> str:
    from rastervec.P3_Vector_Parsing.FastIntoPaddle.fast_detect import FastDetector

    try:
        mask = FastDetector().detect(image)
    except FileNotFoundError as exc:
        return f"FAST weights not found: {exc}"

    heatmap_path = out_dir / f"{stem}_fast_heatmap.png"
    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(heatmap_path)
    flagged = float((mask >= threshold).mean())
    return (
        f"min={mask.min():.4f} max={mask.max():.4f} mean={mask.mean():.4f} "
        f"pct_above_{threshold:g}={flagged * 100:.2f}%\nheatmap saved to {heatmap_path}"
    )


def _run_paddle_detect(image: Image.Image) -> str:
    from rastervec.P3_Vector_Parsing.FastIntoPaddle.paddle_engine import _normalize_bgr

    bgr = _normalize_bgr(np.asarray(image))
    dt_boxes, _elapse = _build_detect_engine().text_detector(bgr)
    lines = [f"{len(dt_boxes)} box(es) detected:"]
    for i, quad in enumerate(dt_boxes):
        pts = [(round(float(x), 1), round(float(y), 1)) for x, y in quad]
        lines.append(f"  [{i}] {pts}")
    return "\n".join(lines)


def _run_paddle_recog(image: Image.Image) -> str:
    from rastervec.P3_Vector_Parsing.FastIntoPaddle.paddle_engine import PaddleRecBackend

    [box] = PaddleRecBackend().recognize_crops([np.asarray(image)])
    return f"text={box.text!r} confidence={box.confidence:.4f} flip_deg={box.flip_deg}"


def _run_paddle_full(image: Image.Image) -> str:
    from rastervec.P3_Vector_Parsing.FastIntoPaddle.paddle_engine import (
        PaddleRecBackend,
        _normalize_bgr,
        _rotate_crop,
    )

    bgr = _normalize_bgr(np.asarray(image))
    dt_boxes, _elapse = _build_detect_engine().text_detector(bgr)
    if len(dt_boxes) == 0:
        return "0 box(es) detected"

    # _rotate_crop's output is cropped straight out of `bgr`, so it's already
    # BGR -- reverse it back before handing to recognize_crops, which does its
    # own RGB->BGR normalization internally (avoids double-flipping channels).
    crops = [_rotate_crop(bgr, np.asarray(quad, dtype=np.float64))[:, :, ::-1] for quad in dt_boxes]
    ocr_boxes = PaddleRecBackend().recognize_crops(crops)

    lines = [f"{len(dt_boxes)} box(es) detected:"]
    for i, (quad, box) in enumerate(zip(dt_boxes, ocr_boxes)):
        pts = [(round(float(x), 1), round(float(y), 1)) for x, y in quad]
        lines.append(f"  [{i}] text={box.text!r} confidence={box.confidence:.4f} corners={pts}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    path = Path(args.path)
    images = _collect_images(path)
    if not images:
        print(f"no PNG files found at {path}")
        return 1

    out_dir = Path(args.out) if args.out else output_dir("verify_ocr")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines = [f"tool={args.tool}", f"input={path}", ""]

    for image_path in images:
        image = Image.open(image_path)
        if args.tool == "fast":
            body = _run_fast(image, args.fast_threshold, out_dir, image_path.stem)
        elif args.tool == "paddle_detect":
            body = _run_paddle_detect(image)
        elif args.tool == "paddle_recog":
            body = _run_paddle_recog(image)
        else:
            body = _run_paddle_full(image)
        block = f"=== {image_path.name} ===\n{body}\n"
        print(block)
        report_lines.append(block)

    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
