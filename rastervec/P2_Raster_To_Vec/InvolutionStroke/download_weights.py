"""Fetch the authors' pretrained InvolutionStroke checkpoints (hosted by ETH
IGL) and check they load into the vendored networks.

    python -m rastervec.P2_Raster_To_Vec.InvolutionStroke.download_weights [--force]

Writes `rastervec/weights/involution_centerline.pt` (~177 MB) and
`involution_intersection.pt` (~15 MB), or wherever
`$INVOLUTIONSTROKE_CENTERLINE_WEIGHTS` / `$INVOLUTIONSTROKE_INTERSECTION_WEIGHTS`
point. An existing file is kept unless `--force`. The raw checkpoints are
loaded as-is (`model/weights.py`) -- upstream's TorchScript export step is
not needed."""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm

from rastervec.P2_Raster_To_Vec.InvolutionStroke.config import CENTERLINE_WEIGHTS_URL, INTERSECTION_WEIGHTS_URL
from rastervec.P2_Raster_To_Vec.InvolutionStroke.inference import (
    default_centerline_path, default_intersection_path,
)


def download(url: str, dest: Path, *, force: bool = False) -> bool:
    """`url` -> `dest` via a `.part` file (renamed only once complete).
    Returns False when `dest` already existed and was kept."""
    if dest.is_file() and not force:
        print(f"keeping existing {dest}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as resp, open(part, "wb") as f:  # noqa: S310 -- fixed https URLs from config
        total = int(resp.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    shutil.move(str(part), str(dest))
    return True


def verify(centerline: Path, intersection: Path) -> None:
    from rastervec.P2_Raster_To_Vec.InvolutionStroke.model.weights import load_centerline, load_intersection

    c = load_centerline(centerline)
    i = load_intersection(intersection)
    print(f"ok: centerline ResNextUNetLarge, {sum(p.numel() for p in c.parameters()):,} params; "
          f"intersection SceneGraphVectorModel (d_dim {i.d_dim}), {sum(p.numel() for p in i.parameters()):,} params")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="re-download even when the files exist")
    args = ap.parse_args(argv)
    centerline, intersection = default_centerline_path(), default_intersection_path()
    download(CENTERLINE_WEIGHTS_URL, centerline, force=args.force)
    download(INTERSECTION_WEIGHTS_URL, intersection, force=args.force)
    verify(centerline, intersection)
    return 0


if __name__ == "__main__":
    sys.exit(main())
