from __future__ import annotations

import numpy as np
from PIL import Image

from rastervec.OCR.fast_detect import FastDetector


class _FakeComputePool:
    """Stands in for a `multiprocessing.managers.SyncManager`-hosted `Pool`
    proxy: runs jobs inline, in-process, so tests stay fast and
    deterministic while still exercising `detect_tiled`'s dispatch path."""

    def starmap(self, fn, args_list):
        return [fn(*args) for args in args_list]


def _fake_detect(self, image: "Image.Image") -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def test_detect_tiled_local_and_compute_paths_match(monkeypatch):
    monkeypatch.setattr(FastDetector, "detect", _fake_detect)
    image = Image.new("RGB", (50, 50), (10, 20, 30))
    detector = FastDetector(weights_path="fake.pth")

    local_mask = detector.detect_tiled(
        image, block_size=32, scale=1.0, show_progress=False,
    )
    compute_mask = detector.detect_tiled(
        image, block_size=32, scale=1.0, show_progress=False,
        compute=_FakeComputePool(),
    )

    assert local_mask.shape == compute_mask.shape == (50, 50)
    assert np.array_equal(local_mask, compute_mask)


def test_detect_tiled_no_rotation_signature():
    import inspect

    params = inspect.signature(FastDetector.detect_tiled).parameters
    assert "n_rotations" not in params
    assert "compute" in params


def test_detect_tiled_calls_detect_once_per_tile_no_rotation(monkeypatch):
    calls = []

    def counting_detect(self, image: "Image.Image") -> np.ndarray:
        calls.append(image.size)
        return np.zeros((image.height, image.width), dtype=np.float32)

    monkeypatch.setattr(FastDetector, "detect", counting_detect)
    image = Image.new("RGB", (64, 32), (0, 0, 0))
    FastDetector(weights_path="fake.pth").detect_tiled(
        image, block_size=32, scale=1.0, show_progress=False,
    )
    # 64x32 at block_size=32 -> 2 cols x 1 row = 2 tiles, one call each (no rotation sweep)
    assert len(calls) == 2
