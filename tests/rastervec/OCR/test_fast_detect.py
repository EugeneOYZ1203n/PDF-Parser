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

    def imap(self, fn, args_list):
        return (fn(*args) for args in args_list)


class _FakeCounter:
    def __init__(self, value: int = 0) -> None:
        self.value = value


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
        image, block_size=32, scale=1.0, overlap=0.0, show_progress=False,
    )
    # 64x32 at block_size=32, no overlap -> 2 cols x 1 row = 2 tiles, one call each (no rotation sweep)
    assert len(calls) == 2


def test_tile_starts_single_tile_when_total_fits():
    from rastervec.OCR.fast_detect import _tile_starts

    assert _tile_starts(20, 32, 27) == [0]


def test_tile_starts_overlap_covers_edge_exactly():
    from rastervec.OCR.fast_detect import _tile_starts

    # 64px, 32px tiles, stride 27 -> [0, 27], last start extended to 32 so
    # the final tile (32-64) reaches the far edge exactly.
    assert _tile_starts(64, 32, 27) == [0, 27, 32]


def test_tile_starts_no_extra_tile_when_grid_already_reaches_edge():
    from rastervec.OCR.fast_detect import _tile_starts

    # 64px, 32px tiles, stride 32 (no overlap) -> grid already ends exactly
    # at the far edge, no extra tile appended.
    assert _tile_starts(64, 32, 32) == [0, 32]


def test_detect_tiled_overlap_stitches_by_max(monkeypatch):
    """Two overlapping tiles disagree in their shared region -- the
    stitched mask must take the max, never the second tile's overwrite."""
    calls = []

    def alternating_detect(self, image: "Image.Image") -> np.ndarray:
        value = 0.2 if len(calls) == 0 else 0.9
        calls.append(image.size)
        return np.full((image.height, image.width), value, dtype=np.float32)

    monkeypatch.setattr(FastDetector, "detect", alternating_detect)
    image = Image.new("RGB", (64, 32), (0, 0, 0))
    mask = FastDetector(weights_path="fake.pth").detect_tiled(
        image, block_size=32, scale=1.0, overlap=0.15, show_progress=False,
    )
    # The overlap band between the two tiles must reflect the max (0.9),
    # not the first tile's lower value or a blind overwrite.
    assert mask[:, 27:32].min() > 0.8


def test_tile_has_candidate():
    from rastervec.OCR.fast_detect import _tile_has_candidate

    assert _tile_has_candidate((0, 0, 10, 10), [(5, 5, 15, 15)]) is True
    assert _tile_has_candidate((0, 0, 10, 10), [(20, 20, 30, 30)]) is False
    assert _tile_has_candidate((0, 0, 10, 10), []) is False


def test_detect_tiled_candidate_bboxes_skips_non_candidate_tiles(monkeypatch):
    calls = []

    def counting_detect(self, image: "Image.Image") -> np.ndarray:
        calls.append(image.size)
        return np.ones((image.height, image.width), dtype=np.float32)

    monkeypatch.setattr(FastDetector, "detect", counting_detect)
    image = Image.new("RGB", (64, 32), (0, 0, 0))
    detector = FastDetector(weights_path="fake.pth")

    # 64x32 at block_size=32 -> 2 cols x 1 row; only a candidate bbox inside
    # the left tile (x in [0, 32)) is given, so the right tile is skipped.
    mask = detector.detect_tiled(
        image, block_size=32, scale=1.0, show_progress=False,
        candidate_bboxes=[(5.0, 5.0, 10.0, 10.0)],
    )
    assert len(calls) == 1
    assert np.all(mask[:, :32] == 1.0)
    assert np.all(mask[:, 32:] == 0.0)


def test_detect_tiled_candidate_bboxes_none_detects_every_tile(monkeypatch):
    calls = []

    def counting_detect(self, image: "Image.Image") -> np.ndarray:
        calls.append(image.size)
        return np.zeros((image.height, image.width), dtype=np.float32)

    monkeypatch.setattr(FastDetector, "detect", counting_detect)
    image = Image.new("RGB", (64, 32), (0, 0, 0))
    FastDetector(weights_path="fake.pth").detect_tiled(
        image, block_size=32, scale=1.0, overlap=0.0, show_progress=False, candidate_bboxes=None,
    )
    assert len(calls) == 2


def test_detect_tiled_progress_counter_local_increments_per_tile(monkeypatch):
    monkeypatch.setattr(FastDetector, "detect", _fake_detect)
    image = Image.new("RGB", (64, 32), (10, 20, 30))
    detector = FastDetector(weights_path="fake.pth")
    counter = _FakeCounter()

    detector.detect_tiled(
        image, block_size=32, scale=1.0, overlap=0.0, show_progress=False, progress_counter=counter,
    )
    # 64x32 at block_size=32, no overlap -> 2 tiles.
    assert counter.value == 2


def test_detect_tiled_progress_counter_with_compute_uses_imap_and_increments(monkeypatch):
    monkeypatch.setattr(FastDetector, "detect", _fake_detect)
    image = Image.new("RGB", (64, 32), (10, 20, 30))
    detector = FastDetector(weights_path="fake.pth")
    counter = _FakeCounter()

    mask = detector.detect_tiled(
        image, block_size=32, scale=1.0, overlap=0.0, show_progress=False,
        compute=_FakeComputePool(), progress_counter=counter,
    )
    assert counter.value == 2
    assert mask.shape == (32, 64)


def test_detect_tiled_candidate_bboxes_with_compute_pool_skips_dispatch(monkeypatch):
    monkeypatch.setattr(FastDetector, "detect", _fake_detect)
    image = Image.new("RGB", (64, 32), (10, 20, 30))
    detector = FastDetector(weights_path="fake.pth")

    mask = detector.detect_tiled(
        image, block_size=32, scale=1.0, show_progress=False,
        compute=_FakeComputePool(), candidate_bboxes=[(5.0, 5.0, 10.0, 10.0)],
    )
    assert mask.shape == (32, 64)
    assert np.all(mask[:, 32:] == 0.0)
