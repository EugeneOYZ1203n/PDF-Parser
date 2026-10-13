"""Stand-ins for the two pretrained networks, so the tests need no weights.

* `fake_centerline`: what a perfect centerline network returns -- the
  white-on-black input's 1 px skeleton at 255;
* `pairing_by_angle`: a geometric pairing network -- half-edges meeting at a
  node score `-cos` of the angle between their outgoing directions, so the
  involutive matching pairs the ones that continue straight through."""
from __future__ import annotations

import numpy as np
from skimage.morphology import skeletonize


def fake_centerline(tile: np.ndarray) -> np.ndarray:
    return np.where(skeletonize(tile > 127), 255, 0).astype(np.uint8)


def _out_dir(pts: np.ndarray) -> np.ndarray:
    """Direction a half-edge leaves its node (point 0) in."""
    k = min(len(pts) - 1, 6)
    d = pts[k] - pts[0]
    n = float(np.hypot(*d))
    return d / n if n > 1e-9 else np.zeros(2)


def pairing_by_angle(lines: np.ndarray, R: np.ndarray) -> np.ndarray:
    n = R.shape[0]
    dirs = [np.zeros(2)] + [_out_dir(lines[i]) for i in range(1, n)]
    scores = np.full((n, n), -np.inf, np.float32)
    for i in range(n):
        for j in range(n):
            if R[i, j]:
                scores[i, j] = 0.0 if i == j else -float(np.dot(dirs[i], dirs[j]))
    return scores


FAKE_MODELS = {"centerline": fake_centerline, "pairing": pairing_by_angle}
