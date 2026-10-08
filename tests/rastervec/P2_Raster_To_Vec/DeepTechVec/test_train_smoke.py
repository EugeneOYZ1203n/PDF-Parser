from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest
import torch

from rastervec.P2_Raster_To_Vec.DeepTechVec import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepTechVec import train, train_data as td
from rastervec.P2_Raster_To_Vec.DeepTechVec.model.network import build_model, load_weights


def _fake_dataset(root, n_pages=2):
    """A directory in DeepVectoriser's prep_dataset format (written directly,
    no DeepVectoriser import): random straight lines."""
    rng = np.random.default_rng(0)
    (root / "layers").mkdir(parents=True)
    keys = []
    for p in range(n_pages):
        gray = np.full((300, 300), 255, np.uint8)
        pieces, bboxes = [], []
        for _ in range(12):
            a = rng.uniform(10, 290, 2)
            b = np.clip(a + rng.uniform(-80, 80, 2), 5, 295)
            cv2.line(gray, tuple(int(v) for v in a), tuple(int(v) for v in b), 0, 2)
            pieces.append(geo.line_to_cubic(a, b))
            bboxes.append([*np.minimum(a, b), *np.maximum(a, b)])
        key = f"fake_p{p}__L1"
        ok, png = cv2.imencode(".png", gray)
        png.tofile(str(root / "layers" / f"{key}.gray.png"))
        ys, xs = np.nonzero(gray < 200)
        np.savez_compressed(root / "layers" / f"{key}.strokes.npz", pieces=np.array(pieces, np.float32),
                            offsets=np.arange(13, dtype=np.int64), widths=np.full(12, 2.0, np.float32),
                            bboxes=np.array(bboxes, np.float32), ink_pts=np.stack([xs, ys], 1).astype(np.int32),
                            color=np.zeros(3, np.uint8), origin=np.zeros(2, np.int64),
                            page_shape=np.array([300, 300], np.int64))
        keys.append(key)
    index = {"train": keys[:-1], "val": keys[-1:], "n_strokes": {k: 12 for k in keys},
             "pages": keys, "val_pages": keys[-1:], "px_per_pt": 300 / 72, "val_frac": 0.5}
    (root / "index.json").write_text(json.dumps(index))
    return root


def _args(data, out, prim, epochs):
    return ["--data", str(data), "--out", str(out), "--prim", prim, "--device", "cpu", "--tiny",
            "--epochs", str(epochs), "--steps-per-epoch", "2", "--batch", "4", "--val-crops", "4", "--workers", "0"]


@pytest.mark.parametrize("prim", ["line", "curve"])
def test_train_runs_and_resumes(tmp_path, prim):
    data = _fake_dataset(tmp_path / "prep")
    out = tmp_path / "w" / f"dtv_{prim}.pth"
    assert train.main(_args(data, out, prim, epochs=1)) == 0
    assert out.is_file() and (tmp_path / "w" / f"dtv_{prim}.pth.last.ckpt").is_file()
    assert load_weights(out).kind == prim  # adapter-loadable, knows its kind
    log = tmp_path / "w" / f"dtv_{prim}_train_log.csv"
    rows = list(csv.DictReader(open(log)))
    assert len(rows) == 1 and np.isfinite(float(rows[0]["loss"])) and rows[0]["val_iou"] != ""

    assert train.main(_args(data, out, prim, epochs=2) + ["--resume"]) == 0
    rows = list(csv.DictReader(open(log)))
    assert [r["epoch"] for r in rows] == ["0", "1"]


def test_loss_decreases_when_overfitting_one_batch():
    torch.manual_seed(0)
    layer_gray = np.full((64, 64), 255, np.uint8)
    layer_gray[20:22, 8:56] = 0
    layer_gray[30:58, 40:42] = 0
    patch = td.Patch(layer_gray, geo.sort_prims([geo.Prim(np.array([[8.0, 21.0], [56.0, 21.0]]), 2.0),
                                                 geo.Prim(np.array([[41.0, 30.0], [41.0, 58.0]]), 2.0)]))
    cfg = {**train.TINY_CONFIG, "prim_kind": "line"}
    model = build_model(cfg)
    batch = td.build_batch([patch] * 4, cfg["n_prim"], "line")
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss, _ = train.train_step(model, batch, "cpu")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]


def test_noam_schedule_warms_up_then_decays():
    lrs = [train.noam_lr(s, 6, 100, 1.0) for s in (1, 50, 100, 400)]
    assert lrs[0] < lrs[1] < lrs[2] and lrs[3] < lrs[2]
