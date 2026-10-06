from __future__ import annotations

import csv
import json

import numpy as np
import torch

from rastervec.P2_Raster_To_Vec.DeepVectoriser import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepVectoriser import train, train_data as td
from rastervec.P2_Raster_To_Vec.DeepVectoriser.model.liu_model import build_model, load_weights


def _fake_prep(root, n_pages=2):
    """A prep_dataset-shaped directory of random straight lines."""
    rng = np.random.default_rng(0)
    keys = []
    for p in range(n_pages):
        gray = np.full((300, 300), 255, np.uint8)
        strokes = []
        for _ in range(12):
            a = rng.uniform(10, 290, 2)
            b = np.clip(a + rng.uniform(-80, 80, 2), 5, 295)
            pts = geo.sample_stroke(geo.line_to_cubic(a, b)[None], 64)
            ij = np.round(pts - 0.5).astype(int)
            gray[ij[:, 1], ij[:, 0]] = 0
            strokes.append((geo.line_to_cubic(a, b)[None], 1.0))
        key = f"fake_p{p}__L1"
        td.save_layer(root / "layers", key, gray, strokes, (0, 0, 0))
        keys.append(key)
    index = {"train": keys[:-1], "val": keys[-1:], "n_strokes": {k: 12 for k in keys},
             "pages": keys, "val_pages": keys[-1:], "px_per_pt": 300 / 72, "val_frac": 0.5}
    (root / "index.json").write_text(json.dumps(index))
    return root


def _args(data, out, joint_epochs):
    return ["--data", str(data), "--out", str(out), "--device", "cpu", "--tiny",
            "--bootstrap-epochs", "1", "--supervise-epochs", "1", "--epochs", str(joint_epochs),
            "--bootstrap-crops", "8", "--steps-per-epoch", "2", "--batch", "4", "--val-crops", "4",
            "--workers", "0"]


def test_train_runs_all_stages_and_resumes(tmp_path):
    data = _fake_prep(tmp_path / "prep")
    out = tmp_path / "w" / "deep_vectoriser.pth"
    assert train.main(_args(data, out, joint_epochs=1)) == 0
    assert out.is_file() and (tmp_path / "w" / "deep_vectoriser.pth.last.ckpt").is_file()
    load_weights(out)  # adapter-loadable
    rows = list(csv.DictReader(open(tmp_path / "w" / "train_log.csv")))
    assert [r["stage"] for r in rows] == ["bootstrap", "supervise", "joint"]
    assert all(np.isfinite(float(r["loss"])) for r in rows)
    assert rows[-1]["val_chamfer_px"] != ""

    assert train.main(_args(data, out, joint_epochs=2) + ["--resume"]) == 0
    rows = list(csv.DictReader(open(tmp_path / "w" / "train_log.csv")))
    assert [(r["stage"], r["epoch"]) for r in rows[3:]] == [("joint", "1")]


def test_joint_step_loss_decreases_when_overfitting_one_batch():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    gray = np.full((64, 64), 255, np.uint8)
    gray[20:22, 8:56] = 0
    gray[30:58, 40:42] = 0
    crop = td.Crop(gray, *td._sorted_pairs([geo.line_to_cubic((8, 21), (56, 21))[None],
                                            geo.line_to_cubic((41, 30), (41, 58))[None]], [2.0, 2.0]))
    cfg = train.TINY_CONFIG
    model = build_model(cfg)
    b = train._to_device(td.build_batch([crop, td.augment(crop, rng)], cfg["n_stroke"], cfg["max_prims"]), "cpu")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(40):
        loss, _terms = train.step_joint(model, b, cap=64, beta=64.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
