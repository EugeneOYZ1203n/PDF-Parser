from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import torch

from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import geometry as geo
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec import train, train_data as td
from rastervec.P2_Raster_To_Vec.ImplicitSketchVec.model.network import build_model, load_weights


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


def _args(data, out, stage, epochs, *extra):
    return ["--data", str(data), "--out", str(out), "--stage", stage, "--device", "cpu", "--tiny",
            "--epochs", str(epochs), "--steps-per-epoch", "2", "--batch", "2", "--val-crops", "2",
            "--workers", "0", *extra]


def test_three_stages_run_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(td, "SIZES", (64,))
    monkeypatch.setattr(train.td, "SIZES", (64,))
    data = _fake_dataset(tmp_path / "prep")
    w = tmp_path / "w"
    dfp, ndc, joint = w / "isv_dfp.pth", w / "isv_ndc.pth", w / "isv.pth"
    stream_sizes = train.CropStream.__init__.__defaults__
    assert train.main(_args(data, dfp, "dfp", 1)) == 0
    assert train.main(_args(data, ndc, "ndc", 1)) == 0
    assert train.main(_args(data, joint, "joint", 1, "--init-dfp", str(dfp), "--init-ndc", str(ndc))) == 0
    model = load_weights(joint)  # adapter-loadable
    for name, src in (("dfp", dfp), ("ndc", ndc)):
        assert load_weights(src).cfg == model.cfg
    rows = {p: list(csv.DictReader(open(w / f"{p}_train_log.csv"))) for p in ("isv_dfp", "isv_ndc", "isv")}
    assert rows["isv_dfp"][0]["val_center_err_px"] != "" and rows["isv_dfp"][0]["val_edge_f1"] == ""
    assert rows["isv_ndc"][0]["val_edge_f1"] != "" and rows["isv_ndc"][0]["val_center_err_px"] == ""
    assert rows["isv"][0]["val_edge_f1"] != "" and rows["isv"][0]["val_center_err_px"] != ""
    assert train.main(_args(data, dfp, "dfp", 2, "--resume")) == 0
    assert [r["epoch"] for r in csv.DictReader(open(w / "isv_dfp_train_log.csv"))] == ["0", "1"]
    assert stream_sizes is not None


def test_joint_needs_the_stage_weights(tmp_path):
    data = _fake_dataset(tmp_path / "prep")
    try:
        train.main(_args(data, tmp_path / "j.pth", "joint", 1, "--init-dfp", str(tmp_path / "none.pth"),
                         "--init-ndc", str(tmp_path / "none.pth")))
    except SystemExit as exc:
        assert "needs trained DFP weights" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_each_stage_overfits_one_batch():
    torch.manual_seed(0)
    gray = np.full((48, 48), 255, np.uint8)
    gray[20:22, 6:42] = 0
    crop = td.Crop(gray, [geo.line_to_cubic((6, 21), (42, 21))[None]])
    batch = train._to_device(td.build_batch([crop, crop]), "cpu")
    for stage in ("dfp", "ndc"):
        model = build_model(train.TINY_CONFIG)
        opt = torch.optim.Adam(train.stage_params(model, stage), lr=3e-3)
        losses = []
        for _ in range(40):
            loss, _ = train.step(model, stage, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < 0.6 * losses[0], (stage, losses[0], losses[-1])
