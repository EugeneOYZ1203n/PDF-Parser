from __future__ import annotations

import numpy as np
import torch

from rastervec.P2_Raster_To_Vec.DeepVectoriser import geometry as geo
from rastervec.P2_Raster_To_Vec.DeepVectoriser import train_data as td
from rastervec.P2_Raster_To_Vec.DeepVectoriser.model import losses as L

BIG = 40.0  # logit so confident BCE ~ 0


def test_feature_loss_zero_at_exact_match_and_padding_counts():
    gt = torch.rand(2, 4, 4)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    enc = {"endpoints": gt.clone(), "logit": torch.where(valid, BIG, -BIG)}
    assert float(L.feature_loss(enc, gt, valid, beta=128)) < 1e-6
    # a confident prediction in a padding slot is penalised
    enc["logit"][0, 3] = BIG
    assert float(L.feature_loss(enc, gt, valid, beta=128)) > 1.0
    # padding endpoints don't matter
    enc["logit"][0, 3] = -BIG
    enc["endpoints"][1, 2] = 0.0
    assert float(L.feature_loss(enc, gt, valid, beta=128)) < 1e-6


def test_recon_and_primitive_losses_zero_at_exact_match():
    target = (torch.rand(3, 1, 16, 16) > 0.5).float()
    logits = torch.where(target > 0, BIG, -BIG)
    assert float(L.recon_loss_sum(logits, target)) < 1e-6
    gt = torch.rand(2, 4, 6)
    mask = torch.tensor([[True, True, True, False], [True, False, False, False]])
    n = mask.sum(1)
    eos = torch.full((2, 4), -BIG)
    eos[0, 2] = BIG
    eos[1, 0] = BIG
    assert float(L.primitive_loss_sum(gt.clone(), eos, gt, mask, n, beta=64)) < 1e-6
    wrong_eos = eos.clone()
    wrong_eos[0, 2] = -BIG
    assert float(L.primitive_loss_sum(gt.clone(), wrong_eos, gt, mask, n, beta=64)) > 1.0


def test_build_batch_orders_and_formats_gt():
    s = 64
    crop = td.Crop(np.full((s, s), 255, np.uint8), widths=[1.0, 2.0])
    strokes = [geo.line_to_cubic((40, 10), (5, 10))[None],
               np.stack([geo.line_to_cubic((2, 30), (20, 30)), geo.line_to_cubic((20, 30), (20, 50))])]
    crop.strokes, crop.widths = td._sorted_pairs(strokes, crop.widths)
    b = td.build_batch([crop], n_stroke=4, max_prims=3)
    # sorted lexicographically by start endpoint; the reversed line now starts at x=5
    assert np.allclose(b["endpoints"][0, 0] * s, (2, 30, 20, 50))
    assert np.allclose(b["endpoints"][0, 1] * s, (5, 10, 40, 10))
    assert b["valid"].tolist() == [[True, True, False, False]]
    assert b["n_prims"].tolist() == [2, 1]
    # seq_in starts with curve_0 = start x3, then the previous curves
    assert np.allclose(b["seq_in"][0, 0] * s, [2, 30] * 3)
    assert np.allclose(b["seq_in"][0, 1], b["curves"][0, 0])
    assert np.allclose(b["curves"][0, 1, 4:] * s, (20, 50))
    assert b["raster"].shape == (2, s, s) and b["raster"][0].any()
