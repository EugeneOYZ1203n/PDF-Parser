# Reading `train.py`'s output

`train.py` trains DeepTechVec's primitive network (Egiazarian et al., ECCV
2020) on the dataset **DeepVectoriser's** `prep_dataset.py` writes. There is
no DeepTechVec prep step: targets are derived from that dataset's vector
strokes on the fly.

```
python -m rastervec.P2_Raster_To_Vec.DeepTechVec.train --data data/deepvec --prim line  [--threads N] [--resume]
python -m rastervec.P2_Raster_To_Vec.DeepTechVec.train --data data/deepvec --prim curve [--threads N] [--resume]
```

The paper trains separate line and quadratic-curve models. Each `--prim`
is its own process with its own `--out` (default
`rastervec/weights/deep_tech_vec_<prim>.pth`), checkpoint and log, so both
can run at once, and alongside DeepVectoriser's trainer. They all share
`<data>/cache/`. On a CPU, split the cores between the processes with
`--threads`.

The excerpts below come from a `--tiny` smoke run on a synthetic dataset:

```
--tiny --epochs 1 --steps-per-epoch 3 --batch 8 --val-crops 8 --workers 0
```

---

## 1. Start banner

```
========================================================================
DeepTechVec training (line) | device cpu | 16 threads | data …\data
train layers 1, val layers 1 | out …\line.pth
model {'prim_kind': 'line', 'n_prim': 10, 'res_ch': 16, 'n_dec': 1, 'n_heads': 2, 'ff_dim': 32, 'dropout': 0.1}
========================================================================
```

`model` is the architecture stored in the checkpoint. Paper defaults:
`n_prim 10`, `res_ch 64` (c), `n_res 1`, `n_dec 8`, `n_heads 4`, `ff_dim 512`.
`d_emb` isn't listed because it follows from the primitive kind: 6 for
lines, 8 for curves.

## 2. Validation set

```
val: 8 patches, 18 GT primitives, 0 truncated to n_prim=10
```

The validation patches are fixed, un-augmented 64 × 64 patches from the
val pages. "truncated" counts patches that had more GT primitives than
`n_prim` even after re-drawing; only their `n_prim` longest primitives
are kept. The paper chose `n_prim = 10` because 97 % of its patches fit.
If many patches are truncated (dense hatching or text), the network can't
represent those patches; consider raising `n_prim` in `config.MODEL_DEFAULTS`
for a fresh run.

## 3. The epoch bar

```
line ep 1/1: 100%|##########| 3/3 [..., loss=2.9874, L_cls=0.9618, L_loc=2.0256, overflow=0.0000]
```

| term | meaning |
|---|---|
| `loss` | Eq. 2: mean over the `n_prim` rows of `L_cls + L_loc` |
| `L_cls` | BCE of the confidences. Placeholder rows have target 0. |
| `L_loc` | `(1−λ)·L1 + λ·L2²` on the normalised parameters (coordinates and width divided by 64), placeholder rows included (their targets are 0) |
| `overflow` | the fraction of training patches truncated to `n_prim` |

Both terms should fall steadily. `L_cls` falling first means the network is
learning how many primitives a patch holds. `L_loc` then shrinks as their
positions settle.

## 4. The epoch line

```
[ep 1/1] steps=3, lr=4.841e-06, loss=2.987434, ..., val_loss=2.9533, val_iou=0.0761, val_chamfer_px=17.1034, seconds=0.3
  new best val IoU 0.0761 -> …\line.pth
```

| field | meaning |
|---|---|
| `lr` | the Noam learning rate after this epoch: it rises linearly for `--warmup` (4000) optimiser steps, then decays as step^−½. In a short run it stays tiny; that is expected. |
| `val_loss` | Eq. 2 on the validation patches |
| `val_iou` | IoU of the predicted vs. GT primitives, each drawn at its own width. This is the metric the paper plots (Fig. 11/12), and higher is better. **`--out` keeps the best one.** |
| `val_chamfer_px` | mean nearest distance between predicted and GT primitives in px (lower is better). Half the patch size (32 px) means one side was empty. |

The validation numbers are the network alone: no refinement and no
merging. The paper's ablation (Table 3) shows refinement adds a lot on top
of them.

## 5. Files

| file | |
|---|---|
| `--out` | the best-val-IoU weights; the adapter and `predict.py` load this |
| `<out>.last.ckpt` | the full state after every epoch, which `--resume` continues from |
| `<out stem>_train_log.csv` | one row per epoch: every value above |

A non-finite loss prints `TRAINING FAILED at epoch=… step=…`, saves
`<out>.last.ckpt` and exits 2. The failed epoch restarts on `--resume`.

## 6. Trying the model

```
python -m rastervec.P2_Raster_To_Vec.DeepTechVec.predict --pdf X.pdf --pages 0 --clip x0,y0,x1,y1 \
    [--weights rastervec/weights/deep_tech_vec_line.pth] [--refine-iters 0]
```

This writes a `scripts/pipeline_report_viewer.py` folder with these layers:
- `primitives/raw (network)`: the network output
- `refine/refined`: after refinement
- `merge/merged vectors`: the final output
- `original/vectors`: the PDF's own vectors, for comparison

OCR never runs in `predict.py`. `--refine-iters 0` skips refinement, which
is the slowest stage (the paper's Table 4 makes the same point).
