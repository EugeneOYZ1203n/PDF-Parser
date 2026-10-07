# Reading `train.py`'s output

`train.py` trains the DeepVectoriser model (Liu et al., AAAI-22) on a
`prep_dataset.py` folder, in three stages:

1. **bootstrap**
2. **supervise**
3. **joint**

This page walks through everything it prints, in order, and how to tell a
healthy run from a broken one.

```
python -m rastervec.P2_Raster_To_Vec.DeepVectoriser.train --data data/deepvec [--threads N] [--resume]
```

The excerpts below come from a real CPU run, a `--tiny` model on the 4
reference PDFs plus 3 synthetic A1 sheets:

```
--tiny --device cpu --bootstrap-crops 160 --bootstrap-epochs 2 --supervise-epochs 2
--epochs 3 --steps-per-epoch 20 --val-crops 32 --batch 8
```

Long paths are shortened to `…`.

---

## 1. Start banner

```
========================================================================
DeepVectoriser training | device cpu | 16 threads | data …\deepvec
train layers 19, val layers 4 | out …\deep_vectoriser.pth
model {'d_model': 32, 'd_emb': 8, 'n_stroke': 16, ... 'max_prims': 4, 'dropout': 0.1}
========================================================================
```

| field | meaning |
|---|---|
| `device` | `cpu` or `cuda`; `(AMP)` is added when mixed precision is on (CUDA only) |
| `16 threads` | torch's CPU threads for the model (CPU only); change it with `--threads` |
| `train layers` / `val layers` | (page × colour layer) samples from `index.json`. Val layers come from **held-out pages**, so val scores measure drawings the model never trained on |
| `out` | where the best weights go; the P2 adapter loads `rastervec/weights/deep_vectoriser.pth` by default |
| `model` | the network sizes. They are saved in the checkpoint, so `--resume` and inference rebuild the same model |

## 2. Setup

```
decode train layers: 100%|##########| 19/19 [...]
building fixed crop sets ...
bootstrap crops: 100%|##########| 160/160 [...]
decode val layers: 100%|##########| 4/4 [...]
val crops: 100%|##########| 32/32 [...]
bootstrap: 160 crops, 1515 strokes | val: 32 crops
```

- **`decode … layers`**: prep stores each layer mask as a compressed PNG.
  Each one is decoded **once** into `--cache-dir` (default `<data>/cache/`)
  and memory-mapped from then on. A re-run skips layers that are already
  decoded, so this bar is near-instant. An old dataset with full-page
  `.gray.npy` files needs no decode.
- **`bootstrap crops`**: the fixed crop set the bootstrap and supervise stages
  train on (`--bootstrap-crops`; the paper uses 5,000).
- **`val crops`**: the fixed validation crops (`--val-crops`). They are the
  same every epoch, so val numbers are comparable across epochs.
- **`bootstrap: N crops, M strokes`**: M is the number of learnable stroke
  embeddings the bootstrap stage creates, one per ground-truth stroke.

Other lines you may see:

| line | meaning |
|---|---|
| `resumed from <out>.last.ckpt: stage S, epoch E, best B` | `--resume` picked up where the last run stopped |
| `WARNING: no val pages in index.json -- validating on training-page crops` | Too few pages for a held-out split. Val then measures memorisation, not generalisation, so add pages |
| `WARNING: optimizer 'X' state not restored (...)` | The checkpoint's optimizer state didn't fit, so Adam starts fresh. That's harmless and the weights are still restored |

## 3. Stage banners

```
========================================================================
stage 1/3: bootstrap -- epochs 0..1, optimizers ['decoder']
========================================================================
```

| stage | what trains | idea |
|---|---|---|
| 1 `bootstrap` | the shared UNet + Stroke Decoder (and the per-stroke embeddings) | Learn an embedding per GT stroke such that the decoder can redraw that stroke's raster from it |
| 2 `supervise` | the Stroke Encoder + Stroke Vectorizer | The encoder learns to predict each stroke's endpoints + the stage-1 embedding; the vectorizer learns to turn an embedding into Bézier curves |
| 3 `joint` | everything, end to end | Fresh random, augmented 64–256 px crops every step; the encoder's own predictions feed the other modules |

`epochs a..b` are 0-based. After `--resume`, the first stage shown starts
part-way.

## 4. The per-epoch progress bar

```
bootstrap ep 1/2:  50%|#####     | 10/20 [00:39<00:40,  4.08s/batch, loss=0.0221, L_R=0.2211]
supervise ep 2/2: 100%|##########| 20/20 [00:57<00:00, ..., loss=1475.2326, L_F=63.6689, L_emb=0.1917, L_P=235.2287]
joint ep 1/3:  55%|#####5    | 11/20 [04:22<02:59, 19.89s/batch, loss=1693.0473, L_F=104.6655, L_R=0.1729, L_P=264.7274]
```

The numbers after the timing are **running means over the epoch so far**, so
they settle as the epoch progresses.

| term | stages | what it measures (paper eq.) | scale |
|---|---|---|---|
| `L_R` | bootstrap, joint | stroke-raster reconstruction error: how well a single stroke is redrawn, as a per-pixel mean of L2/L1 (Eq. 6) | 0 to about 0.3; it starts around 0.2 |
| `L_F` | supervise, joint | encoder loss: endpoint error × β + "is there a stroke in this slot" confidence BCE (Eq. 5) | grows with β (see below) |
| `L_emb` | supervise | MSE between the encoder's embeddings and the stage-1 learned ones | small, and it should shrink |
| `L_P` | supervise, joint | vectorizer loss: Bézier control-point error × β + end-of-stroke BCE (Eq. 9) | grows with β |
| `loss` | all | what is actually optimised: bootstrap `0.1·L_R`; supervise `L_F + L_emb + 6·L_P`; joint Eq. 10 (`L_F + (0.1·ΣL_R + 6·ΣL_P)/n`) | **not comparable across stages** |

**β = crop size in px (64–256).** `L_F` and `L_P` multiply their coordinate
error by β, so they are big numbers by design. In the **joint** stage every
step uses a random crop size, so step-to-step values jump around. Judge the
**trend across epochs**, not single steps or the bar's live values. The
bootstrap and supervise stages always use 128 px crops, so their values are
steadier.

## 5. The epoch summary line

After each epoch, the model is evaluated on the val crops and one line is
printed. The same values are appended to `train_log.csv`.

```
[bootstrap ep 1/2] steps=20, loss=0.021294, L_R=0.21294, val_endpoint_l1_px=44.6205, seconds=83.0
[supervise ep 2/2] steps=20, loss=1475.232617, L_F=63.668913, L_emb=0.191667, L_P=235.228671, val_endpoint_l1_px=38.1832, val_chamfer_px=39.0267, val_raster_iou=0.0173, seconds=57.8
[joint ep 1/3] steps=20, loss=1658.931219, L_F=102.341183, L_R=0.17337, L_P=259.428784, val_endpoint_l1_px=41.1445, val_chamfer_px=40.0652, val_raster_iou=0.012, seconds=515.7
```

| field | meaning | good direction |
|---|---|---|
| `steps` | Batches that actually trained. Below the bar's total means some batches had no strokes and were skipped. A few is fine; many means the crops miss the ink | — |
| `loss`, `L_*` | the epoch means of the bar's terms (above) | down |
| `val_endpoint_l1_px` | Mean endpoint error in **pixels** on the val crops: how far the encoder's predicted stroke ends are from the true ones. Every stage. In bootstrap the encoder isn't trained yet, so this stays flat | down |
| `val_chamfer_px` | **The main score.** The full model runs on each val crop (exactly as inference does) and its strokes are compared with the GT strokes: the average distance, in px, from each predicted point to the nearest true stroke and back. Supervise and joint only | down; under about 2 px is good, and 64 (half a crop) means nothing useful came out |
| `val_raster_iou` | Predicted and GT strokes are both redrawn 2 px wide, then intersection ÷ union. Supervise and joint only | up, from 0 to 1 |
| `seconds` | the epoch's wall time, including validation | — |

```
  new best val Chamfer 40.065 px -> …\deep_vectoriser.pth
```

This line appears only in the **joint** stage, whenever `val_chamfer_px`
beats the best so far. It means the weights file was just overwritten with
this epoch's model. That file is what the P2 adapter and `predict.py` use.

## 6. The end

```
========================================================================
done. best val Chamfer <best> px | weights <out> | log <out dir>	rain_log.csv
========================================================================
```

If no joint epoch ever ran or improved, the last model is saved anyway and you
see `saved final model -> …`. The exit code is 0.

## 7. Healthy run vs red flags

**Healthy:**
- In bootstrap, `L_R` falls steadily (0.21 → 0.19 in the sample after two
  short epochs).
- In supervise, `L_emb` shrinks and `val_chamfer_px` starts to drop.
- In joint, `val_chamfer_px` keeps falling over epochs, `val_raster_iou`
  keeps rising, and every so often you get a `new best` line. Plateaus are
  normal.

**Red flags:**

| what you see | likely cause | what to do |
|---|---|---|
| train `loss` falls but `val_chamfer_px` rises for several epochs | overfitting: too few pages | prep more PDFs; the best checkpoint is already kept |
| `val_chamfer_px` stuck near 64 | the model predicts nothing (or garbage) | train longer; check `val_endpoint_l1_px` is falling; check `index.json` has many strokes |
| `val_raster_iou` stuck at about 0 for many joint epochs | same as above | same as above |
| `steps` far below the bar total | most crops have no strokes | check the prep layers; very sparse pages give empty crops |
| `TRAINING FAILED ... non-finite loss` | the loss became NaN/inf | `--resume` (the failed epoch restarts); if it repeats, lower `--lr` |
| `TRAINING FAILED ... out of memory` | the batch is too big | lower `--batch` and/or `--max-decode-strokes`, and keep the effective batch with `--accum` |

A failure prints `TRAINING FAILED at stage=… epoch=… step=… crop size=…`,
saves `<out>.last.ckpt`, and exits with code **2**. Run again with `--resume`.

## 8. Training on CPU

- **It is slow.** In the sample (tiny model, batch 8, 16 threads) bootstrap
  takes about 4 s/batch, supervise about 3 s/batch and joint about
  20 s/batch. The full-size model is far slower. The defaults (20 + 20 + 50
  epochs, 5,000 bootstrap crops, 1,000 joint steps per epoch) are GPU-scale.
  For a first CPU run, cut them down, for example
  `--bootstrap-crops 1000 --bootstrap-epochs 5 --supervise-epochs 5 --epochs 10 --steps-per-epoch 200`,
  and watch `val_chamfer_px`.
- **`--threads N`** sets torch's CPU threads (the default is torch's own
  choice, about one per core). **`--workers N`** (default 2) is the number of
  extra processes building joint-stage crops. They share the CPU, so if the
  joint bar stalls waiting for data, raise `--workers`. If the machine is
  oversubscribed, lower `--threads`.
- `--amp` and `--grad-checkpoint` are GPU memory and speed options. They do
  nothing useful on CPU.
- You can stop at any time (Ctrl+C) and continue with `--resume`. The
  checkpoint is written after every epoch, so the epoch in progress is lost.

## 9. Files written

| file | contents |
|---|---|
| `--out` (default `rastervec/weights/deep_vectoriser.pth`) | the best joint-stage model by val Chamfer, which the P2 backend loads |
| `<out>.last.ckpt` | the full training state after the latest epoch (model, embeddings, optimizers, stage/epoch) for `--resume` |
| `<out dir>/train_log.csv` | one row per epoch: `time, stage, epoch, steps, loss, L_F, L_R, L_P, L_emb, val_endpoint_l1_px, val_chamfer_px, val_raster_iou, seconds`. It is appended across runs and resumes, so open it in a spreadsheet to plot the trends |
| `<data>/cache/*.gray.npy` | the decoded layer masks (`--cache-dir`). It is safe to delete; it is rebuilt on the next run |

To look at what a trained model actually does on a drawing, use `predict.py`:

```
python -m rastervec.P2_Raster_To_Vec.DeepVectoriser.predict --pdf X.pdf --pages 0 --clip 100,100,400,300
```

It writes a viewer folder and prints the `pipeline_report_viewer.py` command
to open it.
