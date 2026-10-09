# Reading `train.py`'s output

`train.py` trains ImplicitSketchVec (Yan et al., SIGGRAPH 2024) on the
dataset **DeepVectoriser's** `prep_dataset.py` writes. There is no
ImplicitSketchVec prep step: every target (six UDFs, edge flags, vertex map,
skeleton, keypoints) is generated from that dataset's vector strokes on the
fly, after augmentation.

It trains in the paper's three stages, one stage per run:

```
python -m rastervec.P2_Raster_To_Vec.ImplicitSketchVec.train --data data/deepvec --stage dfp   [--threads N] [--resume]
python -m rastervec.P2_Raster_To_Vec.ImplicitSketchVec.train --data data/deepvec --stage ndc   [--threads N] [--resume]
python -m rastervec.P2_Raster_To_Vec.ImplicitSketchVec.train --data data/deepvec --stage joint [--threads N] [--resume]
```

| stage | trains | input → target | default `--out` |
|---|---|---|---|
| `dfp` | Distance Field Prediction | raster → six UDFs at 2× (L_line) | `implicit_sketch_vec_dfp.pth` |
| `ndc` | Line Reconstruction (2D NDC) | *ground-truth* centerline UDF + 1 % noise → edge flags, vertex map, skeleton (L_rec) | `implicit_sketch_vec_ndc.pth` |
| `joint` | both, end to end | raster → … (L_rec + Σ L_line), starting from the two files above | `implicit_sketch_vec.pth`, which the adapter loads |

`dfp` and `ndc` don't depend on each other, so run them **at the same time**
as separate processes. On a CPU, split the cores between them with
`--threads`. Run `joint` once both have finished. All of them, plus
DeepVectoriser's and DeepTechVec's trainers, share `<data>/cache/`.

`--basic` builds the paper's reduced-size model (128-channel DFP), which is
easier on a CPU.

The excerpts below come from a `--tiny` smoke run on a synthetic dataset
(`--epochs 1 --steps-per-epoch 3 --batch 2 --val-crops 2 --workers 0`).

---

## 1. Banner

```
========================================================================
ImplicitSketchVec training, stage dfp | device cpu | 16 threads | data …\data
train layers 1, val layers 1 | out …\dfp.pth
model {'dfp_ch': 16, 'dfp_cardinality': 2, ..., 'ndc_trunk_layers': 1}
========================================================================
```

`model` is the architecture stored in every checkpoint. The joint stage
refuses to start unless both stage files exist. It names the missing one and
tells you which `--stage` to run, or which `--init-dfp` / `--init-ndc` to pass.

## 2. The epoch bar and line

```
[dfp ep 1/1]   steps=3, loss=19.69, L_line=19.69, L_line_center_px=2.23, val_center_err_px=2.22, seconds=2.8
[ndc ep 1/1]   steps=3, loss=1.92, L_edge=1.68, L_vertex=0.46, L_skel=0.74, val_edge_precision=0.0022, val_edge_recall=0.337, val_edge_f1=0.0043, ...
[joint ep 1/1] ... val_center_err_px=2.21, val_edge_precision=0.007, val_edge_recall=0.69, val_edge_f1=0.014, ...
```

| term | meaning |
|---|---|
| `L_line` | Eq. 2, **summed over the six UDFs**: the masked L1 error in px, near strokes only (GT centerline UDF < 4.5 px) |
| `L_line_center_px` | the centerline UDF's share of `L_line`, i.e. its mean error in px |
| `L_edge` | cross-entropy over the 4 edge-flag classes per 0.5 px cell, near strokes only |
| `L_vertex` | squared error of the vertex position inside the cell (in cell units), on cells the stroke passes through |
| `L_skel` | BCE of the 1-cell skeleton; it has weight 0.01 in the loss |
| `val_center_err_px` | mean centerline UDF error near strokes on the validation crops (dfp, joint). **Lower is better; `dfp` keeps its best.** |
| `val_edge_precision` / `recall` / `f1` | per flagged cell edge (right and bottom edges counted separately), predicted vs GT. For `ndc` the network is fed the GT UDF; for `joint` it gets the DFP's prediction. **`ndc` and `joint` keep their best F1.** |

A healthy `ndc` run quickly reaches high recall with low precision (it flags
everything near ink), and precision then climbs. `joint` starts near the two
stages' own numbers and should improve on `ndc`'s F1, because it learns to
read the DFP's imperfect UDF.

## 3. Files

| file | |
|---|---|
| `--out` | the best weights for this stage |
| `<out>.last.ckpt` | the full state after every epoch, which `--resume` continues from |
| `<out stem>_train_log.csv` | one row per epoch: every value above |

A non-finite loss prints `TRAINING FAILED at stage=… epoch=… step=… crop
size=…`, saves `<out>.last.ckpt` and exits 2.

## 4. Trying the model

```
python -m rastervec.P2_Raster_To_Vec.ImplicitSketchVec.predict --pdf X.pdf --pages 0 --clip x0,y0,x1,y1 \
    [--weights rastervec/weights/implicit_sketch_vec.pth]
```

This writes a `scripts/pipeline_report_viewer.py` folder. OCR never runs.
Its layers:
- `dfp/udf centerline`: heat map
- `dfp/keypoints *`, `dfp/under-sampling map`
- `ndc/raw edges`: the DC graph straight from the flags
- `refine/refined edges`: after every graph step
- `topology/refined regions`
- `group/strokes`
- `fit/fitted vectors`
- `original/vectors`: the PDF's own vectors, for comparison

## 5. Training on GPU

The repo's `requirements.txt` pins the **CPU** torch wheel (`+cpu`). On the
training machine install a CUDA build instead, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu126`
(match your driver), then check `python -c "import torch; print(torch.cuda.is_available())"`.
`--device` defaults to `cuda` whenever it is available.

On CUDA, `train.py` turns on by default:

- **mixed precision** -- bf16 where the GPU supports it (Ampere and newer;
  no loss scaling needed), else fp16 with a `GradScaler`. Losses are always
  computed in fp32. `--no-amp` trains in fp32 (TF32 matmuls still on).
- **cuDNN autotuning** (`cudnn.benchmark`) and **TF32** matmuls/convolutions.
- **fused Adam**, **pinned host memory** and non-blocking host-to-GPU copies.
- **fewer host syncs** -- loss terms are summed on the GPU and the bar's
  running means are refreshed every `--log-every` steps (default 20), not
  every step. The epoch line is exact.
- `dfp` and `ndc` can share one GPU as two processes if memory allows.

The banner shows the GPU and the AMP dtype, e.g.
`device cuda (NVIDIA GeForce RTX 4090) | AMP bfloat16`.
If the data side can't keep up (GPU utilisation low, the bar stalls between
steps), raise `--workers`; the batches are built on CPU worker processes.

```
python -m rastervec.P2_Raster_To_Vec.ImplicitSketchVec.train --data data/deepvec --stage dfp --workers 6
```

A checkpoint trained on GPU loads unchanged for CPU inference (weights are
saved device-free and loaded with `map_location="cpu"`); `--resume` works
across CPU/GPU and AMP on/off.
