# InvolutionStroke (P2 backend)

Raster → stroke vectorization with the **authors' pretrained weights** from
Gerstner, Magne & Sorkine-Hornung, *"Stroke Vectorization via Involution
Prediction"*, VMV 2026. The reference code is at
<https://github.com/verven/InvolutionStrokeVectorization> (vendored @
`178eede602bfb488b46cfc38ea2176ff902a4da2`).

> **Licence caveat:** the upstream repository has **no LICENSE file**, so by
> default all rights are reserved. The vendored code (`_upstream/`, `model/`)
> and the checkpoints are fine for internal research use. Check with the
> authors before redistributing either one.

## Running it

```
.venv/Scripts/python.exe -m rastervec.P2_Raster_To_Vec.InvolutionStroke.download_weights
.venv/Scripts/python.exe -m rastervec.P2_Raster_To_Vec.InvolutionStroke.predict --pdf X.pdf [--pages 0] [--clip x0,y0,x1,y1]
.venv/Scripts/python.exe -m rastervec.core.pipeline --pdf X.pdf --page 0 --p2 InvolutionStroke
```

`download_weights.py` fetches the two checkpoints from ETH IGL into
`rastervec/weights/` (gitignored) and checks that they load:

- `involution_centerline.pt`, about 177 MB.
- `involution_intersection.pt`, about 15 MB.

To use other files, set `$INVOLUTIONSTROKE_CENTERLINE_WEIGHTS` and
`$INVOLUTIONSTROKE_INTERSECTION_WEIGHTS`. If the weights are missing, the
backend raises `FileNotFoundError` naming the download command.

## Flow

For each embedded image:

1. **Front end.** Colour separation → tiled OCR → text removal. These are
   copied from ImplicitSketchVec, originally DeepVectoriser / Junction.
2. **Per ink layer.** The binary layer mask is resampled to 300 dpi and cut
   into 1024 px tiles with a 96 px overlap. Each tile runs upstream's whole
   `main.process_image` chain:
   - **Centerline UNet** (`model/centerline.py`, `ResNextUNetLarge`).
   - **Planar line graph** (`_upstream/graph_extraction`, pure Python:
     networkx / shapely / skimage) → polylines.
   - **Half-edge pairing network** (`model/intersection.py`,
     `SceneGraphVectorModel`) → involutive max-weight matching → strokes.
3. **Stitching.** Strokes are clipped to the core of their tile. Ends from
   different tiles that meet on a core border are joined.
4. **Fitting.** Strokes are cut at sharp corners. Straight runs become lines,
   with corners moved to where neighbouring lines intersect. Curved runs are
   pre-smoothed by about 1 px, then fitted with Schneider Béziers.

All tunables are in `config.py`.

## Deviations from upstream

- **No TorchScript, Lightning or torchvision.** The two network classes are
  vendored and the raw checkpoints load into them with `strict=True`.
  torchvision's `Bottleneck` is re-implemented with identical parameter names.
- **Pairing inputs.** Only the inputs the released `SceneGraphVectorModel`
  reads are built: polylines, R and n_objects. Upstream's segmentation and
  line-drawing rasters are only consumed by its image-backbone variant.
- **Centerline padding.** The centerline input is padded to a multiple of
  16, not to the next power of two. The network is fully convolutional.
- **Tiling.** Upstream works on whole 512–1024 px images. We tile, clip to
  the tile cores and join strokes across seams.
- **Python 3.10 / numpy 1.x ports in `_upstream/`.** These are mechanical,
  and each file's header lists its own:
  - `np.concat` → `np.concatenate`
  - `np.bool` → `np.bool_`
  - `a[*idx]` → `a[tuple(idx)]`
  - `print` → debug log
  - debug-only SVG plotting dropped
- **Output smoothing is ours.** Upstream emits dense polylines, with an
  optional `--smooth`, which is `SMOOTH_LINES` here.

## Known behaviour

- **Plain crossings may not be paired.** The released pairing model often
  ends both strokes at an isolated `+` crossing instead of pairing them
  straight through. Its training data allows a stroke to end at any
  intersection. Crossings inside denser drawings pair as expected (see
  `test_model.py::test_real_weights_resolve_crossings`).
- **Speed (CPU).** The centerline UNet takes about 35–55 s per 1024 px tile.
  A full page at 300 dpi is tens of tiles per ink layer.
  `TARGET_PX_PER_PT` is the main speed/detail knob.
- **Training scale.** The networks were trained on sketches rasterized at
  512–1024 px, whereas drafting lines at 300 dpi are only about 1–3 px wide.
  Tune the scale on real drawings.
