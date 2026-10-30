# `deep_tech_vec`: Deep Vectorization of Technical Drawings (optional raster_vectorize backend)

Raster → line / quadratic-Bézier primitives with the **authors' pretrained
weights** from Egiazarian et al., *"Deep Vectorization of Technical
Drawings"*, ECCV 2020
(`references/Deep Vectorization of Technical Drawings 2003.05471v3.pdf`).
The reference code is at
<https://github.com/Vahe1994/Deep-Vectorization-of-Technical-Drawings>; the
network here is copied from commit `63cbc673ee9176925d1ae18e981341df68cb993d`.

> **Licence:** upstream is **MPL-2.0**. `model/network.py` is derived from
> upstream files and stays under MPL-2.0 (see its header). MPL is file-level
> copyleft, so the rest of this folder is not affected.

## Running it

```
.venv/Scripts/python.exe -m raster_parser.phase_raster_vectorize.deep_tech_vec.download_weights [--prim line|curve|all]
.venv/Scripts/python.exe -m raster_parser.phase_raster_vectorize.deep_tech_vec.predict --pdf X.pdf [--pages 0] [--clip x0,y0,x1,y1] [--weights W.weights]
python -c "from raster_parser import extract; print(extract('X.pdf', p2='deep_tech_vec'))"
```

`download_weights.py` fetches the authors' checkpoints from their public
Yandex Disk links (resolved through the Yandex Disk API, with retries, and
checked against a pinned sha256). It saves them to `raster_parser/weights/`
(gitignored) and checks that they load:

- `deep_tech_vec_lines.weights`, about 149 MB. This is the default model
  (upstream: floor plans, PFP).
- `deep_tech_vec_curves.weights`, about 231 MB (upstream: ABC CAD).

To use another file, set `$DEEPTECHVEC_WEIGHTS_PATH`, or pass `--weights` to
`predict.py`. The kind (lines or curves) is read off the checkpoint's output
layer. If the weights are missing, the backend raises `FileNotFoundError`
naming the download command. Nothing is trained in this repo.

## Flow

For each embedded image:

1. **Front end.** The shared raster prelude: OCR → colour CLAHE →
   colour layers → text removal.
2. **Per ink layer.** The binary layer mask is resampled to 300 dpi and cut
   into 64 px patches with a 16 px overlap. Patches with no ink are skipped.
3. **Network (upstream).** Each patch is turned into `[ink, x·mask, y·mask]`.
   The network gives 10 rows of `coords…, width, confidence`. Everything is
   scaled × 64, and rows with confidence < 0.5 are dropped.
4. **Refinement and merging (ours).** These follow the paper (Sec. 3.3/3.4)
   in `refine.py` and `merge.py`. Upstream's own refinement and merging code
   is CUDA-bound and isn't vendored. Each patch is clipped to its core
   before merging.

Deviations from the paper are listed in `docs/p2_paper_deviations.md`.
