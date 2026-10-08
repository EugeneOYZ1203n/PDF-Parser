# P2 learned backends: where the implementations differ from their papers

This page lists every place where the two from-scratch P2 backends depart
from, or fill gaps in, the paper they implement:

- `rastervec/P2_Raster_To_Vec/ImplicitSketchVec/` follows **Yan et al.,
  "Deep Sketch Vectorization via Implicit Surface Extraction"**, SIGGRAPH
  2024 (`references/Deep Sketch Vectorization 3658197.pdf`).
- `rastervec/P2_Raster_To_Vec/DeepTechVec/` follows **Egiazarian et al.,
  "Deep Vectorization of Technical Drawings"**, ECCV 2020
  (`references/Deep Vectorization of Technical Drawings 2003.05471v3.pdf`).

The same choices are marked "ours" in each backend's `config.py` and module
docstrings. Shared by both backends:

- Neither has a prep script. Both train on the dataset
  `DeepVectoriser/prep_dataset.py` writes, through copied readers.
- Both derive every training target from that dataset's vector strokes on
  the fly.
- Both run DeepVectoriser's copied colour separation → tiled OCR → text
  removal front end before the model.

---

## ImplicitSketchVec (Yan et al., SIGGRAPH 2024)

### Training data and targets
1. **Data.** We use our technical-drawing layer masks at 300 dpi, where the
   paper uses Quick Draw / Creative Sketch sketches. There is no brush-style
   or paper-texture augmentation. Each training crop is 128–256 px with a
   random rotation and flip, applied to both the raster and the vector
   ground truth before targets are computed (the paper's
   "augment the vector, then compute the UDF" order).
2. **Ground-truth vertex map.** The paper takes the point on the path
   closest to the cell centre. We pick the nearest of points sampled every
   0.1 px along the strokes, which is within about 0.05 px of the exact point.
3. **Keypoint labels.** The paper follows Puhachov et al. 2021 without
   giving rules. Ours:
   - an *end* is a stroke end no other stroke touches;
   - a *sharp* point is a turn of more than 30°, at a joint inside a stroke
     or where two stroke ends meet;
   - a *junction* is three or more ends meeting, a stroke ending on
     another stroke's interior (T), or two strokes crossing (X).
4. **Distance cap and sparse fields.** All UDFs are clamped at 8 px; the
   paper gives no cap. Outside the 4.5 px loss mask, the five
   non-centreline UDFs are left at that cap, because the loss never reads
   them there.
5. **Grid-line nudge.** Cell assignment adds a 1e-6 nudge. Without it, a
   stroke lying exactly on a grid line (common for axis-aligned CAD lines)
   flickers between two cells on float noise. That reads as repeated
   crossings and creates false under-sampled regions.

### Networks
6. **Distance Field Prediction layout is ours.** The paper only states its
   changes to Puhachov's network: 256 inner channels, cardinality 8, and
   gradually increasing dilation. Our layout:
   - a full-resolution stem;
   - a stride-2 downsample;
   - eight ResNeXt bottleneck blocks at half resolution, with dilations
     1, 1, 2, 2, 4, 4, 8, 8;
   - a ×4 transposed convolution up to 2× resolution;
   - a skip connection from the stem;
   - a light 32-channel head;
   - the paper's 2×2 padding-1 domain-conversion layer (Appendix B).

   `--basic` reduces the DFP to 128 channels, as the paper's basic variant does.
7. **Line Reconstruction sizes.** The paper fixes the structure (2×2 lattice
   → cell conversion, three branches of three residual convolutions at
   dilation 1/2/3, a trunk, 1×1 heads). The 64 channels and 6 trunk blocks
   are ours.
8. **Loss normalisation.** The masked L1 (Eq. 2) is averaged over the masked
   points instead of over every lattice point, and summed over the six UDFs.
9. **Optimiser and schedule are ours**: Adam at lr 1e-4, batch 8, 20 epochs
   of 500 steps, gradient clipping at 1.0. The paper gives none of these.
10. **NDC training noise.** The `ndc` stage trains on the ground-truth
    centerline UDF plus Gaussian noise with a std of 1 % of the UDF's own
    std. The paper only uses that noise level in its robustness test
    (Appendix C).
11. **Validation and best-checkpoint metric are ours.** The `dfp` stage
    keeps the lowest masked centerline error; `ndc` and `joint` keep the
    highest edge-flag F1.

### Post-processing (paper Sec. 7)
12. **Edge-flag refinement (Fig. 5)** skeletonises the flag occupancy only
    where it is two cells thick (cells in a fully occupied 2×2 block).
    Zhang–Suen thinning everywhere deletes the corner cells of every
    staircase, which breaks every diagonal line.
13. **Broken-line repair (Fig. 6).** Instead of the paper's four 3×3
    detection kernels, we join open (degree-1) ends in adjacent cells
    directly: 4-neighbours first, then diagonal.
14. **Topology surgery (Fig. 7)** has two additions:
    - open ends lying right next to an under-sampled region are also
      reconnected, as branches lost into it;
    - a region with no truncated edges is left as it is. The paper's
      procedure would delete, for example, two whole lines closer than one
      0.5 px cell.
15. **Keypoint detection.** The thresholds behind "threshold + local
    maximum" are ours: a keypoint UDF below 1 px that is minimal within
    2 px. An under-sampled cell is one whose predicted USM distance is
    below 0.5 px.
16. **DC downsampling (Fig. 10)** is done on the graph rather than on the
    grid. A 2×2 block of cells becomes one coarse vertex: its single open
    endpoint, otherwise the mean of its vertices. Blocks that are
    under-sampled, or contain more than one separate piece of line, keep
    their fine vertices. The factor is 2, back to the native pixel grid;
    the paper shows 3× and 8×.
17. **Junction splitting is ours (new step).** At every vertex of degree 3
    or more, arms that continue nearly straight through it are paired. Arm
    directions are measured a few vertices out, and pairs need a dot
    product below −0.8. The remaining arms end there. This keeps crossing
    and T-ing lines straight instead of letting line grouping bend them.
18. **Line grouping.** The paper computes all-pairs shortest paths. We
    differ in three ways:
    - the longest shortest path per component comes from a double Dijkstra,
      which is exact on trees;
    - a component above 4000 vertices falls back to linear chain
      decomposition through degree-2 vertices;
    - paths that meet end to end at a degree-2 vertex are then chained, so a
      closed outline is one stroke.
19. **Smoothing.** The Schneider fit always tries up to 8 Newton
    reparameterisation rounds before splitting. The original only tries
    when the error is already within 4× the tolerance. The default fit
    tolerance (0.75 px) is ours. RDP is available via `FIT_MODE = "rdp"`.
20. **No interactive topology-editing GUI** (paper Sec. 8.4).

### Inference
21. **Tiling.** We use 256 px tiles with a 32 px overlap. Each tile's core
    is stitched in, with neighbouring tiles splitting their actual overlap
    at its middle. The DC data is harvested sparsely into one page-wide
    graph, so no dense page-sized cell grid is ever built. The paper runs
    whole images.
22. **Stroke width** comes from the ink's distance transform (2 × the median
    along the stroke). The paper outputs no widths.
23. **Debug rendering.** The dense DC-domain debug layers (raw edges,
    refined edges, under-sampling map) are rendered as raster overlays.

---

## DeepTechVec (Egiazarian et al., ECCV 2020)

### Pipeline
1. **Cleaning U-Net (paper step 1) is not run.** The input is already a
   clean binary layer mask from colour separation, and the shared dataset
   has no degraded/clean pairs to train it on.

### Primitive network
2. **Full-width attention heads.** The paper gives d_emb = 6 for lines with
   4 heads, and 6 doesn't divide by 4. So each head attends at the full
   d_emb width, projected d_emb → 4·d_emb → d_emb. This is our reading of
   the paper.
3. **Stem stride.** The stem's stride of 2 (32×32 = 1024 feature tokens) is
   ours.
4. **Normalisation.** GroupNorm replaces BatchNorm.
5. **Loss weight.** λ = 0.5 in L_loc is ours (the paper says "weighted sum").

### Training
6. **Data.** We use our data at 300 dpi instead of PFP/ABC plus synthetic
   data. The random rotation (any angle) and scale (0.8–1.25, ours) are
   drawn on the fly instead of pre-computed.
7. **Targets from cubic ground truth.** Lines come from flattening each
   cubic to 0.5 px and merging consecutive chords within 3°. Quadratics come
   from a least-squares fit per cubic, split until within 0.5 px.
8. **Overflow.** A patch with more than n_prim = 10 primitives is re-drawn
   up to 5 times, then only the 10 longest are kept.
9. **Optimiser.** The batch of 128, Adam with the Transformer schedule and
   4000 warmup steps follow the paper. Gradient clipping is ours.
10. **Validation.** The best checkpoint is chosen by validation IoU.

### Refinement (Sec. 3.3, Appendix I)
11. **Coverage.** The paper uses analytic integrals; we use the soft
    rasterisation `clamp(w/2 + ½ − d, 0, 1)`. Curves are flattened to 8
    segments.
12. **Potential.** φ(r) is the paper's (R_c 1 px, R_f 32 px, λ_f 0.02), but
    the far Gaussian runs on a ¼-resolution grid with a truncated kernel,
    for speed.
13. **Curve parameterisation.** A curve is parameterised by its point at
    t = ½ plus the two arm lengths and angles. The paper uses the curve's
    intersection with the control-angle bisector.
14. **Connected-area mask c_k.** It is symmetric (the max of the two
    normal-direction extents) instead of the paper's two-sided rectangle.
    For curves, the run along the extended parabola is approximated with
    sampled points.
15. **Collinearity term.** Directions in E^rdn are summed in doubled-angle
    form, so a segment and its reverse count the same.
16. **Gradients.** Per-pixel gradients are analytic (envelope theorem), with
    autograd only from parameters to vertices. This is mathematically the
    same mean-field split, just faster.
17. **Join/move heuristics are ours.** The paper only names them ("join
    lined-up primitives, move collapsed ones onto uncovered ink"). Our
    rules:
    - only lines are joined, using the merge thresholds, every 20 iterations;
    - a collapsed primitive moves onto the most uncovered ink pixel with a
      2 px length.
18. **Iterations and steps are ours**: 50 iterations by default (0
    disables refinement), Adam steps of 0.1 px and 0.01 rad, and
    λ_pos = 2.0.

### Merging (Sec. 3.4, Appendix C)
19. **Thresholds are ours.** Lines are linked when the shorter one's ends
    lie within 1.5 px of the longer one's line, the angle is under 5°, the
    gap is at most 2 px, and the widths are within 2×. A dangling end is cut
    if it is under 5 % of the line. For curves, the distance is 2 px and the
    fit error 1 px.
20. **Curve pair fit.** It uses t_b = s_b = ½, matching the midpoint
    definition above. The pair is oriented so that Q's start is nearest P's
    end.

### Inference
21. **Patching.** Patches are 64 px with a 16 px overlap (the overlap is
    ours). Each patch's primitives are clipped to its core before merging,
    with neighbouring patches splitting their actual overlap at its middle.
22. **Output width.** It is the network's predicted width, converted to pt.
