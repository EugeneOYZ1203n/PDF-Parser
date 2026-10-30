# `junction`: classical centreline tracing

A raster_vectorize backend (`p2="junction"`) that traces line work along its centreline,
instead of outlining it as v1_raster_to_vec does. No machine learning is involved apart from the
PaddleOCR used to find text.

## What it does, per embedded image

1. **Raster prelude**: `commons.raster_text.prepare_images`, shared by every raster backend.
   - Tiled PaddleOCR on the raw image.
   - Colour CLAHE on LAB lightness.
   - Colour layers: one achromatic layer, plus one per hue peak.
   - Each recognised word's ink is erased.
2. **Tracing**, per ink colour layer and per spatial component (`components.py`), with
   `tracer.pipeline.run` at native resolution:
   - binarise the layer;
   - split off solid fills (ink at least `FILL_MIN_WIDTH_PT` wide) and outline them
     (`tracer/fills.py`);
   - skeletonise the rest;
   - build a node / edge graph: junction clusters, barb pruning, junction-zone collapse
     (`tracer/skeleton_graph.py`);
   - join strokes straight through junctions;
   - simplify with Douglas-Peucker. Line width is fixed at 1.0 pt.
3. **Diff** (debug only): the traced vectors are rasterised and compared with their own ink
   (`diff.py`).

The output is `Vector`s coloured by their layer plus the OCR `Text`s, in page space. Pixels map
to the page through the image's placement matrix (`Image.transform`); without one, the image is
assumed to fill `Image.bbox` upright.

The tracer reproduces the centreline core of Dosch, Tombre, Ah-Soon and Masini, "A complete
system for the analysis of architectural drawings" (IJDAR 3(2), 2000), sections 2–3.

## Entry points

- `adapter.extract(images, page, settings)`: the backend (`core/registry.py` key `junction`).
- `python -m raster_parser.phase_raster_vectorize.junction.predict --pdf X.pdf [--pages 0,1]
  [--clip x0,y0,x1,y1] [--dpi 300]` runs it on part of a page and writes every debug layer, ready
  for `scripts/pipeline_report_viewer.py`.

## Needs

OpenCV, scikit-image, SciPy and PaddleOCR, all in the core dependencies.

## Tests and parity

`tests/raster_parser/phase_raster_vectorize/junction/` holds the ported source tests, plus
`test_source_parity.py` (slow). That test checks the output equals what the original Junction
implementation (the project this port came from) produced on the same rasterised fixture pages (`tests/references/p2_golden/`).
