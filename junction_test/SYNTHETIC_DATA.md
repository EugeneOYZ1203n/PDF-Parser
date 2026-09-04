# Synthetic data — what junction_test needs to model

Assumption for everything below: **native + vector text has already been removed** upstream
before this stage. The synthetic data must still model the *residue* that imperfect removal
leaves (see §"Text-removal residue").

## Why the current generator is insufficient

`synthetic.generate` produces one diagram type and misses most of what makes AEC
vectorization hard. Concretely, it does not model:

1. **Diagram variety** — only a small rectilinear floor plan. No sections/elevations,
   reflected ceiling plans, site plans, MEP overlays, enlarged details, stair sections,
   curved-wall / radial plans.
2. **Curved walls** — arcs appear only as r=28–52 px door swings. No large-radius wall
   arcs, no arc↔straight tangency, no ellipses, no serpentine corridors.
3. **Freeform curves** — no Bézier / spline geometry (topo contours, curved furniture,
   roads).
4. **Double-line walls / poché** — real walls are a parallel line pair, often hatched or
   filled between. Here every wall is a single stroke, so the hardest near-parallel
   separation case never occurs.
5. **Deliberate overlap** — interior walls only butt onto outer walls (clean T). No forced
   X crossings, no dense grids, no "two lines touch but are unrelated".
6. **Line weight** — two weights only (`wall_t` 5–9, `thin_t` 1–3). No AEC weight ladder,
   no weight variety within a class.
7. **Dash vocabulary** — one dash/gap (10/8). No hidden-line, centerline (dash-dot),
   phantom (dash-dot-dot), long-dash; dashes never deliberately cross other geometry.
8. **Color** — ink is a single gray value (40). No colored layers; color preservation and
   per-layer processing cannot be exercised at all.
9. **Scale** — fixed 512², fixed implicit DPI, near-square aspect. Long sections (wide
   aspect), tiny dense details, and the `Params.max_work_px` downscale path are never
   tested.
10. **Text-removal residue** — text is simply absent. Real upstream removal leaves partial
    glyph strokes and punches **white rectangular holes through lines** where text
    overlapped; tracing must bridge those.
11. **Dimension lines** — no compound {thin line + end ticks/arrows + extension lines +
    mid-line gap where the number was}.
12. **Symbols** — only door + window. No furniture, fixtures, stairs-in-section, north
    arrow, section/elevation marks, grid bubbles, break lines.
13. **Degradations** — Gaussian noise + ±8° rotate only. No line breaks/gaps, JPEG
    blocking, blur/bleed, skew, salt-pepper, low-DPI aliasing, scanner speckle.
14. **Junction ground truth** — `Junction.directions` is `[]` and there is no type label
    (L / T / X / Y / endpoint / coincident-unrelated), so overlap-disambiguation accuracy
    is unmeasurable.

## Ground-truth schema (extend `types_.py` `GroundTruth`)

Per primitive: `type` (line | arc | bezier | rect/polygon), geometry, `width`,
`color` (RGB), `layer`, `dash_style` (solid | dashed | hidden | center | phantom) +
`dash_array`, `role` (wall | curved_wall | poche_edge | opening | door_swing | door_leaf |
stair_tread | stringer | dimension_line | extension_line | terminator | fixture |
centerline | hidden_edge | mep_route | contour | property_line | grid | hatch |
symbol_stroke | border | break_line).

Per junction: `xy`, `type` (L | T | X | Y | star | endpoint | coincident_unrelated),
`arm_angles`, `members` (primitive ids), `is_true_connection` (bool — false for
`coincident_unrelated` and for a dashed line merely crossing a solid).

Per sample `meta.json`: archetype, DPI, difficulty-factor values, logical→pixel transform,
degradation list.

## Diagram archetypes (procedural builders — sample as a weighted mix)

| # | Archetype | Primarily stresses |
|---|---|---|
| 1 | Rectilinear floor plan — **double-line** walls + poché, rooms, openings | T/L junctions, near-parallel wall pairs, hatch-not-traced |
| 2 | Curved-wall / radial plan — wall arcs tangent to straight walls, curved corridors | arc fitting, arc↔line tangency **through** junctions |
| 3 | Building section / elevation — very wide aspect, long floor/roof lines, repeated verticals, poché slabs, hatch | scale + memory, long thin extent, hatch rejection |
| 4 | Stair plan + stair section — tread run + stringers + up-arrow + break line | dense equal-spacing near-parallel separation |
| 5 | Door / window enlarged detail — swing arcs, shared-centre radii, jamb ticks, small rects | small features, arc+radii symbol, tick-vs-wall junction |
| 6 | Dimension-line network — thin lines, tick/arrow terminators, extension lines, mid gaps | terminator symbols after text removal, thin lines, gap bridging |
| 7 | Reflected ceiling / tile grid — dense regular crossing grid, fixtures | **maximum X-junction density** |
| 8 | MEP overlay — multi-colour dashed routing crossing solid structure | colour layers, dashed-through-overlap, heavy crossings |
| 9 | Site plan — Bézier topo contours, curved roads, property lines (phantom), north arrow | Bézier/spline fitting, long smooth curves, dash-dot |
| 10 | Detail callout sheet — several disjoint small details on one sheet, mixed dash styles, poché | dash-style classification, multi-diagram, fills |

## Essential randomised parameters

- **Primitives**: straight; circular arc (radius 10 px → half-page); ellipse arc; cubic
  Bézier; closed rect/polygon. Enforce arc↔line **tangency** at shared endpoints in
  archetypes 2 and 9.
- **Line weights**: AEC ladder — 0.13 / 0.18 / 0.25 / 0.35 / 0.50 / 0.70 / 1.00 mm,
  converted to px at the sample DPI; ≥3 distinct weights per drawing (heavy walls, medium
  structure, light dimensions/hatch).
- **Dash styles**: solid, dashed, hidden, center (dash-dot), phantom (dash-dot-dot); store
  the real `dash_array`. **Must be recovered as one primitive + `dash_array`, never as many
  short colinear segments.** Deliberately route some dashed lines across solids.
- **Colour**: mostly near-black on off-white; blue/red/green/magenta sub-layers for
  MEP / annotation / site; anti-aliasing on. Provide the per-layer split in GT so
  colour-separated processing can be scored.
- **Junctions**: explicitly synthesise and label L, T, X (both strokes continue), Y,
  star (≥5 arms), endpoint, and **coincident_unrelated**. `overlap_density`
  (crossings / 10 kpx²) is a first-class difficulty knob: low / med / high.
- **Near-parallel**: wall pairs and stair treads 1–8 px apart — the double-line-collapse
  stressor. Knob: `parallel_gap_px`.
- **Scale / DPI**: render each logical diagram at 150 / 300 / 600 DPI; GT stays in logical
  coords + transform. Include ≥1 archetype per sample batch with aspect ratio > 4:1.
- **Text-removal residue** (text is pre-removed, imperfectly): leftover partial glyph
  strokes (reject target); white axis-aligned rectangles punched through strokes where a
  text box sat (bridge target); severity knob `residue_level`.
- **Symbols**: parametric door / window / fixture / plumbing-fixture / north-arrow /
  section-mark / grid-bubble / break-line glyphs; rendered into the raster, tagged
  `symbol_stroke` so a method is scored on *excluding* them (`extract_remainder`).
- **Hatching / poché**: 45° line hatch, cross-hatch, and solid fill between double-wall
  lines and in section slabs — must **not** vectorise to thousands of segments.
- **Degradations** (toggle per sample; define a "clean" split and a "robust" split):
  Gaussian + scanner speckle, JPEG blocking, morphological line breaks (gap-length knob),
  blur/ink-bleed, rotation ±15°, perspective skew, salt-pepper.

## Dataset layout

```
junction_test/data/synthetic_aec/
  train/  <id>/{image.png, gt.json, meta.json}      # for any learned method
  val/    <id>/{...}
  test/   <id>/{...}                                # FROZEN benchmark, fixed seeds
```

`test/` spans every archetype × {overlap low/med/high} × {clean, noisy} × {150, 300, 600
DPI} with full GT — this is the ablation benchmark referenced in `JUNCTION_ABLATION.md`.

## Implementation notes (later, separate task — not part of this doc)

- New `junction_test/synthetic_aec.py`: a `Diagram` builder with one layout function per
  archetype, `render(dpi) -> ndarray` (PyMuPDF `fitz.Shape` or PIL, anti-aliased),
  `ground_truth() -> GroundTruth`. Fully seeded / deterministic. Keep the existing
  `synthetic.generate` as archetype #1's ancestor / a smoke fixture.
- Extend `types_.py` per the schema above; extend `metrics.py` per `JUNCTION_ABLATION.md`
  §5; keep `smoke.py` green on archetype #1.
- Self-check: re-render GT primitives, assert SSIM vs raster > threshold; assert histograms
  (junction type, weight, dash style, archetype, colour) hit target coverage.
