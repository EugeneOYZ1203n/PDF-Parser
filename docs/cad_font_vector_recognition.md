# CAD-Vector Character Recognition

## Problem

Some architectural/engineering shop drawings encode text as raw PDF vector
line-art per character (a "CAD font") rather than as real PDF text objects —
each glyph is hand-drawn as a small set of `Vector` path primitives (lines,
curves, rects/quads). Neither `rastervec`'s native-text extraction (Phase 1)
nor its OCR-based classification (Phase 3 backends) recovers this kind of
text: there is no PDF text object to read, and it isn't raster/pixel content
either, so pixel OCR is the wrong tool for it.

The approach explored here instead treats each character, under a given
font, as a small topological graph — points (endpoints, corners, stroke
intersections) connected by edges (line/curve segments) — and aims to
recognize CAD-vector text by matching a labelled library of these character
graphs against the vector content of a test PDF via affine-transform
estimation, rather than by rendering pixels and running OCR.

This is a 12-step algorithm. Only **steps 1-3** are implemented today: a
labelling tool to build the character+baseline ground-truth file (step 1),
and graph construction + a complexity-sorted character bank with anchor
points (steps 2-3). Steps 4-12 — extracting and clustering candidate vector
groups from an unlabelled test PDF, affine-matching them against the
character bank, and consolidating matches into a final text/non-text split —
are recorded below as a fixed spec for future work, not yet implemented.

## Assumptions

### Primary

Under the same font, each character forms a graph of endpoints and edges.

### Secondary

- Text tends to share a common baseline: a line stretching across the page,
  with multiple characters sitting on it at the same rotation.
- Complex characters (more segments/edges) are more trustworthy matches than
  simple ones — a single stroke is highly ambiguous on its own, while a
  character with many distinct junctions is much less likely to match by
  coincidence.

## Status

| Steps | Status | Where |
|---|---|---|
| 1 | **Implemented** | `scripts/label/CAD_font_label.py` |
| 2-3 | **Implemented** | `rastervec/Evaluation/CadFont/` (`geometry.py`, `graph.py`, `character_bank.py`) + `rastervec/notebooks/cad_font_matching_lab.ipynb` (section 3) |
| 4-12 | **Future work — documented only, no code** | — |

## Process

### Step 1 — Label file generation `[IMPLEMENTED]`

One character maps to a list of `Vector`s, plus assignment to one baseline
(a line stretching across the page at the base of the character) and that
baseline's own direction. Built with `scripts/label/CAD_font_label.py`, a
Tkinter tool with two modes: "Baseline" (drag out a new baseline, or select
an existing one to see/toggle which characters are currently assigned to
it) and "Label" (identical to the existing `scripts/label/vector_label.py`
flow — select vectors, apply a text label — independent of baseline state).
Output is a JSON `LabelSet` (`rastervec/Evaluation/Labelling/
label_schema.py`) with `source="cad_font"` entries plus a `baselines` list.

### Step 2 — Character graph construction `[IMPLEMENTED]`

Build a planar graph from a labelled character's vectors: flatten every
`Vector.items` primitive into straight-line segments (a cubic bezier is
always sampled into a fixed 5 points — 2 real endpoints + 3 generated
interior points, i.e. 4 straight-line segments; a rect/quad into its 4
corner points connected edge-to-edge), compute every pairwise segment
intersection and split segments there so the final edge set is planar
(edges touch only at shared endpoints), then reduce bit-identical points to
one graph node each (exact dedup, not a tolerance merge) and connect any
two *distinct* nodes within `epsilon` of each other that aren't already
connected (an added edge, not a point merge — see below), before running a
Douglas-Peucker simplification pass per chain of non-junction nodes. An
original data vertex is an ordinary Douglas-Peucker candidate, not
unconditionally protected — only a real graph junction (degree > 2)
structurally survives every time — subject to one floor: each connected
component of the graph keeps at least 2 of its own original vertices (or
however many it has, if fewer than 2 exist), restoring the geometrically
most significant dropped one(s) first when plain simplification would
otherwise breach that floor.
`epsilon` is not a tunable constant: it's derived per character as half the
length of the shortest post-split segment, used for both the
connect-nearby-points step and the Douglas-Peucker tolerance. No position
normalization beyond: the assigned baseline is rotated to the line `y=0`,
and the character's own leftmost point is translated to `x=0`. Implemented
in `rastervec/Evaluation/CadFont/geometry.py` (the flatten/split/dedupe/
connect pipeline) and `graph.py` (`CharGraph`, `build_char_graph`'s
epsilon-derivation + Douglas-Peucker orchestration, and `GraphBuildStats`,
the per-character epsilon/points-removed diagnostics it returns alongside
the graph).

### Step 3 — Character → graph mapping, complexity, anchors `[IMPLEMENTED]`

Map every labelled character to its `CharGraph`; sort characters by
complexity (`num_nodes * num_edges * max_degree`); select up to 3
non-collinear, highest-degree anchor points per character (2 if the graph
has fewer than 3 non-collinear nodes). Implemented in
`rastervec/Evaluation/CadFont/character_bank.py`
(`CharacterTemplate`/`build_character_bank`) and
`graph.py` (`complexity`/`select_anchor_points`). Exercised end-to-end, with
per-character debug stats and a raw-vectors-vs-graph gallery, by
`rastervec/notebooks/cad_font_matching_lab.ipynb`'s section 3 (there is no
separate character-graph-only notebook any more -- the matching lab covers
this as its own first step before matching).

### Step 4 — Vector extraction + seqno spatial clustering on the test PDF `[FUTURE]`

Extract every `Vector` from each page of an unlabelled test PDF, then group
them into candidate vector groups via "seqno consecutive spatial
clustering" — clustering that respects both spatial proximity and
adjacency in PDF content-stream draw order (`Vector.seqno`), analogous in
spirit to `rastervec/notebooks/similarity_single_line_cad_text.ipynb`'s own
candidate-run extraction, but producing groups to graph-match rather than
groups to delta-chain-match.

### Step 5 — Candidate graph construction `[FUTURE]`

Build a `CharGraph` for each candidate vector group from step 4, using the
same flatten/split/dedupe/connect/Douglas-Peucker pipeline as step 2
(`matching.py::build_candidate_graph`). No baseline is known yet at this
point, so no baseline-relative normalization is possible until a baseline
hypothesis exists (see step 7.1) — candidate graphs start in raw page
space, and their own `epsilon` is derived from that raw-page-space
geometry directly (same formula as step 2, no external scale needed).

### Step 6 — Affine matching `[FUTURE]`

6.1. Test character graphs from the bank most-complex-first (reusing step
3's complexity sort).
6.2. Prune: only test a candidate vector-group graph against a character
graph when the candidate's `max_degree >= character's max_degree` AND
`num_nodes >= character's num_nodes` AND `num_edges >= character's
num_edges` — a match cannot exist otherwise.
6.3. Test every possible affine transform:
&nbsp;&nbsp;6.3.1. Using the character's stored anchor points, test every
combination of candidate-graph nodes with matching degree as the
correspondence.
&nbsp;&nbsp;6.3.2. With 3 anchor points, the correspondence is enough to
check whether an affine transform is even possible (and to solve for one).
6.4. Record the winning transform's translation, scale, and rotation.
6.5. Score similarity as a function of (MSE of matched endpoint pairs,
normalized by the transform's scale) plus how well edge counts and degrees
match between the character graph and the matched subset. The MSE term is
weighted per matched point's provenance — original data vertex / true
intersection-junction / `select_anchor_points`'s synthesized baseline
anchor (`matching.py::score_match`'s `*_point_weight` kwargs, all default
1.0) — since a synthesized anchor has no real candidate counterpart, only
a reference line it should lie near, so its distance is measured to the
nearest point on the matched candidate edge rather than to a single
candidate node.

### Step 7 — State update per candidate `[FUTURE]`

7.1. If no baseline already exists near the candidate (within some
tolerance), create one from the newly predicted character's implied
rotation.
7.2. If a nearby baseline does exist, increase its confidence, scaled by
the candidate's similarity score (not including any baseline-match bonus).

### Step 8 — Iterate `[FUTURE]`

Continue steps 6-7 until every character template has been tested against
every eligible candidate vector group.

### Step 9 — Baseline pruning `[FUTURE]`

Look at all baselines; remove any baseline with low confidence, and every
text candidate associated with it.

### Step 10 — Local consistency check `[FUTURE]`

For each surviving baseline, find text candidates on it whose scale or
rotation differs from nearby candidates on the same baseline (flag them as
suspect).

### Step 11 — Final candidate selection `[FUTURE]`

For each vector, select the single highest-confidence text candidate that
references it (ranked by similarity, character complexity, and baseline
confidence together).

### Step 12 — Output split `[FUTURE]`

Split vectors into those classified as text and those not classified;
label and group the text ones by character.

## Open design notes for future steps

- How step 5's un-normalized (page-space) candidate graphs become
  baseline-relative once a baseline hypothesis exists from step 7.1 — most
  likely: re-run step 5's flatten/split/dedupe/connect/Douglas-Peucker
  pipeline through `character_bank.to_baseline_relative_vectors` once a
  candidate baseline is known, mirroring exactly how a labelled character
  is normalized today.
- What "nearby" means in steps 7.1 ("no baseline exists... within some
  tolerance") and 10 ("candidates whose scale or rotation is different from
  nearby candidates") — likely a page-distance-along-baseline-direction
  tolerance plus an angular tolerance, but not yet pinned down.
- `complexity()`'s "Num endpoints" factor (see step 3 / `graph.py`) is
  interpreted as every graph node (leaves and internal junction/
  intersection nodes alike), not only degree-1 leaves — this reading is
  documented at the function itself and should carry through consistently
  into steps 4-12's own graph-size comparisons (step 6.2's pruning test).
