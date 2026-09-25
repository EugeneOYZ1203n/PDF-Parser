# CAD-font matching: speedup research + complexity analysis

Research and analysis only -- **no algorithm code in `rastervec/Evaluation/CadFont/` changes
as a result of this doc**. It exists to ground a future speedup decision in real published
techniques and an accurate accounting of where `matching.py`'s time actually goes, rather
than guessing. See `docs/cad_font_vector_recognition.md` for the algorithm this doc is
analyzing (steps 5-6.5).

## 1. Current algorithm, with real complexity terms

Notation: one character/candidate graph has `n` nodes, `s` raw (pre-split) segments, `c`
critical points (`graph.critical_point_indices`), `t` template nodes, `m` candidate nodes,
`N` templates in the bank.

| Step | Function | Complexity | Notes |
|---|---|---|---|
| Segment-crossing split | `geometry.split_at_intersections` | O(s²) | All-pairs test; already documented in-code as fine since `s` is small per character. |
| Exact dedup | `geometry.dedupe_exact_points` | O(s) | Single dict-keyed pass. |
| Epsilon-connect | `geometry.connect_nearby_points` | O(n log n) | `cKDTree.query_pairs`. |
| RDP simplify | `graph._rdp_chain` | O(m²) worst case per chain of length `m` | Naive recursive max-scan; see §1.1. |
| Original-vertex floor rescue | `graph._rescue_original_vertex_floor` | O(dropped × chain length), ≤2 rescue rounds per component | Negligible at character scale. |
| Anchor-pair correspondence search | `matching.enumerate_anchor_correspondences` | **O(c²)** | The real hot path -- see §1.2. |
| Per-correspondence transform fit | `matching.fit_similarity_transform` | O(1) | Closed-form 2×2 SVD. |
| Per-correspondence node mapping | `matching.map_template_nodes_to_candidate` | O(t log m) | `cKDTree.query` over all `t` template nodes. |
| Per-correspondence scoring | `matching.score_match` | O(t) | Each synthetic point's reference-line check is O(local candidate degree), a small constant. |
| **One template vs one candidate** | `matching.match_template_against_candidate` | **O(c² · t log m)** | Dominated by the correspondence count. |
| **Full bank vs one candidate** | notebook's full-bank loop | **O(N · c² · t log m)** | `c²` is paid independently by every one of the `N` templates against the *same* candidate graph -- the biggest structural inefficiency (§2 addresses this directly). |

### 1.1 RDP's O(m²) is a non-issue here

`_rdp_chain` re-scans every remaining interior point on each recursive split, giving O(m²)
worst case for a chain of `m` points (vs. the O(m log m) achievable with Hershberger &
Snoyink's stack-based reformulation of Douglas-Peucker[^hs92]). Not worth adopting: a
character's own chains are a handful of points long (bezier interior samples between two
real vertices, or a short run of near-collinear original vertices) -- `m` is never large
enough for the quadratic term to matter in practice.

### 1.2 Why c² is the real target

`match_template_against_candidate` only ever searches the template's **first 2** anchor
slots combinatorially (the 3rd anchor falls out for free from the node mapping); each slot
is filtered by "candidate degree ≥ that anchor's own degree" before enumeration, so the
practical cost is `O(|slot0| · |slot1|)`, bounded by `c²`. For a busy candidate cluster
(many real intersections from `split_at_intersections`, e.g. dense CAD line art), `c` can be
large enough that this term dominates every other step combined -- and because the notebook's
full-bank loop calls this once per template against the *same* candidate graph, the cost is
paid `N` separate times for work that's substantially about the *same* candidate geometry
each time.

## 2. Applicable literature

Each entry: what it is, why it's a plausible fit here, and the honest tradeoff -- this is a
survey to inform a decision, not a recommendation to adopt all of them.

### RANSAC (Fischler & Bolles, 1981)[^ransac]

Randomized minimal-sample hypothesize-and-verify model fitting: repeatedly draw the smallest
possible sample needed to fit a model (here: 2 point correspondences, since that's exactly
what `fit_similarity_transform` needs), fit, then check how well the rest of the data agrees
(the existing `score_match`/consensus-style scoring already does this check). The standard
iteration-count formula bounds how many random samples are needed for a given confidence:

```
k = log(1 - p) / log(1 - w^n)
```

`p` = desired probability of finding a correct sample at least once, `w` = expected fraction
of "inlier" candidate points, `n = 2` (points needed per hypothesis). This directly replaces
`enumerate_anchor_correspondences`'s exhaustive `O(c²)` scan with a fixed-size random sample
`k`, independent of `c`, with early termination once a sufficiently good match is found.

**Tradeoff**: turns the expected-case cost from `O(c²)` to `O(k)`, but (a) worst case (no
real match exists in this candidate at all) still costs the same unless capped with a hard
iteration limit, accepting a chance of missing a real match; (b) doesn't amortize across the
full-bank loop's `N` templates -- each template still runs its own independent random search
against the candidate.

### Geometric hashing (Lamdan & Wolfson, 1988)[^gh]

Precompute an invariant-indexed hash table over a *model* once, so a query basis looks up
plausible correspondences in ~O(1) instead of scanning every candidate pair. Classically:
every ordered pair (or triple, for full affine invariance) of model points defines a local
coordinate basis; every other model point's coordinates in that basis are computed and
hashed; at query time, a query basis (here: the template's own 2 anchors) looks up the same
hash table directly.

**Why this fits especially well here**: the full-bank loop matches **many templates against
the same one candidate graph** -- building the candidate's own invariant index costs O(c²)
*once*, then every template's lookup is ~O(1) (or O(bucket size), typically small), replacing
`O(N · c²)` with `O(c² + N)`. This is the single biggest asymptotic win available for this
codebase's actual usage pattern (one candidate, many templates), unlike RANSAC which
re-samples per template.

**Tradeoff**: raw distance isn't a valid invariant under this pipeline's *similarity*
transform (scale varies), so the hash key must be built from a scale-invariant quantity
(e.g. a rounded distance-ratio to a reference point, or the existing degree-pair filter used
as a coarse bucket key) rather than distance directly -- more implementation care than RANSAC,
and bucket-width tuning affects both recall (too narrow, real matches missed) and precision
(too wide, buckets degenerate back toward the original O(c²)).

### Pose clustering / generalized Hough transform (Stockman, 1987)[^stockman]

Cast every tested correspondence's *implied* `(scale, rotation, translation)` into a coarse
voting histogram first -- cheap, since it only needs `fit_similarity_transform`'s O(1) output,
not the O(t log m) node-mapping + scoring step. Only bins that accumulate enough votes get
fully scored. This defers the expensive per-hypothesis work to a small fraction of
hypotheses without discarding any correspondence outright (unlike RANSAC's early-exit, every
hypothesis still casts a vote).

**Tradeoff**: still pays the full `O(c²)` cost to cast votes (cheaper per-vote than full
scoring, but not eliminated); requires tuning histogram bin widths for scale/rotation/
translation, and a real match near a bin boundary can under-count if binning is naive
(mitigated by soft-binning/interpolated voting, at added implementation cost).

### 4PCS (Aiger, Mitra, Cohen-Or, 2008)[^4pcs]

A RANSAC-family point-set registration algorithm keyed on ratio-invariant point *quadruples*
rather than pairs, solving the harder problem of registering two point sets with **no known
correspondence or topology** at all. Cited for completeness as the established technique
closest in spirit to "randomized congruent-set sampling," but this pipeline's own 2-anchor +
degree-filter approach is already a tighter fit for a *known-topology* graph -- 4PCS spends
effort recovering structure (which points even *might* correspond) that `split_at_
intersections`/`connect_nearby_points` + degree filtering already gives for free here.

### VF2 / VF2++ (Cordella et al., 2004[^vf2]; Jüttner & Madarási, 2018[^vf2pp])

The canonical (sub)graph isomorphism search: depth-first search over partial node
assignments, pruned by degree/adjacency-consistency at each step. NP-complete in general,
though VF2++ substantially improves practical performance via smarter match ordering and
degree-based pruning.

**Why it's not recommended here**: VF2-family algorithms reason over pure topology
(adjacency + degree), discarding the continuous geometric constraint (real point
coordinates) this pipeline already exploits almost for free once 2 anchors are fixed via
`fit_similarity_transform`. A geometry-aware search is strictly more informative than a
topology-only one for this problem, so replacing the anchor-based approach with generic
subgraph isomorphism would very likely be a step backward, not forward. Cited for
completeness as "the standard answer" to subgraph matching, not as a suggested substitution.

### Locality-sensitive hashing[^lsh]

Generalizes `map_template_nodes_to_candidate`'s nearest-neighbor step to sub-linear
approximate search in high-dimensional or very large point sets, trading exactness for
speed.

**Why it's not needed yet**: `cKDTree` already gives *exact* `O(log m)` queries in 2D, and
`m` (candidate node count) stays small at single-cluster scale. LSH would only become
relevant if candidate graphs grow much larger than one hand-picked cluster (e.g. whole-page
matching against thousands of nodes at once) -- noted here as a future option, not a current
need.

### Batched KD-tree queries (engineering, not literature)

Not a published technique -- a straightforward vectorization: stack every tested
hypothesis's transformed template points into one array and issue a single `cKDTree.query`
call instead of one call per hypothesis inside the correspondence loop, removing per-call
Python/NumPy overhead. Zero change to algorithmic complexity, but a real, low-risk wall-clock
win that composes with any of the above (it doesn't change *which* hypotheses get tested,
only how cheaply each one's node-mapping step runs).

## 3. Recommendation ranking

If a future round implements one of these, suggested order (by expected speedup × the risk
of getting the implementation subtly wrong):

1. **Geometric hashing** -- amortizes the dominant `O(c²)` cost across the full-bank loop's
   `N` templates (this codebase's actual usage pattern: one candidate, many templates), the
   single biggest asymptotic win available.
2. **Batched KD-tree queries** -- near-zero risk, stacks with anything else on this list,
   worth doing regardless of which (if any) of the others get adopted.
3. **RANSAC-style anchor-pair sampling** -- simple to reason about and implement, but only
   helps a single template × candidate pair at a time, not the amortized-across-`N` case
   geometric hashing already covers.
4. **Pose clustering** -- more implementation and tuning complexity (bin widths, soft
   binning) for a benefit that mostly overlaps with what geometric hashing already gets more
   directly.
5. **4PCS / VF2 / VF2++ / LSH** -- documented above as "known, but not a good fit for this
   specific known-topology, single-cluster-scale problem" rather than recommended.

[^hs92]: J. Hershberger, J. Snoyink, "Speeding Up the Douglas-Peucker Line-Simplification Algorithm," Proc. 5th Intl. Symp. on Spatial Data Handling, 1992.
[^ransac]: M. A. Fischler, R. C. Bolles, "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography," Communications of the ACM 24(6), 1981.
[^gh]: Y. Lamdan, H. J. Wolfson, "Geometric Hashing: A General and Efficient Model-Based Recognition Scheme," ICCV 1988.
[^stockman]: G. Stockman, "Object Recognition and Localization via Pose Clustering," Computer Vision, Graphics, and Image Processing 40(3), 1987.
[^4pcs]: D. Aiger, N. J. Mitra, D. Cohen-Or, "4-Points Congruent Sets for Robust Pairwise Surface Registration," ACM Transactions on Graphics (SIGGRAPH) 27(3), 2008.
[^vf2]: L. P. Cordella, P. Foggia, C. Sansone, M. Vento, "A (Sub)Graph Isomorphism Algorithm for Matching Large Graphs," IEEE Transactions on Pattern Analysis and Machine Intelligence 26(10), 2004.
[^vf2pp]: A. Jüttner, P. Madarási, "VF2++ -- An Improved Subgraph Isomorphism Algorithm," Discrete Applied Mathematics 242, 2018.
[^lsh]: Standard survey reference: A. Andoni, P. Indyk, "Near-Optimal Hashing Algorithms for Approximate Nearest Neighbor in High Dimensions," Communications of the ACM 51(1), 2008.
