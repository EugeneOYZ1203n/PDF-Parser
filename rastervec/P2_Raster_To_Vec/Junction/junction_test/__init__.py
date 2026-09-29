"""junction_test: the classical raster->vector tracer Junction runs per
component -- binarize -> skeleton + distance transform -> chains (barb
pruning) -> Douglas-Peucker -> regularize.

Originally a spike reproducing Dosch, Tombre, Ah-Soon, Masini, "A complete
system for the analysis of architectural drawings", IJDAR 3(2):102-116, 2000
(Sections 2-3); stripped down to the centerline core. The spike's text/
graphics separation, thick/thin split, junction repair, LSD/Hough
alternatives, Rosin-West + arc fitting, dashed-line / staircase / symbol
recognition, remainder extraction and junction classification were removed
(none reached Junction's output, or they dropped real geometry).

Imports nothing from `rastervec/`.
"""
