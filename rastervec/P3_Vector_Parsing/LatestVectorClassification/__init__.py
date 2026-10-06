"""LatestVectorClassification P3 backend -- the merged VectorClassification +
CollinearVectorClass pipeline: layer/color/width classification with
collinear-drawing, length-outlier and crossed-grid removal, then PaddleOCR
detect -> quad-angle upright crop -> 0/180 classifier + recognise ->
low-score retry -> quad ink-ownership text/drawing split (see parse.py)."""
