"""Evaluation stage: the pipeline's actual intended final phase.

Consolidates one page's text/line/remainder outputs into a
`ReconstructedPage`, then rebuilds a PDF from all pages' outputs so the
extraction pipeline's result can be scored against the original -- this is
the pipeline's actual last step, distinct from `renderer.py` (which only
renders pixels, e.g. OCR input or debug-app overlays, and is never itself
a pipeline stage).

Not implemented yet -- the real benchmarking suite (Conversion/Labelling/
Evaluate, under this same Evaluation/ package) implements and exercises
this stage. `Evaluation.reconstruct_page`/`build_pdf` are the intended
entry points; they'll be filled in alongside that work rather than left as
placeholder NotImplementedError bodies here.
"""
from __future__ import annotations


class Evaluation:
    """Reconstructs pipeline output into a PDF for evaluation against the
    original. Not implemented yet -- see this module's docstring."""
