"""The pipeline orchestrator: Phase 1 (native read) -> Phase 2 (raster -> vector, pluggable)
and Phase 3 (vector parsing, pluggable) -> Phase 4 (output organisation).

`pipeline.run_pipeline` runs one page; `registry` maps the `p2` / `p3` names to
backend modules; `interfaces` documents what a backend implements; `settings.RunSettings`
carries the run options every backend receives. `raster_parser.extract()` is the public
entry point built on top of this.
"""
