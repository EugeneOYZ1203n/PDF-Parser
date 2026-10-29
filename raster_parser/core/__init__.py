"""The pipeline orchestrator: native_read -> raster_vectorize (raster -> vector, pluggable)
and vector_classify (vector parsing, pluggable) -> output_assemble.

`pipeline.run_pipeline` runs one page; `registry` maps the `p2` / `p3` names to
backend modules; `interfaces` documents what a backend implements; `models.settings.RunSettings`
carries the run options every backend receives. `raster_parser.extract()` is the public
entry point built on top of this.
"""
