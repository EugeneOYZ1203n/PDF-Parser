"""`commons.logging_setup`: the package logger and its per-bucket debug loggers."""

from __future__ import annotations

import logging


def test_per_bucket_debug_logs_are_opt_in():
    import raster_parser.commons.helpers.clustering  # noqa: F401
    import raster_parser.commons.helpers.layer_separation  # noqa: F401

    for name in ("raster_parser.clustering", "raster_parser.layer_separation"):
        assert logging.getLogger(name).getEffectiveLevel() == logging.INFO
