"""Stdlib logging configuration shared by every pipeline stage.

The `raster_parser` logger defaults to DEBUG (set on import, unless a level was already set), so
the stage progress lines and bars (`commons/progress.py`) show by default. A host app silences
them with `logging.getLogger("raster_parser").setLevel(logging.INFO)`.
"""

from __future__ import annotations

import logging

_ROOT_NAME = "raster_parser"

if logging.getLogger(_ROOT_NAME).level == logging.NOTSET:
    logging.getLogger(_ROOT_NAME).setLevel(logging.DEBUG)


class _TqdmHandler(logging.StreamHandler):
    """A StreamHandler that writes through `tqdm.write`, so records don't break an open bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm

            tqdm.write(self.format(record), file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)


def configure_logging(level: int | str = logging.DEBUG, *, stream=None) -> None:
    """Attach a single handler to the raster_parser root logger.

    Idempotent: calling this more than once (e.g. once per test) does not
    stack duplicate handlers.
    """
    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)

    if not logger.handlers:
        handler = _TqdmHandler(stream)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(level)


def get_logger(stage: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT_NAME}.{stage}")
