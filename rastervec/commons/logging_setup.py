"""Stdlib logging configuration shared by every pipeline stage."""
from __future__ import annotations

import logging

_ROOT_NAME = "rastervec"


def configure_logging(level: int | str = logging.INFO, *, stream=None) -> None:
    """Attach a single StreamHandler to the rastervec root logger.

    Idempotent: calling this more than once (e.g. once per test) does not
    stack duplicate handlers.
    """
    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
            )
        )
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(level)


def get_logger(stage: str) -> logging.Logger:
    return logging.getLogger(f"{_ROOT_NAME}.{stage}")
