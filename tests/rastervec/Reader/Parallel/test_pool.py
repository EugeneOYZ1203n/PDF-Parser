from __future__ import annotations

import os

from rastervec.Reader.Parallel.pool import (
    _WORKER_ENV,
    default_worker_count,
    run_parallel,
    worker_init,
)


def _square(x: int) -> int:
    return x * x


def test_run_parallel_serial_preserves_order():
    assert run_parallel([3, 1, 2], _square, workers=1, desc="") == [9, 1, 4]


def test_run_parallel_single_item_stays_serial():
    # workers>1 but one item -> serial branch, no pool / no warmup
    assert run_parallel([5], _square, workers=4) == [25]


def test_worker_init_pins_threads(monkeypatch):
    for key in _WORKER_ENV:
        monkeypatch.delenv(key, raising=False)
    worker_init()
    for key, value in _WORKER_ENV.items():
        assert os.environ[key] == value


def test_worker_init_does_not_override_explicit(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    worker_init()
    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_default_worker_count_is_sane():
    n = default_worker_count()
    assert 1 <= n <= 4
