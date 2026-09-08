from __future__ import annotations

import os

from rastervec.Reader.Parallel import pool as pool_mod
from rastervec.Reader.Parallel.pool import (
    _WORKER_ENV,
    compute_pool,
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


def test_compute_pool_zero_yields_none():
    with compute_pool(0) as compute:
        assert compute is None


def test_compute_pool_builds_and_tears_down_a_usable_proxy(monkeypatch):
    # keep the model warmup out of a pool-plumbing test
    monkeypatch.setattr(pool_mod, "warmup", lambda: None)
    with compute_pool(1) as compute:
        assert compute is not None
        assert hasattr(compute, "starmap") and hasattr(compute, "apply")
        assert compute.apply(abs, (-3,)) == 3
        assert compute.starmap(pow, [(2, 3), (3, 2)]) == [8, 9]
