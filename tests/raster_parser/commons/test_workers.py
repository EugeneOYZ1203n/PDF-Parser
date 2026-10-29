"""`commons.workers`: the spawn process pool and per-worker thread pinning the v1 steps share."""

from __future__ import annotations

import os

from raster_parser.commons import workers


def test_pin_blas_threads_sets_unset_vars_only(monkeypatch):
    for name in workers.BLAS_THREAD_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    workers.pin_blas_threads()
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "4"


def test_pin_blas_threads_can_disable_mkldnn(monkeypatch):
    monkeypatch.delenv("FLAGS_use_mkldnn", raising=False)
    workers.pin_blas_threads()
    assert "FLAGS_use_mkldnn" not in os.environ
    workers.pin_blas_threads(paddle=True)
    assert os.environ["FLAGS_use_mkldnn"] == "0"


def _square(x):
    return x * x


def test_spawn_pool_runs_jobs_in_spawned_processes():
    with workers.spawn_pool(2) as pool:
        assert pool._mp_context.get_start_method() == "spawn"
        assert sorted(pool.map(_square, [1, 2, 3])) == [1, 4, 9]
