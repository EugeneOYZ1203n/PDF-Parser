from __future__ import annotations

import os

from rastervec.Reader.Parallel import pool as pool_mod
from rastervec.Reader.Parallel.pool import (
    _WORKER_ENV,
    _progress_postfix,
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


class _FakeCounter:
    def __init__(self, value: int = 0) -> None:
        self.value = value


def test_run_parallel_progress_counter_none_is_unaffected():
    # progress_counter=None (the default) must not change run_parallel's
    # own behavior at all.
    assert run_parallel([3, 1, 2], _square, workers=1, desc="") == [9, 1, 4]


def test_run_parallel_serial_with_progress_counter_still_returns_correct_results():
    counter = _FakeCounter()

    def _work(x: int) -> int:
        counter.value += 1
        return _square(x)

    result = run_parallel([3, 1, 2], _work, workers=1, desc="", progress_counter=counter)
    assert result == [9, 1, 4]
    assert counter.value == 3


def test_progress_postfix_noop_when_counter_none():
    class _FakePbar:
        def set_postfix_str(self, *a, **k):
            raise AssertionError("should not be called when counter is None")

    with _progress_postfix(_FakePbar(), None):
        pass


def test_progress_postfix_sets_final_counter_value_on_exit():
    import tqdm as tqdm_mod

    counter = _FakeCounter(value=5)
    with tqdm_mod.tqdm(total=1, disable=True) as pbar:
        with _progress_postfix(pbar, counter):
            pass
    assert pbar.postfix is not None
    assert "5" in pbar.postfix


def test_worker_init_pins_threads(monkeypatch):
    # keep the model warmup out of a pool-plumbing test
    monkeypatch.setattr(pool_mod, "warmup", lambda: None)
    for key in _WORKER_ENV:
        monkeypatch.delenv(key, raising=False)
    worker_init()
    for key, value in _WORKER_ENV.items():
        assert os.environ[key] == value


def test_worker_init_does_not_override_explicit(monkeypatch):
    monkeypatch.setattr(pool_mod, "warmup", lambda: None)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    worker_init()
    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_worker_init_warms_model_caches(monkeypatch):
    calls = []
    monkeypatch.setattr(pool_mod, "warmup", lambda: calls.append(1))
    worker_init()
    assert calls == [1]


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
