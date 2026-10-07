from __future__ import annotations

from rastervec.commons.models import Vector
from rastervec.core import registry
from rastervec.core.pipeline import run_pipeline


def _fake_p2(images, page, *, debug_out=None, on_debug_layer=None):
    if debug_out is not None:
        debug_out["images_seen"] = len(images)
    if on_debug_layer is not None:
        on_debug_layer("phase2", "fake", "#000000", b"%PDF-fake-p2")
    return [], []


def _fake_p3(vectors_p1, vectors_p2, page, *, debug_out=None, on_debug_layer=None, **_kwargs):
    fake_vector = Vector(
        type="s", items=[("l", (0.0, 0.0), (10.0, 10.0))], color=(0, 0, 0), fill=None,
        width=1.0, dashes=None, closePath=None, lineCap=0, lineJoin=0, even_odd=False,
        stroke_opacity=None, fill_opacity=None, layer=None, rect=(0.0, 0.0, 10.0, 10.0),
        scissor=None, seqno=0, blendmode=None, isolated=False, knockout=False, opacity=None,
        page_index=page.meta.index,
    )
    if debug_out is not None:
        debug_out["vectors_in"] = len(vectors_p1) + len(vectors_p2)
    if on_debug_layer is not None:
        on_debug_layer("phase3", "fake", "#ffffff", b"%PDF-fake-p3")
    return [fake_vector], []


def _synthetic_pdf_path(synthetic_pdf_factory, tmp_pdf_path) -> str:
    doc = synthetic_pdf_factory([{
        "width": 200, "height": 100,
        "texts": [{"point": (10, 20), "text": "hello"}],
    }])
    return tmp_pdf_path(doc)


def test_run_pipeline_wires_phase4_and_combines_output(
    synthetic_pdf_factory, tmp_pdf_path, monkeypatch,
):
    monkeypatch.setitem(registry.P2_REGISTRY, "FakeP2", _fake_p2)
    monkeypatch.setitem(registry.P3_REGISTRY, "FakeP3", _fake_p3)
    path = _synthetic_pdf_path(synthetic_pdf_factory, tmp_pdf_path)

    result = run_pipeline(path, 0, p2="FakeP2", p3="FakeP3", verbose=True)

    assert "phase1" in result.step_durations
    assert "phase2" in result.step_durations
    assert "phase3" in result.step_durations
    assert "phase4" in result.step_durations

    # Final texts = phase1 native text (the synthetic word) + p2 texts ([])
    # + p3 texts ([]); final vectors = p3's own output (organize_outputs's
    # combination contract).
    assert [t.text for t in result.texts] == ["hello"]
    assert len(result.vectors) == 1
    assert result.vectors[0].bbox == (0.0, 0.0, 10.0, 10.0)
    assert result.p2 == "FakeP2"
    assert result.p3 == "FakeP3"


def test_run_pipeline_forwards_debug_out_only_when_verbose(
    synthetic_pdf_factory, tmp_pdf_path, monkeypatch,
):
    monkeypatch.setitem(registry.P2_REGISTRY, "FakeP2", _fake_p2)
    monkeypatch.setitem(registry.P3_REGISTRY, "FakeP3", _fake_p3)
    path = _synthetic_pdf_path(synthetic_pdf_factory, tmp_pdf_path)

    quiet = run_pipeline(path, 0, p2="FakeP2", p3="FakeP3", verbose=False)
    assert quiet.extra == {}

    verbose = run_pipeline(path, 0, p2="FakeP2", p3="FakeP3", verbose=True)
    # The synthetic page has no embedded images, so Phase 2 sees none.
    assert verbose.extra["p2_debug"]["images_seen"] == 0
    assert "vectors_in" in verbose.extra["p3_debug"]


def test_run_pipeline_forwards_keep_debug_arrays_by_signature(
    synthetic_pdf_factory, tmp_pdf_path, monkeypatch,
):
    seen: dict = {}

    def p2_with_flag(images, page, *, keep_debug_arrays=True):
        seen["p2"] = keep_debug_arrays
        return [], []

    def p3_with_flag(vectors_p1, vectors_p2, page, *, keep_debug_arrays=True):
        seen["p3"] = keep_debug_arrays
        return [], []

    monkeypatch.setitem(registry.P2_REGISTRY, "FlagP2", p2_with_flag)
    monkeypatch.setitem(registry.P3_REGISTRY, "FlagP3", p3_with_flag)
    monkeypatch.setitem(registry.P2_REGISTRY, "FakeP2", _fake_p2)
    monkeypatch.setitem(registry.P3_REGISTRY, "FakeP3", _fake_p3)
    path = _synthetic_pdf_path(synthetic_pdf_factory, tmp_pdf_path)

    run_pipeline(path, 0, p2="FlagP2", p3="FlagP3", keep_debug_arrays=False)
    assert seen == {"p2": False, "p3": False}
    run_pipeline(path, 0, p2="FlagP2", p3="FlagP3")
    assert seen == {"p2": True, "p3": True}
    # A backend that doesn't declare it never receives it.
    run_pipeline(path, 0, p2="FakeP2", p3="FakeP3", keep_debug_arrays=False)


def test_run_pipeline_streams_on_debug_layer_independent_of_verbose(
    synthetic_pdf_factory, tmp_pdf_path, monkeypatch,
):
    monkeypatch.setitem(registry.P2_REGISTRY, "FakeP2", _fake_p2)
    monkeypatch.setitem(registry.P3_REGISTRY, "FakeP3", _fake_p3)
    path = _synthetic_pdf_path(synthetic_pdf_factory, tmp_pdf_path)

    seen: list[tuple] = []
    result = run_pipeline(
        path, 0, p2="FakeP2", p3="FakeP3", verbose=False,
        on_debug_layer=lambda *layer: seen.append(layer),
    )

    assert result.extra == {}  # verbose=False: no debug_out/extra stashing
    assert seen == [
        ("phase2", "fake", "#000000", b"%PDF-fake-p2"),
        ("phase3", "fake", "#ffffff", b"%PDF-fake-p3"),
    ]


def test_run_pipeline_collects_p3_substep_durations(
    synthetic_pdf_factory, tmp_pdf_path, monkeypatch,
):
    def _timed_p3(vectors_p1, vectors_p2, page, *, step_durations=None):
        step_durations["fake_step"] = 0.25
        return [], []

    monkeypatch.setitem(registry.P2_REGISTRY, "FakeP2", _fake_p2)
    monkeypatch.setitem(registry.P3_REGISTRY, "TimedP3", _timed_p3)
    monkeypatch.setitem(registry.P3_REGISTRY, "FakeP3", _fake_p3)
    path = _synthetic_pdf_path(synthetic_pdf_factory, tmp_pdf_path)

    timed = run_pipeline(path, 0, p2="FakeP2", p3="TimedP3")
    assert timed.substep_durations == {"fake_step": 0.25}
    assert set(timed.step_durations) == {"phase1", "phase2", "phase3", "phase4"}

    untimed = run_pipeline(path, 0, p2="FakeP2", p3="FakeP3")
    assert untimed.substep_durations == {}


def test_every_p3_backend_accepts_step_durations():
    import inspect

    for name, fn in registry.P3_REGISTRY.items():
        assert "step_durations" in inspect.signature(fn).parameters, name


def _broken_p2(images, page):
    raise ValueError("boom in p2")


def test_failed_p2_under_verbose_does_not_cascade_into_unbound_local(
    synthetic_pdf_factory, tmp_pdf_path, monkeypatch,
):
    monkeypatch.setitem(registry.P2_REGISTRY, "BrokenP2", _broken_p2)
    monkeypatch.setitem(registry.P3_REGISTRY, "FakeP3", _fake_p3)
    path = _synthetic_pdf_path(synthetic_pdf_factory, tmp_pdf_path)

    result = run_pipeline(path, 0, p2="BrokenP2", p3="FakeP3", verbose=True)

    outcomes = {o.name: o for o in result.step_outputs} if isinstance(result.step_outputs, list) \
        else result.step_outputs
    assert outcomes["phase2"].status == "error"
    assert "boom in p2" in outcomes["phase2"].error
    assert outcomes["phase3"].status == "ok"
    assert [t.text for t in result.texts] == ["hello"]
    assert len(result.vectors) == 1


def _p3_must_not_run(*_args, **_kwargs):
    raise AssertionError("P3 ran despite stop_after")


def test_stop_after_phase2_skips_p3_and_outputs_raw_vectors(
    synthetic_pdf_factory, tmp_pdf_path, monkeypatch,
):
    monkeypatch.setitem(registry.P2_REGISTRY, "FakeP2", _fake_p2)
    monkeypatch.setitem(registry.P3_REGISTRY, "NoP3", _p3_must_not_run)
    path = _synthetic_pdf_path(synthetic_pdf_factory, tmp_pdf_path)
    layers = []

    result = run_pipeline(
        path, 0, p2="FakeP2", p3="NoP3", verbose=True, stop_after="phase2",
        on_debug_layer=lambda *a: layers.append(a[0]),
    )

    assert set(result.step_durations) == {"phase1", "phase2", "phase4"}
    assert layers == ["phase2"]
    assert [t.text for t in result.texts] == ["hello"]
    assert len(result.vectors) == len(result.extra["phase1"].vectors)
    assert result.extra["p3_debug"] == {}


def test_stop_after_phase1_skips_p2_and_p3(synthetic_pdf_factory, tmp_pdf_path, monkeypatch):
    monkeypatch.setitem(registry.P2_REGISTRY, "NoP2", _p3_must_not_run)
    monkeypatch.setitem(registry.P3_REGISTRY, "NoP3", _p3_must_not_run)
    path = _synthetic_pdf_path(synthetic_pdf_factory, tmp_pdf_path)

    result = run_pipeline(path, 0, p2="NoP2", p3="NoP3", stop_after="phase1")

    assert set(result.step_durations) == {"phase1", "phase4"}
    assert [t.text for t in result.texts] == ["hello"]


def test_stop_after_rejects_unknown_phase(synthetic_pdf_factory, tmp_pdf_path):
    path = _synthetic_pdf_path(synthetic_pdf_factory, tmp_pdf_path)
    import pytest

    with pytest.raises(ValueError, match="stop_after"):
        run_pipeline(path, 0, stop_after="phase4")


def _p2_two_vectors(images, page):
    def _v(seqno, y):
        return Vector(
            type="s", items=[("l", (0.0, y), (10.0, y))], color=(1, 0, 0), fill=None,
            width=1.0, dashes=None, closePath=None, lineCap=0, lineJoin=0, even_odd=False,
            stroke_opacity=None, fill_opacity=None, layer=None, rect=(0.0, y, 10.0, y),
            scissor=None, seqno=seqno, blendmode=None, isolated=False, knockout=False,
            opacity=None, page_index=page.meta.index,
        )
    # P2 backends number from 0, colliding with Phase 1's own seqnos.
    return [_v(0, 50.0), _v(1, 60.0)], []


def _p3_passthrough(vectors_p1, vectors_p2, page, **_kwargs):
    # Deliberately P2-last, so the final order can only come from P4's sort.
    return list(vectors_p1) + list(vectors_p2), []


def test_run_pipeline_paints_p2_vectors_below_p1_vectors(
    synthetic_pdf_factory, tmp_pdf_path, monkeypatch,
):
    monkeypatch.setitem(registry.P2_REGISTRY, "P2Two", _p2_two_vectors)
    monkeypatch.setitem(registry.P3_REGISTRY, "P3Pass", _p3_passthrough)
    doc = synthetic_pdf_factory([{
        "width": 200, "height": 100,
        "drawings": [{"lines": [((5, 5), (100, 5)), ((5, 15), (100, 15))], "color": (0, 0, 0)}],
    }])
    path = tmp_pdf_path(doc)

    result = run_pipeline(path, 0, p2="P2Two", p3="P3Pass")

    p2_out = [v for v in result.vectors if v.color == (1, 0, 0)]
    p1_out = [v for v in result.vectors if v.color != (1, 0, 0)]
    assert len(p2_out) == 2 and p1_out
    # every P2 seqno sits below every P1 seqno, P2's own order kept ...
    assert max(v.seqno for v in p2_out) < min(v.seqno for v in p1_out)
    assert [v.bbox[1] for v in p2_out] == [50.0, 60.0]
    # ... so in paint order P2 comes first and P1 is drawn on top.
    assert result.vectors[:2] == p2_out
