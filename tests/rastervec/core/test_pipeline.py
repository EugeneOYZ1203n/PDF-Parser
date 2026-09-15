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
    # extract_images always returns at least the whole-page render.
    assert verbose.extra["p2_debug"]["images_seen"] >= 1
    assert "vectors_in" in verbose.extra["p3_debug"]


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
