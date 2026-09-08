from __future__ import annotations

from rastervec.Evaluation.Evaluate.golden_schema import (
    GoldenCase,
    GoldenCaseBank,
    add_case,
    drop_cases,
    load_cases,
    save_cases,
    upsert_case,
)


def _case(stage="classification", label="positive", **overrides) -> GoldenCase:
    fields = dict(
        stage=stage, label=label, pdf_path="a.pdf", page_index=0,
        signature="1:0.0:0.0:1.0:1.0", bbox=(0.0, 0.0, 1.0, 1.0), role="kept",
    )
    fields.update(overrides)
    return GoldenCase(**fields)


def test_field_defaults():
    case = GoldenCase(
        stage="word_split", label="negative", pdf_path="a.pdf", page_index=2,
        signature="sig", bbox=(0.0, 0.0, 1.0, 1.0),
    )
    assert case.note == ""
    assert case.role is None
    assert case.word_bboxes is None


def test_save_and_load_round_trip(tmp_path):
    bank = GoldenCaseBank(cases=[
        _case(stage="classification", label="positive", role="kept"),
        _case(stage="fast", label="negative", role="dropped"),
        _case(
            stage="word_split", label="positive", role=None,
            word_bboxes=[(0.0, 0.0, 1.0, 1.0), (2.0, 0.0, 3.0, 1.0)],
        ),
    ])
    path = str(tmp_path / "cases.json")
    save_cases(bank, path)
    loaded = load_cases(path)
    assert loaded == bank


def test_upsert_case_replaces_same_stage_label():
    bank = GoldenCaseBank(cases=[_case(stage="fast", label="positive", note="first")])
    updated = upsert_case(bank, _case(stage="fast", label="positive", note="second"))
    assert len(updated.cases) == 1
    assert updated.cases[0].note == "second"


def test_upsert_case_appends_new_stage_label():
    bank = GoldenCaseBank(cases=[_case(stage="fast", label="positive")])
    updated = upsert_case(bank, _case(stage="fast", label="negative", role="dropped"))
    assert len(updated.cases) == 2
    assert {c.label for c in updated.cases} == {"positive", "negative"}


def test_add_case_keeps_several_per_stage_label_but_dedupes_by_signature():
    bank = GoldenCaseBank(cases=[_case(stage="fast", label="positive", signature="a", note="first")])
    bank = add_case(bank, _case(stage="fast", label="positive", signature="b", note="second"))
    assert len(bank.cases) == 2

    bank = add_case(bank, _case(stage="fast", label="positive", signature="a", note="replaced"))
    assert len(bank.cases) == 2
    assert {(c.signature, c.note) for c in bank.cases} == {("a", "replaced"), ("b", "second")}


def test_add_case_labels_kept_candidate_negative():
    bank = add_case(GoldenCaseBank(), _case(label="negative", role="kept"))
    assert bank.cases[0].label == "negative"
    assert bank.cases[0].role == "kept"


def test_drop_cases_removes_by_index_and_ignores_out_of_range():
    bank = GoldenCaseBank(cases=[_case(signature=str(i)) for i in range(4)])
    dropped = drop_cases(bank, [1, 3, 99])
    assert [c.signature for c in dropped.cases] == ["0", "2"]


def test_drop_cases_supports_negative_index():
    bank = GoldenCaseBank(cases=[_case(signature=str(i)) for i in range(3)])
    dropped = drop_cases(bank, [-1])
    assert [c.signature for c in dropped.cases] == ["0", "1"]
