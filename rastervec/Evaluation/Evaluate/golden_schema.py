"""Sidecar JSON schema for hand-curated golden/regression test cases.

Unlike `Evaluation/Labelling/label_schema.py`'s `LabelSet` (exhaustive
ground truth for one whole PDF, scored by the benchmark), a `GoldenCase`
is a single curated snapshot of one cluster's/segmentation's output at one
of three pipeline stages (`"classification"`, `"fast"`, `"word_split"`),
labelled by a human `"positive"` ("this region is text / the split is
right") or `"negative"` ("it isn't"). The label is the curator's verdict
and is assigned freely from the *combined* candidate pool -- a candidate
the pipeline currently keeps/passes can be labelled `"negative"` (a known
false positive) and one it drops labelled `"positive"` (a known miss).
What the pipeline actually decided is stored separately in `role` and is
what the regression check compares. Replaying a case later
(`Evaluation/Evaluate/golden_regression.py`) checks that a fresh pipeline
run still produces matching output for that same region, within a small
tolerance -- a regression guard against silent pipeline drift, not a
correctness oracle.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Stage = Literal["classification", "fast", "word_split"]
CaseLabel = Literal["positive", "negative"]
CaseRole = Literal["kept", "dropped", "passed"]


class GoldenCase(BaseModel):
    stage: Stage
    label: CaseLabel
    pdf_path: str
    page_index: int
    note: str = ""
    # Debug/provenance only -- label_schema.cluster_signature()'s member
    # count + rounded bbox. Replay matching itself is IoU-based (see
    # golden_regression.compare_case), not signature-based: an unrelated
    # member-count shift would make an exact-equality signature falsely
    # read as "cluster not found" even when the region is unchanged.
    signature: str
    # Page-space bbox of the captured cluster -- the IoU match anchor on
    # replay.
    bbox: tuple[float, float, float, float]
    # classification/fast only; None for word_split.
    role: CaseRole | None = None
    # word_split only: page-space bbox per word, in reading order.
    word_bboxes: list[tuple[float, float, float, float]] | None = None


class GoldenCaseBank(BaseModel):
    """Every curated case, across every stage and PDF -- one shared file
    (see golden_regression's storage-location note), since a bank only
    ever holds a handful of self-describing cases rather than an
    exhaustive per-PDF labelling."""

    cases: list[GoldenCase] = Field(default_factory=list)


def upsert_case(bank: GoldenCaseBank, case: GoldenCase) -> GoldenCaseBank:
    """Replace any existing case with the same (stage, label) key, else
    append -- lets the curation notebook be re-run/re-picked idempotently
    instead of accumulating duplicates. Caps a stage at one positive + one
    negative; use `add_case` to keep several of each."""
    kept = [c for c in bank.cases if not (c.stage == case.stage and c.label == case.label)]
    return GoldenCaseBank(cases=[*kept, case])


def add_case(bank: GoldenCaseBank, case: GoldenCase) -> GoldenCaseBank:
    """Append `case`, replacing only an existing case with the same
    (stage, label, signature) -- unlike `upsert_case` (one case per
    (stage, label)), a stage can hold several curated positives and
    negatives at once, while re-running a notebook pick stays idempotent."""
    key = (case.stage, case.label, case.signature)
    kept = [c for c in bank.cases if (c.stage, c.label, c.signature) != key]
    return GoldenCaseBank(cases=[*kept, case])


def drop_cases(bank: GoldenCaseBank, indices: Iterable[int]) -> GoldenCaseBank:
    """Return a bank without the cases at the given positions in
    `bank.cases` (as numbered by the curation notebook's view cell).
    Negative indices count from the end; out-of-range indices are ignored."""
    n = len(bank.cases)
    drop = {i + n if i < 0 else i for i in indices}
    return GoldenCaseBank(cases=[c for i, c in enumerate(bank.cases) if i not in drop])


def save_cases(bank: GoldenCaseBank, path: str) -> None:
    Path(path).write_text(bank.model_dump_json(indent=2), encoding="utf-8")


def load_cases(path: str) -> GoldenCaseBank:
    return GoldenCaseBank.model_validate_json(Path(path).read_text(encoding="utf-8"))
