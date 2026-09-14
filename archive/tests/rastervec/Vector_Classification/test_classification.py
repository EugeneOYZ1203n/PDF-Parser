"""Conservation property for `classify_vectors`: every input `Vector` must
end up in exactly one of `(flattened text_clusters, drawing_vectors)` --
nothing dropped silently, nothing duplicated. Checked by `id()` identity
(not `==`) since `Vector` is a plain dataclass and two distinct instances
could compare equal.

Every filter-step-level unit test that used to live here targeted the
deleted `VectorPath`/`id()`-keyed-lineage internals; the fixed 12-step
chain itself is exercised end-to-end here instead, and each individual
filter function has its own direct unit tests in `cluster_filters.py`/
`group_filters.py`/`item_filters.py`'s own test modules.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from rastervec.pipelines.sub_pipelines.vector_classification import classify_vectors
from rastervec.Reader.reader import Reader
from rastervec.Vector.vector import extract_vectors

REFERENCES_DIR = Path(__file__).resolve().parents[2] / "references"
REFERENCE_PDFS = sorted(REFERENCES_DIR.glob("test_pdfs_*.pdf"))


class _Meta:
    width = 200
    height = 200


class _Page:
    meta = _Meta()


def _flatten_cluster(cluster) -> list:
    return [v for group in cluster for v in group]


def _assert_conserved(vectors: list, page) -> None:
    result = classify_vectors(vectors, page)

    text_vectors = [v for cluster in result.text_clusters for v in _flatten_cluster(cluster)]
    drawing_vectors = result.drawing_vectors

    text_ids = [id(v) for v in text_vectors]
    drawing_ids = [id(v) for v in drawing_vectors]

    assert len(set(text_ids)) == len(text_ids), "a Vector appears more than once across text_clusters"
    assert len(set(drawing_ids)) == len(drawing_ids), "a Vector appears more than once in drawing_vectors"
    assert not (set(text_ids) & set(drawing_ids)), "a Vector was kept AND dropped"

    combined_ids = set(text_ids) | set(drawing_ids)
    input_ids = {id(v) for v in vectors}
    assert combined_ids == input_ids, (
        f"{len(input_ids - combined_ids)} input vector(s) lost, "
        f"{len(combined_ids - input_ids)} unexplained extra vector(s)"
    )
    assert len(text_vectors) + len(drawing_vectors) == len(vectors)


def test_conserves_a_single_small_vector(vector):
    lone = vector(kind="l", bbox=(0, 0, 3, 6), color=(0, 0, 0))
    _assert_conserved([lone], _Page())


def test_conserves_an_oversized_vector_dropped_at_step_1(vector):
    # max dimension 40 = 20% of the 200x200 page's smaller side -- exceeds
    # MAX_DIMENSION_FRACTION (10%), dropped at the very first filter step.
    oversized = vector(kind="l", bbox=(0, 0, 40, 40), color=(0, 0, 0))
    small = vector(kind="l", bbox=(150, 150, 152, 152), color=(0, 0, 0), seqno=50)
    _assert_conserved([oversized, small], _Page())


def test_conserves_a_duplicate_run_dropped_at_seq_dedupe(vector):
    duplicates = [
        vector(kind="re", bbox=(i * 20, 0, i * 20 + 3, 6), fill=(0, 0, 0), seqno=i)
        for i in range(5)
    ]
    other = vector(kind="l", bbox=(150, 150, 158, 158), color=(0, 0, 0), seqno=50)
    _assert_conserved(duplicates + [other], _Page())


def test_conserves_a_perimeter_only_cluster(vector):
    ring = [
        vector(kind="l", bbox=(0, 0, 20, 1), color=(0, 0, 0), seqno=0),
        vector(kind="l", bbox=(0, 19, 20, 20), color=(0, 0, 0), seqno=1),
        vector(kind="l", bbox=(0, 0, 1, 20), color=(0, 0, 0), seqno=2),
        vector(kind="l", bbox=(19, 0, 20, 20), color=(0, 0, 0), seqno=3),
    ]
    centered = vector(kind="l", bbox=(150, 150, 154, 154), color=(0, 0, 0), seqno=50)
    _assert_conserved(ring + [centered], _Page())


def test_conserves_a_realistic_mixed_population(vector):
    # A grab-bag meant to touch several filter steps at once: an oversized
    # frame, a run of identical tick marks, a sparse/perimeter ring, and a
    # handful of varied small shapes that should survive as text
    # candidates -- the conservation property must hold regardless of which
    # steps actually drop what.
    frame = vector(kind="l", bbox=(0, 0, 45, 1), color=(0, 0, 0), seqno=0)
    ticks = [
        vector(kind="re", bbox=(i * 3, 50, i * 3 + 1, 52), fill=(0, 0, 0), seqno=10 + i)
        for i in range(6)
    ]
    varied = [
        vector(kind="l", bbox=(100 + i * 5, 100, 100 + i * 5 + 3, 106), color=(0, 0, 0), seqno=20 + i)
        for i in range(4)
    ] + [
        vector(kind="re", bbox=(100 + i * 5, 110, 100 + i * 5 + 3, 116), fill=(0, 0, 0), seqno=30 + i)
        for i in range(4)
    ]

    _assert_conserved([frame] + ticks + varied, _Page())


@pytest.mark.skipif(not REFERENCE_PDFS, reason="tests/references/test_pdfs_*.pdf not generated")
@pytest.mark.parametrize("pdf_path", REFERENCE_PDFS, ids=lambda p: p.stem)
def test_conserves_vectors_extracted_from_reference_pdf(pdf_path):
    with Reader(str(pdf_path)) as reader:
        page = reader.get_page(0)
        vectors = extract_vectors(page)
        _assert_conserved(vectors, page)
