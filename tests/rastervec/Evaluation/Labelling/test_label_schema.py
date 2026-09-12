from __future__ import annotations

from rastervec.Evaluation.Labelling.label_schema import (
    GeometryAnnotation,
    LabelEntry,
    LabelSet,
    cluster_signature,
    load_labels,
    path_signature,
    save_labels,
)
def test_cluster_signature_deterministic_for_same_members(vector):
    cluster = [vector(bbox=(0, 0, 1, 1)), vector(bbox=(2, 2, 3, 3))]
    assert cluster_signature(cluster) == cluster_signature(list(cluster))


def test_cluster_signature_differs_for_different_bboxes(vector):
    a = [vector(bbox=(0, 0, 1, 1))]
    b = [vector(bbox=(5, 5, 6, 6))]
    assert cluster_signature(a) != cluster_signature(b)


def test_path_signature_stable_across_rebuilds(vector):
    a = vector(bbox=(1, 2, 3, 4), seqno=7, items=[("l", (1, 2), (3, 4))])
    b = vector(bbox=(1, 2, 3, 4), seqno=7, items=[("l", (1, 2), (3, 4))])
    assert path_signature(a) == path_signature(b)


def test_path_signature_differs_for_different_vectors(vector):
    a = vector(bbox=(1, 2, 3, 4), seqno=7)
    assert path_signature(a) != path_signature(vector(bbox=(1, 2, 3, 4), seqno=8))
    assert path_signature(a) != path_signature(vector(bbox=(5, 6, 7, 8), seqno=7))
    assert path_signature(a) != path_signature(vector(bbox=(1, 2, 3, 4), seqno=7, color=(1.0, 0.0, 0.0)))


def test_save_and_load_labels_round_trip(tmp_path):
    labels = LabelSet(
        pdf_path="foo.pdf",
        entries=[
            LabelEntry(
                page_index=0, cluster_bbox=(0, 0, 10, 10),
                cluster_signature="1:0.0:0.0:10.0:10.0", label_id="abc-123", text="Hello",
                source="vector", vector_signatures=["abc123", "def456"],
            )
        ],
        geometry_entries=[
            GeometryAnnotation(page_index=0, kind="l", points=[(0.0, 0.0), (1.0, 1.0)]),
            GeometryAnnotation(
                page_index=0, kind="c",
                points=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            ),
        ],
    )
    out_path = str(tmp_path / "labels.json")

    save_labels(labels, out_path)
    restored = load_labels(out_path)

    assert restored == labels


def test_label_source_accepts_native_vector_raster(tmp_path):
    for i, source in enumerate(("native", "vector", "raster")):
        entry = LabelEntry(
            page_index=0, cluster_bbox=(0, 0, 1, 1),
            cluster_signature=f"sig{i}", label_id=f"id{i}", text="x", source=source,
        )
        assert entry.source == source
