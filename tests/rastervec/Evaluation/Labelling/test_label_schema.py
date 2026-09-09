from __future__ import annotations

from rastervec.Evaluation.Labelling.label_schema import (
    LabelEntry,
    LabelSet,
    cluster_signature,
    load_labels,
    save_labels,
)
def test_cluster_signature_deterministic_for_same_members(vector):
    cluster = [vector(bbox=(0, 0, 1, 1)), vector(bbox=(2, 2, 3, 3))]
    assert cluster_signature(cluster) == cluster_signature(list(cluster))


def test_cluster_signature_differs_for_different_bboxes(vector):
    a = [vector(bbox=(0, 0, 1, 1))]
    b = [vector(bbox=(5, 5, 6, 6))]
    assert cluster_signature(a) != cluster_signature(b)


def test_save_and_load_labels_round_trip(tmp_path):
    labels = LabelSet(
        pdf_path="foo.pdf",
        entries=[
            LabelEntry(
                page_index=0, cluster_bbox=(0, 0, 10, 10),
                cluster_signature="1:0.0:0.0:10.0:10.0", text="Hello",
                source="manual",
            )
        ],
    )
    out_path = str(tmp_path / "labels.json")

    save_labels(labels, out_path)
    restored = load_labels(out_path)

    assert restored == labels
