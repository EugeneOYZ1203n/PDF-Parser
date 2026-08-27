from __future__ import annotations

from rastervec.Evaluation.Labelling.label_schema import (
    LabelEntry,
    LabelSet,
    cluster_signature,
    load_labels,
    save_labels,
)
from rastervec.models import VectorPath


def _make_path(*, bbox=(0, 0, 1, 1)) -> VectorPath:
    return VectorPath(
        seq=0, item_index=0, kind="l", fill_rule="s",
        points=[(bbox[0], bbox[1]), (bbox[2], bbox[3])], bbox=bbox,
        stroke_color=(0, 0, 0), fill_color=None, stroke_opacity=None,
        fill_opacity=None, stroke_width=1.0, dashes=None, closed=False,
        layer=None, page_index=0,
    )


def test_cluster_signature_deterministic_for_same_members():
    cluster = [_make_path(bbox=(0, 0, 1, 1)), _make_path(bbox=(2, 2, 3, 3))]
    assert cluster_signature(cluster) == cluster_signature(list(cluster))


def test_cluster_signature_differs_for_different_bboxes():
    a = [_make_path(bbox=(0, 0, 1, 1))]
    b = [_make_path(bbox=(5, 5, 6, 6))]
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
