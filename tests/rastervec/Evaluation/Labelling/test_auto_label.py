from __future__ import annotations

from rastervec.Evaluation.Labelling.auto_label import auto_label_pdf


def test_auto_label_pdf_recovers_known_text(synthetic_pdf_factory, tmp_pdf_path):
    # A large page (not the usual 200x100 test default): Vector_Classification's
    # MAX_DIMENSION_FRACTION drop steps are tuned for real-size architectural
    # drawings and compare each group's own dimension against a fraction of
    # the page's smaller side -- on a tiny page, a whole converted word's
    # vector-path bbox can itself exceed that fraction and get dropped as
    # "oversized", never reaching text_candidates.
    doc = synthetic_pdf_factory(
        [{"width": 2000, "height": 1500, "texts": [{"point": (100, 200), "text": "Hello", "fontsize": 20}]}]
    )
    path = tmp_pdf_path(doc)

    labels = auto_label_pdf(path, 0)

    assert labels.pdf_path == path
    texts = {e.text for e in labels.entries}
    assert "Hello" in texts
    for entry in labels.entries:
        assert entry.source == "auto"
        assert entry.page_index == 0
