from __future__ import annotations

from types import SimpleNamespace

from rastervec.Evaluation.Evaluate.legacy_adapter import to_texts


def _fake_word(*, word="Hello", bbox=(0.0, 0.0, 10.0, 5.0), rotate=0):
    x0, y0, x1, y1 = bbox
    return SimpleNamespace(x0=x0, y0=y0, x1=x1, y1=y1, word=word, rotate=rotate)


def _fake_elements(words=None, vectors=None):
    return SimpleNamespace(words=words or [], vectors=vectors or [])


def test_to_texts_wraps_each_word():
    elements = _fake_elements(
        words=[
            _fake_word(word="Hello", bbox=(0.0, 0.0, 10.0, 5.0), rotate=0),
            _fake_word(word="World", bbox=(20.0, 20.0, 30.0, 25.0), rotate=90),
        ],
    )

    results = to_texts(elements, page_index=2)

    assert len(results) == 2
    assert results[0].text == "Hello"
    assert results[0].bbox == (0.0, 0.0, 10.0, 5.0)
    assert round(results[0].angle()) % 360 == 0
    assert results[0].page_index == 2
    assert results[0].source == "ocr"
    assert results[1].text == "World"
    assert round(results[1].angle()) % 360 == 90


def test_to_texts_empty():
    assert to_texts(_fake_elements()) == []
