"""The report / benchmark input forms and which ground truth each is scored
on -- shared by `scripts/generate_pipeline_report.py` (which overlays to
draw per form) and the benchmark scorer (`Evaluate/benchmark_run_loading.py`).

| form | pipeline input | text types scored | vector GT (vs Phase 2's vectors) |
|---|---|---|---|
| original | the source page as-is | native text, original_vector, original_raster | original_raster |
| vectorised | `conversion.convert_page_to_vector_text` | native_to_vector, original_vector | -- |
| rasterised | the label folder's `rasterised.pdf`, else `conversion.rasterise_page` | vector_to_raster, original_raster, native_to_raster | vector_to_raster + original_raster |

On the original page the native words are read straight from the PDF (P1),
not OCR'd, so that row is reported as "native text" rather than
`native_to_vector` -- and every text the run outputs (native + OCR) is a
prediction there, while the other forms score their OCR output only.
`vector_to_raster` / `native_to_raster` are never scored on the original:
those lines and words exist there as real PDF vectors / text."""
from __future__ import annotations

FORMS: tuple[str, ...] = ("original", "vectorised", "rasterised")

FORM_TITLES: dict[str, str] = {
    "original": "Original PDF", "vectorised": "Vectorised", "rasterised": "Rasterised",
}

FORM_TEXT_TYPES: dict[str, tuple[str, ...]] = {
    "original": ("native_to_vector", "original_vector", "original_raster"),
    "vectorised": ("native_to_vector", "original_vector"),
    "rasterised": ("vector_to_raster", "original_raster", "native_to_raster"),
}

# GT geometry types each form's Phase 2 vectors are scored against.
FORM_VECTOR_GT: dict[str, tuple[str, ...]] = {
    "original": ("original_raster",),
    "vectorised": (),
    "rasterised": ("vector_to_raster", "original_raster"),
}


def all_texts_are_predictions(form: str) -> bool:
    """Original: native + OCR texts; other forms: OCR output only."""
    return form == "original"


def type_label(form: str, text_type: str) -> str:
    """Display name of a text type on a form."""
    if form == "original" and text_type == "native_to_vector":
        return "native text"
    return text_type
