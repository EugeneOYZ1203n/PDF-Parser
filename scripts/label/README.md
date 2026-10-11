# Labelling tools (`scripts/label/`)

Ground-truth labelling for benchmarking extraction, bundled per PDF into
`outputs/labels/<stem>_label/` (gitignored). Three kinds of label:

- **native** — derived automatically from the page's own native text (no human, no pipeline run);
- **vector** — human-typed text on the page's real extracted `Vector`s;
- **raster** — human-drawn text boxes / lines / curves over embedded raster images (content with
  no vector backing), plus auto line/curve geometry taken from the original vectors.

Each tool is a plain script — run it directly, not with `python -m`. Scripts import their shared
Tk helpers as `from _common import ...`: running a script puts its own folder first on
`sys.path`, so `scripts/label/` needs no `__init__.py`. The library code lives in the package:

| Module | Role |
|---|---|
| `raster_parser.evaluation.labelling.label_schema` | Sidecar JSON format: `LabelEntry` (text label: `label_id`, `cluster_bbox`, `text`, `source`, `expected_rotation`, `vector_signatures`), `GeometryAnnotation` (`"l"`/`"c"` line/curve with paint attrs, `source` `"auto"`/`"manual"`), `LabelSet`; `save_labels`/`load_labels`; `path_signature` re-matches a label's exact vectors across fresh extractions. |
| `raster_parser.evaluation.labelling.native_label` | `native_label_pdf` — one label per native text line, independent of any pipeline; `attach_vector_signatures`. |
| `raster_parser.evaluation.labelling.raster_label` | `raster_geometry_for_page` (auto geometry), `sync_text_from_vector_labels`, `embedded_images_for_page`. |
| `raster_parser.evaluation.conversion` | `convert_page_text_only` / `_drawings_only` / `_to_vector_text` — re-express native text as vector paths without touching existing drawing geometry. |
| `raster_parser.P1_Reading_Native` | `Reader`, `extract_native_text`, `extract_vectors`. |

All label coordinates are unrotated, CropBox-relative page space. The GUIs build the overlay
transform with `_common.get_display_matrix` (`rotation_matrix * zoom`, overlays only — the page
pixmap is rendered zoom-only because `get_pixmap()` already applies `/Rotate`).

## Windows / PowerShell path gotcha

Do **not** end a quoted path argument with a backslash:

```powershell
# BROKEN - the \" is read as an escaped quote, a literal " ends up in the path
python scripts/label/label_viewer.py '.\outputs\labels\My Drawing_label\'
# OK - no trailing separator, or forward slashes
python scripts/label/label_viewer.py '.\outputs\labels\My Drawing_label'
python scripts/label/label_viewer.py 'outputs/labels/My Drawing_label'
```

## `master_label.py` — the whole workflow over one PDF

Native step (auto text labels + `vectorised.pdf`) → raster geometry step (auto line/curve
labels from the original vectors + `rasterised.pdf`) → text re-sync (pulls `vector_label`'s
human text labels into `raster_labels.json`) → opens `vector_label` → re-sync → opens
`raster_label` → prints a summary. Re-running skips pages already done for the native /
raster-geometry steps (tracked in `manifest.json`); the interactive steps always reopen.

```bash
uv run python scripts/label/master_label.py "references/<stem>.pdf" --dpi 300
```

| arg | meaning |
|---|---|
| `pdf` | source PDF (required) |
| `--dpi N` | raster step render resolution (default 300) |

Produces `outputs/labels/<stem>_label/`: `original.pdf`, `vectorised.pdf`, `rasterised.pdf`,
`native_labels.json`, `vector_labels.json`, `raster_labels.json`, `manifest.json`.
Not unit-tested end to end (it chains three Tk event loops).

## `native_label.py` — auto text labels (no GUI)

`native_label_pdf` + a `convert_page_text_only` sidecar PDF + `attach_vector_signatures`,
merged onto any existing `--out` file by `label_id`.

```bash
uv run python scripts/label/native_label.py "references/<stem>.pdf" --page 0
```

| arg | meaning |
|---|---|
| `pdf` | input PDF (required) |
| `--page N` | 0-based page index |
| `--out PATH` | label JSON to load/save (default `outputs/labels/<stem>.json`) |

## `vector_label.py` — label vectors (GUI)

A flat pool of raw `Vector`s (no clustering). Click / ctrl-click / drag-select unlabelled
vectors + Apply to create a label; click an already-labelled vector to re-select its set and
edit it in place. One checkbox per `(layer, color, width)` bucket
(`_common.separate_by_layer_color_width`) filters visibility; "Hide labelled" hides labelled
vectors. Continuous rotation slider (2.5° snap) with a direction-arrow overlay. `<`/`>` (or
PageUp/PageDown) change page; labels save on every page change and on close.

```bash
uv run python scripts/label/vector_label.py "references/<stem>.pdf" --page 0
```

Same args as `native_label.py`.

## `raster_label.py` — label embedded images (GUI)

Scoped to content with **no vector backing** (scanned insets, stamps, photos): steps through
every embedded raster image in the PDF, cropped, with three tools — **Text** (drag a box),
**Line** (drag two points), **Curve** (click start, two controls, end; `Escape` cancels).
Right-click deletes. A PDF with no embedded images has nothing to show.

```bash
uv run python scripts/label/raster_label.py "references/<stem>.pdf"
```

| arg | meaning |
|---|---|
| `pdf` | input PDF (required) |
| `--out PATH` | label JSON to load/save (default `outputs/labels/<stem>.json`) |

## `view_native_labels.py` — native labels in the vector editor

Runs `native_label_pdf`, merges onto any existing `--out` file, then opens `vector_label`'s
window on it. Same args as `native_label.py`; `--out` defaults to
`outputs/labels/<stem>_p<N>_native_labels.json`.

## `label_viewer.py` — read-only overlay of a label folder

Overlays a `master_label.py` folder's three label sets on one page view, one toggle each:
Native (bbox), Vector (bbox + its backing `Vector`s), Raster geometry (dashed = auto, solid =
hand-traced), Raster text (dashed = synced from vector labels, solid = manual). No editing.

```bash
uv run python scripts/label/label_viewer.py "references/<stem>.pdf"   # or the <stem>_label folder
```

## Tests

`tests/scripts/label/` (helpers in `_common.py`) and
`tests/raster_parser/evaluation/` (the library modules above).
