# Labelling (`scripts/label/`)

A label file holds one thing: **original vector text**, i.e. text drawn as vector paths in the
source PDF, with its true reading typed in by a human. Every other kind of ground truth is
derived when the benchmark runs:

- **native text**: from the PDF's own text layer (`evaluation/labelling/native_label.py`);
- **raster ground truth**: from these labels and the page's vectors.

So there is one tool, `manual_vector_text_label.py`. Run it directly, not with `python -m`. It
imports its Tk helpers as `from _common import ...`: running a script puts its own folder first
on `sys.path`, so `scripts/label/` needs no `__init__.py`.

| Module | Role |
|---|---|
| `raster_parser.evaluation.labelling.label_schema` | The JSON format. `LabelSet` (`schema_version`, `pdf_path`, `entries`) of `LabelEntry` (`page_index`, `label_id`, `text`, `bbox`, `expected_rotation`, `vector_signatures`, `source`). `LabelEntry.from_text` / `to_text` convert to and from a `Text`. `path_signature` matches a label's vectors across fresh extractions. `load_labels` also reads version 1 files. |
| `raster_parser.evaluation.labelling.seed_labels` | Draft labels from `v2_vector_class`: `seed_page`, `seed_entries`; `prepare_page` (a page's vectors + drafts) is what the tool's worker runs. |
| `raster_parser.evaluation.labelling.native_label` | `native_label_pdf`: one native-text label per line, made by the benchmark and never saved. |

All label coordinates are in unrotated, CropBox-relative page space. The tool builds the overlay
transform with `_common.get_display_matrix`: `rotation_matrix * zoom`, applied to overlays only,
because `get_pixmap()` already applies `/Rotate` to the page image.

## `manual_vector_text_label.py`

```bash
uv run python scripts/label/manual_vector_text_label.py "references/<stem>.pdf" --page 0
```

| Arg | Meaning |
|---|---|
| `pdf` | input PDF (required) |
| `--page N` | 0-based page to open on |
| `--out PATH` | label JSON to load and save (default `outputs/labels/<stem>.json`) |
| `--no-seed` | don't pre-fill unlabelled pages |

**Background processing.** The page you open shows at once. One worker process then prepares
every page (`seed_labels.prepare_page`: its vectors and, unless `--no-seed`, its draft labels),
the open page first and then outwards. `<` / `>`, PageUp / PageDown and the page box only go to a
page that is ready; the bar counts the pages ready.

**Draft labels.** The first time a page with no labels is shown, it gets the draft labels
`v2_vector_class` read from it, each tied to the vectors it was read from. Fix the text and
rotation, delete wrong drafts, and add labels for the text it missed. Drafts are saved like any
other label, so review a page before moving on.

**Label list.** The side panel lists every label in the file: page, text, rotation and number of
vectors. Select a row (or click one of a label's vectors) to select that label:
- its vectors turn orange and the label bar shows its text and rotation;
- only the `(layer, color, width)` buckets its vectors are in stay visible ("Show all" or
  Escape brings the others back);
- clicking an unlabelled vector adds it to the label, clicking one of its own vectors removes it,
  and a drag adds (or, over only its own vectors, removes) many at once -- the list updates
  straight away. A label always keeps at least one vector; Delete removes the label;
- Apply (or Enter) saves the text and rotation.

A row on another page takes you there once that page is ready. Ctrl- or shift-click rows to
select several labels at once; Delete (the button, or the Delete key in the list) removes them all
after asking to confirm.

**New labels.** With no label selected, click, ctrl-click or drag over unlabelled vectors, type
the text, set the rotation, then Apply.

**Display.** Every raw `Vector` on the page is shown and can be clicked; nothing is clustered.
One checkbox per `(layer, color, width)` bucket (`_common.separate_by_layer_color_width`)
filters what is shown, and "Hide labelled" hides vectors that belong to another label. The
rotation slider snaps to 2.5° and draws a direction arrow on the page. Labels are saved on every
page change and on close.

## Using the labels

List the label JSON in a benchmark config's `input_files`
(`scripts/report_configs/README.md`). Its `pdf_path` names the PDF. The benchmark scores the
run's text against these labels as `original_vector`, and against labels it derives from the
PDF's native text as `native_to_vector`.

## Windows / PowerShell path gotcha

Do **not** end a quoted path argument with a backslash: `'.\outputs\labels\'` passes a literal
`"`. Leave the trailing separator off, or use forward slashes.
