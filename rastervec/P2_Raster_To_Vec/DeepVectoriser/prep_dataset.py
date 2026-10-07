"""Build DeepVectoriser's training set from PDFs -- run once, then `train.py`.

    python -m rastervec.P2_Raster_To_Vec.DeepVectoriser.prep_dataset \\
        --pdf-dir PDFS/ [--pdf C.pdf ...] [--pages 0,2] [--dpi 300] [--val-frac 0.1] \\
        [--workers 4] --out data/deepvec

`--pdf-dir` takes every `*.pdf` under the directory (recursive); `--pdf` adds
single files. Per page:

  render   page -> RGB at --dpi (`get_pixmap`, same as master_label.py's
           rasterised.pdf)
  gt       `raster_geometry_for_page` -> strokes in px (rotation-aware),
           chained per drawing (train_data.chain_annotations)
  colors   DBSCAN color layers (copied Junction color separation)
  layers   each GT stroke -> every color layer it lies on (pixel vote); per
           ink layer with GT strokes, the layer mask (masks.layer_image,
           resampled to the canonical 300 dpi) cropped to its content, as a
           lossless PNG, + its strokes are saved (train_data.save_layer); a
           layer with no strokes is listed in the manifest but not saved

No text cleanup: the P2 backend erases OCR'd text before the model runs, but
prep does no OCR/erase (and imports no OCR code), so text glyph strokes stay
in the ground truth.

`--workers N` (default 1) prepares N pages at once in separate processes.
Memory is per worker: roughly 0.7-1 GB for an A1 sheet at 300 dpi (RGB render
+ layer labels + one layer mask).

Resumable: a page whose manifest (`<out>/pages/<key>.json`) exists is
skipped. A failing page is logged (pdf, page, stage, traceback -> console +
`<out>/prep_log.txt`) and the run continues; a summary prints at the end.
`index.json` is rebuilt from every page manifest at the end of each run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from rastervec.P2_Raster_To_Vec.DeepVectoriser import train_data as td
from rastervec.P2_Raster_To_Vec.DeepVectoriser.color_separation import separate_colors
from rastervec.P2_Raster_To_Vec.DeepVectoriser.config import TARGET_PX_PER_PT
from rastervec.P2_Raster_To_Vec.DeepVectoriser.masks import layer_image

_LOG = logging.getLogger("rastervec.P2.DeepVectoriser.prep")


class _StageError(RuntimeError):
    def __init__(self, stage: str, exc: BaseException) -> None:
        super().__init__(f"{stage}: {type(exc).__name__}: {exc}")
        self.stage = stage


def _say(msg: str, log_file) -> None:
    tqdm.write(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def page_key(pdf: Path, page_index: int, root: Path | None = None) -> str:
    """`<stem>_p<N>`; under `root` the relative path (`sub__name_p<N>`), so
    same-named PDFs in different subfolders never collide."""
    name = pdf.stem
    if root is not None:
        name = "__".join(pdf.relative_to(root).with_suffix("").parts)
    return f"{name}_p{page_index}"


def prep_page(pdf: Path, page_index: int, out: Path, dpi: float, *, key: str | None = None,
              log=lambda _m: None) -> dict:
    """Process one page; returns its manifest (also written to disk)."""
    key = key or page_key(pdf, page_index)
    steps: list[str] = []
    stage = "render+gt"
    t0 = time.perf_counter()
    try:
        rgb, strokes, rotation = td.page_ground_truth(str(pdf), page_index, dpi)
        steps.append(f"render {rgb.shape[1]}x{rgb.shape[0]}")
        steps.append(f"gt {len(strokes)} strokes")
        log(f"[{key}] " + " -> ".join(steps))

        stage = "colors"
        layers = separate_colors(rgb)
        del rgb
        steps.append(f"colors {len(layers.ink_layers())} ink layer(s)")
        if layers.n_layers == 0:
            raise RuntimeError("no color layers (blank page?)")

        stage = "layers"
        scale = TARGET_PX_PER_PT / (dpi / 72.0)
        ink_layers = layers.ink_layers()
        owner = td.assign_layers(strokes, layers.labels, ink_layers)
        saved = []
        for layer in ink_layers:
            mine = [s for s, o in zip(strokes, owner) if layer in o]
            lkey = f"{key}__L{layer}"
            entry = {"key": lkey, "n_strokes": len(mine),
                     "color": [int(c) for c in layers.centroids_rgb[layer]]}
            if not mine:  # never trained on (write_index skips it) -- don't store its mask
                saved.append({**entry, "saved": False})
                continue
            gray = layer_image(layers.labels, layer, scale)
            stored = td.save_layer(out / "layers", lkey, gray, td.scale_strokes(mine, scale),
                                   layers.centroids_rgb[layer],
                                   seed=int(hashlib.md5(lkey.encode()).hexdigest()[:8], 16))
            saved.append({**entry, "saved": True, "page_shape": list(gray.shape), **stored})
            del gray
        n_saved = sum(1 for s in saved if s["saved"])
        steps.append(f"saved {n_saved} layer(s) ({len(saved) - n_saved} empty skipped), "
                     f"{sum(s['n_strokes'] for s in saved)} strokes")
    except Exception as exc:  # noqa: BLE001 -- re-raised with the stage attached
        raise _StageError(stage, exc) from exc

    manifest = {
        "pdf": str(pdf), "page": page_index, "key": key, "dpi": dpi, "rotation": rotation,
        "px_per_pt": TARGET_PX_PER_PT, "layers": saved,
        "seconds": round(time.perf_counter() - t0, 2),
    }
    (out / "pages").mkdir(parents=True, exist_ok=True)
    (out / "pages" / f"{key}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"[{key}] " + " -> ".join(steps) + f" ({manifest['seconds']}s)")
    return manifest


def _init_worker() -> None:
    """Pool initializer: one BLAS/OMP thread per worker (N workers x all
    cores each would oversubscribe the CPU)."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _prep_job(pdf: Path, page_index: int, out: Path, dpi: float, key: str) -> tuple:
    """One page in a worker process. Returns plain data -- `("ok", key,
    log_lines)` or `("fail", key, stage, message, traceback, log_lines)` --
    since `_StageError` doesn't survive pickling; the main process does all
    the logging."""
    lines: list[str] = []
    try:
        prep_page(pdf, page_index, out, dpi, key=key, log=lines.append)
        return ("ok", key, lines)
    except _StageError as exc:
        return ("fail", key, exc.stage, str(exc), traceback.format_exc(), lines)


def _is_val(page_key_: str, val_frac: float) -> bool:
    h = int(hashlib.md5(page_key_.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < val_frac


def write_index(out: Path, val_frac: float) -> dict:
    """Rebuild index.json from every page manifest; split by page (hashed,
    deterministic). With >= 2 pages at least one is held out for val."""
    manifests = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "pages").glob("*.json"))]
    val_pages = {m["key"] for m in manifests if _is_val(m["key"], val_frac)}
    if not val_pages and len(manifests) >= 2 and val_frac > 0:
        val_pages = {sorted(m["key"] for m in manifests)[-1]}
    if len(val_pages) == len(manifests) and manifests:
        val_pages.discard(sorted(val_pages)[0])
    train, val = [], []
    for m in manifests:
        for layer in m["layers"]:
            if layer["n_strokes"] == 0:
                continue
            (val if m["key"] in val_pages else train).append(layer["key"])
    index = {
        "px_per_pt": TARGET_PX_PER_PT, "val_frac": val_frac,
        "pages": [m["key"] for m in manifests], "val_pages": sorted(val_pages),
        "train": train, "val": val,
        "n_strokes": {layer["key"]: layer["n_strokes"] for m in manifests for layer in m["layers"]},
    }
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return index


def _parse_pages(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    pages = [int(p) for p in spec.split(",") if p.strip()]
    bad = [p for p in pages if not 0 <= p < n]
    if bad:
        raise SystemExit(f"--pages {bad} out of range (document has {n} pages)")
    return pages


def collect_pdfs(pdf_dir: str | None, pdfs: list[str]) -> list[tuple[Path, Path | None]]:
    """`(pdf, key root)` for every `*.pdf` under `pdf_dir` (recursive, sorted),
    then each extra `--pdf` (root None -> keyed by its stem)."""
    if not pdf_dir and not pdfs:
        raise SystemExit("give --pdf-dir DIR and/or --pdf FILE")
    sources: list[tuple[Path, Path | None]] = []
    if pdf_dir:
        root = Path(pdf_dir)
        if not root.is_dir():
            raise SystemExit(f"--pdf-dir is not a directory: {root}")
        found = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")
        if not found:
            raise SystemExit(f"no .pdf files under {root}")
        sources += [(p, root) for p in found]
    for pdf in pdfs:
        p = Path(pdf)
        if not p.is_file():
            raise SystemExit(f"PDF not found: {p}")
        sources.append((p, None))
    return sources


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pdf-dir", default=None, help="directory of source PDFs (searched recursively)")
    ap.add_argument("--pdf", action="append", default=[], help="extra source PDF (repeatable)")
    ap.add_argument("--pages", default=None, help="comma-separated page indices, every PDF (default: all)")
    ap.add_argument("--dpi", type=float, default=300.0, help="render dpi (default 300)")
    ap.add_argument("--val-frac", type=float, default=0.1, help="fraction of pages held out (default 0.1)")
    ap.add_argument("--workers", type=int, default=1,
                    help="pages prepared in parallel processes (default 1; ~0.7-1 GB RAM each on A1 @ 300 dpi)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    from rastervec.commons.logging_setup import configure_logging

    configure_logging(logging.DEBUG if args.verbose else logging.WARNING)
    import fitz

    sources = collect_pdfs(args.pdf_dir, args.pdf)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[Path, int, str]] = []
    for pdf, root in sources:
        with fitz.open(pdf) as doc:
            n = doc.page_count
        jobs += [(pdf, i, page_key(pdf, i, root)) for i in _parse_pages(args.pages, n)]
    keys = [key for *_x, key in jobs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise SystemExit(f"page keys collide (same PDF given twice?): {dupes[:5]}")

    ok, skipped, failed = [], [], []
    where = {key: (pdf, i) for pdf, i, key in jobs}

    def report_failure(key: str, stage: str, msg: str, tb: str, log_file) -> None:
        failed.append(key)
        pdf, i = where[key]
        _say(f"[{key}] FAILED at stage '{stage}' ({pdf} page {i}): {msg}", log_file)
        log_file.write(tb + "\n")
        log_file.flush()
        tqdm.write(tb)

    with open(out / "prep_log.txt", "a", encoding="utf-8") as log_file:
        workers = max(1, args.workers)
        _say(f"=== prep_dataset {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(sources)} PDF(s), "
             f"{len(jobs)} page(s), {workers} worker(s) -> {out}", log_file)
        todo = []
        for pdf, i, key in jobs:
            if (out / "pages" / f"{key}.json").exists():
                skipped.append(key)
            else:
                todo.append((pdf, i, key))
        bar = tqdm(total=len(jobs), initial=len(skipped), desc="prep pages", unit="page")
        if workers == 1 or len(todo) <= 1:
            for pdf, i, key in todo:
                bar.set_postfix_str(key)
                try:
                    prep_page(pdf, i, out, args.dpi, key=key, log=lambda m: _say(m, log_file))
                    ok.append(key)
                except _StageError as exc:
                    report_failure(key, exc.stage, str(exc), traceback.format_exc(), log_file)
                bar.update(1)
        else:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=min(workers, len(todo)), mp_context=ctx,
                                     initializer=_init_worker) as pool:
                futures = [pool.submit(_prep_job, pdf, i, out, args.dpi, key) for pdf, i, key in todo]
                for fut in as_completed(futures):
                    result = fut.result()
                    key = result[1]
                    bar.set_postfix_str(key)
                    if result[0] == "ok":
                        for line in result[2]:
                            _say(line, log_file)
                        ok.append(key)
                    else:
                        for line in result[5]:
                            _say(line, log_file)
                        report_failure(key, result[2], result[3], result[4], log_file)
                    bar.update(1)
        bar.close()
        index = write_index(out, args.val_frac)
        _say(f"=== done: {len(ok)} ok, {len(skipped)} skipped (already done), {len(failed)} failed"
             + (f": {failed}" if failed else ""), log_file)
        _say(f"=== index.json: {len(index['train'])} train layer(s), {len(index['val'])} val layer(s), "
             f"val pages {index['val_pages']}", log_file)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
