"""Browse a `prep_dataset.py` training set (Tkinter, read-only).

    python -m raster_parser.phase_raster_vectorize.deep_vectoriser.view_dataset --data data/deepvec \\
        [--key KEY] [--seed 0]

Reads only the dataset folder (`index.json`, `pages/*.json`, `layers/*`) --
the source PDFs need not exist -- and never writes to it (the PNG masks are
decoded in memory, not into `train.py`'s `<data>/cache/`).

Left: every (page, color layer) entry, filterable by key / split; layers
prep listed but didn't store (no GT strokes) show greyed out. Centre: the
layer mask tinted with its color, GT strokes on top (one hue per stroke,
drawn at their width or 1 px), toggleable endpoints (green start, red
end), piece joints, stroke bboxes and the stored ink samples; hover a stroke
for its details. Right: the page manifest + layer statistics, and the
**crop mode** -- a training crop drawn exactly as `train.py` draws one
(`train_data.sample_crop`, optionally `train_data.augment`), its strokes
numbered in the decoder's slot order, its window outlined on the layer
(un-augmented crops only: rot90/flip move it).

Keys: Up/Down (or j/k) layer, PageUp/PageDown next/previous page, f fit,
+/- zoom, wheel zoom at cursor, drag pan, c crop mode, n new crop.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from raster_parser.models.deep_vectoriser import LayerRow, DatasetInfo, LoadedLayer
from raster_parser.commons.logging_setup import configure_logging, get_logger
from raster_parser.phase_raster_vectorize.deep_vectoriser import geometry as geo
from raster_parser.phase_raster_vectorize.deep_vectoriser import train_data as td
from raster_parser.phase_raster_vectorize.deep_vectoriser.config import (
    MODEL_DEFAULTS,
    TARGET_PX_PER_PT,
)

_LAYER_RE = re.compile(r"__L(\d+)$")
_PIECE_BINS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, None))

_LOG = get_logger("deep_vectoriser.view_dataset")


# ---------------------------------------------------------------------------
# Data (pure, no Tk)
# ---------------------------------------------------------------------------


def dataset_px_per_pt(info: DatasetInfo) -> float:
    """The dataset's stored scale (pixels per page point)."""
    return float(info.index.get("px_per_pt", TARGET_PX_PER_PT))


def load_dataset(data_dir) -> DatasetInfo:
    data_dir = Path(data_dir)
    index_path = data_dir / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"{data_dir} has no index.json -- not a prep_dataset.py output folder?")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    train, val = set(index.get("train", [])), set(index.get("val", []))
    pages: dict[str, dict] = {}
    rows: list[LayerRow] = []
    for path in sorted((data_dir / "pages").glob("*.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        pages[m["key"]] = m
        for entry in m.get("layers", []):
            key = entry["key"]
            saved = bool(entry.get("saved", entry.get("n_strokes", 0) > 0))
            split = (
                "train"
                if key in train
                else "val"
                if key in val
                else "unindexed"
                if saved
                else "unsaved"
            )
            match = _LAYER_RE.search(key)
            rows.append(
                LayerRow(
                    key=key,
                    page_key=m["key"],
                    layer=int(match.group(1)) if match else -1,
                    split=split,
                    n_strokes=int(entry.get("n_strokes", 0)),
                    color=tuple(int(c) for c in entry.get("color", (0, 0, 0))),
                    saved=saved,
                    pdf=m.get("pdf", ""),
                    page=int(m.get("page", -1)),
                )
            )
    rows.sort(key=lambda r: (r.page_key, r.layer, r.key))
    return DatasetInfo(data_dir, index, pages, rows)


def read_layer(data_dir, key: str) -> LoadedLayer:
    """One stored layer, decoded in memory (unlike `td.load_layer`, which
    writes the decoded mask into `<data>/cache/`)."""
    layers_dir = Path(data_dir) / "layers"
    legacy = layers_dir / f"{key}.gray.npy"
    if legacy.is_file():
        gray = np.load(legacy, mmap_mode="r")
    else:
        png = layers_dir / f"{key}.gray.png"
        if not png.is_file():
            raise FileNotFoundError(f"no mask for {key} in {layers_dir}")
        gray = cv2.imdecode(np.fromfile(str(png), np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"could not decode {png}")
    z = np.load(layers_dir / f"{key}.strokes.npz")
    pieces, offsets = z["pieces"].astype(np.float64), z["offsets"]
    strokes = [pieces[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
    data = td.LayerData(key, gray, strokes, z["widths"], z["bboxes"], z["ink_pts"])
    origin = tuple(int(v) for v in z["origin"]) if "origin" in z.files else (0, 0)
    page_shape = (
        tuple(int(v) for v in z["page_shape"]) if "page_shape" in z.files else tuple(gray.shape)
    )
    color = tuple(int(c) for c in z["color"]) if "color" in z.files else (0, 0, 0)
    return LoadedLayer(data, color, origin, page_shape)  # type: ignore[arg-type]


def stroke_polylines(strokes, per_piece: int = 8) -> list[np.ndarray]:
    return [geo.sample_stroke(s, per_piece) for s in strokes]


def layer_stats(layer: td.LayerData, max_prims: int = MODEL_DEFAULTS["max_prims"]) -> dict:
    n_pieces = np.array([len(s) for s in layer.strokes], int)
    bins = {}
    for lo, hi in _PIECE_BINS:
        sel = (n_pieces >= lo) & (n_pieces <= hi) if hi is not None else n_pieces >= lo
        bins[f"{lo}" if lo == hi else f"{lo}-{hi}" if hi else f"{lo}+"] = int(sel.sum())

    def spread(a) -> dict:
        a = np.asarray(a, float)
        return (
            {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}
            if len(a)
            else {}
        )

    return {
        "strokes": len(layer.strokes),
        "pieces": int(n_pieces.sum()),
        "pieces_per_stroke": bins,
        "over_max_prims": int((n_pieces > max_prims).sum()),
        "width_px": spread(layer.widths),
        "mask_shape": list(layer.gray.shape),
        "ink_fraction": float((np.asarray(layer.gray) < 200).mean()) if layer.gray.size else 0.0,
        "ink_samples": len(layer.ink_pts),
    }


def sample_training_crop(
    layer: td.LayerData,
    rng: np.random.Generator,
    size: "int | None" = None,
    augment: bool = False,
    cfg: "dict | None" = None,
) -> td.Crop:
    """A crop as `train.py` draws it: `sample_crop`, then `augment` if asked.
    `size=None` picks one of `td.SIZES` like the trainer."""
    cfg = cfg or MODEL_DEFAULTS
    if size is None:
        size = int(rng.choice(td.SIZES))
    crop = td.sample_crop(layer, size, rng, cfg["n_stroke"], cfg["max_prims"])
    if augment:
        aug = td.augment(crop, rng)
        aug.n_found = crop.n_found  # window is meaningless after rot90/flip
        crop = aug
    return crop


def stroke_colors(n: int) -> list[tuple[int, int, int]]:
    """Distinct-ish RGB per stroke index (golden-ratio hue walk)."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.85, 0.85)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def tint(gray: np.ndarray, rgb, enabled: bool = True) -> np.ndarray:
    """Mask (ink dark) -> RGB, ink drawn in `rgb` (or gray when disabled)."""
    if not enabled:
        return np.repeat(gray[:, :, None], 3, axis=2)
    alpha = (255.0 - gray.astype(np.float32))[:, :, None] / 255.0
    col = np.asarray(rgb, np.float32)[None, None, :]
    # a near-white layer color would vanish -- darken it to stay visible
    if col.mean() > 200:
        col = col * 0.6
    return (255.0 * (1.0 - alpha) + col * alpha).astype(np.uint8)


def point_segment_dist(p, pts: np.ndarray) -> float:
    """Min distance from point `p` to polyline `pts` (N, 2)."""
    if len(pts) == 1:
        return float(np.hypot(*(pts[0] - p)))
    a, b = pts[:-1], pts[1:]
    ab = b - a
    denom = np.maximum((ab**2).sum(1), 1e-12)
    t = np.clip(((p - a) * ab).sum(1) / denom, 0.0, 1.0)
    proj = a + ab * t[:, None]
    return float(np.sqrt(((proj - p) ** 2).sum(1)).min())


# ---------------------------------------------------------------------------
# Tk app
# ---------------------------------------------------------------------------
class DatasetViewerApp:
    CROP_VIEW = 512

    def __init__(self, info: DatasetInfo, *, key: "str | None" = None, seed: int = 0) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk = tk, ttk
        self.info = info
        self.seed = seed
        self.crop_counter = 0
        self.layer: "LoadedLayer | None" = None
        self.row: "LayerRow | None" = None
        self.polys: list[np.ndarray] = []
        self.colors: list[tuple[int, int, int]] = []
        self.crop: "td.Crop | None" = None
        self.zoom = 1.0
        self.ox = self.oy = 0.0
        self.hover: "int | None" = None
        self._drag: "tuple[int, int, float, float] | None" = None
        self._photo = self._crop_photo = None
        self._swatches: dict = {}
        self._redraw_pending = False
        self._auto_fit = True  # refit on resize until the user zooms/pans

        self.root = tk.Tk()
        self.root.title(f"DeepVectoriser dataset -- {info.data_dir}")
        self.root.geometry("1500x900")
        self._build()
        self._populate()
        if key:
            if self.tree.exists(key):
                self._select(key)
            else:
                _LOG.warning(f"--key {key!r} not found")
        if not self.tree.selection():
            children = self.tree.get_children()
            if children:
                self._select(children[0])

    # -- layout -------------------------------------------------------------
    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(self.root)
        bar.pack(side="top", fill="x")
        self.v = {
            name: tk.BooleanVar(value=default)
            for name, default in (
                ("tint", True),
                ("strokes", True),
                ("thin", False),
                ("ends", True),
                ("joints", False),
                ("bboxes", False),
                ("ink", False),
                ("crop_mode", False),
                ("augment", False),
                ("show_window", True),
                ("hide_unsaved", False),
            )
        }
        for name, text in (
            ("tint", "tint mask"),
            ("strokes", "strokes"),
            ("thin", "thin (1 px)"),
            ("ends", "endpoints"),
            ("joints", "piece joints"),
            ("bboxes", "bboxes"),
            ("ink", "ink samples"),
        ):
            ttk.Checkbutton(bar, text=text, variable=self.v[name], command=self.redraw).pack(
                side="left", padx=3
            )
        ttk.Button(bar, text="fit (f)", command=self.fit).pack(side="left", padx=6)
        ttk.Checkbutton(
            bar, text="crop mode (c)", variable=self.v["crop_mode"], command=self._toggle_crop_mode
        ).pack(side="left", padx=6)

        self.status = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status, anchor="w").pack(side="bottom", fill="x")

        body = ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, width=330)
        body.add(left, weight=0)
        self.filter_var = tk.StringVar()
        self.filter_var.trace_add("write", lambda *_: self._populate())
        ttk.Entry(left, textvariable=self.filter_var).pack(fill="x", padx=2, pady=2)
        splits = ttk.Frame(left)
        splits.pack(fill="x")
        self.split_var = tk.StringVar(value="all")
        for s in ("all", "train", "val"):
            ttk.Radiobutton(
                splits, text=s, value=s, variable=self.split_var, command=self._populate
            ).pack(side="left")
        ttk.Checkbutton(
            splits, text="hide unsaved", variable=self.v["hide_unsaved"], command=self._populate
        ).pack(side="left", padx=4)
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(
            tree_frame, columns=("split", "strokes"), show="tree headings", selectmode="browse"
        )
        self.tree.heading("#0", text="layer key")
        self.tree.heading("split", text="split")
        self.tree.heading("strokes", text="strokes")
        self.tree.column("#0", width=210)
        self.tree.column("split", width=60, anchor="center")
        self.tree.column("strokes", width=60, anchor="e")
        self.tree.tag_configure("unsaved", foreground="#999999")
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())
        self.count_var = tk.StringVar()
        ttk.Label(left, textvariable=self.count_var).pack(fill="x")

        self.canvas = tk.Canvas(body, background="#ebebeb", highlightthickness=0)
        body.add(self.canvas, weight=1)

        right = ttk.Frame(body, width=self.CROP_VIEW + 20)
        body.add(right, weight=0)
        self.info_text = tk.Text(right, width=60, height=22, wrap="none", font=("Consolas", 9))
        self.info_text.pack(fill="x")
        self.crop_frame = ttk.Frame(right)
        ctl = ttk.Frame(self.crop_frame)
        ctl.pack(fill="x")
        ttk.Label(ctl, text="size").pack(side="left")
        self.size_var = tk.StringVar(value="random")
        ttk.Combobox(
            ctl,
            textvariable=self.size_var,
            width=7,
            state="readonly",
            values=["random", *map(str, td.SIZES)],
        ).pack(side="left", padx=2)
        ttk.Checkbutton(ctl, text="augment", variable=self.v["augment"]).pack(side="left", padx=2)
        ttk.Checkbutton(
            ctl, text="window on layer", variable=self.v["show_window"], command=self.redraw
        ).pack(side="left", padx=2)
        ttk.Button(ctl, text="new crop (n)", command=self.new_crop).pack(side="left", padx=4)
        self.crop_canvas = tk.Canvas(
            self.crop_frame,
            width=self.CROP_VIEW,
            height=self.CROP_VIEW,
            background="white",
            highlightthickness=1,
        )
        self.crop_canvas.pack(pady=2)
        self.crop_info = tk.StringVar()
        ttk.Label(self.crop_frame, textvariable=self.crop_info, justify="left").pack(fill="x")

        c = self.canvas
        c.bind("<Configure>", lambda _e: self.fit() if self._auto_fit else self.redraw())
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_drag", None))
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Motion>", self._on_motion)
        c.bind("<Leave>", lambda _e: self._set_hover(None, None))
        r = self.root
        for seq, fn in (
            ("j", lambda: self._step(1)),
            ("k", lambda: self._step(-1)),
            ("<Next>", lambda: self._step_page(1)),
            ("<Prior>", lambda: self._step_page(-1)),
            ("f", self.fit),
            ("plus", lambda: self._zoom_center(1.25)),
            ("equal", lambda: self._zoom_center(1.25)),
            ("minus", lambda: self._zoom_center(0.8)),
            ("c", self._key_crop_mode),
            ("n", self.new_crop),
        ):
            r.bind(
                seq,
                lambda e, fn=fn: (
                    None if isinstance(e.widget, (self.tk.Entry, self.ttk.Entry)) else fn()
                ),
            )

    # -- list ---------------------------------------------------------------
    def _swatch(self, rgb):
        if rgb not in self._swatches:
            img = self.tk.PhotoImage(width=14, height=14)
            img.put("#%02x%02x%02x" % rgb, to=(0, 0, 14, 14))
            self._swatches[rgb] = img
        return self._swatches[rgb]

    def _populate(self) -> None:
        keep = self.tree.selection()
        self.tree.delete(*self.tree.get_children())
        text = self.filter_var.get().strip().lower()
        split = self.split_var.get()
        shown = 0
        for row in self.info.rows:
            if text and text not in row.key.lower():
                continue
            if split != "all" and row.split != split:
                continue
            if self.v["hide_unsaved"].get() and not row.saved:
                continue
            self.tree.insert(
                "",
                "end",
                iid=row.key,
                text=row.key,
                image=self._swatch(row.color),
                values=(row.split, row.n_strokes),
                tags=() if row.saved else ("unsaved",),
            )
            shown += 1
        n_saved = sum(r.saved for r in self.info.rows)
        self.count_var.set(
            f"{shown} shown / {len(self.info.rows)} layers ({n_saved} saved, "
            f"{len(self.info.pages)} pages)"
        )
        if keep and self.tree.exists(keep[0]):
            self.tree.selection_set(keep[0])
            self.tree.see(keep[0])

    def _select(self, key: str) -> None:
        self.tree.selection_set(key)
        self.tree.focus(key)
        self.tree.see(key)

    def _step(self, delta: int) -> None:
        items = self.tree.get_children()
        if not items:
            return
        sel = self.tree.selection()
        i = items.index(sel[0]) if sel else -1
        self._select(items[max(0, min(len(items) - 1, i + delta))])

    def _step_page(self, delta: int) -> None:
        items = self.tree.get_children()
        sel = self.tree.selection()
        if not items or not sel:
            return
        by_key = {r.key: r for r in self.info.rows}
        i = items.index(sel[0])
        cur = by_key[items[i]].page_key
        j = i
        while 0 <= j + delta < len(items) and by_key[items[j]].page_key == cur:
            j += delta
        target = by_key[items[j]].page_key
        if delta < 0:  # land on the first layer of that page
            while j > 0 and by_key[items[j - 1]].page_key == target:
                j -= 1
        self._select(items[j])

    def _on_select(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        row = next(r for r in self.info.rows if r.key == sel[0])
        if row is self.row:
            return
        self.row, self.layer, self.crop, self.hover = row, None, None, None
        self.polys, self.colors = [], []
        if row.saved:
            try:
                self.layer = read_layer(self.info.data_dir, row.key)
            except (OSError, RuntimeError) as exc:
                self.status.set(f"{row.key}: {exc}")
            else:
                self.polys = stroke_polylines(self.layer.data.strokes)
                self.colors = stroke_colors(len(self.polys))
        self._show_info()
        self._auto_fit = True
        self.fit()
        self._draw_crop()

    def _show_info(self) -> None:
        row = self.row
        lines = [
            f"key        {row.key}",
            f"split      {row.split}",
            f"pdf        {row.pdf}",
            f"page       {row.page}",
        ]
        m = self.info.pages.get(row.page_key, {})
        lines += [
            f"dpi        {m.get('dpi')}  (stored at {dataset_px_per_pt(self.info) * 72:.0f} dpi)",
            f"rotation   {m.get('rotation')}",
            f"prep time  {m.get('seconds')} s",
            f"color      {row.color}  #%02x%02x%02x" % row.color,
        ]
        if self.layer is None:
            lines.append(
                "\n(not stored: prep found no GT strokes on this layer)"
                if not row.saved
                else "\n(could not load)"
            )
        else:
            lines += [
                f"origin     {self.layer.origin}  (crop on page, px)",
                f"page shape {self.layer.page_shape}  (h, w px)",
            ]
            st = layer_stats(self.layer.data)
            lines += [
                "",
                f"strokes    {st['strokes']}   pieces {st['pieces']}",
                "pieces/stroke "
                + "  ".join(f"{k}:{v}" for k, v in st["pieces_per_stroke"].items()),
                f"> max_prims ({MODEL_DEFAULTS['max_prims']})  {st['over_max_prims']}  (split in training)",
            ]
            if st["width_px"]:
                w = st["width_px"]
                lines.append(
                    f"width px   min {w['min']:.2f}  median {w['median']:.2f}  max {w['max']:.2f}"
                )
            lines += [
                f"mask       {st['mask_shape'][1]} x {st['mask_shape'][0]} px, "
                f"ink {st['ink_fraction'] * 100:.2f} %",
                f"ink samples {st['ink_samples']}",
            ]
        self.info_text.configure(state="normal")
        self.info_text.delete("1.0", "end")
        self.info_text.insert("1.0", "\n".join(lines))
        self.info_text.configure(state="disabled")

    # -- view transform -----------------------------------------------------
    def _canvas_size(self) -> tuple[int, int]:
        return max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())

    def fit(self) -> None:
        if self.layer is None:
            self.redraw()
            return
        h, w = self.layer.data.gray.shape
        cw, ch = self._canvas_size()
        self.zoom = max(1e-3, min(cw / max(w, 1), ch / max(h, 1)) * 0.98)
        self.ox = (w - cw / self.zoom) / 2.0
        self.oy = (h - ch / self.zoom) / 2.0
        self.redraw()

    def _zoom_at(self, sx: float, sy: float, factor: float) -> None:
        self._auto_fit = False
        x, y = self.ox + sx / self.zoom, self.oy + sy / self.zoom
        self.zoom = float(np.clip(self.zoom * factor, 1e-3, 64.0))
        self.ox, self.oy = x - sx / self.zoom, y - sy / self.zoom
        self.redraw()

    def _zoom_center(self, factor: float) -> None:
        cw, ch = self._canvas_size()
        self._zoom_at(cw / 2, ch / 2, factor)

    def _on_wheel(self, e) -> None:
        self._zoom_at(e.x, e.y, 1.25 if e.delta > 0 else 0.8)

    def _on_press(self, e) -> None:
        self._drag = (e.x, e.y, self.ox, self.oy)

    def _on_drag(self, e) -> None:
        if self._drag:
            self._auto_fit = False
            x0, y0, ox, oy = self._drag
            self.ox = ox - (e.x - x0) / self.zoom
            self.oy = oy - (e.y - y0) / self.zoom
            self.redraw()

    # -- drawing ------------------------------------------------------------
    def redraw(self) -> None:
        if not self._redraw_pending:
            self._redraw_pending = True
            self.root.after_idle(self._redraw_now)

    def _visible(self, cw: int, ch: int) -> np.ndarray:
        b = self.layer.data.bboxes
        if len(b) == 0:
            return np.zeros(0, int)
        x1, y1 = self.ox + cw / self.zoom, self.oy + ch / self.zoom
        return np.nonzero(
            (b[:, 2] >= self.ox) & (b[:, 0] <= x1) & (b[:, 3] >= self.oy) & (b[:, 1] <= y1)
        )[0]

    def _to_screen(self, pts: np.ndarray) -> np.ndarray:
        """Data px -> canvas px, fixed-point (cv2 `shift=4`)."""
        return np.round((pts - (self.ox, self.oy)) * self.zoom * 16).astype(np.int32)

    def _redraw_now(self) -> None:
        from PIL import Image, ImageTk

        self._redraw_pending = False
        cw, ch = self._canvas_size()
        img = np.full((ch, cw, 3), 235, np.uint8)
        layer = self.layer
        if layer is not None:
            z = self.zoom
            gray = layer.data.gray
            h, w = gray.shape
            sx0, sy0 = max(0, int(math.floor(self.ox))), max(0, int(math.floor(self.oy)))
            sx1 = min(w, int(math.ceil(self.ox + cw / z)))
            sy1 = min(h, int(math.ceil(self.oy + ch / z)))
            if sx1 > sx0 and sy1 > sy0:
                tx0, ty0 = int(round((sx0 - self.ox) * z)), int(round((sy0 - self.oy) * z))
                tx1, ty1 = int(round((sx1 - self.ox) * z)), int(round((sy1 - self.oy) * z))
                if tx1 > tx0 and ty1 > ty0:
                    sub = np.asarray(gray[sy0:sy1, sx0:sx1])
                    interp = cv2.INTER_NEAREST if z >= 1 else cv2.INTER_AREA
                    rs = tint(
                        cv2.resize(sub, (tx1 - tx0, ty1 - ty0), interpolation=interp),
                        layer.color,
                        self.v["tint"].get(),
                    )
                    cx0, cy0 = max(0, tx0), max(0, ty0)
                    cx1, cy1 = min(cw, tx1), min(ch, ty1)
                    if cx1 > cx0 and cy1 > cy0:
                        img[cy0:cy1, cx0:cx1] = rs[cy0 - ty0 : cy1 - ty0, cx0 - tx0 : cx1 - tx0]
            self._draw_overlays(img, cw, ch)
        self._photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        if layer is None:
            msg = (
                "no layer"
                if self.row is None
                else (
                    "not stored -- no GT strokes on this layer"
                    if not self.row.saved
                    else "could not load"
                )
            )
            self.canvas.create_text(cw / 2, ch / 2, text=msg, fill="#666666", font=("Segoe UI", 12))
        self._draw_hover()
        self._update_status()

    def _draw_overlays(self, img: np.ndarray, cw: int, ch: int) -> None:
        data, z = self.layer.data, self.zoom
        vis = self._visible(cw, ch)
        if self.v["ink"].get() and len(data.ink_pts):
            p = self._to_screen(data.ink_pts.astype(float) + 0.5) // 16
            ok = (p[:, 0] >= 0) & (p[:, 0] < cw) & (p[:, 1] >= 0) & (p[:, 1] < ch)
            for x, y in p[ok]:
                cv2.circle(img, (int(x), int(y)), 1, (220, 0, 220), -1)
        if self.v["bboxes"].get():
            for i in vis:
                a, b = self._to_screen(data.bboxes[i].reshape(2, 2).astype(float)).tolist()
                cv2.rectangle(img, tuple(a), tuple(b), (150, 150, 150), 1, cv2.LINE_8, shift=4)
        if self.v["strokes"].get():
            thin = self.v["thin"].get()
            for i in vis:
                width = 1 if thin else max(1, int(round(float(data.widths[i]) * z)))
                cv2.polylines(
                    img,
                    [self._to_screen(self.polys[i]).reshape(-1, 1, 2)],
                    False,
                    self.colors[i],
                    width,
                    cv2.LINE_AA,
                    shift=4,
                )
        if self.v["joints"].get():
            for i in vis:
                s = data.strokes[i]
                if len(s) > 1:
                    for x, y in self._to_screen(s[1:, 0]):
                        cv2.circle(
                            img, (int(x), int(y)), 2 * 16, (0, 90, 255), -1, cv2.LINE_AA, shift=4
                        )
        if self.v["ends"].get():
            for i in vis:
                s = data.strokes[i]
                if len(s):
                    a, b = self._to_screen(np.stack([s[0, 0], s[-1, 3]]))
                    cv2.circle(
                        img, (int(a[0]), int(a[1])), 3 * 16, (0, 170, 0), -1, cv2.LINE_AA, shift=4
                    )
                    cv2.circle(
                        img, (int(b[0]), int(b[1])), 3 * 16, (220, 0, 0), -1, cv2.LINE_AA, shift=4
                    )
        if self.crop is not None and self.crop.window and self.v["show_window"].get():
            x0, y0, win = self.crop.window
            a, b = self._to_screen(np.array([[x0, y0], [x0 + win, y0 + win]], float)).tolist()
            cv2.rectangle(img, tuple(a), tuple(b), (255, 140, 0), 2, cv2.LINE_8, shift=4)

    # -- hover --------------------------------------------------------------
    def _on_motion(self, e) -> None:
        if self.layer is None or self._drag:
            self._update_status(e.x, e.y)
            return
        p = np.array([self.ox + e.x / self.zoom, self.oy + e.y / self.zoom])
        tol = 6.0 / self.zoom
        b = self.layer.data.bboxes
        best, best_d = None, tol
        if len(b):
            near = np.nonzero(
                (b[:, 0] - tol <= p[0])
                & (b[:, 2] + tol >= p[0])
                & (b[:, 1] - tol <= p[1])
                & (b[:, 3] + tol >= p[1])
            )[0]
            for i in near[:500]:
                d = point_segment_dist(p, self.polys[i])
                if d <= best_d:
                    best, best_d = int(i), d
        self._set_hover(best, (e.x, e.y))
        self._update_status(e.x, e.y)

    def _set_hover(self, i, pos) -> None:
        if i == self.hover and (i is None or pos is None):
            return
        self.hover, self._hover_pos = i, pos
        self._draw_hover()

    def _draw_hover(self) -> None:
        c = self.canvas
        c.delete("hover")
        i = self.hover
        if i is None or self.layer is None or i >= len(self.polys):
            return
        data = self.layer.data
        pts = (self.polys[i] - (self.ox, self.oy)) * self.zoom
        flat = pts.ravel().tolist()
        if len(flat) >= 4:
            c.create_line(*flat, fill="#ffd400", width=5, tags="hover")
            c.create_line(*flat, fill="#000000", width=1, tags="hover")
        s = data.strokes[i]
        bb = data.bboxes[i]
        text = (
            f"stroke {i}\npieces {len(s)}   width {float(data.widths[i]):.2f} px\n"
            f"length {geo.stroke_length(s):.1f} px\n"
            f"bbox ({bb[0]:.1f}, {bb[1]:.1f}) - ({bb[2]:.1f}, {bb[3]:.1f})\n"
            f"start ({s[0, 0, 0]:.1f}, {s[0, 0, 1]:.1f})  end ({s[-1, 3, 0]:.1f}, {s[-1, 3, 1]:.1f})"
        )
        x, y = getattr(self, "_hover_pos", None) or (pts[0, 0], pts[0, 1])
        t = c.create_text(
            x + 14, y + 14, text=text, anchor="nw", font=("Consolas", 9), tags="hover"
        )
        x0, y0, x1, y1 = c.bbox(t)
        cw, ch = self._canvas_size()
        dx = min(0, cw - x1 - 4)
        dy = min(0, ch - y1 - 4)
        if dx or dy:
            c.move(t, dx if dx else 0, dy if dy else 0)
            x0, y0, x1, y1 = c.bbox(t)
        r = c.create_rectangle(
            x0 - 4, y0 - 3, x1 + 4, y1 + 3, fill="#ffffe0", outline="#888888", tags="hover"
        )
        c.tag_lower(r, t)

    def _update_status(self, sx: "float | None" = None, sy: "float | None" = None) -> None:
        parts = [f"zoom {self.zoom:.3g}x"]
        if self.layer is not None:
            parts.append(f"{len(self.polys)} strokes")
            if sx is not None:
                x, y = self.ox + sx / self.zoom, self.oy + sy / self.zoom
                px, py = x + self.layer.origin[0], y + self.layer.origin[1]
                k = dataset_px_per_pt(self.info)
                parts.append(
                    f"crop px ({x:.1f}, {y:.1f})  page px ({px:.1f}, {py:.1f})  "
                    f"page pt ({px / k:.1f}, {py / k:.1f}) [rotated display space]"
                )
        self.status.set("   |   ".join(parts))

    # -- crop mode ----------------------------------------------------------
    def _key_crop_mode(self) -> None:
        self.v["crop_mode"].set(not self.v["crop_mode"].get())
        self._toggle_crop_mode()

    def _toggle_crop_mode(self) -> None:
        if self.v["crop_mode"].get():
            self.crop_frame.pack(fill="both", expand=True)
            if self.crop is None:
                self.new_crop()
        else:
            self.crop_frame.pack_forget()
            self.crop = None
            self.redraw()

    def new_crop(self) -> None:
        if not self.v["crop_mode"].get():
            self.v["crop_mode"].set(True)
            self.crop_frame.pack(fill="both", expand=True)
        if self.layer is None:
            self.crop = None
            self._draw_crop()
            return
        rng = np.random.default_rng([self.seed, self.crop_counter])
        self.crop_counter += 1
        size = None if self.size_var.get() == "random" else int(self.size_var.get())
        self.crop = sample_training_crop(self.layer.data, rng, size, self.v["augment"].get())
        self._crop_meta = (self.crop_counter - 1, self.v["augment"].get())
        self._draw_crop()
        self.redraw()

    def _draw_crop(self) -> None:
        from PIL import Image, ImageTk

        c = self.crop_canvas
        c.delete("all")
        crop = self.crop
        if crop is None or self.layer is None:
            self.crop_info.set("" if self.layer is not None else "no stored layer selected")
            return
        n = crop.gray.shape[0]
        scale = self.CROP_VIEW / n
        rgb = tint(
            cv2.resize(
                crop.gray, (self.CROP_VIEW, self.CROP_VIEW), interpolation=cv2.INTER_NEAREST
            ),
            self.layer.color,
            self.v["tint"].get(),
        )
        colors = stroke_colors(len(crop.strokes))
        for j, (s, w) in enumerate(zip(crop.strokes, crop.widths)):
            pts = np.round(geo.sample_stroke(s, 8) * scale * 16).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(
                rgb,
                [pts],
                False,
                colors[j],
                max(1, int(round(w * scale))) if not self.v["thin"].get() else 1,
                cv2.LINE_AA,
                shift=4,
            )
            for pt, col in ((s[0, 0], (0, 170, 0)), (s[-1, 3], (220, 0, 0))):
                cv2.circle(
                    rgb,
                    tuple(int(v) for v in np.round(pt * scale * 16)),
                    3 * 16,
                    col,
                    -1,
                    cv2.LINE_AA,
                    shift=4,
                )
        self._crop_photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        c.create_image(0, 0, anchor="nw", image=self._crop_photo)
        for j, s in enumerate(crop.strokes):  # decoder slot order
            x, y = s[0, 0] * scale
            c.create_text(
                x + 5, y - 5, text=str(j), anchor="sw", font=("Consolas", 8, "bold"), fill="#000000"
            )
        n_stroke = MODEL_DEFAULTS["n_stroke"]
        idx, aug = getattr(self, "_crop_meta", (0, False))
        lines = [
            f"crop #{idx} (seed {self.seed})  {n}x{n} px  augment {'on' if aug else 'off'}",
            f"strokes {len(crop.strokes)} kept / {crop.n_found} found (n_stroke {n_stroke})"
            + ("  -- TRUNCATED" if crop.n_found > n_stroke else ""),
        ]
        if crop.window:
            x0, y0, win = crop.window
            lines.append(
                f"window ({x0}, {y0}) side {win}"
                + (f"  (shrunk from {n}, white-padded)" if win < n else "")
            )
        else:
            lines.append("window not shown (augmented: rot90/flip moved it)")
        lines.append("numbers = slot order (sorted, oriented); green start, red end")
        self.crop_info.set("\n".join(lines))

    def run(self) -> None:
        self.root.mainloop()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="prep_dataset.py output folder")
    ap.add_argument("--key", default=None, help="layer key to open first")
    ap.add_argument(
        "--seed", type=int, default=0, help="crop-mode seed (default 0; crops are reproducible)"
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_arg_parser().parse_args(argv)
    info = load_dataset(args.data)
    if not info.rows:
        raise SystemExit(f"{args.data} has no page manifests (pages/*.json)")
    _LOG.info(
        f"{len(info.pages)} page(s), {len(info.rows)} layer(s) "
        f"({len(info.index.get('train', []))} train / {len(info.index.get('val', []))} val)"
    )
    DatasetViewerApp(info, key=args.key, seed=args.seed).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
