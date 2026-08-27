"""Right pane: layer checkboxes plus collapsible sub-filter groups."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from rastervec.Evaluation.inspector.layers import LayerSpec, SubFilterSpec, NONE_OPTION


class ControlPanel(ttk.Frame):
    def __init__(self, master, layers: list[LayerSpec], on_change=None, **kwargs):
        super().__init__(master, **kwargs)
        self.layers = layers
        self.on_change = on_change

        self._layer_vars: dict[str, tk.BooleanVar] = {}
        # option_vars[layer_key][subfilter_key][option_value] -> BooleanVar
        self._option_vars: dict[str, dict[str, dict[str, tk.BooleanVar]]] = {}
        self._subfilter_frames: dict[tuple[str, str], ttk.Frame] = {}

        self._build_scrollable_area()
        self._build_layer_sections()

    def _build_scrollable_area(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")

        self.inner = ttk.Frame(canvas)
        self._inner_window = canvas.create_window((0, 0), window=self.inner, anchor="nw")

        def on_inner_configure(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def on_canvas_configure(event):
            canvas.itemconfig(self._inner_window, width=event.width)

        self.inner.bind("<Configure>", on_inner_configure)
        canvas.bind("<Configure>", on_canvas_configure)

    def _build_layer_sections(self) -> None:
        for layer in self.layers:
            self._option_vars[layer.key] = {}

            section = ttk.Frame(self.inner)
            section.pack(fill="x", anchor="w", pady=(6, 0))

            var = tk.BooleanVar(value=layer.enabled_default)
            self._layer_vars[layer.key] = var
            cb = ttk.Checkbutton(
                section,
                text=layer.label,
                variable=var,
                command=self._notify_change,
            )
            cb.pack(anchor="w")

            for sub in layer.subfilters:
                self._build_subfilter_group(section, layer, sub)

    def _build_subfilter_group(self, parent, layer: LayerSpec, sub: SubFilterSpec) -> None:
        group = ttk.Frame(parent)
        group.pack(fill="x", anchor="w", padx=(16, 0))

        header = ttk.Frame(group)
        header.pack(fill="x", anchor="w")

        options_frame = ttk.Frame(group)
        options_frame.pack(fill="x", anchor="w", padx=(12, 0))
        self._subfilter_frames[(layer.key, sub.key)] = options_frame

        expanded = tk.BooleanVar(value=True)

        def toggle():
            if expanded.get():
                options_frame.pack_forget()
                expanded.set(False)
                toggle_btn.config(text="+")
            else:
                options_frame.pack(fill="x", anchor="w", padx=(12, 0))
                expanded.set(True)
                toggle_btn.config(text="-")

        toggle_btn = ttk.Button(header, text="-", width=2, command=toggle)
        toggle_btn.pack(side="left")
        ttk.Label(header, text=sub.label).pack(side="left", padx=4)

        self._option_vars[layer.key][sub.key] = {}

        if not sub.dynamic and sub.static_options:
            for value, label in sub.static_options:
                self._add_option_row(options_frame, layer.key, sub.key, value, label, sub.render_as)

    def _add_option_row(self, parent, layer_key: str, sub_key: str, value: str, label: str, render_as: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", anchor="w")

        var = tk.BooleanVar(value=False)
        self._option_vars[layer_key][sub_key][value] = var

        if render_as == "swatch" and value != NONE_OPTION[0]:
            swatch = tk.Canvas(row, width=14, height=14, highlightthickness=1, highlightbackground="#666")
            swatch.pack(side="left", padx=(0, 4))
            swatch.create_rectangle(0, 0, 14, 14, fill=label, outline="")
            text = str(value)
        else:
            text = label

        ttk.Checkbutton(row, text=text, variable=var, command=self._notify_change).pack(side="left", anchor="w")

    def refresh_dynamic_options(self, layer_key: str, subfilter_key: str, options: dict[str, str]) -> None:
        """options: value -> hex color (or display label for non-swatch)."""
        frame = self._subfilter_frames.get((layer_key, subfilter_key))
        if frame is None:
            return
        for child in frame.winfo_children():
            child.destroy()
        self._option_vars[layer_key][subfilter_key] = {}
        for value, label in options.items():
            self._add_option_row(frame, layer_key, subfilter_key, value, label, "swatch")

    def get_active_layers(self) -> set[str]:
        return {key for key, var in self._layer_vars.items() if var.get()}

    def get_active_filters(self, layer_key: str) -> dict[str, set[str]]:
        result = {}
        for sub_key, options in self._option_vars.get(layer_key, {}).items():
            checked = {value for value, var in options.items() if var.get()}
            result[sub_key] = checked
        return result

    def _notify_change(self) -> None:
        if self.on_change:
            self.on_change()
