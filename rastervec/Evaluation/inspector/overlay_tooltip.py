"""Simple mouse-following Tk tooltip, used by `overlay_canvas.py::PageView`
for hover metadata."""
from __future__ import annotations

import tkinter as tk


class Tooltip:
    """Simple mouse-following tooltip for the canvas."""

    def __init__(self, parent: tk.Widget):
        self.parent = parent

        self.window: tk.Toplevel | None = None
        self.label: tk.Label | None = None

    def show(
        self,
        x: int,
        y: int,
        text: str,
    ) -> None:

        if self.window is None:
            self.window = tk.Toplevel(
                self.parent
            )

            self.window.overrideredirect(
                True
            )

            self.window.attributes(
                "-topmost",
                True,
            )

            self.label = tk.Label(
                self.window,
                text=text,
                justify="left",
                anchor="w",
                padx=8,
                pady=6,
                bg="#ffffe0",
                fg="#111111",
                relief="solid",
                borderwidth=1,
                font=("TkDefaultFont", 9),
            )

            self.label.pack()

        else:
            assert self.label is not None

            self.label.config(
                text=text
            )

        self.window.geometry(
            f"+{x + 15}+{y + 15}"
        )

        self.window.deiconify()

    def hide(self) -> None:
        if self.window is not None:
            self.window.withdraw()
