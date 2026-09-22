"""CLI/path-resolution for the PDF layer inspector -- mostly Tk-independent
(only `pick_pdf_with_dialog`'s file picker touches Tk) and has no dependency
on `InspectorApp`'s own internals, unlike everything else in `inspector.py`."""
from __future__ import annotations

import argparse
import os
import tkinter as tk
from tkinter import filedialog

from rastervec.Evaluation.inspector.inspector import InspectorApp, REFERENCES_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Inspect the layers and objects "
            "contained in a PDF."
        )
    )

    parser.add_argument(
        "pdf",
        nargs="?",
        help="Path to the PDF to inspect.",
    )

    return parser.parse_args()


def pick_initial_pdf() -> str | None:
    """Pick the first PDF in the references directory."""

    if not os.path.isdir(
        REFERENCES_DIR
    ):
        return None

    pdfs = sorted(
        filename
        for filename in os.listdir(
            REFERENCES_DIR
        )
        if filename.lower().endswith(
            ".pdf"
        )
    )

    if not pdfs:
        return None

    return os.path.join(
        REFERENCES_DIR,
        pdfs[0],
    )


def pick_pdf_with_dialog() -> str | None:
    """Open a PDF selection dialog."""

    root = tk.Tk()
    root.withdraw()

    try:
        return filedialog.askopenfilename(
            title="Open PDF",
            filetypes=[
                ("PDF files", "*.pdf"),
            ],
        )
    finally:
        root.destroy()


def resolve_pdf_path(
    requested_path: str | None,
) -> str | None:
    """Resolve the PDF path.

    Priority:

        1. CLI argument
        2. First PDF in references/
        3. File picker
    """

    if requested_path:

        path = os.path.abspath(
            os.path.expanduser(
                requested_path
            )
        )

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"PDF not found:\n{path}"
            )

        if not path.lower().endswith(
            ".pdf"
        ):
            raise ValueError(
                f"Expected a PDF file:\n{path}"
            )

        return path

    path = pick_initial_pdf()

    if path:
        return path

    return pick_pdf_with_dialog()


def main() -> None:
    args = parse_args()

    try:
        pdf_path = resolve_pdf_path(
            args.pdf
        )

    except Exception as exc:
        print(
            f"Error: {exc}"
        )

        raise SystemExit(1)

    if not pdf_path:
        return

    root = tk.Tk()

    app: InspectorApp | None = None

    try:
        app = InspectorApp(
            root,
            pdf_path,
        )

        root.mainloop()

    finally:

        if app is not None:
            app.close()

        try:
            root.destroy()

        except tk.TclError:
            pass
