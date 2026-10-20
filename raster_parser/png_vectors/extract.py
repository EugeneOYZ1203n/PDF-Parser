"""Autotrace PNGs (and a page's embedded images) into shared `Vector`s.

Moved from `raster_parser/extract_vectors_from_png.py`, which re-exports it; the tracing it
uses is in `trace.py`. Placing an image's vectors on the page is described in
`docs/UniqueCasesDuringExtractionOfVectors/`.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import io
from pathlib import Path
import tempfile
from contextlib import nullcontext

from raster_parser.commons.models import Vector
from raster_parser.commons.progress import bar, stage
from raster_parser.png_vectors.trace import (
    TraceVector,
    draw_vectors,
    pdf_to_vectors,
    png_to_svg,
    svg_to_pdf,
    trace_vector_to_vector,
)

import pymupdf

from PIL import Image

import statistics as st
import numpy as np

TRANSPARENCY_THRESHOLD = 0.8


def _calculate_vector_transparency(
    vectors: list[TraceVector],
    alpha: np.ndarray,
    tmp_dir: Path | None = None,
    dpi: int = 300,
) -> list[TraceVector]:
    """
    Calculate fill opacity for each vector using the image alpha channel.

    Vectors with less than 5% average opacity are dropped.
    Vectors without filled pixels are kept unchanged.
    """
    height, width = alpha.shape

    result: list[TraceVector] = []

    for i, vector in enumerate(vectors):

        pdf_path = tmp_dir / f"vector{i}.pdf"

        draw_vectors(
            [vector],
            pdf_path,
            page_width=width * 72 / dpi,
            page_height=height * 72 / dpi,
        )

        doc = pymupdf.open(pdf_path)
        page = doc[0]

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(dpi / 72, dpi / 72),
            colorspace=pymupdf.csGRAY,
            alpha=False,
        )

        fill_mask = np.frombuffer(
            pix.samples,
            dtype=np.uint8,
        ).reshape(
            pix.height,
            pix.width,
        )

        doc.close()

        # PDF background is white, vector is black.
        fill_mask = fill_mask < 128

        if not np.any(fill_mask):
            result.append(vector)
            continue

        vector_alpha = alpha[fill_mask]

        opaque_fraction = st.mode(vector_alpha) / 255.0

        vector.fill_opacity = opaque_fraction
        result.append(vector)

    return result

def _transform_vectors_to_matrix(
    vectors: list[Vector],
    matrix: pymupdf.Matrix,
    image_size: tuple[int, int],
    dpi: int,
) -> list[Vector]:
    """Image vectors mapped into page coordinates through one placement's matrix.

    Points (lines, curves, quad corners) are mapped exactly. A rectangle (`"re"`) stays a
    rectangle while the placement keeps the axes (scale, flip, 90-degree turns); under any other
    rotation or skew it becomes the quad (`"qu"`) of its four mapped corners. `rect` and
    `scissor` become the axis-aligned bounds of their mapped corners.
    """

    image_width, image_height = image_size

    # Vector coordinates are in PDF points, not PNG pixels.
    image_width_pt = image_width * 72 / dpi
    image_height_pt = image_height * 72 / dpi
    keeps_axes = (matrix.b == 0 and matrix.c == 0) or (matrix.a == 0 and matrix.d == 0)

    def transform_point(point) -> tuple[float, float]:
        # Generated-PDF coordinates -> the unit image square -> page coordinates.
        p = pymupdf.Point(point[0] / image_width_pt, point[1] / image_height_pt) * matrix
        return (p.x, p.y)

    def corners(r) -> list[tuple[float, float]]:
        x0, y0, x1, y1 = r
        return [transform_point(c) for c in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]  # ul ur lr ll

    def transform_rect(r) -> tuple[float, float, float, float]:
        pts = corners(r)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))

    def transform_item(item: tuple) -> tuple:
        op = item[0]
        if op == "l":
            return (op, transform_point(item[1]), transform_point(item[2]))
        if op == "c":
            return (op, *[transform_point(pt) for pt in item[1:5]], *item[5:])
        if op == "re":
            if keeps_axes:
                return (op, transform_rect(item[1]), *item[2:])
            return ("qu", tuple(corners(item[1])))
        if op == "qu":
            return (op, tuple(transform_point(pt) for pt in item[1]))
        return item

    return [
        replace(
            vector,
            rect=transform_rect(vector.rect),
            scissor=transform_rect(vector.scissor) if vector.scissor is not None else None,
            items=[transform_item(item) for item in vector.items],
        )
        for vector in vectors
    ]


def extract_vectors_from_png(
    png_path: Path,
    tmp_dir: Path | None = None,
    dpi: int = 300,
) -> list[Vector]:
    """Autotrace a PNG and return its vectors.

    Pipeline:
        PNG -> SVG -> PDF -> MuPDF drawings -> TraceVector -> Vector

    The returned vectors are in the coordinate system of the generated
    vector PDF and are not transformed to any external bounding box.
    """
    with stage(f"png_vectors › autotrace {Path(png_path).name}"):
        return [trace_vector_to_vector(v) for v in _trace_png(png_path, tmp_dir, dpi)]


def _trace_png(png_path: Path, tmp_dir: Path | None, dpi: int) -> list[TraceVector]:
    png_path = Path(png_path)

    if not png_path.is_file():
        raise FileNotFoundError(f"image not found: {png_path}")

    if tmp_dir is not None:
        work_dir_context = nullcontext(Path(tmp_dir))
    else:
        work_dir_context = tempfile.TemporaryDirectory()

    with work_dir_context as work_dir:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        svg_path = work_dir / f"{png_path.stem}.svg"
        pdf_path = work_dir / f"{png_path.stem}.pdf"

        png_to_svg(
            png_path,
            svg_path,
            color_count=2,
        )

        svg_to_pdf(
            svg_path,
            pdf_path,
        )

        vectors = pdf_to_vectors(
            pdf_path,
            page_index=0,
            dpi=dpi,
            split_strokes=True,
        )

    return vectors

def _extract_image_vectors_worker(
    job: tuple[
        int,
        Path,
        Path,
        int,
    ],
) -> tuple[int, list[Vector], tuple[int, int]]:
    """
    Extract vectors from one unique image xref.

    Returns:
        (
            imgxref,
            vectors,
            image_size,
        )
    """

    imgxref, png_path, tmp_dir, dpi = job

    vectors = _trace_png(png_path, tmp_dir, dpi)

    with Image.open(png_path) as image:
        image_size = image.size

        if "A" in image.getbands():
            alpha = np.array(image.getchannel("A"), dtype=np.uint8)

            vectors = _calculate_vector_transparency(
                vectors,
                alpha,
                tmp_dir=tmp_dir,
            )

    return imgxref, [trace_vector_to_vector(v) for v in vectors], image_size

def get_image_polys(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    tmp_dir: str | Path = "experiments/tmp_vectors/imgs",
    dpi: int = 300,
    workers: int = 1,
) -> dict[int, dict[tuple[float, float, float, float], list[Vector]]]:
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if workers < 1:
        raise ValueError("workers must be >= 1")

    # ---------------------------------------------------------
    # Collect unique image xrefs.
    # ---------------------------------------------------------

    image_xrefs = set()
    smask_xrefs = set()

    for imgobj in page.get_images():
        imgxref = imgobj[0]
        smask = imgobj[1]

        image_xrefs.add(imgxref)
        smask_xrefs.add(smask)

    image_xrefs = list(image_xrefs - smask_xrefs)

    # ---------------------------------------------------------
    # Prepare one job per unique image.
    # ---------------------------------------------------------

    jobs = []

    for imgxref in image_xrefs:
        img = doc.extract_image(imgxref)

        image_tmp_dir = tmp_dir / f"xref_{imgxref}"
        image_tmp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        png_path = image_tmp_dir / "image.png"

        image = pymupdf.Pixmap(img["image"])
        smask_xref = img["smask"]

        if smask_xref:
            smask = pymupdf.Pixmap(doc.extract_image(smask_xref)["image"])
            image = pymupdf.Pixmap(image, smask)

        image.pil_image().save(
            png_path,
            format="PNG",
        )

        jobs.append(
            (
                imgxref,
                png_path,
                image_tmp_dir,
                dpi,
            )
        )

    extracted: dict[
        int,
        tuple[
            list[Vector],
            tuple[int, int],
        ],
    ] = {}

    if workers == 1:
        results = (
            _extract_image_vectors_worker(job)
            for job in jobs
        )

        for imgxref, vectors, image_size in bar(results, desc="autotrace images", total=len(jobs), unit="image"):
            extracted[imgxref] = (
                vectors,
                image_size,
            )

    else:
        with ProcessPoolExecutor(
            max_workers=workers,
        ) as executor:
            results = executor.map(
                _extract_image_vectors_worker,
                jobs,
            )

            for imgxref, vectors, image_size in bar(results, desc="autotrace images", total=len(jobs), unit="image"):
                extracted[imgxref] = (
                    vectors,
                    image_size,
                )

    vector_dict: dict[
        int,
        dict[tuple[float, float, float, float], list[Vector]],
    ] = {}

    for imgxref in image_xrefs:
        vectors, image_size = extracted[imgxref]

        vectors_by_rect: dict[
            tuple[float, float, float, float],
            list[Vector],
        ] = {}

        image_placements = page.get_image_rects(
            imgxref,
            transform=True,
        )

        for rect, matrix in image_placements:
            rect_vectors = _transform_vectors_to_matrix(
                vectors,
                matrix,
                image_size,
                dpi,
            )

            rect_key = (
                rect.x0,
                rect.y0,
                rect.x1,
                rect.y1,
            )

            vectors_by_rect[rect_key] = rect_vectors

        vector_dict[imgxref] = vectors_by_rect

    return vector_dict




