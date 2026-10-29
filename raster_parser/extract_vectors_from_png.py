"""Autotrace PNGs (and a page's embedded images) into shared `Vector`s.

The tracing is `commons.v1_commons.autotrace`. An image's vectors are placed on the page through
the image's placement matrix (`page.get_image_rects(xref, transform=True)`).
"""

import tempfile
from dataclasses import replace
from pathlib import Path

from raster_parser.commons.helpers.geometry import PDF_POINTS_PER_INCH
from raster_parser.commons.helpers.pixmap import pixmap_to_array, zoom_matrix
from raster_parser.models.vector import Vector
from raster_parser.commons.v1_commons.autotrace import autotrace_drawing, traced_vector
from raster_parser.commons.v1_commons.render import draw_vectors
from raster_parser.commons.workers import spawn_pool

import pymupdf

from PIL import Image

import statistics as st
import numpy as np


def _calculate_vector_transparency(
    vectors: list[Vector],
    alpha: np.ndarray,
    tmp_dir: Path | None = None,
    dpi: int = 300,
) -> list[Vector]:
    """
    Set each vector's fill opacity from the image alpha channel: the most common alpha under the
    pixels the vector fills, as a fraction of 255.

    `vectors` must be in the generated PDF's own space traced at `dpi` (as `autotrace_drawing`
    returns them), and `alpha` is the image's alpha channel at that same resolution. Vectors
    that fill no pixel are kept unchanged; none is dropped. Each vector is rendered alone to a
    PDF under `tmp_dir` (a throwaway folder when None).
    """
    if tmp_dir is None:
        with tempfile.TemporaryDirectory(prefix="vector_alpha_") as scratch:
            return _calculate_vector_transparency(vectors, alpha, Path(scratch), dpi)

    height, width = alpha.shape

    result: list[Vector] = []

    for i, vector in enumerate(vectors):
        pdf_path = tmp_dir / f"vector{i}.pdf"

        draw_vectors(
            [vector],
            pdf_path,
            page_width=width * PDF_POINTS_PER_INCH / dpi,
            page_height=height * PDF_POINTS_PER_INCH / dpi,
        )

        with pymupdf.open(pdf_path) as doc:
            pix = doc[0].get_pixmap(
                matrix=zoom_matrix(dpi),
                colorspace=pymupdf.csGRAY,
                alpha=False,
            )
            fill_mask = pixmap_to_array(pix)[:, :, 0]

        # PDF background is white, vector is black.
        fill_mask = fill_mask < 128

        if not np.any(fill_mask):
            result.append(vector)
            continue

        vector_alpha = alpha[fill_mask]

        opaque_fraction = st.mode(vector_alpha) / 255.0

        result.append(traced_vector(replace(vector, fill_opacity=opaque_fraction)))

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
    image_width_pt = image_width * PDF_POINTS_PER_INCH / dpi
    image_height_pt = image_height * PDF_POINTS_PER_INCH / dpi
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
        PNG -> SVG -> PDF -> MuPDF drawings -> Vector

    The returned vectors are in the coordinate system of the generated
    vector PDF and are not transformed to any external bounding box.
    """
    return _trace_png(png_path, tmp_dir, dpi)


def _trace_png(png_path: Path, tmp_dir: Path | None, dpi: int) -> list[Vector]:
    """Autotrace `png_path`; with `tmp_dir` the SVG / PDF intermediates are kept there."""
    png_path = Path(png_path)
    if not png_path.is_file():
        raise FileNotFoundError(f"image not found: {png_path}")
    if tmp_dir is None:
        return autotrace_drawing(png_path, dpi=dpi)
    work_dir = Path(tmp_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    return autotrace_drawing(
        png_path,
        dpi=dpi,
        svg_out=work_dir / f"{png_path.stem}.svg",
        pdf_out=work_dir / f"{png_path.stem}.pdf",
    )


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
                dpi=dpi,
            )

    return imgxref, vectors, image_size


def get_image_polys(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    tmp_dir: str | Path | None = None,
    dpi: int = 300,
    workers: int = 1,
) -> dict[int, dict[tuple[float, float, float, float], list[Vector]]]:
    """Vectors of every embedded image on `page`, per image xref and placement rectangle.

    `tmp_dir` keeps the per-image PNG / SVG / PDF intermediates there; by default they live in a
    throwaway folder. `workers > 1` traces the images in spawned worker processes.
    """
    if tmp_dir is None:
        with tempfile.TemporaryDirectory(prefix="image_polys_") as scratch:
            return get_image_polys(doc, page, scratch, dpi, workers)

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
        results = (_extract_image_vectors_worker(job) for job in jobs)

        for imgxref, vectors, image_size in results:
            extracted[imgxref] = (
                vectors,
                image_size,
            )

    else:
        with spawn_pool(workers) as executor:
            results = executor.map(
                _extract_image_vectors_worker,
                jobs,
            )

            for imgxref, vectors, image_size in results:
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
