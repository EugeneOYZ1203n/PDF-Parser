from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import io
from pathlib import Path
import tempfile
from contextlib import nullcontext

from raster_parser.conversion.png_to_vectors.png_to_svg import png_to_svg
from raster_parser.conversion.png_to_vectors.svg_to_pdf import svg_to_pdf
from raster_parser.rendering.pdf_render.dto_reconstruct_pipeline import draw_vectors
from raster_parser.helpers.geometry import PDF_POINTS_PER_INCH
from raster_parser.models import Vector
from raster_parser.serialisation.deserialization import deserialize_vector
from raster_parser.autotrace.drawing_autotrace import (
    pdf_to_vectors,
)
from raster_parser.models import VectorDTO

import pymupdf
from PIL import Image

import statistics as st
import numpy as np

TRANSPARENCY_THRESHOLD = 0.8
# A vector rendered black-on-white covers a pixel when its gray value is below mid-gray.
COVERED_GRAY_MAX = 128


def _calculate_vector_transparency(
    vectors: list[VectorDTO],
    alpha: np.ndarray,
    tmp_dir: Path | None = None,
    dpi: int = 300,
) -> list[VectorDTO]:
    """
    Calculate fill opacity for each vector using the image alpha channel.

    Vectors with less than 5% average opacity are dropped.
    Vectors without filled pixels are kept unchanged.
    """
    height, width = alpha.shape

    result: list[VectorDTO] = []

    for i, vector_dto in enumerate(vectors):
        vector = deserialize_vector(vector_dto)

        pdf_path = tmp_dir / f"vector{i}.pdf"

        draw_vectors(
            [vector],
            pdf_path,
            page_width=width * PDF_POINTS_PER_INCH / dpi,
            page_height=height * PDF_POINTS_PER_INCH / dpi,
        )

        doc = pymupdf.open(pdf_path)
        page = doc[0]

        pix = page.get_pixmap(
            matrix=pymupdf.Matrix(dpi / PDF_POINTS_PER_INCH, dpi / PDF_POINTS_PER_INCH),
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
        fill_mask = fill_mask < COVERED_GRAY_MAX

        if not np.any(fill_mask):
            result.append(vector_dto)
            continue

        vector_alpha = alpha[fill_mask]

        opaque_fraction = st.mode(vector_alpha) / 255.0

        vector_dto.fill_opacity = opaque_fraction
        result.append(vector_dto)

    return result


def _transform_vectors_to_matrix(
    vectors: list[VectorDTO],
    matrix: pymupdf.Matrix,
    image_size: tuple[int, int],
    dpi: int,
) -> None:
    """Transform image vectors into page coordinates using a PyMuPDF matrix.

    ``matrix`` is the image's placement transform from ``page.get_image_rects(xref,
    transform=True)``: it maps the image's unit square (0..1 on both axes) onto the page.
    Item ops: ``"l"`` line, ``"c"`` cubic Bezier, ``"re"`` rectangle, ``"qu"`` quad.
    """

    image_width, image_height = image_size

    # VectorDTO coordinates are in PDF points, not PNG pixels.
    image_width_pt = image_width * PDF_POINTS_PER_INCH / dpi
    image_height_pt = image_height * PDF_POINTS_PER_INCH / dpi

    def transform_point(point: list[float]) -> list[float]:
        # Convert generated-PDF coordinates into the unit image square.
        p = pymupdf.Point(
            point[0] / image_width_pt,
            point[1] / image_height_pt,
        )

        # Map unit-image coordinates to page coordinates.
        p *= matrix

        return [p.x, p.y]

    def transform_rect(
        r: list[float],
    ) -> list[float]:
        # Transform all four corners because the image may be rotated.
        points = [
            transform_point([r[0], r[1]]),
            transform_point([r[2], r[1]]),
            transform_point([r[0], r[3]]),
            transform_point([r[2], r[3]]),
        ]

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        return [
            min(xs),
            min(ys),
            max(xs),
            max(ys),
        ]

    for vector in vectors:
        vector.x0, vector.y0, vector.x1, vector.y1 = transform_rect(
            [
                vector.x0,
                vector.y0,
                vector.x1,
                vector.y1,
            ]
        )

        transformed_items = []

        for item in vector.items:
            op = item["op"]
            args = item["args"]

            if op == "l":
                transformed_items.append(
                    {
                        "op": op,
                        "args": [
                            transform_point(args[0]),
                            transform_point(args[1]),
                        ],
                    }
                )

            elif op == "c":
                transformed_items.append(
                    {
                        "op": op,
                        "args": [transform_point(point) for point in args],
                    }
                )

            elif op == "re":
                transformed_items.append(
                    {
                        "op": op,
                        "args": [
                            transform_rect(args[0]),
                            *args[1:],
                        ],
                    }
                )

            elif op == "qu":
                quad = args[0]

                transformed_items.append(
                    {
                        "op": op,
                        "args": [
                            [transform_point(point) for point in quad],
                            *args[1:],
                        ],
                    }
                )

            else:
                transformed_items.append(item)

        vector.items = transformed_items


def extract_vectors_from_png(
    png_path: Path,
    tmp_dir: Path | None = None,
    dpi: int = 300,
) -> list[VectorDTO]:
    """Autotrace a PNG and return its non-black vectors.

    Pipeline:
        PNG -> SVG -> PDF -> MuPDF drawings -> Vector -> VectorDTO

    The returned vectors are in the coordinate system of the generated
    vector PDF and are not transformed to any external bounding box.
    """
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

    return [VectorDTO.from_vector(vector) for vector in vectors]


def _extract_image_vectors_worker(
    job: tuple[
        int,
        Path,
        Path,
        int,
    ],
) -> tuple[int, list[VectorDTO], tuple[int, int]]:
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

    vectors = extract_vectors_from_png(
        png_path=png_path,
        tmp_dir=tmp_dir,
        dpi=dpi,
    )

    with Image.open(png_path) as image:
        image_size = image.size

        if "A" in image.getbands():
            alpha = np.array(image.getchannel("A"), dtype=np.uint8)

            vectors = _calculate_vector_transparency(
                vectors,
                alpha,
                tmp_dir=tmp_dir,
            )

    return imgxref, vectors, image_size


def get_image_polys(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    tmp_dir: str | Path = "experiments/tmp_vectors/imgs",
    dpi: int = 300,
    workers: int = 1,
) -> dict[int, dict[tuple[float, float, float, float], list[VectorDTO]]]:
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if workers < 1:
        raise ValueError("workers must be >= 1")

    # ---------------------------------------------------------
    # Collect unique image xrefs.
    # ---------------------------------------------------------

    # An xref is a PDF object's number in the file's cross-reference table; images are
    # identified by it. An image's smask (soft mask) is a second grayscale image holding its
    # alpha channel, stored as its own xref; masks are merged into their image below, never
    # traced on their own.
    image_xrefs = set()
    smask_xrefs = set()

    # page.get_images() rows start (xref, smask_xref, width, height, ...).
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
            list[VectorDTO],
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
        with ProcessPoolExecutor(
            max_workers=workers,
        ) as executor:
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
        dict[tuple[float, float, float, float], list[VectorDTO]],
    ] = {}

    for imgxref in image_xrefs:
        vectors, image_size = extracted[imgxref]

        vectors_by_rect: dict[
            tuple[float, float, float, float],
            list[VectorDTO],
        ] = {}

        image_placements = page.get_image_rects(
            imgxref,
            transform=True,
        )

        for rect, matrix in image_placements:
            rect_vectors = deepcopy(vectors)

            _transform_vectors_to_matrix(
                rect_vectors,
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
