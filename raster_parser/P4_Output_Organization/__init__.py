"""Phase 4: combine every phase's output (`organize.organize_outputs`), plus the final
reconstructed page for tools that want to look at it (`render.render_output_pdf`)."""
from raster_parser.P4_Output_Organization.organize import organize_outputs
from raster_parser.P4_Output_Organization.render import render_output_page, render_output_pdf

__all__ = ["organize_outputs", "render_output_page", "render_output_pdf"]
