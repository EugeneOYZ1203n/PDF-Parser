"""Phase 4: final output -- the combined `(texts, vectors)` lists
(`organize.organize_outputs`) and the final reconstructed PDF
(`render.render_output_pdf`)."""
from rastervec.P4_Output_Organization.organize import organize_outputs
from rastervec.P4_Output_Organization.render import (
    output_specs,
    render_output_page,
    render_output_pdf,
)

__all__ = ["organize_outputs", "output_specs", "render_output_page", "render_output_pdf"]
