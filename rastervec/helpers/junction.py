"""JunctionDetector helper: interface-only stub for a future phase.

Trains and runs a small CNN that finds "junction" points in line diagrams
(a center point with several black lines radiating out at random angles,
on a white background) and derives line connections/widths from them. Not
implemented yet; method bodies intentionally raise NotImplementedError. No
torch/numpy imports yet -- deferred to when this is actually implemented.
"""
from __future__ import annotations

from rastervec.models import JunctionPoint, LineVector


class JunctionDetector:
    """Synthetic-data CNN training + inference for junction detection,
    plus the geometric line-connection and line-width steps that consume
    its output."""

    def generate_synthetic_data(
        self, n: int
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Generate n synthetic training samples: a black center point
        with a random number of black lines radiating out at random
        angles, on a white background, plus their junction labels."""
        raise NotImplementedError

    def train(self, data: "np.ndarray", labels: "np.ndarray") -> "torch.nn.Module":
        """Train the junction-detector CNN on synthetic data."""
        raise NotImplementedError

    def infer(self, image: "np.ndarray") -> list[JunctionPoint]:
        """Run the trained CNN over an image to find junction points."""
        raise NotImplementedError

    def probe_directions(
        self, image: "np.ndarray", point: JunctionPoint
    ) -> list[float]:
        """Probe 360 degrees around a junction point to find the angles
        of lines radiating from it."""
        raise NotImplementedError

    def connect(
        self, junctions: list[JunctionPoint], image: "np.ndarray"
    ) -> list[LineVector]:
        """Connect junction points along their probed directions to the
        closest other junction that is on (or nearly on) that line."""
        raise NotImplementedError

    def measure_width(
        self, image: "np.ndarray", point: JunctionPoint, direction: float
    ) -> float:
        """Expand a sampling radius outward from a junction along a
        direction until circumference-sampled points are no longer the
        line's color; the radius at that point is the line width."""
        raise NotImplementedError
