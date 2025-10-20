"""Linear/uniform mass axis generator with constant spacing."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import AxisType, MassAxis
from .base_generator import BaseAxisGenerator


class LinearAxisGenerator(BaseAxisGenerator):
    """Mass axis generator for uniform/constant spacing.

    Creates equidistant m/z bins with constant spacing across the entire
    range. This is the simplest and most common approach for mass axis
    generation.
    """

    def generate_axis(self, min_mz: float, max_mz: float, target_bins: int) -> MassAxis:
        """Generate uniform mass axis with constant spacing.

        Parameters
        ----------
        min_mz : float
            Minimum m/z value
        max_mz : float
            Maximum m/z value
        target_bins : int
            Number of bins

        Returns
        -------
        MassAxis
            Generated mass axis with uniform spacing
        """
        mz_values = np.linspace(min_mz, max_mz, target_bins)

        return MassAxis(
            mz_values=mz_values,
            min_mz=float(mz_values[0]),
            max_mz=float(mz_values[-1]),
            num_bins=len(mz_values),
            axis_type=AxisType.CONSTANT,
        )

    def get_axis_type(self) -> AxisType:
        """Return the axis type for uniform spacing."""
        return AxisType.CONSTANT

    def generate_axis_bins(
        self, min_mz: float, max_mz: float, num_bins: int
    ) -> npt.NDArray[np.floating[Any]]:
        """Generate uniform axis with fixed number of bins."""
        return np.linspace(min_mz, max_mz, num_bins)

    def generate_axis_width(
        self,
        min_mz: float,
        max_mz: float,
        width_da: float,
        reference_mz: float = 500.0,
    ) -> npt.NDArray[np.floating[Any]]:
        """Generate uniform axis based on constant mass width."""
        # For uniform spacing, width is constant across the range
        num_bins = int((max_mz - min_mz) / width_da) + 1
        return np.linspace(min_mz, max_mz, num_bins)
