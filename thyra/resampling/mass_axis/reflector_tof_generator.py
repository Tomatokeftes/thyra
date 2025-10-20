"""Reflector TOF mass axis generator with bin size ∝ m/z spacing."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ..types import AxisType, MassAxis
from .base_generator import BaseAxisGenerator


class ReflectorTOFAxisGenerator(BaseAxisGenerator):
    """Mass axis generator for Reflector Time-of-Flight analyzers (like timsTOF).

    Reflector TOF has bin size ∝ m/z, providing constant relative
    resolution (R = m/Δm). This is optimal for most MS applications as
    it maintains constant relative mass accuracy across the entire mass
    range.
    """

    def generate_axis(
        self,
        min_mz: float,
        max_mz: float,
        target_bins: int,
        reference_mz: float = 500.0,
        reference_width: float = 0.1,
    ) -> MassAxis:
        """Generate Reflector TOF mass axis with m/z-proportional spacing.

        Parameters
        ----------
        min_mz : float
            Minimum m/z value
        max_mz : float
            Maximum m/z value
        target_bins : int
            Target number of bins
        reference_mz : float
            Reference m/z for width specification (default: 500.0)
        reference_width : float
            Mass width at reference m/z (default: 0.1)

        Returns
        -------
        MassAxis
            Generated mass axis with Reflector TOF spacing
        """
        # For Reflector TOF: bin_width = k * mz
        # where k is the relative resolution (constant)
        # relative_resolution = reference_width / reference_mz  # Used for scaling

        # Generate axis by solving: integral of 1/mz from min_mz to mz = target_position
        # Integral: ln(mz) - ln(min_mz) = target_position * (ln(max_mz) - ln(min_mz)) / target_bins

        ln_min = np.log(min_mz)
        ln_max = np.log(max_mz)
        # ln_range = ln_max - ln_min  # Used for scaling

        # Create uniform grid in ln(mz) space
        ln_values = np.linspace(ln_min, ln_max, target_bins + 1)
        mz_values = np.exp(ln_values)

        # Use bin centers
        mz_centers = (mz_values[:-1] + mz_values[1:]) / 2

        return MassAxis(
            mz_values=mz_centers,
            min_mz=float(mz_centers[0]),
            max_mz=float(mz_centers[-1]),
            num_bins=len(mz_centers),
            axis_type=AxisType.REFLECTOR_TOF,
        )

    def calculate_width_at_mz(
        self,
        mz: float,
        reference_mz: float = 500.0,
        reference_width: float = 0.1,
    ) -> float:
        """Calculate expected bin width at given m/z for Reflector TOF.

        Parameters
        ----------
        mz : float
            Target m/z value
        reference_mz : float
            Reference m/z for width specification
        reference_width : float
            Width at reference m/z

        Returns
        -------
        float
            Expected bin width at target m/z
        """
        return reference_width * (mz / reference_mz)

    def get_axis_type(self) -> AxisType:
        """Return the axis type for Reflector TOF."""
        return AxisType.REFLECTOR_TOF

    def generate_axis_bins(
        self, min_mz: float, max_mz: float, num_bins: int
    ) -> npt.NDArray[np.floating[Any]]:
        """Generate Reflector TOF axis with fixed number of bins."""
        axis = self.generate_axis(min_mz, max_mz, num_bins)
        return axis.mz_values

    def generate_axis_width(
        self,
        min_mz: float,
        max_mz: float,
        width_da: float,
        reference_mz: float = 500.0,
    ) -> npt.NDArray[np.floating[Any]]:
        """Generate Reflector TOF axis based on mass width at reference m/z."""
        # For Reflector TOF: constant relative resolution R = m/Δm
        # relative_resolution = reference_mz / width_da  # Used for scaling

        # Generate axis in log space for constant relative resolution
        ln_min = np.log(min_mz)
        ln_max = np.log(max_mz)

        # Calculate number of bins needed for desired resolution
        ln_step = width_da / reference_mz  # Δ(ln(m)) ≈ Δm/m for small Δm
        num_bins = int((ln_max - ln_min) / ln_step) + 1

        ln_values = np.linspace(ln_min, ln_max, num_bins)
        return np.exp(ln_values)
