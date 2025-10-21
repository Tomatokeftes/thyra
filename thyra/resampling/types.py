"""Data types and enums for the resampling module."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np
import numpy.typing as npt


class ResamplingMethod(Enum):
    """Available resampling methods."""

    NONE = "none"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    TIC_PRESERVING = "tic_preserving"
    LINEAR_INTERPOLATION = "linear_interpolation"


class AxisType(Enum):
    """Mass axis spacing characteristics."""

    CONSTANT = "constant"  # Equidistant spacing
    LINEAR_TOF = "linear_tof"  # sqrt(m/z) relationship
    REFLECTOR_TOF = "reflector_tof"  # Linear m/z relationship
    ORBITRAP = "orbitrap"  # m/z^(3/2) relationship
    FTICR = "fticr"  # m/z^2 relationship
    UNKNOWN = "unknown"


@dataclass
class MassAxis:
    """Represents a mass axis with metadata."""

    mz_values: npt.NDArray[np.floating[Any]]
    min_mz: float
    max_mz: float
    num_bins: int
    axis_type: AxisType

    @property
    def spacing(self) -> npt.NDArray[np.floating[Any]]:
        """Calculate spacing between consecutive m/z values."""
        return np.diff(self.mz_values)

    def resolution_at(self, mz: float) -> float:
        """Calculate resolution at given m/z."""
        idx = int(np.searchsorted(self.mz_values, mz))
        if idx > 0 and idx < len(self.mz_values):
            delta_mz = float(self.mz_values[idx] - self.mz_values[idx - 1])
            return mz / delta_mz
        return 0.0


@dataclass
class ResamplingConfig:
    """Configuration for resampling operations."""

    method: Optional[ResamplingMethod] = None  # Auto-detect from instrument
    axis_type: Optional[AxisType] = None  # Auto-detect from instrument
    target_bins: Optional[int] = None  # Number of target bins
    mass_width_da: Optional[float] = None  # Mass width in Da at reference
    reference_mz: float = 500.0  # Reference m/z for width spec
    min_mz: Optional[float] = None  # Override mass range
    max_mz: Optional[float] = None  # Override mass range
