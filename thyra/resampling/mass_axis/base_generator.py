"""Abstract base class for mass axis generators."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt


class BaseAxisGenerator(ABC):
    """Abstract base class for mass axis generators."""

    @abstractmethod
    def generate_axis_bins(
        self, min_mz: float, max_mz: float, num_bins: int
    ) -> npt.NDArray[np.floating[Any]]:
        """Generate axis with fixed number of bins.

        Parameters
        ----------
        min_mz : float
            Minimum m/z value
        max_mz : float
            Maximum m/z value
        num_bins : int
            Number of bins to generate

        Returns
        -------
        np.ndarray
            Generated mass axis
        """
        pass

    @abstractmethod
    def generate_axis_width(
        self,
        min_mz: float,
        max_mz: float,
        width_da: float,
        reference_mz: float = 500.0,
    ) -> npt.NDArray[np.floating[Any]]:
        """Generate axis based on mass width at reference m/z using analyzer physics.

        Parameters
        ----------
        min_mz : float
            Minimum m/z value
        max_mz : float
            Maximum m/z value
        width_da : float
            Mass width in Da at reference m/z
        reference_mz : float
            Reference m/z for width specification

        Returns
        -------
        np.ndarray
            Generated mass axis with physics-based spacing
        """
        pass
