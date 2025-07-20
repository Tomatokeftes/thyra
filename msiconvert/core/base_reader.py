# msiconvert/core/base_reader.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class BaseMSIReader(ABC):
    """Abstract base class for reading MSI data formats."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the MSI dataset."""
        pass

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Return the dimensions of the MSI dataset (x, y, z)."""
        pass

    @abstractmethod
    def get_common_mass_axis(self) -> NDArray[np.float64]:
        """
        Return the common mass axis for all spectra.

        This method must always return a valid array.
        If no common mass axis can be created, implementations should raise an exception.
        """
        pass

    @abstractmethod
    def iter_spectra(
        self, batch_size: Optional[int] = None
    ) -> Generator[
        Tuple[Tuple[int, int, int], NDArray[np.float64], NDArray[np.float64]],
        None,
        None,
    ]:
        """
        Iterate through spectra with optional batch processing.

        Args:
            batch_size: Optional batch size for spectrum iteration

        Yields:
            Tuple containing:

                - Coordinates (x, y, z) using 0-based indexing
                - m/z values array

                - Intensity values array
        """
        pass

    @staticmethod
    def map_mz_to_common_axis(
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
        common_axis: NDArray[np.float64],
    ) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
        """
        Map m/z values to indices in the common mass axis with high accuracy.

        This method ensures exact mapping of m/z values to the common mass axis
        without interpolation, preserving the original intensity values.

        Args:
            mzs: NDArray[np.float64] - Array of m/z values
            intensities: NDArray[np.float64] - Array of intensity values
            common_axis: NDArray[np.float64] - Common mass axis (sorted array of unique m/z values)

        Returns:
            Tuple of (indices in common mass axis, corresponding intensities)
        """
        if mzs.size == 0 or intensities.size == 0:
            return np.array([], dtype=int), np.array([])

        # Use searchsorted to find indices in common mass axis
        indices = np.searchsorted(common_axis, mzs)

        # Ensure indices are within bounds
        indices = np.clip(indices, 0, len(common_axis) - 1)

        # Verify that we're actually finding the right m/z values
        max_diff = (
            1e-6  # A very small tolerance threshold for floating point differences
        )
        indices_valid = np.abs(common_axis[indices] - mzs) <= max_diff

        # Return only the valid indices and their corresponding intensities
        return indices[indices_valid], intensities[indices_valid]

    @abstractmethod
    def close(self) -> None:
        """Close all open file handles."""
        pass

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Return the shape of the dataset (x, y, z dimensions).

        Returns:
            Tuple of (x, y, z) dimensions
        """
        return self.get_dimensions()

    @property
    def n_spectra(self) -> int:
        """
        Return the total number of spectra in the dataset.

        Returns:
            Total number of spectra
        """
        # Default implementation counts actual spectra
        count = 0
        for _ in self.iter_spectra():
            count += 1
        return count

    @property
    def mass_range(self) -> Tuple[float, float]:
        """
        Return the mass range (min_mass, max_mass) of the dataset.

        Returns:
            Tuple of (min_mass, max_mass) values
        """
        mass_axis = self.get_common_mass_axis()
        if len(mass_axis) == 0:
            return (0.0, 0.0)
        return (float(np.min(mass_axis)), float(np.max(mass_axis)))

    def get_pixel_size(self) -> Optional[Tuple[float, float]]:
        """
        Extract pixel size from format-specific metadata.

        Returns:
            Optional[Tuple[float, float]]: Pixel size as (x_size, y_size) in micrometers,
                                         or None if not available in metadata.

        Notes:
            - For ImzML: Extracts from cvParam IMS:1000046 and IMS:1000047
            - For Bruker: Extracts from MaldiFrameLaserInfo table (BeamScanSizeX/Y)
            - Default implementation returns None (no automatic detection)
        """
        return None
