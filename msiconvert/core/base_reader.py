# msiconvert/core/base_reader.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ..metadata.types import ComprehensiveMetadata, EssentialMetadata

if TYPE_CHECKING:
    from .base_extractor import MetadataExtractor


class BaseMSIReader(ABC):
    """Abstract base class for reading MSI data formats."""

    def __init__(self, data_path: Path, **kwargs):
        """
        Initialize the reader with the path to the data.

        Args:
            data_path: Path to the data file or directory
            **kwargs: Additional reader-specific parameters
        """
        self.data_path = Path(data_path)
        self._metadata_extractor: Optional["MetadataExtractor"] = None

    @abstractmethod
    def _create_metadata_extractor(self) -> "MetadataExtractor":
        """Create format-specific metadata extractor."""
        pass

    @property
    def metadata_extractor(self) -> "MetadataExtractor":
        """Lazy-loaded metadata extractor."""
        if self._metadata_extractor is None:
            self._metadata_extractor = self._create_metadata_extractor()
        return self._metadata_extractor

    def get_essential_metadata(self) -> EssentialMetadata:
        """Get essential metadata for processing."""
        return self.metadata_extractor.get_essential()

    def get_comprehensive_metadata(self) -> ComprehensiveMetadata:
        """Get complete metadata."""
        return self.metadata_extractor.get_comprehensive()

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

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
