"""
Base reader interface for MSI data.

This module provides the abstract base class that all MSI readers must implement
to maintain compatibility with the spatialdata_converter.py interface.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np


class BaseMSIReader(ABC):
    """
    Abstract base class for MSI (Mass Spectrometry Imaging) data readers.

    This interface ensures compatibility with the spatialdata_converter.py and
    provides a consistent API for all reader implementations.
    """

    def __init__(self, data_path: Path, **kwargs):
        """
        Initialize the reader with the path to the data.

        Args:
            data_path: Path to the data file or directory
            **kwargs: Additional reader-specific parameters
        """
        self.data_path = Path(data_path)

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about the dataset.

        Returns:
            Dict containing metadata information including:
                - source: str - path to the data source
                - Any other format-specific metadata
        """
        pass

    @abstractmethod
    def get_dimensions(self) -> Tuple[int, int, int]:
        """
        Return the dimensions of the MSI dataset using 0-based indexing.

        Returns:
            Tuple of (x, y, z) dimensions where:
                - x: number of pixels in x dimension
                - y: number of pixels in y dimension
                - z: number of pixels in z dimension (1 for 2D data)
        """
        pass

    @abstractmethod
    def get_common_mass_axis(self) -> np.ndarray:
        """
        Return the common mass axis composed of all unique m/z values.

        This should collect all unique m/z values across the dataset to create
        an accurate common mass axis for sparse matrix construction.

        Returns:
            np.ndarray: Array of unique m/z values in ascending order
        """
        pass

    @abstractmethod
    def iter_spectra(
        self, batch_size: Optional[int] = None
    ) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """
        Iterate through all spectra in the dataset with optional batching.

        This method should yield spectra one by one for memory-efficient processing.
        All coordinates should use 0-based indexing for internal consistency.

        Args:
            batch_size: Optional batch size for processing optimization

        Yields:
            Tuple containing:
                - Tuple[int, int, int]: Coordinates (x, y, z) using 0-based indexing
                - np.ndarray: m/z values array
                - np.ndarray: Intensity values array
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close all open file handles and connections.

        This method should clean up any resources used by the reader.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
