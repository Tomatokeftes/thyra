# msiconvert/core/base_reader.py
from abc import ABC, abstractmethod
import numpy as np
import logging
from typing import Dict, Any, Tuple, Generator, Optional

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
    def get_common_mass_axis(self) -> np.ndarray:
        """
        Return the common mass axis for all spectra.
        
        This should include all unique m/z values across the entire dataset,
        ensuring complete accuracy without approximation or interpolation.
        """
        pass
    
    @abstractmethod
    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
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
    def map_mz_to_common_axis(mzs: np.ndarray, intensities: np.ndarray, common_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map m/z values to indices in the common mass axis with high accuracy.
        
        This method ensures exact mapping of m/z values to the common mass axis
        without interpolation, preserving the original intensity values.
        
        Args:
            mzs: Array of m/z values
            intensities: Array of intensity values
            common_axis: Common mass axis (sorted array of unique m/z values)
        
        Returns:
            Tuple of (indices in common mass axis, corresponding intensities)
        """
        if mzs.size == 0 or intensities.size == 0:
            return np.array([], dtype=int), np.array([])
            
        # Use searchsorted to find indices in common mass axis
        indices = np.searchsorted(common_axis, mzs)
        
        # Ensure indices are within bounds
        # This is safe because we're not changing the values, just ensuring valid indexing
        indices = np.clip(indices, 0, len(common_axis) - 1)
        
        # Verify that we're actually finding the right m/z values
        # If the m/z value differs too much, we might want to consider it as not found
        # This is important for maintaining accuracy
        max_diff = 1e-6  # A very small tolerance threshold for floating point differences
        indices_valid = np.abs(common_axis[indices] - mzs) <= max_diff
        
        # Return only the valid indices and their corresponding intensities
        return indices[indices_valid], intensities[indices_valid]
    
    @abstractmethod
    def close(self) -> None:
        """Close all open file handles."""
        pass