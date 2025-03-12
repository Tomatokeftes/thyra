# msiconvert/core/base_reader.py
from abc import ABC, abstractmethod
import numpy as np
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
        """Return the common mass axis for all spectra."""
        pass
    
    @abstractmethod
    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """
        Iterate through spectra with progress tracking.
        
        Args:
            batch_size: Optional batch size for spectrum iteration
        
        Yields:
            Tuple containing:
                - Coordinates (x, y, z)
                - Indices in common mass axis
                - Intensity values array
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close all open file handles."""
        pass
    
    @staticmethod
    def map_mz_to_common_axis(mzs: np.ndarray, intensities: np.ndarray, common_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map m/z values to indices in the common mass axis.
        
        Args:
            mzs: Array of m/z values
            intensities: Array of intensity values
            common_axis: Precomputed common mass axis
        
        Returns:
            Tuple of (indices in common mass axis, corresponding intensities)
        """
        if mzs.size == 0 or intensities.size == 0:
            return np.array([], dtype=int), np.array([])
            
        # Use searchsorted to find indices in common mass axis
        indices = np.searchsorted(common_axis, mzs)
        
        # Ensure indices are within bounds
        indices = np.clip(indices, 0, len(common_axis) - 1)
        
        return indices, intensities
