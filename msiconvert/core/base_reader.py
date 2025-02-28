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
    def iter_spectra(self) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """
        Iterate through spectra.
        
        Yields:
        -------
        Tuple containing:
            - Coordinates (x, y, z)
            - m/z values array
            - Intensity values array
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close all open file handles."""
        pass