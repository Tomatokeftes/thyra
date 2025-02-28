# msiconvert/readers/imzml_reader.py
import numpy as np
from typing import Dict, Any, Tuple, Generator, Optional
from pathlib import Path
from pyimzml.ImzMLParser import ImzMLParser

from ..core.base_reader import BaseMSIReader

class ImzMLReader(BaseMSIReader):
    """Reader for imzML format files."""
    
    def __init__(self, imzml_path: Path):
        self.imzml_path = imzml_path
        self.ibd_path = imzml_path.with_suffix('.ibd')
        
        if not self.ibd_path.exists():
            raise ValueError(f"Corresponding .ibd file not found for {imzml_path}")
        
        self.ibd_file = open(self.ibd_path, mode="rb")
        self.parser = ImzMLParser(
            filename=str(imzml_path),
            parse_lib="lxml",
            ibd_file=self.ibd_file
        )
        
        # Determine file mode
        self.is_continuous = "continuous" in self.parser.metadata.file_description.param_by_name
        self.is_processed = "processed" in self.parser.metadata.file_description.param_by_name
        
        if self.is_continuous == self.is_processed:
            raise ValueError("Invalid file mode, expected either 'continuous' or 'processed'.")
        
        # Calculate common mass axis
        self._common_mass_axis = None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the imzML dataset."""
        return {
            'source': str(self.imzml_path),
            'uuid': self.parser.metadata.file_description.cv_params[0][2],
            'file_mode': 'continuous' if self.is_continuous else 'processed'
        }
    
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Return the dimensions of the imzML dataset (x, y, z)."""
        x_max = self.parser.imzmldict['max count of pixels x']
        y_max = self.parser.imzmldict['max count of pixels y']
        z_max = 1  # imzML is typically 2D; set to 1 for compatibility
        return (x_max, y_max, z_max)
    
    def get_common_mass_axis(self) -> np.ndarray:
        """Return the common mass axis for all spectra."""
        if self._common_mass_axis is None:
            if self.is_continuous:
                # Continuous data: all pixels have the same m/z values
                self._common_mass_axis = self.parser.getspectrum(0)[0]
            else:
                # Processed data: collect all unique m/z values
                all_mz_values = np.concatenate([
                    self.parser.getspectrum(idx)[0] 
                    for idx in range(len(self.parser.coordinates))
                ])
                self._common_mass_axis = np.unique(all_mz_values)
        
        return self._common_mass_axis
    
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
        for idx, (x, y, _) in enumerate(self.parser.coordinates):
            try:
                mz_array, intensity_array = self.parser.getspectrum(idx)
                
                # Adjust coordinates to 0-based for internal use
                x_adj = x - 1
                y_adj = y - 1
                z = 0  # imzML is typically 2D
                
                yield ((x_adj, y_adj, z), mz_array, intensity_array)
            except Exception as err:
                print(f"Error processing spectrum {idx} at pixel ({x}, {y}): {err}")
    
    def close(self) -> None:
        """Close all open file handles."""
        if hasattr(self, 'ibd_file') and self.ibd_file:
            self.ibd_file.close()
            self.ibd_file = None
        
        if hasattr(self, 'parser') and hasattr(self.parser, 'm') and self.parser.m:
            self.parser.m.close()