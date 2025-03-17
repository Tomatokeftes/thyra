# msiconvert/readers/imzml_reader.py
import numpy as np
from typing import Dict, Any, Tuple, Generator, Optional, Union
from pathlib import Path
from pyimzml.ImzMLParser import ImzMLParser
from tqdm import tqdm

from ..core.base_reader import BaseMSIReader

from ..core.registry import register_reader

@register_reader('imzml')
class ImzMLReader(BaseMSIReader):
    """Reader for imzML format files."""
    
    def __init__(self, imzml_path: Optional[Union[str, Path]] = None):
        """
        Initialize an ImzML reader.
        
        Parameters:
        -----------
        imzml_path : str or Path, optional
            Path to the imzML file. If not provided, can be set later using read().
        """
        super().__init__() # Call parent constructor with no arguments
        self.filepath = imzml_path # Store filepath as instance variable
        
        # Only initialize the parser if a path is provided
        if imzml_path is not None:
            self._initialize_parser(imzml_path)
            
    def _initialize_parser(self, imzml_path):
        """Initialize the ImzML parser with the given path."""
        if isinstance(imzml_path, str):
            imzml_path = Path(imzml_path)
        
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
        if self.parser.metadata is None:
            raise ValueError("Failed to parse metadata from imzML file.")
            
        # Determine file mode
        self.is_continuous = "continuous" in self.parser.metadata.file_description.param_by_name
        self.is_processed = "processed" in self.parser.metadata.file_description.param_by_name
        
        if self.is_continuous == self.is_processed:
            raise ValueError("Invalid file mode, expected either 'continuous' or 'processed'.")
        
        # Calculate common mass axis
        self._common_mass_axis = None
    
    def can_read(self, filepath: str) -> bool:
        """Check if this reader can read the given file."""
        return isinstance(filepath, str) and filepath.lower().endswith('.imzml')
    
    def read(self, filepath=None):
        """Read the imzML file and return parsed data."""
        if filepath is not None:
            self._initialize_parser(filepath)
        elif self.filepath is not None and not hasattr(self, 'parser'):
            self._initialize_parser(self.filepath)
        elif not hasattr(self, 'parser'):
            raise ValueError("No filepath provided to read method and no parser initialized")
        
        # Read the data from the file
        coordinates = []
        intensities_list = []
        mzs = self.get_common_mass_axis()
        
        for (x, y, z), spectrum_mzs, spectrum_intensities in self.iter_spectra():
            coordinates.append((x, y))
            
            # Process intensity values
            if self.is_continuous:
                intensities_list.append(spectrum_intensities)
            else:
                # For processed data, we need to remap to the common mass axis
                intensities = np.zeros_like(mzs)
                # Simple nearest-neighbor mapping for now
                for i, mz in enumerate(spectrum_mzs):
                    idx = np.abs(mzs - mz).argmin()
                    intensities[idx] = spectrum_intensities[i]
                intensities_list.append(intensities)
        
        # Convert list to array
        intensities = np.vstack(intensities_list)
        
        # Return the data dictionary
        dimensions = self.get_dimensions()
        width, height, _ = dimensions
        
        return {
            'mzs': mzs,
            'intensities': intensities,
            'coordinates': coordinates,
            'width': width,
            'height': height,
            'pixel_size_x': 1.0,  # Default pixel size if not available
            'pixel_size_y': 1.0
        }
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the imzML dataset."""
        if not hasattr(self, 'parser'):
            raise ValueError("Parser not initialized. Call read() first.")
        
        return {
            'source': str(self.imzml_path),
            'uuid': self.parser.metadata.file_description.cv_params[0][2],
            'file_mode': 'continuous' if self.is_continuous else 'processed'
        }
    
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Return the dimensions of the imzML dataset (x, y, z)."""
        if not hasattr(self, 'parser'):
            raise ValueError("Parser not initialized. Call read() first.")
            
        x_max = self.parser.imzmldict['max count of pixels x']
        y_max = self.parser.imzmldict['max count of pixels y']
        z_max = 1  # imzML is typically 2D; set to 1 for compatibility
        return (x_max, y_max, z_max)
    
    def get_common_mass_axis(self) -> np.ndarray:
        """Return the common mass axis for all spectra."""
        if not hasattr(self, 'parser'):
            raise ValueError("Parser not initialized. Call read() first.")
            
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
        
    def get_mzs(self):
        """Get the m/z values."""
        return self.get_common_mass_axis()
    
    def get_intensities(self):
        """Get the intensity values for all spectra."""
        if not hasattr(self, 'parser'):
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")
        
        data = self.read()
        return data['intensities']
    
    def get_coordinates(self):
        """Get the pixel coordinates."""
        if not hasattr(self, 'parser'):
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")
                
        data = self.read()
        return data['coordinates']
    
    def get_pixel_size(self) -> Tuple[float, float]:
        """Get the pixel size in microns."""
        # Default to 1.0 if not available
        return (1.0, 1.0)
    
    def iter_spectra(self, batch_size: Optional[int] = None) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """
        Iterate through spectra with progress monitoring.
        
        Args:
            batch_size: Optional batch size for spectrum iteration (not used in current implementation)
        
        Yields:
            Tuple containing:
                - Coordinates (x, y, z)
                - m/z values array
                - Intensity values array
        """
        if not hasattr(self, 'parser'):
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")
        
        total_spectra = len(self.parser.coordinates)
        dimensions = self.get_dimensions()
        total_pixels = dimensions[0] * dimensions[1] * dimensions[2]
        
        # Log information about spectra vs pixels
        import logging
        logging.info(f"Processing {total_spectra} spectra in a grid of {total_pixels} pixels")
        
        with tqdm(total=total_spectra, desc="Reading spectra", unit="spectrum") as pbar:
            for idx, (x, y, _) in enumerate(self.parser.coordinates):
                try:
                    mz_array, intensity_array = self.parser.getspectrum(idx)
                    
                    # Adjust coordinates to 0-based for internal use
                    x_adj = x - 1
                    y_adj = y - 1
                    z = 0  # imzML is typically 2D
                    
                    yield ((x_adj, y_adj, z), mz_array, intensity_array)
                    pbar.update(1)
                except Exception as err:
                    print(f"Error processing spectrum {idx} at pixel ({x}, {y}): {err}")
                    pbar.update(1)
    
    def close(self) -> None:
        """Close all open file handles."""
        if hasattr(self, 'ibd_file') and self.ibd_file:
            self.ibd_file.close()
            self.ibd_file = None
        
        if hasattr(self, 'parser') and hasattr(self.parser, 'm') and self.parser.m:
            self.parser.m.close()