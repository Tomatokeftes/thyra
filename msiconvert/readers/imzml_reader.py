# msiconvert/readers/imzml_reader.py
import numpy as np
from typing import Dict, Any, Tuple, Generator, Optional, Union, List
from pathlib import Path
from pyimzml.ImzMLParser import ImzMLParser
import logging
from tqdm import tqdm

from ..core.base_reader import BaseMSIReader
from ..core.registry import register_reader

@register_reader('imzml')
class ImzMLReader(BaseMSIReader):
    """Reader for imzML format files with optimizations for performance."""
    
    def __init__(self, imzml_path: Optional[Union[str, Path]] = None, 
                 batch_size: int = 50,
                 cache_coordinates: bool = True):
        """
        Initialize an ImzML reader.
        
        Args:
            imzml_path: Path to the imzML file. If not provided, can be set later using read().
            batch_size: Default batch size for spectrum iteration
            cache_coordinates: Whether to cache coordinates upfront
        """
        self.filepath = imzml_path
        self.batch_size = batch_size
        self.cache_coordinates = cache_coordinates
        self.parser = None
        self.ibd_file = None
        
        # Cached properties
        self._common_mass_axis = None  # Will hold all unique m/z values
        self._dimensions = None
        self._metadata = None
        self._coordinates_cache = {}
        
        # Initialize if path provided
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
        
        # Open the .ibd file for reading
        self.ibd_file = open(self.ibd_path, mode="rb")
        
        # Initialize the parser
        logging.info(f"Initializing ImzML parser for {imzml_path}")
        try:
            self.parser = ImzMLParser(
                filename=str(imzml_path),
                parse_lib="lxml",
                ibd_file=self.ibd_file
            )
        except Exception as e:
            if self.ibd_file:
                self.ibd_file.close()
            logging.error(f"Failed to initialize ImzML parser: {e}")
            raise
        
        if self.parser.metadata is None:
            raise ValueError("Failed to parse metadata from imzML file.")
            
        # Determine file mode
        self.is_continuous = "continuous" in self.parser.metadata.file_description.param_by_name
        self.is_processed = "processed" in self.parser.metadata.file_description.param_by_name
        
        if self.is_continuous == self.is_processed:
            raise ValueError("Invalid file mode, expected either 'continuous' or 'processed'.")
            
        # Cache coordinates if requested
        if self.cache_coordinates:
            self._cache_all_coordinates()
    
    def _cache_all_coordinates(self):
        """Cache all coordinates for faster access."""
        if not hasattr(self, 'parser') or self.parser is None:
            return
            
        logging.info("Caching all coordinates for faster access")
        self._coordinates_cache = {}
        
        for idx, (x, y, z) in enumerate(self.parser.coordinates):
            # Adjust coordinates to 0-based
            self._coordinates_cache[idx] = (x - 1, y - 1, z - 1 if z > 0 else 0)
            
        logging.info(f"Cached {len(self._coordinates_cache)} coordinates")
    
    def can_read(self, filepath: str) -> bool:
        """Check if this reader can read the given file."""
        return isinstance(filepath, str) and filepath.lower().endswith('.imzml')
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the imzML dataset."""
        if not hasattr(self, 'parser') or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")
        
        if self._metadata is None:
            self._metadata = {
                'source': str(self.imzml_path),
                'uuid': self.parser.metadata.file_description.cv_params[0][2],
                'file_mode': 'continuous' if self.is_continuous else 'processed'
            }
            
            # Add additional metadata if available
            if hasattr(self.parser, 'imzmldict'):
                for key, value in self.parser.imzmldict.items():
                    self._metadata[key] = value
                    
        return self._metadata
    
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Return the dimensions of the imzML dataset (x, y, z)."""
        if not hasattr(self, 'parser') or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")
                
        if self._dimensions is None:
            # Get dimensions from metadata if available
            if hasattr(self.parser, 'imzmldict'):
                x_max = self.parser.imzmldict.get('max count of pixels x', 1)
                y_max = self.parser.imzmldict.get('max count of pixels y', 1)
                z_max = self.parser.imzmldict.get('max count of pixels z', 1)
            else:
                # Calculate from coordinates
                x_coordinates = [x for x, _, _ in self.parser.coordinates]
                y_coordinates = [y for _, y, _ in self.parser.coordinates]
                z_coordinates = [z for _, _, z in self.parser.coordinates]
                
                x_max = max(x_coordinates) if x_coordinates else 1
                y_max = max(y_coordinates) if y_coordinates else 1
                z_max = max(z_coordinates) if z_coordinates else 1
                
            self._dimensions = (x_max, y_max, z_max)
            
        return self._dimensions
    
    def get_common_mass_axis(self) -> np.ndarray:
        """
        Return the common mass axis composed of all unique m/z values.
        
        For continuous mode, returns the m/z values from the first spectrum.
        For processed mode, collects all unique m/z values across spectra.
        
        Returns:
            Array of m/z values in ascending order
        """
        if not hasattr(self, 'parser') or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")
                
        if self._common_mass_axis is None:
            if self.is_continuous:
                # For continuous data, all spectra share the same m/z values
                logging.info("Using m/z values from first spectrum (continuous mode)")
                self._common_mass_axis = self.parser.getspectrum(0)[0]
            else:
                # For processed data, collect unique m/z values across spectra
                logging.info("Building common mass axis from all unique m/z values (processed mode)")
                
                # Sample spectra for performance
                total_spectra = len(self.parser.coordinates)
                sample_size = min(50, total_spectra)
                sample_indices = np.linspace(0, total_spectra - 1, sample_size, dtype=int)
                
                all_mzs = []
                for idx in sample_indices:
                    try:
                        mzs, _ = self.parser.getspectrum(idx)
                        if mzs.size > 0:
                            all_mzs.append(mzs)
                    except Exception as e:
                        logging.warning(f"Error getting spectrum {idx}: {e}")
                
                if all_mzs:
                    # Concatenate and find unique values
                    try:
                        # More efficient approach for large arrays
                        combined_mzs = np.concatenate(all_mzs)
                        # Sort and find unique values (more efficient than np.unique for large arrays)
                        combined_mzs.sort()
                        # Use tolerance-based uniqueness to handle precision issues
                        tolerance = 1e-6  # Adjust based on instrument precision
                        mask = np.empty(combined_mzs.size, dtype=bool)
                        mask[0] = True
                        np.greater(np.diff(combined_mzs), tolerance, out=mask[1:])
                        self._common_mass_axis = combined_mzs[mask]
                    except Exception as e:
                        logging.warning(f"Error creating optimized common mass axis: {e}, falling back to standard method")
                        # Fallback to standard np.unique
                        combined_mzs = np.concatenate(all_mzs)
                        self._common_mass_axis = np.unique(combined_mzs)
                else:
                    # Fallback to empty array
                    self._common_mass_axis = np.array([])
            
            logging.info(f"Common mass axis created with {len(self._common_mass_axis)} m/z values")
                
        return self._common_mass_axis
    
    def _map_mz_to_common_axis(self, mzs: np.ndarray, intensities: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map m/z values to indices in the common mass axis.
        
        For continuous mode imzML files, the mapping is straightforward since all
        spectra share the same m/z values. For processed mode, we need to map each
        m/z value to its closest match in the common axis within tolerance.
        
        Args:
            mzs: Array of m/z values
            intensities: Array of intensity values
            
        Returns:
            Tuple of (indices in common mass axis, corresponding intensities)
        """
        if mzs.size == 0 or intensities.size == 0:
            return np.array([], dtype=int), np.array([])
            
        # Get common mass axis (calculate if not already done)
        common_axis = self.get_common_mass_axis()
        if common_axis.size == 0:
            return np.array([], dtype=int), intensities
            
        # Use searchsorted to find indices in common mass axis
        indices = np.searchsorted(common_axis, mzs)
        
        # Filter out indices that are out of bounds
        valid_mask = (indices < len(common_axis))
        
        # Ensure the values are close enough (within tolerance)
        tolerance = 1e-6  # Adjust based on instrument precision
        if np.any(valid_mask):
            exact_matches = np.abs(common_axis[indices[valid_mask]] - mzs[valid_mask]) <= tolerance
            valid_mask[valid_mask] = exact_matches
        
        # Return only valid mappings
        return indices[valid_mask], intensities[valid_mask]
    
    def iter_spectra(self, batch_size: int = None) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """
        Iterate through spectra with progress monitoring and batch processing.
        
        Maps m/z values to the common mass axis using searchsorted for accurate
        representation in the output data structures.
        
        Args:
            batch_size: Number of spectra to process in each batch (None for default)
            
        Yields:
            Tuple of:
                - Coordinates (x, y, z)
                - Indices in common mass axis
                - Intensity values array
        """
        if not hasattr(self, 'parser') or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")
        
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.batch_size
        
        # Ensure common mass axis is built before starting iteration
        common_axis = self.get_common_mass_axis()
        
        total_spectra = len(self.parser.coordinates)
        dimensions = self.get_dimensions()
        total_pixels = dimensions[0] * dimensions[1] * dimensions[2]
        
        # Log information about spectra vs pixels
        logging.info(f"Processing {total_spectra} spectra in a grid of {total_pixels} pixels")
        
        # Process in batches
        with tqdm(total=total_spectra, desc="Reading spectra", unit="spectrum") as pbar:
            if batch_size <= 1:
                # Process one at a time
                for idx in range(total_spectra):
                    try:
                        # Get coordinates (with adjustment to 0-based)
                        if idx in self._coordinates_cache:
                            coords = self._coordinates_cache[idx]
                        else:
                            x, y, z = self.parser.coordinates[idx]
                            # Adjust coordinates to 0-based for internal use
                            coords = (x - 1, y - 1, z - 1 if z > 0 else 0)
                        
                        # Get spectrum data
                        mzs, intensities = self.parser.getspectrum(idx)
                        
                        if mzs.size > 0 and intensities.size > 0:
                            # Map to common mass axis
                            common_indices, mapped_intensities = self._map_mz_to_common_axis(mzs, intensities)
                            
                            if common_indices.size > 0:
                                yield coords, common_indices, mapped_intensities
                        
                        pbar.update(1)
                    except Exception as e:
                        logging.warning(f"Error processing spectrum {idx}: {e}")
                        pbar.update(1)
            else:
                # Process in batches
                for batch_start in range(0, total_spectra, batch_size):
                    batch_end = min(batch_start + batch_size, total_spectra)
                    batch_size_actual = batch_end - batch_start
                    
                    # Process each spectrum in the batch
                    for offset in range(batch_size_actual):
                        idx = batch_start + offset
                        try:
                            # Get coordinates (with adjustment to 0-based)
                            if idx in self._coordinates_cache:
                                coords = self._coordinates_cache[idx]
                            else:
                                x, y, z = self.parser.coordinates[idx]
                                # Adjust coordinates to 0-based for internal use
                                coords = (x - 1, y - 1, z - 1 if z > 0 else 0)
                            
                            # Get spectrum data
                            mzs, intensities = self.parser.getspectrum(idx)
                            
                            if mzs.size > 0 and intensities.size > 0:
                                # Map to common mass axis
                                common_indices, mapped_intensities = self._map_mz_to_common_axis(mzs, intensities)
                                
                                if common_indices.size > 0:
                                    yield coords, common_indices, mapped_intensities
                            
                            pbar.update(1)
                        except Exception as e:
                            logging.warning(f"Error processing spectrum {idx}: {e}")
                            pbar.update(1)
    
    def read(self):
        """
        Read the entire imzML file and return a structured data dictionary.
        
        Returns:
            dict: Dictionary containing:
                - mzs: common m/z values array
                - intensities: array of intensity arrays
                - coordinates: list of (x,y,z) coordinates
                - width: number of pixels in x dimension
                - height: number of pixels in y dimension
                - depth: number of pixels in z dimension
        """
        if not hasattr(self, 'parser') or self.parser is None:
            if self.filepath:
                self._initialize_parser(self.filepath)
            else:
                raise ValueError("Parser not initialized and no filepath available")

        # Get common mass axis
        mzs = self.get_common_mass_axis()
        
        # Get dimensions
        width, height, depth = self.get_dimensions()
        
        # Collect all spectra
        coordinates = []
        intensities = []
        
        # Iterate through all spectra
        for coords, spectrum_indices, spectrum_intensities in self.iter_spectra():
            coordinates.append(coords)
            
            # Convert sparse representation to full array
            full_spectrum = np.zeros(len(mzs), dtype=np.float32)
            full_spectrum[spectrum_indices] = spectrum_intensities
            intensities.append(full_spectrum)
        
        return {
            "mzs": mzs,
            "intensities": np.array(intensities),
            "coordinates": coordinates,
            "width": width,
            "height": height,
            "depth": depth
        }
    
    def close(self) -> None:
        """Close all open file handles."""
        if hasattr(self, 'ibd_file') and self.ibd_file is not None:
            self.ibd_file.close()
            self.ibd_file = None
        
        if hasattr(self, 'parser') and self.parser is not None:
            if hasattr(self.parser, 'm') and self.parser.m is not None:
                self.parser.m.close()
            self.parser = None