# msiconvert/converters/lightweight_converter.py (improved)
import numpy as np
import zarr
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging

from ..core.base_converter import BaseMSIConverter
from ..core.base_reader import BaseMSIReader
from ..core.registry import register_converter

@register_converter('lightweight')
class LightweightConverter(BaseMSIConverter):
    """Lightweight MSI data converter that mimics SpatialData organization without dependencies."""
    
    def __init__(self, reader: BaseMSIReader, output_path: Path, 
                 dataset_id: str = "msi_dataset",
                 pixel_size_um: float = 1.0,
                 compression_level: int = 5,
                 **kwargs):
        super().__init__(
            reader, 
            output_path, 
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            compression_level=compression_level,
            **kwargs
        )
        self.root = None
    
    def _create_data_structures(self) -> Dict[str, Any]:
        """Create Zarr-based data structures for lightweight format."""
        # Create Zarr store
        self.root = zarr.open(str(self.output_path), mode='w')
        
        # Add metadata
        if self._metadata is None:
            raise ValueError("Metadata is not initialized.")
            
        self.add_metadata(self._metadata)
        
        # Create arrays
        self._create_arrays()
        
        # Return buffers for accumulating data
        return {
            'data_buffer': [],
            'indices_buffer': [],
            'current_size': 0
        }
    
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to the Zarr store."""
        # Consolidate checks and assignments
        if self.root is None:
            raise ValueError("Root is not initialized.")
            
        self.root.attrs['metadata'] = metadata
        self.root.attrs['dataset_id'] = self.dataset_id
        self.root.attrs['pixel_size_um'] = self.pixel_size_um
    
    def _create_arrays(self) -> None:
        """Create Zarr arrays for storing the data."""
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized.")
            
        n_x, n_y, n_z = self._dimensions
        
        # Create compressor
        compressor = zarr.Blosc(cname='zstd', clevel=self.compression_level, shuffle=zarr.Blosc.SHUFFLE)
        
        if self.root is None:
            raise ValueError("Root is not initialized.")
            
        # Create array for mass values
        self.root.array(
            'mass_values',
            data=self._common_mass_axis,
            dtype=np.float64,
            compressor=compressor
        )
        
        # Create coordinates array
        coords = np.zeros((n_x * n_y * n_z, 3), dtype=np.int32)
        idx = 0
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    coords[idx] = [x, y, z]
                    idx += 1
        
        self.root.array(
            'coordinates',
            data=coords,
            dtype=np.int32,
            compressor=compressor
        )
        
        # Create sparse array structure
        self.root.create_group('sparse_data')
        
        # We'll use COO format for sparse data
        self.root.create_dataset(
            'sparse_data/data',
            shape=(0,),
            dtype=np.float64,
            compressor=compressor,
            chunks=(10000,)
        )

        self.root.create_dataset(
            'sparse_data/indices',
            shape=(0, 2),  # (pixel_idx, mass_idx)
            dtype=np.int32,
            compressor=compressor,
            chunks=(10000, 2)
        )
    
    def _process_single_spectrum(self, data_structures: Dict[str, Any], 
                               coords: Tuple[int, int, int], 
                               mzs: np.ndarray, intensities: np.ndarray) -> None:
        """Process a single spectrum for the lightweight format."""
        x, y, z = coords
        pixel_idx = self._get_pixel_index(x, y, z)
        
        # Map the m/z values to indices in the common mass axis
        mz_indices = self._map_mass_to_indices(mzs)
        
        # Add to buffers
        for mz_idx, intensity in zip(mz_indices, intensities):
            if intensity > 0:  # Only store non-zero values
                data_structures['data_buffer'].append(intensity)
                data_structures['indices_buffer'].append([pixel_idx, mz_idx])
        
        # Flush buffer if it's full
        if len(data_structures['data_buffer']) >= self._buffer_size:
            data_structures['current_size'] = self._flush_buffer(
                data_structures['data_buffer'], 
                data_structures['indices_buffer'],
                data_structures['current_size']
            )
            data_structures['data_buffer'] = []
            data_structures['indices_buffer'] = []
    
    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Flush any remaining data in buffers."""
        if data_structures['data_buffer']:
            self._flush_buffer(
                data_structures['data_buffer'], 
                data_structures['indices_buffer'],
                data_structures['current_size']
            )
    
    def _flush_buffer(self, data_buffer: List[float], indices_buffer: List[List[int]], current_size: int) -> int:
        """
        Append buffered data to Zarr arrays.
        
        Args:
            data_buffer: List of intensity values
            indices_buffer: List of [pixel_idx, mass_idx] pairs
            current_size: Current size of the arrays
            
        Returns:
            int: New current size after flushing
        """
        if not data_buffer:
            return current_size
            
        if self.root is None:
            raise ValueError("Root is not initialized.")
            
        # Get arrays once to avoid repeated dictionary lookups
        data_array = self.root['sparse_data/data']
        indices_array = self.root['sparse_data/indices']
        
        # Calculate new size
        buffer_length = len(data_buffer)
        new_size = current_size + buffer_length
        
        # Resize arrays to accommodate new data
        data_array.resize(new_size)
        indices_array.resize((new_size, 2))
        
        # Convert data to numpy arrays for efficient bulk insertion
        data_np = np.array(data_buffer, dtype=np.float64)
        indices_np = np.array(indices_buffer, dtype=np.int32)
        
        # Store data in a single operation
        data_array[current_size:new_size] = data_np
        indices_array[current_size:new_size] = indices_np
        
        return new_size
    
    def _save_output(self, data_structures: Dict[str, Any]) -> bool:
        """Save output for lightweight format."""
        try:
            if self.root is None:
                raise ValueError("Root is not initialized.")
                
            # Add any final metadata
            if 'sparse_count' not in self.root.attrs:
                self.root.attrs['sparse_count'] = data_structures['current_size']
                
            # Consolidate metadata for better performance
            zarr.consolidate_metadata(self.root.store)
            logging.info(f"Successfully saved lightweight format to {self.output_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving lightweight format: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False