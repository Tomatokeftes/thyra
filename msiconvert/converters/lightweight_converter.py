# msiconvert/converters/lightweight_converter.py
import numpy as np
import zarr
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from tqdm import tqdm

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
        self.root.attrs['metadata'] = metadata
        self.root.attrs['dataset_id'] = self.dataset_id
        self.root.attrs['pixel_size_um'] = self.pixel_size_um
    
    def _create_arrays(self) -> None:
        """Create Zarr arrays for storing the data."""
        n_x, n_y, n_z = self._dimensions
        
        # Create compressor
        compressor = zarr.Blosc(cname='zstd', clevel=self.compression_level, shuffle=zarr.Blosc.SHUFFLE)
        
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
            dtype=np.float32,
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
        
        Parameters:
        -----------
        data_buffer: List of intensity values
        indices_buffer: List of [pixel_idx, mass_idx] pairs
        current_size: Current size of the arrays
        
        Returns:
        --------
        int: New current size after flushing
        """
        data_array = self.root['sparse_data/data']
        indices_array = self.root['sparse_data/indices']
        
        new_size = current_size + len(data_buffer)
        
        # Create new arrays with the updated size
        data_array.resize(new_size)
        indices_array.resize((new_size, 2))
        
        # Store data
        data_array[current_size:new_size] = np.array(data_buffer, dtype=np.float32)
        indices_array[current_size:new_size] = np.array(indices_buffer, dtype=np.int32)
        
        return new_size
    
    def _save_output(self, data_structures: Dict[str, Any]) -> bool:
        """Save output for lightweight format."""
        try:
            # Consolidate metadata
            zarr.consolidate_metadata(str(self.output_path))
            return True
        except Exception as e:
            logging.error(f"Error consolidating Zarr metadata: {e}")
            return False