# msiconvert/converters/lightweight_converter.py
import numpy as np
import zarr
from pathlib import Path
from typing import Dict, Any, Tuple
from scipy import sparse
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
        super().__init__(reader, output_path, **kwargs)
        self.dataset_id = dataset_id
        self.pixel_size_um = pixel_size_um
        self.compression_level = compression_level
        self.root = None
    
    def convert(self) -> bool:
        """Convert MSI data to the lightweight format."""
        try:
            # Create Zarr store
            self.root = zarr.open(str(self.output_path), mode='w')
            
            # Get dataset dimensions and mass axis
            dimensions = self.reader.get_dimensions()
            mass_values = self.reader.get_common_mass_axis()
            metadata = self.reader.get_metadata()
            
            # Add metadata
            self.add_metadata(metadata)
            
            # Create and fill arrays
            self._create_arrays(dimensions, mass_values)
            self._process_spectra(dimensions, mass_values)
            
            # Consolidate metadata
            zarr.consolidate_metadata(str(self.output_path))
            return True
        except Exception as e:
            print(f"Error during conversion: {e}")
            return False
        finally:
            self.reader.close()
    
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to the Zarr store."""
        self.root.attrs['metadata'] = metadata
        self.root.attrs['dataset_id'] = self.dataset_id
        self.root.attrs['pixel_size_um'] = self.pixel_size_um
    
    def _create_arrays(self, dimensions: Tuple[int, int, int], mass_values: np.ndarray) -> None:
        """Create Zarr arrays for storing the data."""
        n_x, n_y, n_z = dimensions
        n_masses = len(mass_values)
        
        # Create compressor
        compressor = zarr.Blosc(cname='zstd', clevel=self.compression_level, shuffle=zarr.Blosc.SHUFFLE)
        
        # Create array for mass values
        self.root.array(
            'mass_values',
            data=mass_values,
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
    
    def _process_spectra(self, dimensions: Tuple[int, int, int], mass_values: np.ndarray) -> None:
        """Process spectra and store in sparse format with progress monitoring."""
        n_x, n_y, n_z = dimensions
        
        # Prepare for appending data
        data_array = self.root['sparse_data/data']
        indices_array = self.root['sparse_data/indices']
        
        data_buffer = []
        indices_buffer = []
        buffer_size = 100000  # Adjust as needed
        
        # Process each spectrum
        current_size = 0
        flush_count = 0
        
        for (x, y, z), mzs, intensities in self.reader.iter_spectra():
            pixel_idx = z * (n_y * n_x) + y * n_x + x
            
            # Map the m/z values to indices in the common mass axis
            mz_indices = np.searchsorted(mass_values, mzs)
            
            # Add to buffer
            for mz_idx, intensity in zip(mz_indices, intensities):
                if intensity > 0:  # Only store non-zero values
                    data_buffer.append(intensity)
                    indices_buffer.append([pixel_idx, mz_idx])
            
            # Flush buffer if it's full
            if len(data_buffer) >= buffer_size:
                current_size = self._flush_buffer(data_array, indices_array, data_buffer, indices_buffer)
                flush_count += 1
                data_buffer = []
                indices_buffer = []
        
        # Flush any remaining data
        if data_buffer:
            current_size = self._flush_buffer(data_array, indices_array, data_buffer, indices_buffer)
            flush_count += 1
    
    def _flush_buffer(self, data_array, indices_array, data_buffer, indices_buffer):
        """Append buffered data to Zarr arrays."""
        current_size = data_array.shape[0]
        new_size = current_size + len(data_buffer)
        
        # Create new arrays with the updated size
        data_array.resize(new_size)
        indices_array.resize((new_size, 2))
        
        # Store data
        data_array[current_size:new_size] = np.array(data_buffer, dtype=np.float32)
        indices_array[current_size:new_size] = np.array(indices_buffer, dtype=np.int32)
        
        return current_size