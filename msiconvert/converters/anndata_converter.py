# msiconvert/converters/anndata_converter.py
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from pathlib import Path
from typing import Dict, Any, Tuple
from scipy import sparse
from tqdm import tqdm

from ..core.base_converter import BaseMSIConverter
from ..core.base_reader import BaseMSIReader
from ..core.registry import register_converter

@register_converter('anndata')
class AnnDataConverter(BaseMSIConverter):
    """MSI data converter using AnnData without SpatialData dependencies."""
    
    def __init__(self, reader: BaseMSIReader, output_path: Path, 
                 dataset_id: str = "msi_dataset",
                 pixel_size_um: float = 1.0,
                 compression_level: int = 5,
                 **kwargs):
        super().__init__(reader, output_path, **kwargs)
        self.dataset_id = dataset_id
        self.pixel_size_um = pixel_size_um
        self.compression_level = compression_level
    
    def convert(self) -> bool:
        """Convert MSI data to AnnData format."""
        try:
            # Get dataset dimensions and mass axis
            dimensions = self.reader.get_dimensions()
            mass_values = self.reader.get_common_mass_axis()
            metadata = self.reader.get_metadata()
            
            # Create AnnData object with sparse data
            adata = self._create_anndata(dimensions, mass_values)
            
            # Add metadata
            self.add_metadata(adata, metadata)
            
            # Save AnnData to disk with zarr backing for large datasets
            self._save_anndata(adata)
            
            return True
        except Exception as e:
            print(f"Error during conversion: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            self.reader.close()
    
    def _create_anndata(self, dimensions: Tuple[int, int, int], mass_values: np.ndarray) -> AnnData:
        """Create AnnData object from MSI data."""
        n_x, n_y, n_z = dimensions
        n_masses = len(mass_values)
        n_pixels = n_x * n_y * n_z
        
        print(f"Creating AnnData object for {n_pixels} pixels and {n_masses} mass values")
        
        # Create sparse matrix for intensity data (pixels x masses)
        sparse_data = sparse.lil_matrix((n_pixels, n_masses), dtype=np.float32)
        
        # Create coordinate dataframe
        coords = []
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    coords.append({
                        'z': z,
                        'y': y, 
                        'x': x,
                        'pixel_id': str(z * (n_y * n_x) + y * n_x + x)  # Convert to string to avoid the warning
                    })
        
        obs_df = pd.DataFrame(coords)
        obs_df.set_index('pixel_id', inplace=True)
        
        # Create variable dataframe for mass values
        var_df = pd.DataFrame({'mz': mass_values})
        # Convert to string index to avoid the warning
        var_df['mz_str'] = var_df['mz'].astype(str)
        var_df.set_index('mz_str', inplace=True)
        
        # Process spectra
        for (x, y, z), mzs, intensities in self.reader.iter_spectra():
            # Calculate pixel index
            idx = z * (n_y * n_x) + y * n_x + x
            
            # Map the m/z values to indices in the common mass axis
            mz_indices = np.searchsorted(mass_values, mzs)
            
            # Store non-zero intensities in sparse matrix
            for mz_idx, intensity in zip(mz_indices, intensities):
                if intensity > 0 and mz_idx < n_masses:  # Only store non-zero values
                    sparse_data[idx, mz_idx] = intensity
        
        # Convert to CSR format for better performance
        sparse_data = sparse_data.tocsr()
        
        # Create AnnData object
        adata = AnnData(
            X=sparse_data,
            obs=obs_df,
            var=var_df
        )
        
        return adata
    
    def add_metadata(self, adata: AnnData, metadata: Dict[str, Any]) -> None:
        """Add metadata to the AnnData object."""
        # Add dataset-level metadata
        adata.uns['metadata'] = metadata
        adata.uns['dataset_id'] = self.dataset_id
        adata.uns['pixel_size_um'] = self.pixel_size_um
        
        # Add coordinate information to observations
        adata.obs['spatial_x'] = adata.obs['x'] * self.pixel_size_um
        adata.obs['spatial_y'] = adata.obs['y'] * self.pixel_size_um
        if 'z' in adata.obs.columns:
            adata.obs['spatial_z'] = adata.obs['z'] * self.pixel_size_um
    
    def _save_anndata(self, adata: AnnData) -> None:
        """Save AnnData object to disk, using zarr backing for large datasets."""
        # Calculate dimensions for reporting
        n_pixels = adata.n_obs
        n_masses = adata.n_vars
        
        # For very large datasets, use zarr backing
        if n_pixels * n_masses > 1e8:  # Threshold can be adjusted
            print(f"Large dataset detected ({n_pixels} pixels, {n_masses} masses). Using zarr backing.")
            
            # Check AnnData version to use appropriate method
            try:
                # For newer versions of AnnData that support chunks directly
                adata.write_zarr(
                    store=str(self.output_path),
                    chunks=(min(10000, n_pixels), min(1000, n_masses))
                )
            except TypeError:
                # Fallback for older versions that don't support chunks or compressor directly
                print("Using compatible mode for AnnData zarr writing")
                adata.write_zarr(store=str(self.output_path))
        else:
            # For smaller datasets, use standard h5ad format
            adata.write_h5ad(str(self.output_path))