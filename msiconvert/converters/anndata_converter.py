import numpy as np
from anndata import AnnData
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

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
        super().__init__(
            reader, 
            output_path, 
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            compression_level=compression_level,
            **kwargs
        )
    
    def _create_data_structures(self) -> Dict[str, Any]:
        """Create data structures for AnnData conversion."""
        # Create the sparse matrix for intensity data
        sparse_data = self._create_sparse_matrix()
        
        # Create observation dataframe (pixels)
        obs_df = self._create_coordinates_dataframe()
        
        # Create variable dataframe (mass values)
        var_df = self._create_mass_dataframe()
        
        return {
            'sparse_data': sparse_data,
            'obs_df': obs_df,
            'var_df': var_df
        }
    
    def _process_single_spectrum(self, data_structures: Dict[str, Any], 
                               coords: Tuple[int, int, int], 
                               mzs: np.ndarray, intensities: np.ndarray) -> None:
        """Process a single spectrum for AnnData format."""
        x, y, z = coords
        pixel_idx = self._get_pixel_index(x, y, z)
        
        # Map the m/z values to indices in the common mass axis
        mz_indices = self._map_mass_to_indices(mzs)
        
        # Add data to sparse matrix
        self._add_to_sparse_matrix(
            data_structures['sparse_data'], 
            pixel_idx, 
            mz_indices, 
            intensities
        )
    
    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize AnnData structures before saving."""
        # Convert sparse matrix from LIL to CSR format for better performance
        data_structures['sparse_data'] = data_structures['sparse_data'].tocsr()
    
    def _save_output(self, data_structures: Dict[str, Any]) -> bool:
        """Save data to AnnData format."""
        try:
            # Create AnnData object
            adata = AnnData(
                X=data_structures['sparse_data'],
                obs=data_structures['obs_df'],
                var=data_structures['var_df']
            )
            
            # Add metadata
            self.add_metadata(adata)
            
            # Save to disk
            self._save_anndata(adata)
            
            return True
        except Exception as e:
            logging.error(f"Error saving AnnData: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def add_metadata(self, metadata: AnnData) -> None:
        """Add metadata to the AnnData object."""
        # Add dataset-level metadata
        metadata.uns['metadata'] = self._metadata
        metadata.uns['dataset_id'] = self.dataset_id
        metadata.uns['pixel_size_um'] = self.pixel_size_um
        
        # Add spatial coordinates reference (without using deprecated obsm['spatial'])
        metadata.uns['spatial'] = {
            'coordinate_cols': ['x', 'y', 'z'],
            'pixel_size_um': self.pixel_size_um
        }
    
    def _save_anndata(self, adata: AnnData) -> None:
        """Save AnnData object to disk, using zarr backing for large datasets."""
        # Calculate dimensions for reporting
        n_pixels = adata.n_obs
        n_masses = adata.n_vars
        
        # Get the file path as a Path object
        output_path = self.output_path
        
        # For very large datasets, use zarr backing
        if n_pixels * n_masses > 1e8:  # Threshold can be adjusted
            logging.info(f"Large dataset detected ({n_pixels} pixels, {n_masses} masses). Using zarr backing.")
            
            # Check AnnData version to use appropriate method
            try:
                # For newer versions of AnnData that support chunks directly
                adata.write_zarr(
                    store=output_path,
                    chunks=(min(10000, n_pixels), min(1000, n_masses))
                )
            except TypeError:
                # Fallback for older versions that don't support chunks or compressor directly
                logging.info("Using compatible mode for AnnData zarr writing")
                adata.write_zarr(store=output_path)
        else:
            # For smaller datasets, use standard h5ad format
            adata.write_h5ad(output_path)
            logging.info(f"Successfully saved AnnData to h5ad file: {output_path}")