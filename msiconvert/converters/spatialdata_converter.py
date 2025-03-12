# msiconvert/converters/spatialdata_converter.py
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Identity
from shapely.geometry import box
import geopandas as gpd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

from ..core.base_converter import BaseMSIConverter
from ..core.base_reader import BaseMSIReader
from ..core.registry import register_converter

@register_converter('spatialdata')
class SpatialDataConverter(BaseMSIConverter):
    """Converter for MSI data to SpatialData format."""
    
    def __init__(self, reader: BaseMSIReader, output_path: Path, 
                 dataset_id: str = "msi_dataset",
                 pixel_size_um: float = 1.0,
                 handle_3d: bool = False,
                 **kwargs):
        super().__init__(
            reader, 
            output_path, 
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            handle_3d=handle_3d,
            **kwargs
        )
    
    def _create_data_structures(self) -> Dict[str, Any]:
        """Create data structures for SpatialData format."""
        # Return dictionaries to store tables, shapes, and sparse matrices
        tables = {}
        shapes = {}
        
        # If 3D data but we want to treat as 2D slices
        n_x, n_y, n_z = self._dimensions
        
        if n_z > 1 and not self.handle_3d:
            # For 3D data treated as 2D slices, we'll create a structure for each slice
            slices_data = {}
            for z in range(n_z):
                slice_id = f"{self.dataset_id}_z{z}"
                slices_data[slice_id] = {
                    'sparse_data': self._create_sparse_matrix_for_slice(z),
                    'coords_df': self._create_coordinates_dataframe_for_slice(z),
                }
            
            return {
                'mode': '2d_slices',
                'slices_data': slices_data,
                'tables': tables,
                'shapes': shapes,
                'var_df': self._create_mass_dataframe()
            }
        else:
            # For full 3D dataset or single 2D slice
            return {
                'mode': '3d_volume',
                'sparse_data': self._create_sparse_matrix(),
                'coords_df': self._create_coordinates_dataframe(),
                'var_df': self._create_mass_dataframe(),
                'tables': tables,
                'shapes': shapes
            }
    
    def _create_sparse_matrix_for_slice(self, z_value: int) -> sparse.lil_matrix:
        """Create a sparse matrix for a single Z-slice."""
        n_x, n_y, _ = self._dimensions
        n_pixels = n_x * n_y
        n_masses = len(self._common_mass_axis)
        
        logging.info(f"Creating sparse matrix for slice z={z_value} with {n_pixels} pixels and {n_masses} mass values")
        return sparse.lil_matrix((n_pixels, n_masses), dtype=np.float32)
    
    def _create_coordinates_dataframe_for_slice(self, z_value: int) -> pd.DataFrame:
        """Create a coordinates dataframe for a single Z-slice."""
        n_x, n_y, _ = self._dimensions
        
        coords = []
        for y in range(n_y):
            for x in range(n_x):
                pixel_idx = y * n_x + x
                coords.append({
                    'y': y, 
                    'x': x,
                    'instance_id': pixel_idx
                })
        
        coords_df = pd.DataFrame(coords)
        slice_id = f"{self.dataset_id}_z{z_value}"
        coords_df['region'] = f"{slice_id}_pixels"
        # Make sure instance_id is a string to avoid issues
        coords_df['instance_id'] = coords_df['instance_id'].astype(str)
        coords_df.set_index('instance_id', inplace=True)
        
        # Add spatial coordinates
        coords_df['spatial_x'] = coords_df['x'] * self.pixel_size_um
        coords_df['spatial_y'] = coords_df['y'] * self.pixel_size_um
        
        return coords_df
    
    def _process_single_spectrum(self, data_structures: Dict[str, Any], 
                               coords: Tuple[int, int, int], 
                               mzs: np.ndarray, intensities: np.ndarray) -> None:
        """Process a single spectrum for SpatialData format."""
        x, y, z = coords
        
        if data_structures['mode'] == '2d_slices':
            # For 2D slices mode, add data to the appropriate slice
            slice_id = f"{self.dataset_id}_z{z}"
            if slice_id in data_structures['slices_data']:
                slice_data = data_structures['slices_data'][slice_id]
                pixel_idx = y * self._dimensions[0] + x
                
                # Map m/z values to indices
                mz_indices = self._map_mass_to_indices(mzs)
                
                # Add to sparse matrix for this slice
                self._add_to_sparse_matrix(
                    slice_data['sparse_data'], 
                    pixel_idx, 
                    mz_indices, 
                    intensities
                )
        else:
            # For 3D volume mode, add data to the single sparse matrix
            pixel_idx = self._get_pixel_index(x, y, z)
            
            # Map m/z values to indices
            mz_indices = self._map_mass_to_indices(mzs)
            
            # Add to sparse matrix
            self._add_to_sparse_matrix(
                data_structures['sparse_data'], 
                pixel_idx, 
                mz_indices, 
                intensities
            )
    
    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize SpatialData structures by creating tables and shapes."""
        if data_structures['mode'] == '2d_slices':
            # Process each slice
            for slice_id, slice_data in data_structures['slices_data'].items():
                try:
                    # Convert to CSR format for efficiency
                    slice_data['sparse_data'] = slice_data['sparse_data'].tocsr()
                    
                    # Create AnnData for this slice
                    adata = AnnData(
                        X=slice_data['sparse_data'],
                        obs=slice_data['coords_df'],
                        var=data_structures['var_df']
                    )
                    
                    # Store spatial coordinates
                    adata.obsm['spatial'] = slice_data['coords_df'][['x', 'y']].values
                    
                    # Make sure region column exists and is correct
                    region_key = f"{slice_id}_pixels"
                    if 'region' not in adata.obs.columns:
                        adata.obs['region'] = region_key
                    
                    # Make sure instance_key is a string column
                    adata.obs['instance_key'] = adata.obs.index.astype(str)
                    
                    # Create table model
                    table = TableModel.parse(
                        adata,
                        region=region_key,
                        region_key="region",
                        instance_key="instance_key"
                    )
                    
                    # Add to tables and create shapes
                    data_structures['tables'][slice_id] = table
                    data_structures['shapes'][region_key] = self._create_pixel_shapes(adata, is_3d=False)
                
                except Exception as e:
                    logging.error(f"Error processing slice {slice_id}: {e}")
                    import traceback
                    logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
                    raise
        else:
            try:
                # Process the single 3D volume
                # Convert to CSR format
                data_structures['sparse_data'] = data_structures['sparse_data'].tocsr()
                
                # Create AnnData
                adata = AnnData(
                    X=data_structures['sparse_data'],
                    obs=data_structures['coords_df'],
                    var=data_structures['var_df']
                )
                
                # Store spatial coordinates
                if self.handle_3d:
                    adata.obsm['spatial'] = data_structures['coords_df'][['x', 'y', 'z']].values
                else:
                    adata.obsm['spatial'] = data_structures['coords_df'][['x', 'y']].values
                
                # Make sure region column exists and is correct
                region_key = f"{self.dataset_id}_pixels"
                if 'region' not in adata.obs.columns:
                    adata.obs['region'] = region_key
                
                # Ensure instance_key is a string column
                adata.obs['instance_key'] = adata.obs.index.astype(str)
                
                # Create table model
                table = TableModel.parse(
                    adata,
                    region=region_key,
                    region_key="region",
                    instance_key="instance_key"
                )
                
                # Add to tables and create shapes
                data_structures['tables'][self.dataset_id] = table
                data_structures['shapes'][region_key] = self._create_pixel_shapes(adata, is_3d=self.handle_3d)
            
            except Exception as e:
                logging.error(f"Error processing 3D volume: {e}")
                import traceback
                logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
                raise
    
    def _create_pixel_shapes(self, adata: AnnData, is_3d: bool = False) -> ShapesModel:
        """
        Create geometric shapes for pixels with proper transformations.
        
        Parameters:
        -----------
        adata: AnnData object containing coordinates
        is_3d: Whether to handle as 3D data
        
        Returns:
        --------
        ShapesModel: SpatialData shapes model
        """
        coordinates = adata.obsm['spatial']
        
        # Create a list of square geometries for each pixel
        geometries = []
        for i in range(len(adata)):
            if is_3d and coordinates.shape[1] >= 3:  # 3D data
                x, y, z = coordinates[i][:3]
            else:  # 2D data
                x, y = coordinates[i][:2]
                
            # Create a square centered at pixel coordinates
            pixel_box = box(
                x - self.pixel_size_um/2, 
                y - self.pixel_size_um/2,
                x + self.pixel_size_um/2,
                y + self.pixel_size_um/2
            )
            geometries.append(pixel_box)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            geometry=geometries,
            index=adata.obs.index
        )
        
        # Set up transform
        transform = Identity()
        transformations = {self.dataset_id: transform, "global": transform}
        
        # Parse shapes
        shapes = ShapesModel.parse(
            gdf,
            transformations=transformations
        )
        
        return shapes
    
    def _save_output(self, data_structures: Dict[str, Any]) -> bool:
        """Save the data to SpatialData format."""
        try:
            # Create SpatialData object
            sdata = SpatialData(
                tables=data_structures['tables'],
                shapes=data_structures['shapes'],
                images={}  # No images in MSI data
            )
            
            # Add metadata
            self.add_metadata(sdata)
            
            # Write to disk
            sdata.write(str(self.output_path))
            return True
        except Exception as e:
            logging.error(f"Error saving SpatialData: {e}")
            import traceback
            logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
            return False
    
    def add_metadata(self, sdata: SpatialData) -> None:
        """Add metadata to the SpatialData object."""
        # Add dataset metadata if SpatialData supports it
        if hasattr(sdata, 'metadata'):
            sdata.metadata = {
                'dataset_id': self.dataset_id,
                'pixel_size_um': self.pixel_size_um,
                'source': self._metadata.get('source', 'unknown'),
                'msi_metadata': self._metadata
            }