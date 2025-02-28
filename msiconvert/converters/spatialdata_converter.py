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
from typing import Dict, Any, Optional

from ..core.base_converter import BaseMSIConverter
from ..core.base_reader import BaseMSIReader

class SpatialDataConverter(BaseMSIConverter):
    """Converter for MSI data to SpatialData format."""
    
    def __init__(self, reader: BaseMSIReader, output_path: Path, 
                 dataset_id: str = "msi_dataset",
                 pixel_size_um: float = 1.0,
                 handle_3d: bool = False,
                 **kwargs):
        super().__init__(reader, output_path, **kwargs)
        self.dataset_id = dataset_id
        self.pixel_size_um = pixel_size_um
        self.handle_3d = handle_3d
    
    def convert(self) -> bool:
        """Convert MSI data to SpatialData format."""
        try:
            # Get dataset dimensions and mass axis
            dimensions = self.reader.get_dimensions()
            mass_values = self.reader.get_common_mass_axis()
            
            # Create SpatialData object
            sdata = self._create_spatialdata(dimensions, mass_values)
            
            # Write to disk
            sdata.write(str(self.output_path))
            return True
        except Exception as e:
            print(f"Error during conversion: {e}")
            return False
        finally:
            self.reader.close()
    
    def _create_spatialdata(self, dimensions: Tuple[int, int, int], mass_values: np.ndarray) -> SpatialData:
        """Create SpatialData object from MSI data."""
        n_x, n_y, n_z = dimensions
        n_masses = len(mass_values)
        
        tables = {}
        shapes = {}
        images = {}
        
        # If 3D data but we want to treat as 2D slices
        if n_z > 1 and not self.handle_3d:
            for z in range(n_z):
                # Process this z-slice
                slice_id = f"{self.dataset_id}_z{z}"
                adata, shape = self._process_2d_slice(mass_values, z, slice_id)
                tables[slice_id] = adata
                shapes[f"{slice_id}_pixels"] = shape
        else:
            # Process as full 3D dataset or single 2D slice
            adata = self._process_3d_volume(dimensions, mass_values)
            tables[self.dataset_id] = adata
            shapes[f"{self.dataset_id}_pixels"] = self._create_pixel_shapes(adata)
        
        # Create SpatialData object
        sdata = SpatialData(
            tables=tables,
            shapes=shapes,
            images=images
        )
        
        return sdata
    
    def _process_2d_slice(self, mass_values: np.ndarray, z_value: int, slice_id: str):
        """Process a single 2D slice of MSI data."""
        # Implementation details...
        
    def _process_3d_volume(self, dimensions: Tuple[int, int, int], mass_values: np.ndarray) -> AnnData:
        """Process the entire 3D volume of MSI data."""
        n_x, n_y, n_z = dimensions
        n_masses = len(mass_values)
        
        # Create sparse matrix to hold intensity data
        sparse_data = sparse.lil_matrix((n_x * n_y * n_z, n_masses))
        
        # Create coordinate dataframe
        coords = []
        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    coords.append({
                        'z': z,
                        'y': y, 
                        'x': x,
                        'instance_id': z * (n_y * n_x) + y * n_x + x
                    })
        
        coords_df = pd.DataFrame(coords)
        coords_df['region'] = f"{self.dataset_id}_pixels"
        coords_df.set_index('instance_id', inplace=True)
        
        # Create variable dataframe
        var_df = pd.DataFrame({'mass': mass_values})
        var_df.set_index('mass', inplace=True)
        
        # Fill the sparse matrix with intensity data
        for (x, y, z), mzs, intensities in self.reader.iter_spectra():
            idx = z * (n_y * n_x) + y * n_x + x
            
            # Map the m/z values to indices in the common mass axis
            mz_indices = np.searchsorted(mass_values, mzs)
            sparse_data[idx, mz_indices] = intensities
        
        # Convert to CSR format for better performance
        sparse_data = sparse_data.tocsr()
        
        # Create AnnData
        adata = AnnData(
            X=sparse_data,
            obs=coords_df,
            var=var_df
        )
        
        # Store spatial coordinates
        adata.obsm['spatial'] = coords_df[['x', 'y', 'z']].values
        
        # Create table model
        adata.obs["instance_key"] = adata.obs.index.astype(str)
        table = TableModel.parse(
            adata,
            region=f"{self.dataset_id}_pixels",
            region_key="region",
            instance_key="instance_key"
        )
        
        return table
    
    def _create_pixel_shapes(self, adata) -> ShapesModel:
        """Create geometric shapes for pixels with proper transformations."""
        coordinates = adata.obsm['spatial']
        
        # Create a list of square geometries for each pixel
        geometries = []
        for i in range(len(adata)):
            if coordinates.shape[1] >= 3:  # 3D data
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
    
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to the SpatialData object."""
        # Implementation details...