# msiconvert/converters/spatialdata_converter.py (improved)
import numpy as np
import pandas as pd
from anndata import AnnData #type: ignore
from scipy import sparse
from pathlib import Path
from typing import Dict, Any, Tuple, List
import logging
from numpy.typing import NDArray

# Import SpatialData dependencies
try:
    from spatialdata import SpatialData  #type: ignore
    from spatialdata.models import ShapesModel, TableModel  #type: ignore
    from spatialdata.transformations import Identity  #type: ignore
    from shapely.geometry import box
    import geopandas as gpd
    import xarray as xr
    from geopandas import GeoDataFrame
    SPATIALDATA_AVAILABLE = True
except ImportError:
    logging.warning("SpatialData dependencies not available. SpatialDataConverter will not work.")
    SPATIALDATA_AVAILABLE = False  #type: ignore

from ..core.base_converter import BaseMSIConverter
from ..core.base_reader import BaseMSIReader
from ..core.registry import register_converter

@register_converter('spatialdata')
class SpatialDataConverter(BaseMSIConverter):
    """Converter for MSI data to SpatialData format."""
    
    def __init__(
        self, 
        reader: BaseMSIReader, 
        output_path: Path, 
        dataset_id: str = "msi_dataset",
        pixel_size_um: float = 1.0,
        handle_3d: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the SpatialData converter.
        
        Args:
            reader: MSI data reader
            output_path: Path for output file
            dataset_id: Identifier for the dataset
            pixel_size_um: Size of each pixel in micrometers
            handle_3d: Whether to process as 3D data (True) or 2D slices (False)
            **kwargs: Additional keyword arguments
            
        Raises:
            ImportError: If SpatialData dependencies are not available
        """
        # Check if SpatialData is available
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available. Please install required packages.")
            
        super().__init__(
            reader, 
            output_path, 
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            handle_3d=handle_3d,
            **kwargs
        )

        self._non_empty_pixel_count: int = 0
    
    def _create_data_structures(self) -> Dict[str, Any]:
        """
        Create data structures for SpatialData format.
        
        Returns:
            Dict containing tables, shapes, images, and data arrays
        """
        # Return dictionaries to store tables, shapes, and sparse matrices
        tables: Dict[str, Any] = {}
        shapes: Dict[str, Any] = {}
        images: Dict[str, Any] = {}
        
        # If 3D data but we want to treat as 2D slices
        n_x: int
        n_y: int
        n_z: int
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        n_x, n_y, n_z = self._dimensions
        
        if n_z > 1 and not self.handle_3d:
            # For 3D data treated as 2D slices, we'll create a structure for each slice
            slices_data: Dict[str, Dict[str, Any]] = {}
            for z in range(n_z):
                slice_id: str = f"{self.dataset_id}_z{z}"
                slices_data[slice_id] = {
                    'sparse_data': self._create_sparse_matrix_for_slice(z),
                    'coords_df': self._create_coordinates_dataframe_for_slice(z),
                    'tic_values': np.zeros((n_y, n_x), dtype=np.float64)  # 2D array with conventional (row, col) = (y, x) ordering
                }
            
            # Check if common mass axis is initialized
            if self._common_mass_axis is None:
                raise ValueError("Common mass axis is not initialized")
                
            return {
                'mode': '2d_slices',
                'slices_data': slices_data,
                'tables': tables,
                'shapes': shapes,
                'images': images,
                'var_df': self._create_mass_dataframe(),
                'total_intensity': np.zeros(len(self._common_mass_axis), dtype=np.float64),
                'pixel_count': 0,
            }
        else:
            # For full 3D dataset or single 2D slice
            # Check if common mass axis is initialized
            if self._common_mass_axis is None:
                raise ValueError("Common mass axis is not initialized")
                
            return {
                'mode': '3d_volume',
                'sparse_data': self._create_sparse_matrix(),
                'coords_df': self._create_coordinates_dataframe(),
                'var_df': self._create_mass_dataframe(),
                'tables': tables,
                'shapes': shapes,
                'images': images,
                'tic_values': np.zeros((n_y, n_x, n_z), dtype=np.float64),  # For TIC image
                'total_intensity': np.zeros(len(self._common_mass_axis), dtype=np.float64),
                'pixel_count': 0,  # Count for normalization
            }
    
    def _create_sparse_matrix_for_slice(self, z_value: int) -> sparse.lil_matrix:
        """
        Create a sparse matrix for a single Z-slice.
        
        Args:
            z_value: Z-index of the slice
            
        Returns:
            Sparse matrix for storing intensity values
            
        Raises:
            ValueError: If dimensions or common mass axis are not initialized
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")
            
        n_x: int
        n_y: int
        _z: int
        n_x, n_y, _z = self._dimensions
        n_pixels: int = n_x * n_y
        n_masses: int = len(self._common_mass_axis)
        
        logging.info(f"Creating sparse matrix for slice z={z_value} with {n_pixels} pixels and {n_masses} mass values")
        return sparse.lil_matrix((n_pixels, n_masses), dtype=np.float64)
    
    def _create_coordinates_dataframe_for_slice(self, z_value: int) -> pd.DataFrame:
        """
        Create a coordinates dataframe for a single Z-slice.
        
        Args:
            z_value: Z-index of the slice
            
        Returns:
            DataFrame with pixel coordinates
            
        Raises:
            ValueError: If dimensions are not initialized
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
            
        n_x: int
        n_y: int
        _z: int
        n_x, n_y, _z = self._dimensions
        
        # Pre-allocate arrays for better performance
        pixel_count: int = n_x * n_y
        y_values: NDArray[np.int32] = np.repeat(np.arange(n_y, dtype=np.int32), n_x)
        x_values: NDArray[np.int32] = np.tile(np.arange(n_x, dtype=np.int32), n_y)
        instance_ids: NDArray[np.int32] = np.arange(pixel_count, dtype=np.int32)
        
        # Create DataFrame in one operation
        coords_df: pd.DataFrame = pd.DataFrame({
            'y': y_values,
            'x': x_values,
            'instance_id': instance_ids,
            'region': f"{self.dataset_id}_z{z_value}_pixels"
        })
        
        # Set index efficiently
        coords_df['instance_id'] = coords_df['instance_id'].astype(str)
        coords_df.set_index('instance_id', inplace=True)  # type: ignore
        
        # Add spatial coordinates in a vectorized operation
        coords_df['spatial_x'] = coords_df['x'] * self.pixel_size_um
        coords_df['spatial_y'] = coords_df['y'] * self.pixel_size_um
        
        return coords_df
    
    def _process_single_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64]
    ) -> None:
        """
        Process a single spectrum for SpatialData format.
        
        Args:
            data_structures: Data structures for storing processed data
            coords: (x, y, z) pixel coordinates
            mzs: Array of m/z values
            intensities: Array of intensity values
            
        Raises:
            ValueError: If dimensions are not initialized
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
            
        x: int
        y: int
        z: int
        x, y, z = coords
        
        # Calculate TIC for this pixel (sum of all intensities)
        tic_value: float = float(np.sum(intensities))
        
        # Update total intensity for average spectrum calculation
        mz_indices: NDArray[np.int_] = self._map_mass_to_indices(mzs)
        data_structures['total_intensity'][mz_indices] += intensities
        data_structures['pixel_count'] += 1
        
        if data_structures['mode'] == '2d_slices':
            # For 2D slices mode, add data to the appropriate slice
            slice_id: str = f"{self.dataset_id}_z{z}"
            if slice_id in data_structures['slices_data']:
                slice_data: Dict[str, Any] = data_structures['slices_data'][slice_id]
                pixel_idx: int = y * self._dimensions[0] + x
                
                # Store TIC value for this pixel
                slice_data['tic_values'][y, x] = tic_value
                
                # Add to sparse matrix for this slice
                self._add_to_sparse_matrix(
                    slice_data['sparse_data'], 
                    pixel_idx, 
                    mz_indices, 
                    intensities
                )
        else:
            # For 3D volume mode, add data to the single sparse matrix
            pixel_idx: int = self._get_pixel_index(x, y, z)
            
            # Store TIC value for this pixel
            data_structures['tic_values'][y, x, z] = tic_value
            
            # Add to sparse matrix
            self._add_to_sparse_matrix(
                data_structures['sparse_data'], 
                pixel_idx, 
                mz_indices, 
                intensities
            )
    
    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """
        Finalize SpatialData structures by creating tables, shapes, and images.
        
        Args:
            data_structures: Data structures containing processed data
            
        Raises:
            ImportError: If required SpatialData dependencies are not available
            ValueError: If dimensions are not initialized
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")
            
        import xarray as xr  # type: ignore
        from spatialdata.models import Image2DModel  # type: ignore
        
        # Calculate average mass spectrum using only non-zero pixels
        if data_structures['pixel_count'] > 0:
            avg_spectrum: NDArray[np.float64] = data_structures['total_intensity'] / data_structures['pixel_count']
        else:
            avg_spectrum: NDArray[np.float64] = data_structures['total_intensity'].copy()
        
        # Create a table for the average spectrum
        avg_spectrum_table: pd.DataFrame = pd.DataFrame({
            'spectrum_type': ['average'],
            'description': ['Average mass spectrum across all non-zero pixels']
        })
        avg_spectrum_table.set_index('spectrum_type', inplace=True)  # type: ignore
        
        # Create the AnnData for the average spectrum
        avg_adata: AnnData = AnnData(  # type: ignore
            X=avg_spectrum.reshape(1, -1),
            obs=avg_spectrum_table,
            var=data_structures['var_df']
        )
        
        # Add the average spectrum to tables
        data_structures['tables']['average_spectrum'] = avg_adata

        # Store pixel count for metadata
        self._non_empty_pixel_count = data_structures['pixel_count']
        
        if data_structures['mode'] == '2d_slices':
            # Process each slice
            for slice_id, slice_data in data_structures['slices_data'].items():
                try:
                    # Convert to CSR format for efficiency
                    slice_data['sparse_data'] = slice_data['sparse_data'].tocsr()
                    
                    # Create AnnData for this slice
                    adata: AnnData = AnnData(
                        X=slice_data['sparse_data'],
                        obs=slice_data['coords_df'],
                        var=data_structures['var_df']
                    )
                    
                    # Make sure region column exists and is correct
                    region_key: str = f"{slice_id}_pixels"
                    if 'region' not in adata.obs.columns:
                        adata.obs['region'] = region_key
                    
                    # Make sure instance_key is a string column
                    adata.obs['instance_key'] = adata.obs.index.astype(str)  #type: ignore
                    
                    # Create table model
                    table: 'TableModel' = TableModel.parse(  # type: ignore
                        adata,
                        region=region_key,
                        region_key="region",
                        instance_key="instance_key"
                    )
                    
                    # Add to tables and create shapes
                    data_structures['tables'][slice_id] = table
                    data_structures['shapes'][region_key] = self._create_pixel_shapes(adata, is_3d=False)
                    
                    # Create TIC image for this slice
                    # Use the actual shape of the TIC values array for coordinates
                    tic_values: NDArray[np.float64] = slice_data['tic_values']
                    y_size, x_size = tic_values.shape
                    
                    # Add channel dimension to make it (c, y, x) as required by SpatialData
                    tic_values_with_channel: NDArray[np.float64] = tic_values.reshape(1, y_size, x_size)
                    
                    tic_image: 'xr.DataArray' = xr.DataArray(  # type: ignore
                        tic_values_with_channel,
                        dims=('c', 'y', 'x'),
                        coords={
                            'c': [0],  # Single channel
                            'y': np.arange(y_size) * self.pixel_size_um,
                            'x': np.arange(x_size) * self.pixel_size_um,
                        }
                    )
                    
                    # Create Image2DModel for the TIC image
                    transform: 'Identity' = Identity()  # type: ignore
                    data_structures['images'][f"{slice_id}_tic"] = Image2DModel.parse(  # type: ignore
                        tic_image,
                        transformations={slice_id: transform, "global": transform}
                    )
                    
                except Exception as e:
                    logging.error(f"Error processing slice {slice_id}: {e}")
                    import traceback
                    logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
                    raise
        else:
            try:
                # Process the single 3D volume or 2D slice
                # Convert to CSR format
                data_structures['sparse_data'] = data_structures['sparse_data'].tocsr()
                
                # Create AnnData
                adata: AnnData = AnnData(  # type: ignore
                    X=data_structures['sparse_data'],
                    obs=data_structures['coords_df'],
                    var=data_structures['var_df']
                )
                
                # Make sure region column exists and is correct
                region_key: str = f"{self.dataset_id}_pixels"
                if 'region' not in adata.obs.columns:
                    adata.obs['region'] = region_key
                
                # Ensure instance_key is a string column
                adata.obs['instance_key'] = adata.obs.index.astype(str)  #type: ignore
                
                # Create table model
                table: 'TableModel' = TableModel.parse(  # type: ignore
                    adata,
                    region=region_key,
                    region_key="region",
                    instance_key="instance_key"
                )
                
                # Add to tables and create shapes
                data_structures['tables'][self.dataset_id] = table
                data_structures['shapes'][region_key] = self._create_pixel_shapes(adata, is_3d=self.handle_3d)
                
                # Create TIC image
                if self.handle_3d:
                    # 3D TIC image
                    # Use the actual shape of the TIC values array for coordinates
                    tic_values: NDArray[np.float64] = data_structures['tic_values']
                    z_size, y_size, x_size = tic_values.shape
                    
                    # Add channel dimension for 3D image
                    tic_values_with_channel: NDArray[np.float64] = tic_values.reshape(1, z_size, y_size, x_size)
                    
                    tic_image: 'xr.DataArray' = xr.DataArray(  # type: ignore
                        tic_values_with_channel,
                        dims=('c', 'z', 'y', 'x'),
                        coords={
                            'c': [0],  # Single channel
                            'z': np.arange(z_size) * self.pixel_size_um,
                            'y': np.arange(y_size) * self.pixel_size_um,
                            'x': np.arange(x_size) * self.pixel_size_um,
                        }
                    )
                    
                    # Create Image model for 3D image
                    transform: 'Identity' = Identity()  # type: ignore
                    try:
                        from spatialdata.models import Image3DModel  # type: ignore
                        data_structures['images'][f"{self.dataset_id}_tic"] = Image3DModel.parse(  # type: ignore
                            tic_image,
                            transformations={self.dataset_id: transform, "global": transform}
                        )
                    except (ImportError, AttributeError):
                        # Fallback if Image3DModel is not available
                        logging.warning("Image3DModel not available, using generic image model")
                        from spatialdata.models import ImageModel  # type: ignore
                        data_structures['images'][f"{self.dataset_id}_tic"] = ImageModel.parse(  # type: ignore
                            tic_image,
                            transformations={self.dataset_id: transform, "global": transform}
                        )
                else:
                    # 2D TIC image
                    # Use the actual shape of the TIC values array for coordinates
                    if len(data_structures['tic_values'].shape) == 3:
                        # Take first z-plane from 3D array
                        tic_values: NDArray[np.float64] = data_structures['tic_values'][:, :, 0]
                    else:
                        tic_values: NDArray[np.float64] = data_structures['tic_values']
                    
                    y_size, x_size = tic_values.shape
                    
                    # Add channel dimension to make it (c, y, x)
                    tic_values_with_channel: NDArray[np.float64] = tic_values.reshape(1, y_size, x_size)
                    
                    tic_image: 'xr.DataArray' = xr.DataArray(  # type: ignore
                        tic_values_with_channel,
                        dims=('c', 'y', 'x'),
                        coords={
                            'c': [0],  # Single channel
                            'y': np.arange(y_size) * self.pixel_size_um,
                            'x': np.arange(x_size) * self.pixel_size_um,
                        }
                    )
                    
                    # Create Image2DModel for the TIC image
                    transform: 'Identity' = Identity()  # type: ignore
                    data_structures['images'][f"{self.dataset_id}_tic"] = Image2DModel.parse(  # type: ignore
                        tic_image,
                        transformations={self.dataset_id: transform, "global": transform}
                    )
                
            except Exception as e:
                logging.error(f"Error processing 3D volume: {e}")
                import traceback
                logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
                raise
    
    def _create_pixel_shapes(self, adata: AnnData, is_3d: bool = False) -> 'ShapesModel':
        """
        Create geometric shapes for pixels with proper transformations.
        
        Args:
            adata: AnnData object containing coordinates
            is_3d: Whether to handle as 3D data
            
        Returns:
            SpatialData shapes model
            
        Raises:
            ImportError: If required SpatialData dependencies are not available
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")
            
        # Extract coordinates directly from obs
        x_coords: NDArray[np.float64] = adata.obs['spatial_x'].values  # type: ignore
        y_coords: NDArray[np.float64] = adata.obs['spatial_y'].values  # type: ignore
        
        # Create geometries efficiently using vectorized operations
        half_pixel: float = self.pixel_size_um / 2
        
        # Create geometries list - this can be optimized but must remain a list for geopandas
        geometries: List[Any] = []
        for i in range(len(adata)):
            x, y = x_coords[i], y_coords[i]
            
            # Create a square centered at pixel coordinates
            pixel_box = box(  # type: ignore
                x - half_pixel, 
                y - half_pixel,
                x + half_pixel,
                y + half_pixel
            )
            geometries.append(pixel_box)
        
        # Create GeoDataFrame
        gdf: 'GeoDataFrame' = gpd.GeoDataFrame(geometry=geometries, index=adata.obs.index)  # type: ignore
        
        # Set up transform
        transform: 'Identity' = Identity()  # type: ignore
        transformations: Dict[str, 'Identity'] = {self.dataset_id: transform, "global": transform}
        
        # Parse shapes
        shapes: 'ShapesModel' = ShapesModel.parse(  # type: ignore
            gdf,
            transformations=transformations
        )
        
        return shapes
    
    def _save_output(self, data_structures: Dict[str, Any]) -> bool:
        """
        Save the data to SpatialData format.
        
        Args:
            data_structures: Data structures to save
            
        Returns:
            True if saving was successful, False otherwise
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")
            
        try:
            # Create SpatialData object with images included
            sdata: 'SpatialData' = SpatialData(  # type: ignore
                tables=data_structures['tables'],
                shapes=data_structures['shapes'],
                images=data_structures['images']
            )
            
            # Add metadata
            self.add_metadata(sdata)
            
            # Write to disk
            sdata.write(str(self.output_path))
            logging.info(f"Successfully saved SpatialData to {self.output_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving SpatialData: {e}")
            import traceback
            logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
            return False
    
    def add_metadata(self, metadata: 'SpatialData') -> None:  # type: ignore
        """
        Add metadata to the SpatialData object.
        
        Args:
            metadata: SpatialData object to add metadata to
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._metadata is None:
            raise ValueError("Metadata is not initialized")
            
        # Add dataset metadata if SpatialData supports it
        if hasattr(metadata, 'metadata'):
            metadata.metadata = {  #type: ignore
                'dataset_id': self.dataset_id,
                'pixel_size_um': self.pixel_size_um,
                'source': self._metadata.get('source', 'unknown'),
                'msi_metadata': self._metadata,
                'total_grid_pixels': self._dimensions[0] * self._dimensions[1] * self._dimensions[2],
                'non_empty_pixels': self._non_empty_pixel_count
            }