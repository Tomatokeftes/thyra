import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel
from spatialdata.transformations import Identity
from pathlib import Path
import zarr
from shapely.geometry import box
import geopandas as gpd


def convert_msi_to_spatialdata(
    zarr_path, 
    dataset_id="msi_dataset",
    pixel_size_um=1.0,
    handle_3d=False
):
    """
    Convert MSI data from Zarr store to SpatialData format.
    
    Parameters
    ----------
    zarr_path : str or Path
        Path to the MSI data Zarr store
    dataset_id : str
        Identifier for the dataset
    pixel_size_um : float
        Size of each pixel in micrometers
    handle_3d : bool
        Whether to process as 3D data or as 2D slices
        
    Returns
    -------
    SpatialData
        A SpatialData object containing the MSI data
    """
    # Load MSI data from Zarr store
    zarr_store = zarr.open(str(zarr_path), mode='r')
    
    # Load intensity data - assuming dimensions (c, z, y, x)
    msi_data = zarr_store['0'][:]
    
    # Load mass values
    mass_values = zarr_store['labels/common_mass_axis/0'][:]
    
    # Get dimensions
    n_masses, n_z, n_y, n_x = msi_data.shape
    
    tables = {}
    shapes = {}
    images = {}
    
    # If 3D data but we want to treat as 2D slices
    if n_z > 1 and not handle_3d:
        for z in range(n_z):
            # Extract this z-slice
            slice_data = msi_data[:, z, :, :]
            slice_id = f"{dataset_id}_z{z}"
            
            # Process this slice
            adata, shape = process_2d_slice(
                slice_data, mass_values, z, pixel_size_um, slice_id
            )
            
            tables[slice_id] = adata
            shapes[f"{slice_id}_pixels"] = shape
    else:
        # Handle as full 3D dataset or single 2D slice
        # Reshape data matrix to 2D (features × pixels)
        flattened_data = msi_data.reshape(n_masses, -1)
        
        # Convert to sparse matrix (transposed so pixels are rows, masses are columns)
        sparse_data = sparse.csr_matrix(flattened_data.T)
        
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
        coords_df['region'] = f"{dataset_id}_pixels"
        coords_df.set_index('instance_id', inplace=True)
        
        # Create variable dataframe
        var_df = pd.DataFrame({'mass': mass_values})
        var_df.set_index('mass', inplace=True)
        
        # Create AnnData
        adata = AnnData(
            X=sparse_data,
            obs=coords_df,
            var=var_df
        )
        
        # Store spatial coordinates
        adata.obsm['spatial'] = coords_df[['x', 'y', 'z']].values
        
        # Create table model
        # We need to explicitly provide instance_key
        adata.obs["instance_key"] = adata.obs.index.astype(str)
        tables[dataset_id] = TableModel.parse(
            adata,
            region=f"{dataset_id}_pixels",
            region_key="region",
            instance_key="instance_key"  # Now using the explicit column
        )
        
        # Create shapes
        shapes[f"{dataset_id}_pixels"] = create_pixel_shapes(
            adata, pixel_size_um, dataset_id
        )
    
    # No optical image handling in this version
    
    # Create SpatialData object
    sdata = SpatialData(
        tables=tables,
        shapes=shapes,
        images=images
    )
    
    return sdata


def process_2d_slice(slice_data, mass_values, z_value, pixel_size_um, slice_id):
    """Process a single 2D slice of MSI data"""
    n_masses, n_y, n_x = slice_data.shape
    
    # Reshape to 2D matrix
    flattened_data = slice_data.reshape(n_masses, -1)
    sparse_data = sparse.csr_matrix(flattened_data.T)
    
    # Create coordinate dataframe
    coords = []
    for y in range(n_y):
        for x in range(n_x):
            coords.append({
                'z': z_value,
                'y': y,
                'x': x,
                'instance_id': y * n_x + x
            })
    
    coords_df = pd.DataFrame(coords)
    coords_df['region'] = f"{slice_id}_pixels"
    coords_df.set_index('instance_id', inplace=True)
    
    # Create variable dataframe
    var_df = pd.DataFrame({'mass': mass_values})
    var_df.set_index('mass', inplace=True)
    
    # Create AnnData
    adata = AnnData(
        X=sparse_data,
        obs=coords_df,
        var=var_df
    )
    
    # Store spatial coordinates
    adata.obsm['spatial'] = coords_df[['x', 'y']].values
    
    # Create table model
    adata.obs["instance_key"] = adata.obs.index.astype(str)
    table = TableModel.parse(
        adata,
        region=f"{slice_id}_pixels",
        region_key="region",
        instance_key="instance_key"  # Using explicit column
    )
    
    # Create shapes
    shapes = create_pixel_shapes(adata, pixel_size_um, slice_id)
    
    return table, shapes


def create_pixel_shapes(adata, pixel_size_um, dataset_id):
    """Create geometric shapes for pixels with proper transformations"""

    
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
            x - pixel_size_um/2, 
            y - pixel_size_um/2,
            x + pixel_size_um/2,
            y + pixel_size_um/2
        )
        geometries.append(pixel_box)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        geometry=geometries,
        index=adata.obs.index
    )
    
    # Set up transform (identity transform for original resolution)
    transform = Identity()
    transformations = {dataset_id: transform, "global": transform}  # Add both dataset_id and global
    
    # Parse shapes
    shapes = ShapesModel.parse(
        gdf,
        transformations=transformations
    )
    
    return shapes

# Example usage:
if __name__ == "__main__":
    # Convert MSI data to SpatialData
    sdata = convert_msi_to_spatialdata(
        zarr_path=r"C:\Users\P70078823\Desktop\VisiumHD\pea.zarr",
        dataset_id="my_msi_dataset",
        pixel_size_um=10.0,
        handle_3d=False
    )
    
    # Save the SpatialData object
    sdata.write(r"C:\Users\P70078823\Desktop\VisiumHD\pea_spatialdata.zarr")