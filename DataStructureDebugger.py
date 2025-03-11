import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import anndata as ad
from spatialdata import SpatialData
import zarr
import pandas as pd
import seaborn as sns
from scipy import sparse
import os

# Set the path to your zarr file
zarr_path = r"C:\Users\P70078823\Desktop\MSIConverter\brain_epilepsy2.zarr"

# Try loading with SpatialData
print("Attempting to load as SpatialData...")
try:
    data = SpatialData.read(zarr_path)
    print("Successfully loaded as SpatialData!")
    
    # Explore SpatialData structure
    print("\nSpatialData Structure:")
    print(f"Tables: {list(data.tables.keys()) if hasattr(data, 'tables') else 'None'}")
    print(f"Shapes: {list(data.shapes.keys()) if hasattr(data, 'shapes') else 'None'}")
    print(f"Images: {list(data.images.keys()) if hasattr(data, 'images') else 'None'}")
    
    # If there are tables, explore the first one
    if hasattr(data, 'tables') and len(data.tables) > 0:
        table_key = list(data.tables.keys())[0]
        table = data.tables[table_key]
        print(f"\nTable '{table_key}' information:")
        print(f"Observations (n_obs): {table.n_obs}")
        print(f"Variables (n_vars): {table.n_vars}")
        print(f"Observation columns: {list(table.obs.columns)}")
        print(f"Variable columns: {list(table.var.columns)}")
        
        # Check if spatial coordinates exist
        if 'spatial' in table.obsm:
            print("Found spatial coordinates in obsm['spatial']")
        elif any(col in table.obs.columns for col in ['x', 'y', 'spatial_x', 'spatial_y', 'X_position', 'Y_position']):
            print("Found spatial coordinates in obs columns")
        else:
            print("Could not find spatial coordinates")
            
except Exception as e:
    print(f"Failed to load as SpatialData: {str(e)}")
    
    # Try opening as zarr store directly
    print("\nAttempting to open as zarr store...")
    try:
        z = zarr.open(zarr_path, mode='r')
        print("Successfully opened as zarr!")
        
        # Explore zarr structure
        print("\nZarr Store Structure:")
        print(f"Root groups/arrays: {list(z.keys())}")
        
        # Check if it has table structure
        if 'tables' in z:
            print(f"Tables group exists with keys: {list(z['tables'].keys())}")
            
            # Explore first table
            if len(list(z['tables'].keys())) > 0:
                table_key = list(z['tables'].keys())[0]
                print(f"\nTable '{table_key}' structure:")
                table_group = z['tables'][table_key]
                print(f"Table groups/arrays: {list(table_group.keys())}")
                
                # Check for X matrix
                if 'X' in table_group:
                    print(f"X matrix exists with shape: {table_group['X'].shape}")
                    
                # Check for obs and var
                if 'obs' in table_group:
                    print(f"obs exists with keys: {list(table_group['obs'].keys())}")
                if 'var' in table_group:
                    print(f"var exists with keys: {list(table_group['var'].keys())}")
    except Exception as e2:
        print(f"Failed to open as zarr: {str(e2)}")

print("\nComplete debugging information gathered.")