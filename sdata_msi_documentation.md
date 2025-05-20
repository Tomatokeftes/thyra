# SpatialData MSI Documentation

This document explains how to access Mass Spectrometry Imaging (MSI) data stored in SpatialData objects created by the MSIConverter package, including Total Ion Current (TIC) images, average mass spectra, and other metadata components.

## Table of Contents

- [SpatialData Structure Overview](#spatialdata-structure-overview)
- [Accessing TIC Images](#accessing-tic-images)
- [Accessing Average Mass Spectrum](#accessing-average-mass-spectrum)
- [Pixel Count and Data Representation](#pixel-count-and-data-representation)
- [Accessing Pixel Size Information](#accessing-pixel-size-information)
- [Accessing Raw MSI Data](#accessing-raw-msi-data)
- [Working with 3D Data and Slices](#working-with-3d-data-and-slices)
- [Common Operations](#common-operations)
- [Implementation Notes](#implementation-notes)

## SpatialData Structure Overview

The SpatialData object created by MSIConverter contains the following components:

### Main Components

1. **Tables**: Stores the primary MSI data (m/z values × pixels) and the average spectrum
   - The MSI data table: `"{dataset_id}"` or `"{dataset_id}_z{slice_index}"` for sliced 3D data
   - The average spectrum table: `"average_spectrum"`

2. **Images**: Contains the TIC image
   - TIC image: `"{dataset_id}_tic"` or `"{dataset_id}_z{slice_index}_tic"` for sliced 3D data

3. **Shapes**: Contains the pixel shapes (geometry)
   - Pixel shapes: `"{dataset_id}_pixels"` or `"{dataset_id}_z{slice_index}_pixels"` for sliced 3D data

### Basic Access

```python
from spatialdata import SpatialData

# Load the SpatialData object
sdata = SpatialData.read("path/to/dataset.zarr")

# List available tables
print("Available tables:", list(sdata.tables.keys()))

# List available images
print("Available images:", list(sdata.images.keys()))

# List available shapes
print("Available shapes:", list(sdata.shapes.keys()))

# Access metadata
print("Metadata:", sdata.metadata)
```

## Accessing TIC Images

The Total Ion Current (TIC) image represents the sum of all intensities across the mass spectrum for each pixel.

### Finding TIC Image Keys

```python
# Get all image keys containing "tic"
tic_keys = [key for key in sdata.images.keys() if "tic" in key.lower()]
print("TIC image keys:", tic_keys)
```

### Accessing the TIC Image

```python
# Get the TIC image (assuming single dataset)
tic_key = tic_keys[0]  # e.g., "msi_dataset_tic"
tic_image = sdata.images[tic_key]

# Get the image data as an xarray DataArray
tic_data = tic_image.data

# The image is stored with dimensions (c, y, x) for 2D or (c, z, y, x) for 3D
# where 'c' is the channel dimension (typically just 1 channel for TIC)

# Extract image without channel dimension
tic_array = tic_data.sel(c=0).values

# Get coordinates
y_coords = tic_data.coords['y'].values
x_coords = tic_data.coords['x'].values
```

### Visualizing the TIC Image

```python
import matplotlib.pyplot as plt

# Simple visualization
plt.figure(figsize=(10, 8))
plt.imshow(tic_array, extent=[x_coords.min(), x_coords.max(), y_coords.max(), y_coords.min()])
plt.colorbar(label='TIC Intensity')
plt.xlabel('X position (µm)')
plt.ylabel('Y position (µm)')
plt.title('Total Ion Current (TIC) Image')
plt.show()
```

## Accessing Average Mass Spectrum

The average mass spectrum represents the mean intensity across all non-empty pixels for each m/z value.

### Finding Average Spectrum Keys

```python
# Get all table keys containing "average"
avg_keys = [key for key in sdata.tables.keys() if "average" in key.lower()]
print("Average spectrum keys:", avg_keys)
```

### Accessing the Average Spectrum

```python
# Get the average spectrum table
avg_key = "average_spectrum"  # Default key name
avg_table = sdata.tables[avg_key]

# The table is an AnnData object

# Get m/z values from var DataFrame
mz_values = avg_table.var["mz"].values if "mz" in avg_table.var.columns else np.arange(avg_table.n_vars)

# Get intensity values from X matrix
import scipy.sparse as sparse
if sparse.issparse(avg_table.X):
    intensities = avg_table.X.toarray().flatten()
else:
    intensities = avg_table.X.flatten()
```

### Visualizing the Average Spectrum

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.plot(mz_values, intensities)
plt.xlabel('m/z')
plt.ylabel('Average Intensity')
plt.title('Average Mass Spectrum')
plt.xlim(min(mz_values), max(mz_values))
plt.grid(True, alpha=0.3)
plt.show()
```

## Pixel Count and Data Representation

The MSI data is stored in a rectangular grid format, but not all pixels in the grid may contain actual data. Understanding this is important for correct analysis and interpretation.

### Empty vs. Non-Empty Pixels

- **Total Pixels**: The total number of pixels in the rectangular grid (width × height)
- **Non-Empty Pixels**: The number of pixels that contain actual spectral data
- **Empty Pixels**: Pixels in the grid that don't contain measurements (these exist to maintain a regular grid)

### Average Spectrum Calculation

The average mass spectrum is calculated using **only the non-empty pixels** (pixels with actual data). This provides a more accurate representation of the average spectral intensities by excluding areas where no measurements were taken.

### Checking Pixel Counts

```python
def count_pixels(sdata):
    """Count total and non-empty pixels."""
    # Get TIC image
    tic_key = [key for key in sdata.images.keys() if "tic" in key.lower()][0]
    tic_image = sdata.images[tic_key]
    tic_array = tic_image.data.sel(c=0).values if hasattr(tic_image.data, 'sel') else tic_image.data
    
    # Count total and non-empty pixels
    total_pixels = tic_array.size
    non_empty_pixels = np.count_nonzero(tic_array)
    
    print(f"Total pixels in grid: {total_pixels}")
    print(f"Non-empty pixels with data: {non_empty_pixels}")
    print(f"Percentage of grid with data: {non_empty_pixels/total_pixels*100:.1f}%")
    
    return total_pixels, non_empty_pixels
```

## Accessing Pixel Size Information

Pixel size information is stored in multiple locations within the SpatialData object.

### From SpatialData Metadata

```python
# Access from top-level metadata
if "pixel_size_um" in sdata.metadata:
    pixel_size = sdata.metadata["pixel_size_um"]
    print(f"Pixel size from metadata: {pixel_size} µm")
```

### From Image Coordinates

```python
# Calculate from image coordinates
y_coords = tic_image.data.coords["y"].values
x_coords = tic_image.data.coords["x"].values

if len(y_coords) > 1 and len(x_coords) > 1:
    pixel_size_y = abs(y_coords[1] - y_coords[0])
    pixel_size_x = abs(x_coords[1] - x_coords[0])
    print(f"Pixel size from coordinates: {pixel_size_x} × {pixel_size_y} µm")
```

### From Table Metadata

```python
# Get main MSI data table (usually dataset_id or similar)
main_key = [key for key in sdata.tables.keys() if "average" not in key.lower()][0]
main_table = sdata.tables[main_key]

# Check if pixel size is in table metadata
if "spatial" in main_table.uns and "pixel_size_um" in main_table.uns["spatial"]:
    pixel_size = main_table.uns["spatial"]["pixel_size_um"]
    print(f"Pixel size from table metadata: {pixel_size} µm")
```

## Accessing Raw MSI Data

The primary MSI data consists of intensity values for each m/z value at each pixel position.

### Getting Main Data Table

```python
# Find main MSI data table (excluding average spectrum table)
main_keys = [key for key in sdata.tables.keys() if "average" not in key.lower()]
print("Main MSI data tables:", main_keys)

# Get the main table (assuming single dataset)
main_key = main_keys[0]  # e.g., "msi_dataset"
msi_table = sdata.tables[main_key]
```

### Accessing Intensity Data

```python
# Get the intensity matrix (pixels × m/z values)
X = msi_table.X  # Often a sparse matrix

# Convert to dense matrix if needed (careful with memory for large datasets)
import scipy.sparse as sparse
if sparse.issparse(X) and X.shape[0] * X.shape[1] < 1e8:  # Memory check
    X_dense = X.toarray()
else:
    # Keep as sparse for large datasets
    X_dense = None
    print(f"Matrix is sparse with shape {X.shape} (pixels × m/z values)")

# Get m/z values
mz_values = msi_table.var["mz"].values if "mz" in msi_table.var.columns else np.arange(msi_table.n_vars)
```

### Accessing Pixel Coordinates

```python
# Get pixel coordinates from obs DataFrame
pixel_coords = None

# Check for different coordinate column patterns
if all(col in msi_table.obs.columns for col in ["spatial_x", "spatial_y"]):
    pixel_coords = msi_table.obs[["spatial_x", "spatial_y"]].values
elif all(col in msi_table.obs.columns for col in ["x", "y"]):
    pixel_coords = msi_table.obs[["x", "y"]].values

if pixel_coords is not None:
    print(f"Pixel coordinates available for {len(pixel_coords)} pixels")
```

### Extracting Individual Spectra

```python
# Get spectrum for a specific pixel (by index)
pixel_idx = 0  # First pixel
pixel_spectrum = X[pixel_idx].toarray().flatten() if sparse.issparse(X) else X[pixel_idx]

# Plot the spectrum
plt.figure(figsize=(12, 5))
plt.plot(mz_values, pixel_spectrum)
plt.xlabel('m/z')
plt.ylabel('Intensity')
plt.title(f'Mass Spectrum for Pixel {pixel_idx}')
plt.show()
```

## Working with 3D Data and Slices

For 3D datasets, MSIConverter can either handle them as true 3D volumes or as a series of 2D slices.

### Detecting Dataset Dimensionality

```python
# Check if we have 3D data based on slice naming pattern
slice_keys = [key for key in sdata.tables.keys() if "_z" in key]
is_sliced_3d = len(slice_keys) > 0

# Check if we have true 3D data based on z coordinate
main_key = main_keys[0]
main_table = sdata.tables[main_key]
is_true_3d = "z" in main_table.obs.columns or "spatial_z" in main_table.obs.columns

print(f"Dataset is sliced 3D: {is_sliced_3d}")
print(f"Dataset is true 3D: {is_true_3d}")
```

### Accessing Individual Z-Slices

```python
if is_sliced_3d:
    # Get available z indices
    z_indices = sorted(set(int(key.split("_z")[1].split("_")[0]) for key in slice_keys))
    print(f"Available z-slices: {z_indices}")
    
    # Access a specific z-slice
    z_idx = z_indices[0]  # First slice
    slice_key = f"{dataset_id}_z{z_idx}"
    slice_table = sdata.tables[slice_key]
    
    # Get corresponding TIC image
    slice_tic_key = f"{slice_key}_tic"
    if slice_tic_key in sdata.images:
        slice_tic = sdata.images[slice_tic_key]
        slice_tic_array = slice_tic.data.sel(c=0).values
        
        # Visualize slice
        plt.figure(figsize=(10, 8))
        plt.imshow(slice_tic_array)
        plt.title(f'TIC Image for Z-Slice {z_idx}')
        plt.colorbar(label='TIC Intensity')
        plt.show()
```

### Working with True 3D Data

```python
if is_true_3d:
    # Get z coordinates
    z_coord = main_table.obs["spatial_z"] if "spatial_z" in main_table.obs.columns else main_table.obs["z"]
    z_values = sorted(set(z_coord))
    print(f"Z values: {z_values}")
    
    # Get data for a specific z plane
    z_value = z_values[0]  # First z-plane
    z_mask = z_coord == z_value
    z_indices = np.where(z_mask)[0]
    
    if sparse.issparse(X):
        z_plane_data = X[z_indices].toarray() if len(z_indices) < 1e5 else X[z_indices]
    else:
        z_plane_data = X[z_indices]
    
    print(f"Extracted data for z-plane {z_value} with {len(z_indices)} pixels")
    
    # Calculate TIC for this z-plane
    if sparse.issparse(z_plane_data):
        z_tic = np.array(z_plane_data.sum(axis=1)).flatten()
    else:
        z_tic = z_plane_data.sum(axis=1)
```

## Common Operations

### Recalculating TIC Image

```python
import numpy as np

# Get the main MSI data
main_key = [key for key in sdata.tables.keys() if "average" not in key.lower()][0]
msi_table = sdata.tables[main_key]
X = msi_table.X

# Calculate TIC for each pixel
if sparse.issparse(X):
    tic_values = np.array(X.sum(axis=1)).flatten()
else:
    tic_values = X.sum(axis=1)

# Reshape TIC values to image format
# (Depends on coordinate system, this is a simplified example)
if "spatial_x" in msi_table.obs.columns and "spatial_y" in msi_table.obs.columns:
    x_positions = msi_table.obs["spatial_x"].values
    y_positions = msi_table.obs["spatial_y"].values
    
    x_unique = np.sort(np.unique(x_positions))
    y_unique = np.sort(np.unique(y_positions))
    
    # Create empty image
    recalc_tic = np.zeros((len(y_unique), len(x_unique)))
    
    # Convert to pixel indices
    x_indices = np.round((x_positions - x_unique.min()) / (x_unique[1] - x_unique[0])).astype(int)
    y_indices = np.round((y_positions - y_unique.min()) / (y_unique[1] - y_unique[0])).astype(int)
    
    # Fill image with TIC values
    for i, (x_idx, y_idx, tic) in enumerate(zip(x_indices, y_indices, tic_values)):
        if 0 <= x_idx < len(x_unique) and 0 <= y_idx < len(y_unique):
            recalc_tic[y_idx, x_idx] = tic
```

### Recalculating Average Spectrum

```python
# Calculate average spectrum (using only non-empty pixels)
if sparse.issparse(X):
    # Only include pixels with data (rows with at least one non-zero value)
    nonzero_rows = np.diff(X.indptr) > 0
    if np.any(nonzero_rows):
        X_nonzero = X[nonzero_rows]
        recalc_avg_spectrum = np.array(X_nonzero.mean(axis=0)).flatten()
    else:
        recalc_avg_spectrum = np.array(X.mean(axis=0)).flatten()
else:
    # For dense matrices
    nonzero_rows = np.any(X > 0, axis=1)
    if np.any(nonzero_rows):
        recalc_avg_spectrum = X[nonzero_rows].mean(axis=0)
    else:
        recalc_avg_spectrum = X.mean(axis=0)

# Get m/z values
mz_values = msi_table.var["mz"].values if "mz" in msi_table.var.columns else np.arange(msi_table.n_vars)

# Plot average spectrum
plt.figure(figsize=(12, 5))
plt.plot(mz_values, recalc_avg_spectrum)
plt.xlabel('m/z')
plt.ylabel('Average Intensity')
plt.title('Recalculated Average Mass Spectrum (Non-Empty Pixels Only)')
plt.show()
```

### Extracting Mass Spectrum at Specific m/z Value

```python
# Get intensity values for a specific m/z value
target_mz = 760.5  # Example: m/z value of interest
mz_tolerance = 0.1  # m/z tolerance

# Find the closest m/z index
mz_idx = np.abs(mz_values - target_mz).argmin()
found_mz = mz_values[mz_idx]

if abs(found_mz - target_mz) <= mz_tolerance:
    print(f"Found m/z {found_mz} (requested {target_mz})")
    
    # Extract intensity values for this m/z across all pixels
    if sparse.issparse(X):
        intensity_values = X[:, mz_idx].toarray().flatten()
    else:
        intensity_values = X[:, mz_idx]
    
    # Reshape to image format (simplified example)
    # (Similarly to TIC reshape above)
```

## Implementation Notes

This section provides technical details about how the data is processed and stored.

### Average Spectrum Calculation

The average mass spectrum is calculated using **only non-empty pixels** (those containing actual spectral data). This approach provides a more accurate representation of the average intensity across the sample by:

1. Excluding areas outside the sample or where no measurements were taken
2. Ensuring the average reflects only real data points
3. Preventing artificial dilution of signal intensity from including empty regions

### TIC Image Orientation

The TIC image follows a standard scientific imaging convention where:
- The first dimension (`y`) corresponds to rows (top to bottom)
- The second dimension (`x`) corresponds to columns (left to right)

This orientation (`y`, `x`) is consistent with common image processing libraries and visualization tools, and ensures that images are displayed with the correct spatial orientation.

### Data Sparsity

MSI data is typically very sparse (many pixels have zero intensity at most m/z values). The data is stored in a memory-efficient sparse format that only keeps track of non-zero intensity values. When accessing or processing this data:

1. Keep the data in sparse format whenever possible
2. Be careful when converting to dense representations for large datasets
3. Consider chunked processing for operations on large datasets