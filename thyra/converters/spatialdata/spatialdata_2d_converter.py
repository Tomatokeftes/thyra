# thyra/converters/spatialdata/spatialdata_2d_converter.py

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse

from .base_spatialdata_converter import SPATIALDATA_AVAILABLE, BaseSpatialDataConverter

if SPATIALDATA_AVAILABLE:
    import xarray as xr
    from anndata import AnnData
    from spatialdata.models import Image2DModel, TableModel
    from spatialdata.transformations import Identity


class SpatialData2DConverter(BaseSpatialDataConverter):
    """Converter for MSI data to SpatialData format treating 3D data as
    separate 2D slices."""

    def __init__(self, *args, **kwargs):
        """Initialize 2D converter with handle_3d=False."""
        kwargs["handle_3d"] = False  # Force 2D mode
        super().__init__(*args, **kwargs)

    def _create_data_structures(self) -> Dict[str, Any]:
        """
        Create data structures for 2D slices format.

        Returns:
            Dict containing tables, shapes, images, and data arrays for
            2D slices
        """
        # Return dictionaries to store tables, shapes, and sparse matrices
        tables: Dict[str, Any] = {}
        shapes: Dict[str, Any] = {}
        images: Dict[str, Any] = {}

        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, n_z = self._dimensions

        # Create a structure for each slice
        slices_data: Dict[str, Dict[str, Any]] = {}
        for z in range(n_z):
            slice_id = f"{self.dataset_id}_z{z}"
            slices_data[slice_id] = {
                "sparse_data": self._create_sparse_matrix_for_slice(z),
                "coords_df": self._create_coordinates_dataframe_for_slice(z),
                "tic_values": np.zeros((n_y, n_x), dtype=np.float64),
            }

        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        return {
            "mode": "2d_slices",
            "slices_data": slices_data,
            "tables": tables,
            "shapes": shapes,
            "images": images,
            "var_df": self._create_mass_dataframe(),
            "total_intensity": np.zeros(len(self._common_mass_axis), dtype=np.float64),
            "pixel_count": 0,
        }

    def _create_sparse_matrix_for_slice(self, z_value: int) -> sparse.lil_matrix:
        """
        Create sparse matrix for a single slice.

        Args:
            z_value: Z-index of the slice

        Returns:
            Sparse matrix for this slice
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis is not initialized")

        n_x, n_y, _ = self._dimensions
        n_pixels = n_x * n_y
        n_masses = len(self._common_mass_axis)

        logging.info(
            f"Creating sparse matrix for slice z={z_value} with {n_pixels} "
            f"pixels and {n_masses} mass values"
        )

        return sparse.lil_matrix((n_pixels, n_masses), dtype=np.float64)

    def _create_coordinates_dataframe_for_slice(self, z_value: int) -> pd.DataFrame:
        """
        Create a coordinates dataframe for a single Z-slice.

        Args:
            z_value: Z-index of the slice

        Returns:
            DataFrame with pixel coordinates
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        n_x, n_y, _ = self._dimensions

        # Pre-allocate arrays for better performance
        pixel_count = n_x * n_y
        y_values: NDArray[np.int32] = np.repeat(np.arange(n_y, dtype=np.int32), n_x)
        x_values: NDArray[np.int32] = np.tile(np.arange(n_x, dtype=np.int32), n_y)
        instance_ids: NDArray[np.int32] = np.arange(pixel_count, dtype=np.int32)

        # Create DataFrame in one operation
        coords_df = pd.DataFrame(
            {
                "y": y_values,
                "x": x_values,
                "instance_id": instance_ids,
                "region": f"{self.dataset_id}_z{z_value}_pixels",
            }
        )

        # Set index efficiently
        coords_df["instance_id"] = coords_df["instance_id"].astype(str)
        coords_df.set_index("instance_id", inplace=True)

        # Add spatial coordinates in a vectorized operation
        coords_df["spatial_x"] = coords_df["x"] * self.pixel_size_um
        coords_df["spatial_y"] = coords_df["y"] * self.pixel_size_um

        return coords_df

    def _process_single_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mzs: NDArray[np.float64],
        intensities: NDArray[np.float64],
    ) -> None:
        """
        Process a single spectrum for 2D slices format.
        Delegates to parent's resampling-aware processing.

        Args:
            data_structures: Data structures for storing processed data
            coords: (x, y, z) pixel coordinates
            mzs: Array of m/z values
            intensities: Array of intensity values
        """
        # Delegate to parent's resampling-aware processing
        super()._process_single_spectrum(data_structures, coords, mzs, intensities)

    def _process_resampled_spectrum(
        self,
        data_structures: Dict[str, Any],
        coords: Tuple[int, int, int],
        mz_indices: NDArray[np.int_],
        intensities: NDArray[np.float64],
    ) -> None:
        """
        Process a spectrum with resampled intensities for 2D slices format.

        Args:
            data_structures: Data structures for storing processed data
            coords: (x, y, z) pixel coordinates
            mz_indices: Indices in the common mass axis (all indices for
            resampled)
            intensities: Resampled intensity values
        """
        if self._dimensions is None:
            raise ValueError("Dimensions are not initialized")

        x, y, z = coords

        # Calculate TIC for this pixel
        tic_value = float(np.sum(intensities))

        # Update total intensity for average spectrum calculation
        # Handle both resampled and non-resampled cases
        if len(intensities) == len(data_structures["total_intensity"]):
            # Resampled case - intensities match common mass axis length
            data_structures["total_intensity"] += intensities
        else:
            # Non-resampled case - need to map to indices
            for i, intensity in enumerate(intensities):
                if i < len(mz_indices) and mz_indices[i] < len(
                    data_structures["total_intensity"]
                ):
                    data_structures["total_intensity"][mz_indices[i]] += intensity
        data_structures["pixel_count"] += 1

        # Add data to the appropriate slice
        slice_id = f"{self.dataset_id}_z{z}"
        if slice_id in data_structures["slices_data"]:
            slice_data = data_structures["slices_data"][slice_id]
            pixel_idx = y * self._dimensions[0] + x

            # Store TIC value for this pixel
            slice_data["tic_values"][y, x] = tic_value

            # Add to sparse matrix for this slice
            self._add_to_sparse_matrix(
                slice_data["sparse_data"], pixel_idx, mz_indices, intensities
            )

    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """
        Finalize 2D slices data by creating tables, shapes, and images.

        Args:
            data_structures: Data structures containing processed data
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        # Calculate average mass spectrum
        if data_structures["pixel_count"] > 0:
            avg_spectrum = (
                data_structures["total_intensity"] / data_structures["pixel_count"]
            )
        else:
            avg_spectrum = data_structures["total_intensity"].copy()

        # Store pixel count for metadata
        self._non_empty_pixel_count = data_structures["pixel_count"]

        # Process each slice separately
        for slice_id, slice_data in data_structures["slices_data"].items():
            try:
                # Convert lil_matrix to csr_matrix for efficient access
                sparse_matrix = slice_data["sparse_data"].tocsr()
                logging.info(
                    f"Converted sparse matrix for {slice_id}: "
                    f"{sparse_matrix.nnz:,} non-zero entries"
                )

                # Create AnnData for this slice
                adata = AnnData(
                    X=sparse_matrix,
                    obs=slice_data["coords_df"],
                    var=data_structures["var_df"],
                )

                # Add average spectrum to .uns
                adata.uns["average_spectrum"] = avg_spectrum

                # Make sure region column exists and is correct
                region_key = f"{slice_id}_pixels"
                if "region" not in adata.obs.columns:
                    adata.obs["region"] = region_key

                # Make sure instance_key is a string column
                adata.obs["instance_key"] = adata.obs.index.astype(str)

                # Create table model
                table = TableModel.parse(
                    adata,
                    region=region_key,
                    region_key="region",
                    instance_key="instance_key",
                )

                # Add to tables and create shapes
                data_structures["tables"][slice_id] = table
                data_structures["shapes"][region_key] = self._create_pixel_shapes(
                    adata, is_3d=False
                )

                # Create TIC image for this slice
                tic_values = slice_data["tic_values"]
                y_size, x_size = tic_values.shape

                # Add channel dimension to make it (c, y, x) as required by
                # SpatialData
                tic_values_with_channel = tic_values.reshape(1, y_size, x_size)

                tic_image = xr.DataArray(
                    tic_values_with_channel,
                    dims=("c", "y", "x"),
                    coords={
                        "c": [0],  # Single channel
                        "y": np.arange(y_size) * self.pixel_size_um,
                        "x": np.arange(x_size) * self.pixel_size_um,
                    },
                )

                # Create Image2DModel for the TIC image
                transform = Identity()
                data_structures["images"][f"{slice_id}_tic"] = Image2DModel.parse(
                    tic_image,
                    transformations={
                        slice_id: transform,
                        "global": transform,
                    },
                )

            except Exception as e:
                logging.error(f"Error processing slice {slice_id}: {e}")
                import traceback

                logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
                raise
