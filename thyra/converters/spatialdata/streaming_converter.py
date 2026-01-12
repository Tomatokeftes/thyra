# thyra/converters/spatialdata/streaming_converter.py

"""Streaming SpatialData converter with zero-copy direct Zarr write.

This converter processes MSI data in a memory-efficient streaming manner:
- Two-pass approach: count nnz per row, then write directly to Zarr
- Zero-copy mode writes directly to final output (no scipy matrix in memory)
- Memory stays bounded regardless of dataset size
- Produces identical output to the standard SpatialData2DConverter
"""

import gc
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from numpy.typing import NDArray
from scipy import sparse
from tqdm import tqdm

from .base_spatialdata_converter import SPATIALDATA_AVAILABLE, BaseSpatialDataConverter

if SPATIALDATA_AVAILABLE:
    import xarray as xr
    from anndata import AnnData
    from spatialdata.models import Image2DModel, TableModel
    from spatialdata.transformations import Identity


class StreamingSpatialDataConverter(BaseSpatialDataConverter):
    """Memory-efficient streaming converter for MSI data to SpatialData format.

    Uses a two-pass approach to keep memory bounded during processing:
    - Pass 1: Count non-zeros per row to build indptr
    - Pass 2: Write indices and data directly to Zarr

    In zero_copy mode (default), writes directly to the final output without
    ever loading the sparse matrix into scipy. This means memory usage stays
    bounded regardless of dataset size - even for 1M+ pixels.

    Key advantages:
    - Memory stays bounded during processing regardless of dataset size
    - Zero-copy mode: no scipy matrix created during conversion
    - Handles sparse datasets where not all pixels have spectra
    - Produces identical output to SpatialData2DConverter
    """

    def __init__(
        self,
        *args,
        chunk_size: int = 5000,
        temp_dir: Optional[Path] = None,
        zero_copy: bool = True,
        **kwargs,
    ):
        """Initialize streaming converter.

        Args:
            *args: Arguments passed to BaseSpatialDataConverter
            chunk_size: Number of spectra to process before writing to disk.
                Larger values use more memory but reduce I/O overhead.
                Default: 5000 spectra per chunk.
            temp_dir: Directory for temporary Zarr files. If None, uses
                system temp directory. Cleaned up after conversion.
            zero_copy: If True (default), writes directly to final output
                without creating scipy sparse matrix. This avoids the final
                memory spike but may be slightly slower due to more disk I/O.
            **kwargs: Keyword arguments passed to BaseSpatialDataConverter
        """
        kwargs["handle_3d"] = False  # Force 2D mode for now
        super().__init__(*args, **kwargs)

        self._chunk_size = chunk_size
        self._temp_dir = temp_dir
        self._cleanup_temp = temp_dir is None
        self._zero_copy = zero_copy
        self._zarr_store: Optional[zarr.Group] = None
        self._temp_path: Optional[Path] = None

    def convert(self) -> bool:
        """Stream-convert MSI data to SpatialData format.

        Overrides the base convert() method. Uses zero-copy mode by default
        to avoid memory spikes during conversion.

        Returns:
            True if conversion was successful, False otherwise
        """
        if self._zero_copy:
            return self._convert_zero_copy()
        else:
            return self._convert_with_scipy()

    def _convert_zero_copy(self) -> bool:
        """Convert using zero-copy direct Zarr write (no scipy matrix).

        This method writes directly to the final output Zarr without ever
        creating a scipy sparse matrix in memory. Memory stays bounded
        regardless of dataset size.
        """
        try:
            # Initialize (loads metadata, mass axis, etc.)
            self._initialize_conversion()

            # Stream process and write directly to output - use new direct Zarr method
            result = self._stream_write_direct_zarr()

            logging.info(
                f"Zero-copy conversion complete: {result['total_nnz']:,} non-zeros"
            )
            return True

        except Exception as e:
            logging.error(f"Error during zero-copy conversion: {e}")
            import traceback

            logging.error(f"Detailed traceback:\n{traceback.format_exc()}")
            return False

        finally:
            self.reader.close()

    def _convert_with_scipy(self) -> bool:
        """Convert using scipy sparse matrix (original approach with memory spike)."""
        try:
            # Initialize (loads metadata, mass axis, etc.)
            self._initialize_conversion()

            # Set up temporary storage
            self._setup_temp_storage()

            # Stream process and build CSR in temp storage
            coo_result = self._stream_build_coo()

            # Create data structures from CSR (loads into scipy - causes spike)
            data_structures = self._create_data_structures_from_coo(coo_result)

            # Finalize (create tables, shapes, images)
            self._finalize_data(data_structures)

            # Save output
            success = self._save_output(data_structures)

            return success

        except Exception as e:
            logging.error(f"Error during streaming conversion: {e}")
            import traceback

            logging.error(f"Detailed traceback:\n{traceback.format_exc()}")
            return False

        finally:
            self.reader.close()
            self._cleanup_temp_storage()

    def _setup_temp_storage(self) -> None:
        """Set up temporary Zarr storage for COO components."""
        if self._temp_dir is None:
            self._temp_path = Path(tempfile.mkdtemp(prefix="streaming_coo_"))
            self._cleanup_temp = True
        else:
            self._temp_path = self._temp_dir
            self._cleanup_temp = False

        logging.info(f"Temp storage: {self._temp_path}")

        # Create Zarr store for COO chunks
        zarr_path = self._temp_path / "coo_chunks.zarr"
        self._zarr_store = zarr.open_group(str(zarr_path), mode="w")

    def _cleanup_temp_storage(self) -> None:
        """Clean up temporary storage."""
        if self._cleanup_temp and self._temp_path is not None:
            try:
                shutil.rmtree(self._temp_path, ignore_errors=True)
                logging.debug(f"Cleaned up temp storage: {self._temp_path}")
            except Exception as e:
                logging.warning(f"Failed to cleanup temp storage: {e}")

    def _stream_build_coo(self) -> Dict[str, Any]:
        """Stream-build CSR matrix using two-pass direct Zarr write.

        This approach uses bounded memory by:
        1. Pass 1: Count nnz per row to size arrays and build indptr
        2. Pass 2: Write indices and data directly to Zarr

        Returns:
            Dictionary with matrix info and metadata
        """
        if self._dimensions is None:
            raise ValueError("Dimensions not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis not initialized")
        if self._zarr_store is None:
            raise ValueError("Zarr store not initialized")

        n_x, n_y, n_z = self._dimensions
        n_rows = n_x * n_y * n_z
        n_cols = len(self._common_mass_axis)

        logging.info(
            f"Streaming CSR build (two-pass): {n_rows:,} pixels x {n_cols:,} cols"
        )

        # Track TIC values and total intensity for average spectrum
        tic_values = np.zeros((n_y, n_x), dtype=np.float64)
        total_intensity = np.zeros(n_cols, dtype=np.float64)

        # Set quiet mode on reader
        setattr(self.reader, "_quiet_mode", True)

        total_spectra = self._get_total_spectra_count()

        # ========== PASS 1: Count non-zeros per row ==========
        logging.info(
            f"Pass 1: Counting non-zeros per row ({total_spectra:,} spectra)..."
        )
        nnz_per_row = np.zeros(n_rows, dtype=np.int64)
        total_nnz = 0
        pixel_count = 0

        with tqdm(
            total=total_spectra,
            desc="Pass 1: Counting",
            unit="spectrum",
        ) as pbar:
            for coords, mzs, intensities in self.reader.iter_spectra(
                batch_size=self._buffer_size
            ):
                x, y, z = coords
                pixel_idx = z * (n_x * n_y) + y * n_x + x

                # Resample to get nnz count
                mz_indices, resampled_ints = self._process_spectrum(mzs, intensities)
                nnz = len(mz_indices)

                nnz_per_row[pixel_idx] = nnz
                total_nnz += nnz

                # Update TIC
                tic_value = float(np.sum(resampled_ints))
                if 0 <= y < n_y and 0 <= x < n_x:
                    tic_values[y, x] = tic_value

                # Update total intensity for average spectrum
                if nnz > 0:
                    np.add.at(total_intensity, mz_indices, resampled_ints)

                pixel_count += 1
                pbar.update(1)

        logging.info(f"Pass 1 complete: {total_nnz:,} total non-zeros")

        # Build indptr from nnz counts
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(nnz_per_row)
        del nnz_per_row
        gc.collect()

        # Create CSR component arrays in Zarr
        X_group = self._zarr_store.create_group("X")
        X_group.attrs["encoding-type"] = "csr_matrix"
        X_group.attrs["encoding-version"] = "0.1.0"
        X_group.attrs["shape"] = [n_rows, n_cols]

        # Write indptr immediately (it's relatively small)
        X_group.create_array("indptr", data=indptr.astype(np.int32))

        # Create indices and data arrays (will fill incrementally)
        chunk_size_zarr = min(total_nnz, 1000000)
        indices_arr = X_group.create_array(
            "indices",
            shape=(total_nnz,),
            dtype=np.int32,
            chunks=(chunk_size_zarr,),
        )
        data_arr = X_group.create_array(
            "data",
            shape=(total_nnz,),
            dtype=np.float64,
            chunks=(chunk_size_zarr,),
        )

        # ========== PASS 2: Write data directly to Zarr ==========
        logging.info("Pass 2: Writing data directly to Zarr...")

        # Reset reader for second pass
        reader_path = self.reader.data_path
        self.reader.close()
        self.reader = type(self.reader)(reader_path)
        setattr(self.reader, "_quiet_mode", True)

        # Process in chunks to batch Zarr writes
        chunk_indices: list = []
        chunk_data: list = []
        chunk_start_pos = 0
        spectra_in_chunk = 0

        with tqdm(
            total=total_spectra,
            desc="Pass 2: Writing",
            unit="spectrum",
        ) as pbar:
            for coords, mzs, intensities in self.reader.iter_spectra(
                batch_size=self._buffer_size
            ):
                # Resample spectrum
                mz_indices, resampled_ints = self._process_spectrum(mzs, intensities)

                if len(mz_indices) > 0:
                    chunk_indices.append(mz_indices.astype(np.int32))
                    chunk_data.append(resampled_ints.astype(np.float64))

                spectra_in_chunk += 1

                # Write chunk when full
                if spectra_in_chunk >= self._chunk_size:
                    if chunk_indices:
                        all_indices = np.concatenate(chunk_indices)
                        all_data = np.concatenate(chunk_data)

                        end_pos = chunk_start_pos + len(all_indices)
                        indices_arr[chunk_start_pos:end_pos] = all_indices
                        data_arr[chunk_start_pos:end_pos] = all_data

                        chunk_start_pos = end_pos
                        del all_indices, all_data

                    chunk_indices = []
                    chunk_data = []
                    spectra_in_chunk = 0
                    gc.collect()

                pbar.update(1)

        # Write final chunk
        if chunk_indices:
            all_indices = np.concatenate(chunk_indices)
            all_data = np.concatenate(chunk_data)

            end_pos = chunk_start_pos + len(all_indices)
            indices_arr[chunk_start_pos:end_pos] = all_indices
            data_arr[chunk_start_pos:end_pos] = all_data

        logging.info(f"Pass 2 complete: {total_nnz:,} non-zeros written to Zarr")

        # Calculate average spectrum
        avg_spectrum = total_intensity / max(pixel_count, 1)

        return {
            "total_nnz": total_nnz,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "tic_values": tic_values,
            "avg_spectrum": avg_spectrum,
            "pixel_count": pixel_count,
        }

    def _process_spectrum(
        self, mzs: np.ndarray, intensities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Process a single spectrum - resample and return indices/values.

        Args:
            mzs: Mass values
            intensities: Intensity values

        Returns:
            Tuple of (mz_indices, resampled_intensities)
        """
        if self._resampling_config:
            if hasattr(self, "_resampling_method"):
                from ...resampling import ResamplingMethod

                if self._resampling_method == ResamplingMethod.NEAREST_NEIGHBOR:
                    return self._nearest_neighbor_resample(mzs, intensities)
                else:
                    resampled_ints = self._resample_spectrum(mzs, intensities)
                    mz_indices = np.arange(len(self._common_mass_axis))
                    nonzero = resampled_ints != 0
                    return mz_indices[nonzero], resampled_ints[nonzero]
            else:
                resampled_ints = self._resample_spectrum(mzs, intensities)
                mz_indices = np.arange(len(self._common_mass_axis))
                nonzero = resampled_ints != 0
                return mz_indices[nonzero], resampled_ints[nonzero]
        else:
            # No resampling - use original indices
            mz_indices = self._map_mass_to_indices(mzs)
            return mz_indices, intensities

    def _create_data_structures_from_coo(
        self, coo_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create SpatialData structures from CSR components in Zarr.

        Reads CSR components (indptr, indices, data) from Zarr and creates
        a scipy sparse matrix. This is more memory-efficient than the old
        COO approach because we read pre-built CSR components.

        Args:
            coo_result: Dictionary with matrix info and metadata

        Returns:
            Data structures dictionary for SpatialData creation
        """
        logging.info("Reading CSR components from Zarr...")

        n_rows = coo_result["n_rows"]
        n_cols = coo_result["n_cols"]

        # Read CSR components directly from Zarr
        X_group = self._zarr_store["X"]
        indptr = X_group["indptr"][:]
        indices = X_group["indices"][:]
        data = X_group["data"][:]

        logging.info(f"Loaded CSR components: {len(data):,} entries")

        # Create CSR matrix directly (no COO intermediate)
        sparse_matrix = sparse.csr_matrix(
            (data, indices, indptr),
            shape=(n_rows, n_cols),
            dtype=np.float64,
        )

        del indptr, indices, data
        gc.collect()

        # Convert to CSC if needed
        if self._sparse_format == "csc":
            logging.info("Converting CSR to CSC format...")
            sparse_matrix = sparse_matrix.tocsc()
            gc.collect()

        logging.info(
            f"Created sparse matrix: {sparse_matrix.shape}, {sparse_matrix.nnz:,} nnz"
        )

        # Build data structures
        if self._dimensions is None:
            raise ValueError("Dimensions not initialized")

        n_x, n_y, n_z = self._dimensions

        # Create slice data structure (similar to 2D converter)
        slice_id = f"{self.dataset_id}_z0"

        tables: Dict[str, Any] = {}
        shapes: Dict[str, Any] = {}
        images: Dict[str, Any] = {}

        return {
            "mode": "2d_slices",
            "slices_data": {
                slice_id: {
                    "sparse_matrix": sparse_matrix,
                    "coords_df": self._create_coordinates_dataframe_for_slice(0),
                    "tic_values": coo_result["tic_values"],
                }
            },
            "tables": tables,
            "shapes": shapes,
            "images": images,
            "var_df": self._create_mass_dataframe(),
            "avg_spectrum": coo_result["avg_spectrum"],
            "pixel_count": coo_result["pixel_count"],
        }

    def _create_data_structures(self) -> Dict[str, Any]:
        """Not used in streaming mode - required by ABC."""
        raise NotImplementedError("Streaming converter uses _stream_build_coo instead")

    def _create_coordinates_dataframe_for_slice(self, z_value: int) -> pd.DataFrame:
        """Create a coordinates dataframe for a single Z-slice.

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

    def _finalize_data(self, data_structures: Dict[str, Any]) -> None:
        """Finalize data by creating tables, shapes, and images.

        Args:
            data_structures: Data structures containing processed data
        """
        if not SPATIALDATA_AVAILABLE:
            raise ImportError("SpatialData dependencies not available")

        avg_spectrum = data_structures["avg_spectrum"]
        self._non_empty_pixel_count = data_structures["pixel_count"]

        # Process each slice
        for slice_id, slice_data in data_structures["slices_data"].items():
            try:
                sparse_matrix = slice_data["sparse_matrix"]
                coords_df = slice_data["coords_df"]

                # Create AnnData for this slice
                adata = AnnData(
                    X=sparse_matrix,
                    obs=coords_df,
                    var=data_structures["var_df"],
                )

                # Add average spectrum to .uns
                adata.uns["average_spectrum"] = avg_spectrum

                # Add MSI metadata to .uns
                self._add_metadata_to_uns(adata)

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

                # Add channel dimension to make it (c, y, x) as required by SpatialData
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

        # Add optical images if available
        self._add_optical_images(data_structures)

    def _stream_write_direct(self) -> Dict[str, Any]:
        """Stream CSR data directly to final output (zero-copy approach).

        This method writes directly to the final SpatialData/Zarr output without
        ever creating a scipy sparse matrix in memory.

        Approach:
        1. Pass 1: Count non-zeros per row to build indptr and collect TIC
        2. Create placeholder SpatialData structure (empty X matrix)
        3. Pass 2: Stream-write indices and data directly to output Zarr

        Returns:
            Dictionary with conversion statistics
        """
        if self._dimensions is None:
            raise ValueError("Dimensions not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis not initialized")

        n_x, n_y, n_z = self._dimensions
        n_rows = n_x * n_y * n_z
        n_cols = len(self._common_mass_axis)
        slice_id = f"{self.dataset_id}_z0"
        region_key = f"{slice_id}_pixels"

        logging.info(f"Zero-copy streaming: {n_rows:,} pixels x {n_cols:,} cols")

        # Track TIC values and total intensity for average spectrum
        tic_values = np.zeros((n_y, n_x), dtype=np.float64)
        total_intensity = np.zeros(n_cols, dtype=np.float64)

        # Set quiet mode on reader
        setattr(self.reader, "_quiet_mode", True)
        total_spectra = self._get_total_spectra_count()

        # ========== PASS 1: Count non-zeros per row ==========
        logging.info(f"Pass 1: Counting non-zeros ({total_spectra:,} spectra)...")
        nnz_per_row = np.zeros(n_rows, dtype=np.int64)
        total_nnz = 0
        pixel_count = 0

        with tqdm(
            total=total_spectra,
            desc="Pass 1: Counting",
            unit="spectrum",
        ) as pbar:
            for coords, mzs, intensities in self.reader.iter_spectra(
                batch_size=self._buffer_size
            ):
                x, y, z = coords
                pixel_idx = z * (n_x * n_y) + y * n_x + x

                # Resample to get nnz count
                mz_indices, resampled_ints = self._process_spectrum(mzs, intensities)
                nnz = len(mz_indices)

                nnz_per_row[pixel_idx] = nnz
                total_nnz += nnz

                # Update TIC
                tic_value = float(np.sum(resampled_ints))
                if 0 <= y < n_y and 0 <= x < n_x:
                    tic_values[y, x] = tic_value

                # Update total intensity for average spectrum
                if nnz > 0:
                    np.add.at(total_intensity, mz_indices, resampled_ints)

                pixel_count += 1
                pbar.update(1)

        logging.info(f"Pass 1 complete: {total_nnz:,} total non-zeros")

        # Build indptr from nnz counts
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        indptr[1:] = np.cumsum(nnz_per_row)
        del nnz_per_row
        gc.collect()

        # ========== Create placeholder SpatialData structure ==========
        logging.info("Creating output structure...")

        # Create empty placeholder CSR (correct shape, zero nnz)
        placeholder_csr = sparse.csr_matrix((n_rows, n_cols), dtype=np.float64)

        # Create coordinate dataframe
        coords_df = self._create_coordinates_dataframe_for_slice(0)

        # Create var dataframe
        var_df = self._create_mass_dataframe()

        # Create placeholder AnnData
        adata = AnnData(
            X=placeholder_csr,
            obs=coords_df,
            var=var_df,
        )

        # Calculate and add average spectrum to .uns
        avg_spectrum = total_intensity / max(pixel_count, 1)
        adata.uns["average_spectrum"] = avg_spectrum

        # Add MSI metadata
        self._add_metadata_to_uns(adata)

        # Add region and instance_key columns
        if "region" not in adata.obs.columns:
            adata.obs["region"] = region_key
        adata.obs["instance_key"] = adata.obs.index.astype(str)

        # Create table model
        table = TableModel.parse(
            adata,
            region=region_key,
            region_key="region",
            instance_key="instance_key",
        )

        # Create pixel shapes
        shapes = {region_key: self._create_pixel_shapes(adata, is_3d=False)}

        # Create TIC image
        tic_values_with_channel = tic_values.reshape(1, n_y, n_x)
        tic_image = xr.DataArray(
            tic_values_with_channel,
            dims=("c", "y", "x"),
            coords={
                "c": [0],
                "y": np.arange(n_y) * self.pixel_size_um,
                "x": np.arange(n_x) * self.pixel_size_um,
            },
        )
        transform = Identity()
        tic_model = Image2DModel.parse(
            tic_image,
            transformations={
                slice_id: transform,
                "global": transform,
            },
        )

        images = {f"{slice_id}_tic": tic_model}

        # Create SpatialData with placeholder X
        from spatialdata import SpatialData

        sdata = SpatialData(
            tables={slice_id: table},
            shapes=shapes,
            images=images,
        )

        # Add metadata
        self.add_metadata(sdata)

        # Write to disk (this creates the full structure with empty X)
        if self.output_path.exists():
            shutil.rmtree(self.output_path)
        sdata.write(str(self.output_path))
        logging.info(f"Created output structure at {self.output_path}")

        # Free memory from placeholder structures
        del sdata, table, adata, placeholder_csr
        gc.collect()

        # ========== Overwrite X arrays directly ==========
        logging.info("Setting up X arrays for streaming write...")

        # Open output Zarr WITHOUT consolidated metadata to allow modifications
        # SpatialData writes zarr v3 with consolidated metadata, but we need
        # to modify arrays which requires bypassing the consolidated metadata
        store = zarr.open_group(
            str(self.output_path),
            mode="r+",
            use_consolidated=False,  # Important: bypass consolidated metadata
        )
        X_group = store[f"tables/{slice_id}/X"]

        # Get original dtypes from placeholder (scipy uses int32 for indices)
        orig_indptr_dtype = X_group["indptr"].dtype
        orig_indices_dtype = X_group["indices"].dtype
        orig_data_dtype = X_group["data"].dtype
        logging.debug(
            f"Original dtypes: indptr={orig_indptr_dtype}, "
            f"indices={orig_indices_dtype}, data={orig_data_dtype}"
        )

        # Overwrite indptr with our computed values (matching original dtype)
        del X_group["indptr"]
        X_group.create_array("indptr", data=indptr.astype(orig_indptr_dtype))

        # Delete empty arrays and create correctly-sized ones
        del X_group["indices"]
        del X_group["data"]

        chunk_size_zarr = min(total_nnz, 1000000) if total_nnz > 0 else 1
        indices_arr = X_group.create_array(
            "indices",
            shape=(max(total_nnz, 1),),
            dtype=orig_indices_dtype,
            chunks=(chunk_size_zarr,),
        )
        data_arr = X_group.create_array(
            "data",
            shape=(max(total_nnz, 1),),
            dtype=orig_data_dtype,
            chunks=(chunk_size_zarr,),
        )

        # Update shape attribute
        X_group.attrs["shape"] = [n_rows, n_cols]

        # ========== PASS 2: Stream-write data directly to output ==========
        logging.info("Pass 2: Writing data directly to output Zarr...")

        # Reset reader for second pass
        reader_path = self.reader.data_path
        self.reader.close()
        self.reader = type(self.reader)(reader_path)
        setattr(self.reader, "_quiet_mode", True)

        # Process in chunks to batch Zarr writes
        chunk_indices: list = []
        chunk_data: list = []
        chunk_start_pos = 0
        spectra_in_chunk = 0

        with tqdm(
            total=total_spectra,
            desc="Pass 2: Writing",
            unit="spectrum",
        ) as pbar:
            for coords, mzs, intensities in self.reader.iter_spectra(
                batch_size=self._buffer_size
            ):
                # Resample spectrum
                mz_indices, resampled_ints = self._process_spectrum(mzs, intensities)

                if len(mz_indices) > 0:
                    # Use original dtypes from placeholder (scipy uses int32)
                    chunk_indices.append(mz_indices.astype(orig_indices_dtype))
                    chunk_data.append(resampled_ints.astype(orig_data_dtype))

                spectra_in_chunk += 1

                # Write chunk when full
                if spectra_in_chunk >= self._chunk_size:
                    if chunk_indices:
                        all_indices = np.concatenate(chunk_indices)
                        all_data = np.concatenate(chunk_data)

                        end_pos = chunk_start_pos + len(all_indices)
                        indices_arr[chunk_start_pos:end_pos] = all_indices
                        data_arr[chunk_start_pos:end_pos] = all_data

                        chunk_start_pos = end_pos
                        del all_indices, all_data

                    chunk_indices = []
                    chunk_data = []
                    spectra_in_chunk = 0
                    gc.collect()

                pbar.update(1)

        # Write final chunk
        if chunk_indices:
            all_indices = np.concatenate(chunk_indices)
            all_data = np.concatenate(chunk_data)

            end_pos = chunk_start_pos + len(all_indices)
            indices_arr[chunk_start_pos:end_pos] = all_indices
            data_arr[chunk_start_pos:end_pos] = all_data

        logging.info(f"Pass 2 complete: {total_nnz:,} non-zeros written")

        return {
            "total_nnz": total_nnz,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "pixel_count": pixel_count,
        }

    def _stream_write_direct_zarr(self) -> Dict[str, Any]:
        """Single-pass direct Zarr construction - no SpatialData intermediate.

        This method constructs the SpatialData-compatible Zarr store directly
        without ever using SpatialData.write() or creating scipy matrices.

        Approach:
        1. Create Zarr directory structure with proper attributes
        2. Stream through spectra once, building indptr incrementally
        3. Write indices/data directly as we go
        4. Consolidate metadata at the end

        Returns:
            Dictionary with conversion statistics
        """
        from datetime import datetime

        if self._dimensions is None:
            raise ValueError("Dimensions not initialized")
        if self._common_mass_axis is None:
            raise ValueError("Common mass axis not initialized")

        n_x, n_y, n_z = self._dimensions
        n_rows = n_x * n_y * n_z
        n_cols = len(self._common_mass_axis)
        slice_id = f"{self.dataset_id}_z0"
        region_key = f"{slice_id}_pixels"

        logging.info(f"Direct Zarr streaming: {n_rows:,} pixels x {n_cols:,} cols")

        # Clean output directory
        if self.output_path.exists():
            shutil.rmtree(self.output_path)

        # ========== Create Zarr store structure ==========
        logging.info("Creating Zarr structure...")
        store = zarr.open_group(str(self.output_path), mode="w")

        # Root attributes (SpatialData metadata)
        store.attrs["spatialdata_attrs"] = {
            "version": "0.2",
            "spatialdata_software_version": "0.6.1",
        }
        store.attrs["pixel_size_x_um"] = float(self.pixel_size_um)
        store.attrs["pixel_size_y_um"] = float(self.pixel_size_um)
        store.attrs["pixel_size_units"] = "micrometers"
        store.attrs["coordinate_system"] = "physical_micrometers"
        store.attrs["msi_converter_version"] = "1.9.0"
        store.attrs["conversion_timestamp"] = datetime.now().isoformat()

        # ========== Create table structure ==========
        tables_group = store.create_group("tables")
        table_group = tables_group.create_group(slice_id)

        # Table attributes
        table_group.attrs["encoding-type"] = "anndata"
        table_group.attrs["encoding-version"] = "0.1.0"
        table_group.attrs["spatialdata-encoding-type"] = "ngff:regions_table"
        table_group.attrs["region"] = region_key
        table_group.attrs["region_key"] = "region"
        table_group.attrs["instance_key"] = "instance_key"
        table_group.attrs["version"] = "0.2"

        # Create empty groups required by AnnData
        for group_name in ["layers", "obsm", "obsp", "varm", "varp"]:
            g = table_group.create_group(group_name)
            g.attrs["encoding-type"] = "dict"
            g.attrs["encoding-version"] = "0.1.0"

        # Create 'raw' array (AnnData uses this)
        table_group.create_array("raw", data=np.array(False))

        # ========== Create X group (CSR matrix) ==========
        X_group = table_group.create_group("X")
        X_group.attrs["encoding-type"] = "csr_matrix"
        X_group.attrs["encoding-version"] = "0.1.0"
        X_group.attrs["shape"] = [n_rows, n_cols]

        # Pre-allocate indptr (we know its exact size)
        indptr_arr = X_group.create_array(
            "indptr",
            shape=(n_rows + 1,),
            dtype=np.int32,
            chunks=(min(n_rows + 1, 100000),),
        )

        # Estimate total nnz for initial allocation
        # Use a reasonable estimate based on typical MSI data
        estimated_nnz = n_rows * 500  # Assume ~500 peaks per spectrum on average

        # Create resizable arrays for indices and data
        chunk_size_zarr = min(estimated_nnz, 1000000)
        indices_arr = X_group.create_array(
            "indices",
            shape=(estimated_nnz,),
            dtype=np.int32,
            chunks=(chunk_size_zarr,),
        )
        data_arr = X_group.create_array(
            "data",
            shape=(estimated_nnz,),
            dtype=np.float64,
            chunks=(chunk_size_zarr,),
        )

        # ========== Create obs (coordinates) ==========
        obs_group = table_group.create_group("obs")
        obs_group.attrs["encoding-type"] = "dataframe"
        obs_group.attrs["encoding-version"] = "0.2.0"
        obs_group.attrs["_index"] = "instance_id"
        obs_group.attrs["column-order"] = [
            "y",
            "x",
            "region",
            "spatial_x",
            "spatial_y",
            "instance_key",
        ]

        # Pre-compute coordinate arrays
        y_values = np.repeat(np.arange(n_y, dtype=np.int32), n_x)
        x_values = np.tile(np.arange(n_x, dtype=np.int32), n_y)
        # Use numpy StringDType for proper Zarr 3.x string handling
        str_dtype = np.dtypes.StringDType()
        instance_ids = np.array([str(i) for i in range(n_rows)], dtype=str_dtype)
        spatial_x = x_values.astype(np.float64) * self.pixel_size_um
        spatial_y = y_values.astype(np.float64) * self.pixel_size_um

        obs_group.create_array("y", data=y_values)
        obs_group.create_array("x", data=x_values)
        obs_group.create_array("spatial_x", data=spatial_x)
        obs_group.create_array("spatial_y", data=spatial_y)
        obs_group.create_array("instance_id", data=instance_ids)
        obs_group.create_array("instance_key", data=instance_ids)

        # Region as categorical
        region_group = obs_group.create_group("region")
        region_group.attrs["encoding-type"] = "categorical"
        region_group.attrs["encoding-version"] = "0.2.0"
        region_group.attrs["ordered"] = False
        region_group.create_array(
            "categories", data=np.array([region_key], dtype=str_dtype)
        )
        region_group.create_array("codes", data=np.zeros(n_rows, dtype=np.int8))

        # ========== Create var (mass axis) ==========
        var_group = table_group.create_group("var")
        var_group.attrs["encoding-type"] = "dataframe"
        var_group.attrs["encoding-version"] = "0.2.0"
        var_group.attrs["_index"] = "_index"
        var_group.attrs["column-order"] = ["mz"]

        mz_values = self._common_mass_axis
        mz_index = np.array([f"mz_{i}" for i in range(n_cols)], dtype=str_dtype)
        var_group.create_array("_index", data=mz_index)
        var_group.create_array("mz", data=mz_values)

        # ========== Create uns (metadata) ==========
        uns_group = table_group.create_group("uns")
        uns_group.attrs["encoding-type"] = "dict"
        uns_group.attrs["encoding-version"] = "0.1.0"

        # spatialdata_attrs in uns
        sd_attrs = uns_group.create_group("spatialdata_attrs")
        sd_attrs.attrs["encoding-type"] = "dict"
        sd_attrs.attrs["encoding-version"] = "0.1.0"
        sd_attrs.create_array("region", data=np.array(region_key, dtype=str_dtype))
        sd_attrs.create_array("region_key", data=np.array("region", dtype=str_dtype))
        sd_attrs.create_array(
            "instance_key", data=np.array("instance_key", dtype=str_dtype)
        )

        # essential_metadata
        em_group = uns_group.create_group("essential_metadata")
        em_group.attrs["encoding-type"] = "dict"
        em_group.attrs["encoding-version"] = "0.1.0"
        em_group.create_array("dimensions", data=np.array(self._dimensions))
        em_group.create_array(
            "mass_range", data=np.array([mz_values.min(), mz_values.max()])
        )
        em_group.create_array(
            "source_path", data=np.array(str(self.reader.data_path), dtype=str_dtype)
        )
        em_group.create_array(
            "spectrum_type", data=np.array("processed", dtype=str_dtype)
        )

        # ========== SINGLE PASS: Stream and write data ==========
        logging.info("Streaming data (single pass)...")

        setattr(self.reader, "_quiet_mode", True)
        total_spectra = self._get_total_spectra_count()

        # Track TIC and total intensity
        tic_values = np.zeros((n_y, n_x), dtype=np.float64)
        total_intensity = np.zeros(n_cols, dtype=np.float64)

        # Streaming state
        current_position = 0
        indptr_buffer = np.zeros(n_rows + 1, dtype=np.int32)
        pixel_count = 0

        # Buffers for batched writes
        buffer_indices: list = []
        buffer_data: list = []
        buffer_size = 0
        max_buffer_size = 500000  # Flush every ~500k values

        with tqdm(
            total=total_spectra,
            desc="Streaming",
            unit="spectrum",
        ) as pbar:
            for coords, mzs, intensities in self.reader.iter_spectra(
                batch_size=self._buffer_size
            ):
                x, y, z = coords
                pixel_idx = z * (n_x * n_y) + y * n_x + x

                # Process spectrum
                mz_indices, resampled_ints = self._process_spectrum(mzs, intensities)
                nnz = len(mz_indices)

                # Update TIC
                if nnz > 0:
                    tic_value = float(np.sum(resampled_ints))
                    if 0 <= y < n_y and 0 <= x < n_x:
                        tic_values[y, x] = tic_value

                    # Accumulate for average spectrum
                    np.add.at(total_intensity, mz_indices, resampled_ints)

                    # Add to buffer
                    buffer_indices.append(mz_indices.astype(np.int32))
                    buffer_data.append(resampled_ints.astype(np.float64))
                    buffer_size += nnz

                # Update indptr (row pointer)
                current_position += nnz
                indptr_buffer[pixel_idx + 1] = current_position

                pixel_count += 1

                # Flush buffer if large enough
                if buffer_size >= max_buffer_size:
                    if buffer_indices:
                        all_indices = np.concatenate(buffer_indices)
                        all_data = np.concatenate(buffer_data)

                        start_pos = current_position - buffer_size
                        end_pos = current_position

                        # Resize arrays if needed
                        if end_pos > indices_arr.shape[0]:
                            new_size = max(end_pos * 2, indices_arr.shape[0] * 2)
                            indices_arr.resize((new_size,))
                            data_arr.resize((new_size,))

                        indices_arr[start_pos:end_pos] = all_indices
                        data_arr[start_pos:end_pos] = all_data

                        del all_indices, all_data

                    buffer_indices = []
                    buffer_data = []
                    buffer_size = 0

                pbar.update(1)

        # Flush remaining buffer
        if buffer_indices:
            all_indices = np.concatenate(buffer_indices)
            all_data = np.concatenate(buffer_data)

            start_pos = current_position - buffer_size
            end_pos = current_position

            if end_pos > indices_arr.shape[0]:
                indices_arr.resize((end_pos,))
                data_arr.resize((end_pos,))

            indices_arr[start_pos:end_pos] = all_indices
            data_arr[start_pos:end_pos] = all_data

        total_nnz = current_position

        # Resize arrays to final size
        if total_nnz != indices_arr.shape[0]:
            indices_arr.resize((max(total_nnz, 1),))
            data_arr.resize((max(total_nnz, 1),))

        # Forward-fill indptr for empty pixels
        # CSR format requires indptr to be monotonically non-decreasing.
        # Empty pixels (not visited during iteration) have indptr=0, but should
        # inherit the value from the previous pixel (meaning 0 non-zeros for that row).
        # Use numpy's maximum.accumulate for efficient forward-fill.
        np.maximum.accumulate(indptr_buffer, out=indptr_buffer)

        # Write indptr
        indptr_arr[:] = indptr_buffer

        # Update X shape attribute
        X_group.attrs["shape"] = [n_rows, n_cols]

        logging.info(f"Streaming complete: {total_nnz:,} non-zeros")

        # ========== Add average spectrum ==========
        avg_spectrum = total_intensity / max(pixel_count, 1)
        uns_group.create_array("average_spectrum", data=avg_spectrum)

        # Skip images and shapes for now - can be added lazily later
        # Create empty groups so SpatialData doesn't complain
        store.create_group("images")
        store.create_group("shapes")

        # ========== Consolidate metadata ==========
        logging.info("Consolidating metadata...")
        zarr.consolidate_metadata(str(self.output_path))

        logging.info(f"Direct Zarr write complete: {self.output_path}")

        return {
            "total_nnz": total_nnz,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "pixel_count": pixel_count,
        }
