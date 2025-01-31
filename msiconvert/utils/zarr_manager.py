# from os import sync
import zarr
from typing import List, Tuple
import numpy as np
from contextlib import contextmanager
from dask.diagnostics.progress import ProgressBar
import dask.array as da


from ..utils.temp_store import single_temp_store

# Set a byte size threshold for copying arrays
_DISK_COPY_THRESHOLD = 8 * 10**9

SHAPE = Tuple[int, int, int, int]

class ZarrManager:
    def __init__(self, root: zarr.Group, parser):
        self.root = root
        self.parser = parser
        self.compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
        self.intensities = None
        self.common_mass_axis = None
        self.fast_intensities = None
    
    def _get_xarray_axes(self) -> List[str]:
        """Return axes metadata for Xarray compatibility."""
        return self.root.attrs['multiscales'][0]['axes']

    def create_arrays(self, get_intensity_array_shape, common_mass_axis: np.ndarray) -> None:
        """
        Create the Zarr arrays for intensities and the common mass axis.

        Parameters:
        -----------
        get_intensity_shape : function
            A function that returns the shape of the intensity array.
        common_mass_axis : np.ndarray
            The computed common mass axis array.
        """

        chunk_shape = (1000, 1, 128, 128)
        
        # Create the main Zarr arrays
        self.intensities = self.root.zeros(
            '0',
            shape=get_intensity_array_shape(),
            dtype=np.float64,
            compressor=self.compressor,
            dimension_separator='/',
            chunks = chunk_shape,
            # synchronizer=zarr.ThreadSynchronizer(),
        )
        self.intensities.attrs['_ARRAY_DIMENSIONS'] = self._get_xarray_axes()

        self.common_mass_axis = self.root.array(
            "labels/common_mass_axis/0",
            data=common_mass_axis,
            dtype=np.float64,
            compressor=self.compressor,
        )

    def save_array(self, name: str, data: List[float]): 
        self.root.array(
            name,
            data=np.array(data),
            dtype=np.float64,
            compressor=self.compressor,
        )

    def copy_array(self, source: zarr.Array, destination: zarr.Array) -> None:
        """
        Copy an array using Dask for better parallelism and chunk management, with a progress bar.
        
        Parameters:
        -----------
        source : zarr.Array
            The source array to copy from.
        destination : zarr.Array
            The destination array to copy to.
        optimal_chunk_size : tuple
            The optimal chunk size to use for reducing task graph overhead.
        """

        dask_source = da.from_zarr(source)

        # Correctly compare and rechunk if necessary
        if dask_source.chunks != destination.chunks:
            dask_source = dask_source.rechunk(destination.chunks)

        da.store(dask_source, destination, lock=False)

    @contextmanager
    def temporary_arrays(self):
        """Create a fast temporary Zarr array for efficient writing before copying to main storage."""
        with single_temp_store() as fast_store:
            fast_group = zarr.group(fast_store)
            
            # Temporary fast intensities array
            self.fast_intensities = fast_group.zeros(
                '0',
                shape=self.intensities.shape,
                dtype=self.intensities.dtype,
                chunks=(-1, 1, 1, 1),
            )
            yield

    def copy_to_main_arrays(self):
        # Copy data from temporary arrays to the main arrays
        with ProgressBar():
            self.copy_array(self.fast_intensities, self.intensities)
