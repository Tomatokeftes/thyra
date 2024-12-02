import zarr
from typing import List, Tuple
import numpy as np
from contextlib import contextmanager
from dask.diagnostics import ProgressBar
import dask.array as da
from numcodecs import Blosc

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
        self.mzs = None
        self.lengths = None
        self.fast_intensities = None
        self.fast_mzs = None
    
    def _get_xarray_axes(self) -> List[str]:
        """Return axes metadata for Xarray compatibility."""
        return self.root.attrs['multiscales'][0]['axes']

    def create_arrays(self, get_intensity_shape, get_mz_shape, get_lengths_shape):
        chunk_shape = (1000, 1, 256, 256)
        # Create the main Zarr arrays
        intensities = self.root.zeros(
            '0',
            shape=get_intensity_shape(),
            dtype=np.float32,
            compressor=self.compressor,
            dimension_separator='/',
            chunks = chunk_shape,
        )
        intensities.attrs['_ARRAY_DIMENSIONS'] = self._get_xarray_axes()

        self.root.zeros(
            'labels/mzs/0',
            shape=get_mz_shape(),
            dtype=np.uint32,
            compressor=self.compressor,
            dimension_separator='/',
            chunks = chunk_shape,
        )

        self.root.zeros(
            'labels/lengths/0',
            shape=get_lengths_shape(),
            dtype=np.uint32,
            compressor=self.compressor,
            dimension_separator='/',
            # chunks = chunk_shape,
        )

        # Save references to the arrays
        self.intensities = self.root['0']
        self.mzs = self.root['labels/mzs/0']
        self.lengths = self.root['labels/lengths/0']

    def save_array(self, name: str, data: List[float]): 
        self.root.array(
            name,
            data=np.array(data),
            dtype=np.float32,
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
        with single_temp_store() as fast_store:
            fast_group = zarr.group(fast_store)
            self.fast_intensities = fast_group.zeros(
                '0',
                shape=self.intensities.shape,
                dtype=self.intensities.dtype,
                chunks=(-1, 1, 1, 1),
                compressor=self.compressor,
            )
            self.fast_mzs = fast_group.zeros(
                'mzs',
                shape=self.mzs.shape,
                dtype=np.float64,
                chunks=(-1, 1, 1, 1),
                compressor=self.compressor,
            )
            yield


    def fill_temporary_arrays(self):
        # Fill the temporary arrays with data
    
        for idx, (x, y, _) in enumerate(self.parser.coordinates):
            length = self.parser.mzLengths[idx]
            self.lengths[0, 0, y - 1, x - 1] = length
            spectra = self.parser.getspectrum(idx)
            self.fast_mzs[:length, 0, y - 1, x - 1] = spectra[0]
            self.fast_intensities[:length, 0, y - 1, x - 1] = spectra[1]

    def copy_to_main_arrays(self):
        # Copy data from temporary arrays to the main arrays
        with ProgressBar():
            self.copy_array(self.fast_intensities, self.intensities)
            self.copy_array(self.fast_mzs, self.mzs)

