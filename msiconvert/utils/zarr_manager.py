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
        # Create the main Zarr arrays
        intensities = self.root.zeros(
            '0',
            shape=get_intensity_shape(),
            dtype=self.parser.intensityPrecision,
        )
        intensities.attrs['_ARRAY_DIMENSIONS'] = self._get_xarray_axes()

        self.root.zeros(
            'labels/mzs/0',
            shape=get_mz_shape(),
            dtype=self.parser.mzPrecision,
            compressor=self.compressor,
        )

        self.root.zeros(
            'labels/lengths/0',
            shape=get_lengths_shape(),
            dtype=np.uint32,
            compressor=self.compressor,
        )

        # Save references to the arrays
        self.intensities = self.root['0']
        self.mzs = self.root['labels/mzs/0']
        self.lengths = self.root['labels/lengths/0']

    def copy_array(self, source: zarr.Array, destination: zarr.Array, optimal_chunk_size=(1000, 1, 256, 256)) -> None:
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
        # Convert the source Zarr array to a Dask array
        dask_source = da.from_zarr(source)

        # Rechunk the array to a more optimal size if necessary
        if dask_source.chunksize != optimal_chunk_size:
            dask_source = dask_source.rechunk(optimal_chunk_size)

        # Copy the Dask array to the destination Zarr store
        da.store(dask_source, destination, lock=False)

    @contextmanager
    def temporary_arrays(self):
        # Create temporary arrays within a context manager
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
                dtype=self.mzs.dtype,
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


# class _ContinuousImzMLConvertor(_BaseImzMLConvertor):
    def get_labels(self) -> List[str]:
        return ['mzs/0']

    def get_intensity_shape(self) -> SHAPE:
        return (
            self.parser.mzLengths[0],
            1,
            self.parser.imzmldict['max count of pixels y'],
            self.parser.imzmldict['max count of pixels x'],
        )
    
    def get_mz_shape(self) -> SHAPE:
        "return an int tuple describing the shape of the mzs array"
        return (
            self.parser.mzLengths[0],  # c = m/Z
            1,                         # z
            1,                         # y
            1,                         # x
        )

    def create_zarr_arrays(self):
        intensities = self.root.zeros(
            '0',
            shape=self.get_intensity_shape(),
            dtype=self.parser.intensityPrecision,
        )
        compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        intensities.attrs['_ARRAY_DIMENSIONS'] = _get_xarray_axes(self.root)
        self.root.zeros(
            'labels/mzs/0',
            shape=self.get_mz_shape(),
            dtype=self.parser.mzPrecision,
            compressor=compressor,
        )

    def read_binary_data(self) -> None:
        compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
        intensities = self.root[0]
        mzs = self.root['labels/mzs/0']
        with single_temp_store() as fast_store:
            fast_intensities = zarr.group(fast_store).zeros(
                '0',
                shape=intensities.shape,
                dtype=intensities.dtype,
                chunks=(-1, 1, 1, 1),
                compressor=compressor,
            )
            self.parser.m.seek(self.parser.mzOffsets[0])
            mzs[:, 0, 0, 0] = np.fromfile(
                self.parser.m, count=self.parser.mzLengths[0], dtype=self.parser.mzPrecision
            )
            for idx, (x, y, _) in enumerate(self.parser.coordinates):
                self.parser.m.seek(self.parser.intensityOffsets[idx])
                fast_intensities[:, 0, y - 1, x - 1] = np.fromfile(
                    self.parser.m, count=self.parser.intensityLengths[idx], dtype=self.parser.intensityPrecision
                )
            with ProgressBar():
                copy_array(fast_intensities, intensities)
