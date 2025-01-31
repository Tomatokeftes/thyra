import zarr
from typing import List, Tuple, Callable, Generator
import numpy as np
from contextlib import contextmanager
from dask.diagnostics.progress import ProgressBar
import dask.array as da

from ..utils.temp_store import single_temp_store

SHAPE = Tuple[int, int, int, int]

class ZarrManager:
    def __init__(self, root: zarr.Group, parser: Callable):
        """
        Initialize the ZarrManager.

        Parameters:
        -----------
        root : zarr.Group
            The root Zarr group.
        parser : Callable
            A parser function or object.
        """
        self.root = root
        self.parser = parser
        self.compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=zarr.Blosc.BITSHUFFLE)
        self.intensities: zarr.Array = None
        self.common_mass_axis: zarr.Array = None
        self.temporary_intensities: zarr.Array = None
    
    def _get_xarray_axes(self) -> List[str]:
        """
        Return axes metadata for Xarray compatibility.

        Returns:
        --------
        List[str]
            A list of axis names.
        """
        return self.root.attrs['multiscales'][0]['axes']

    def create_arrays(self, get_intensity_array_shape: Callable[[], SHAPE], common_mass_axis: np.ndarray) -> None:
        """
        Create the Zarr arrays for intensities and the common mass axis.

        Parameters:
        -----------
        get_intensity_array_shape : Callable[[], SHAPE]
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
            chunks=chunk_shape,
        )
        self.intensities.attrs['_ARRAY_DIMENSIONS'] = self._get_xarray_axes()

        self.common_mass_axis = self.root.array(
            "labels/common_mass_axis/0",
            data=common_mass_axis,
            dtype=np.float64,
            compressor=self.compressor,
        )

    def save_array(self, name: str, data: List[float]) -> None:
        """
        Save an array to the Zarr store.

        Parameters:
        -----------
        name : str
            The name of the array.
        data : List[float]
            The data to be saved.
        """
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
        """
        try:
            with ProgressBar():
                dask_source = da.from_zarr(source)
                if dask_source.chunks != destination.chunks:
                    dask_source = dask_source.rechunk(destination.chunks)
                da.store(dask_source, destination, lock=False)
        except Exception as e:
            print(f"Error copying array from temporary to main store: {e}")
            raise



    @contextmanager
    def temporary_arrays(self) -> Generator[None, None, None]:
        """
        Create a fast temporary Zarr array for efficient writing before copying to main storage.

        Yields:
        -------
        None
        """
        with single_temp_store() as fast_store:
            fast_group = zarr.group(fast_store)
            
            # Temporary fast intensities array
            self.temporary_intensities = fast_group.zeros(
                '0',
                shape=self.intensities.shape,
                dtype=self.intensities.dtype,
                chunks=(-1, 1, 1, 1),
            )
            yield
