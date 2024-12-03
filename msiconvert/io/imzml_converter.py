from typing import List
import zarr
from sortedcontainers import SortedSet
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser
from .base_converter import BaseMSIConverter
from .registry import register_converter
from ..utils.zarr_manager import ZarrManager, SHAPE
import numpy as np
from numba import njit

class BaseImzMLConverter(BaseMSIConverter):
    """Base class for imzML-specific converters."""

    def __init__(self, root: zarr.Group, name: str, parser: PyImzMLParser):
        super().__init__(root, name)
        self.parser = parser

    def add_base_metadata(self) -> None:
        """Add imzML-specific metadata to the root group."""
        self.root.attrs['multiscales'] = [{
            'version': '0.4',
            'name': self.name,
            'datasets': [{'path': '0'}],
            'axes': ['c', 'z', 'y', 'x'],
            'type': 'none',
        }]
        self.root.attrs['imzml'] = {
            'source': self.parser.filename,
            'uuid': self.parser.metadata.file_description.cv_params[0][2],
        }
        self.root.create_group('labels').attrs['labels'] = self.get_labels()

    def run(self) -> None:
        """Ensure the ibd file is closed after conversion."""
        try:
            super().run()
        finally:
            self.parser.m.close()

@register_converter('imzml_processed')
class ProcessedImzMLConvertor(BaseImzMLConverter):
    def get_labels(self) -> List[str]:
        return ['mzs/0', 'lengths/0']

    def get_intensity_shape(self) -> SHAPE:
        return (
            max(self.parser.mzLengths),
            1,
            self.parser.imzmldict['max count of pixels y'],
            self.parser.imzmldict['max count of pixels x'],
        )

    def get_mz_shape(self) -> SHAPE:
        "Return an int tuple describing the shape of the mzs array"
        return self.get_intensity_shape()

    def get_lengths_shape(self) -> SHAPE:
        "Return an int tuple describing the shape of the lengths array"
        return (
            1,                                               # c = m/Z
            1,                                               # z = 1
            self.parser.imzmldict['max count of pixels y'],  # y
            self.parser.imzmldict['max count of pixels x'],  # x
        )

    def create_zarr_arrays(self):
        self.zarr_manager = ZarrManager(self.root, self.parser)
        self.zarr_manager.create_arrays(
            self.get_intensity_shape,
            self.get_mz_shape,
            self.get_lengths_shape
        )

    def read_binary_data(self) -> None:
        decimal_places = 5
        all_mz_arrays = []
        all_intensity_arrays = []
        coordinates = []
        
        with self.zarr_manager.temporary_arrays():
            total_spectra = len(self.parser.coordinates)
            with tqdm(total=total_spectra, desc='Processing spectra', unit='spectrum') as pbar:
                for idx, (x, y, _) in enumerate(self.parser.coordinates):
                    length = self.parser.mzLengths[idx]
                    self.zarr_manager.lengths[0, 0, y - 1, x - 1] = length

                    # Get spectra (returns a tuple of arrays)
                    mz_array, intensity_array = self.parser.getspectrum(idx)

                    # Round mz values and convert to float32
                    mz_array = np.round(mz_array, decimals=decimal_places).astype(np.float32)
                    intensity_array = intensity_array.astype(np.float32)

                    # Collect all mz and intensity arrays
                    all_mz_arrays.append(mz_array)
                    all_intensity_arrays.append(intensity_array)
                    coordinates.append((x, y, length))
                    pbar.update(1)

            # Concatenate all mz arrays and find unique values
            all_mz_values = np.concatenate(all_mz_arrays)
            common_mass_axis = np.unique(all_mz_values)
            print(f'Common mass axis length: {len(common_mass_axis)}')

            # Create mapping from m/z value to index
            sorted_mz_values = common_mass_axis

            # Use Numba-accelerated function for mapping
            from numba import njit

            @njit
            def map_mz_indices(mz_values, sorted_mz_values):
                return np.searchsorted(sorted_mz_values, mz_values)

            # Map all m/z values to indices
            all_mz_indices = []
            for mz_array in all_mz_arrays:
                mz_indices = map_mz_indices(mz_array, sorted_mz_values)
                all_mz_indices.append(mz_indices)

            # Save common mass axis
            self.zarr_manager.save_array('common_mass_axis', common_mass_axis)

            # Write data back to Zarr arrays
            with tqdm(total=total_spectra, desc='Writing data', unit='spectrum') as pbar:
                for idx, (x, y, length) in enumerate(coordinates):
                    mz_indices = all_mz_indices[idx]
                    intensity_array = all_intensity_arrays[idx]
                    self.zarr_manager.fast_mzs[:length, 0, y - 1, x - 1] = mz_indices
                    self.zarr_manager.fast_intensities[:length, 0, y - 1, x - 1] = intensity_array
                    pbar.update(1)

            # Copy data to main arrays
            self.zarr_manager.copy_to_main_arrays()



        

# @register_converter('imzml_continuous')
# class _ContinuousImzMLConvertor(_BaseImzMLConvertor):
#     def get_labels(self) -> List[str]:
#         return ['mzs/0']

#     def get_intensity_shape(self) -> SHAPE:
#         return (
#             self.parser.mzLengths[0],
#             1,
#             self.parser.imzmldict['max count of pixels y'],
#             self.parser.imzmldict['max count of pixels x'],
#         )
    
#     def get_mz_shape(self) -> SHAPE:
#         "return an int tuple describing the shape of the mzs array"
#         return (
#             self.parser.mzLengths[0],  # c = m/Z
#             1,                         # z
#             1,                         # y
#             1,                         # x
#         )

#     def create_zarr_arrays(self):
#         intensities = self.root.zeros(
#             '0',
#             shape=self.get_intensity_shape(),
#             dtype=self.parser.intensityPrecision,
#         )
#         compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
#         intensities.attrs['_ARRAY_DIMENSIONS'] = _get_xarray_axes(self.root)
#         self.root.zeros(
#             'labels/mzs/0',
#             shape=self.get_mz_shape(),
#             dtype=self.parser.mzPrecision,
#             compressor=compressor,
#         )

#     def read_binary_data(self) -> None:
#         compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
#         intensities = self.root[0]
#         mzs = self.root['labels/mzs/0']
#         with single_temp_store() as fast_store:
#             fast_intensities = zarr.group(fast_store).zeros(
#                 '0',
#                 shape=intensities.shape,
#                 dtype=intensities.dtype,
#                 chunks=(-1, 1, 1, 1),
#                 compressor=compressor,
#             )
#             self.parser.m.seek(self.parser.mzOffsets[0])
#             mzs[:, 0, 0, 0] = np.fromfile(
#                 self.parser.m, count=self.parser.mzLengths[0], dtype=self.parser.mzPrecision
#             )
#             for idx, (x, y, _) in enumerate(self.parser.coordinates):
#                 self.parser.m.seek(self.parser.intensityOffsets[idx])
#                 fast_intensities[:, 0, y - 1, x - 1] = np.fromfile(
#                     self.parser.m, count=self.parser.intensityLengths[idx], dtype=self.parser.intensityPrecision
#                 )
#             with ProgressBar():
#                 copy_array(fast_intensities, intensities)
