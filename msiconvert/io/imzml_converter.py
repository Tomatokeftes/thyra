from typing import List
import zarr
from tqdm import tqdm
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser
from .base_converter import BaseMSIConverter
from .registry import register_converter
from ..utils.zarr_manager import ZarrManager, SHAPE
import numpy as np

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
        return ['common_mass_axis']

    def get_common_mass_axis(self) -> None:
        """Create a common mass axis for all spectra."""
        print('Creating common mass axis for processed imzML')

        all_mz_values = np.concatenate([self.parser.getspectrum(idx)[0] for idx, _ in enumerate(self.parser.coordinates)])
        self.common_mass_axis = np.unique(all_mz_values)
        print(f'Common mass axis length: {self.common_mass_axis.shape[0]}')

    def get_intensity_array_shape(self) -> SHAPE:
        return (
            self.common_mass_axis.shape[0],
            1,
            self.parser.imzmldict['max count of pixels y'],
            self.parser.imzmldict['max count of pixels x'],
        )

    def create_zarr_arrays(self):
        self.zarr_manager = ZarrManager(self.root, self.parser)
        self.zarr_manager.create_arrays(
            self.get_intensity_array_shape,
            self.common_mass_axis,
        )

    def read_binary_data(self) -> None:
        
        with self.zarr_manager.temporary_arrays():
            total_spectra = len(self.parser.coordinates)
            with tqdm(total=total_spectra, desc='Processing spectra', unit='spectrum') as pbar:
                for idx, (x, y, _) in enumerate(self.parser.coordinates):
                    mz_array, intensity_array = self.parser.getspectrum(idx)

                    # Enforce intensity array to be float64
                    intensity_array = intensity_array.astype(np.float64)

                    # Map m/z values to common mass axis
                    mz_indices = np.searchsorted(self.common_mass_axis, mz_array)

                    # Store intensities in temporary arrays
                    self.zarr_manager.fast_intensities[mz_indices, 0, y - 1, x - 1] = intensity_array

                    pbar.update(1)

                self.zarr_manager.copy_to_main_arrays()
                
@register_converter('imzml_continuous')
class ContinuousImzMLConvertor(BaseImzMLConverter):
    def get_labels(self) -> List[str]:
        return ['common_mass_axis']

    def get_common_mass_axis(self) -> None:
        """Extract the common mass axis from the continuous imzML file."""
        print('Extracting common mass axis for continuous imzML')

        # Read the m/z values from the first pixel (all pixels have the same m/z values)
        self.common_mass_axis = self.parser.getspectrum(0)[0]
        print(f'Common mass axis length: {self.common_mass_axis.shape[0]}')

    def get_intensity_array_shape(self) -> SHAPE:
        return (
            self.common_mass_axis.shape[0],  # Number of m/z channels
            1,
            self.parser.imzmldict['max count of pixels y'],
            self.parser.imzmldict['max count of pixels x'],
        )

    def create_zarr_arrays(self):
        self.zarr_manager = ZarrManager(self.root, self.parser)
        self.zarr_manager.create_arrays(
            self.get_intensity_array_shape,
            self.common_mass_axis,
        )

    def read_binary_data(self) -> None:
        """Read continuous imzML spectra and store them in the Zarr array efficiently."""
        with self.zarr_manager.temporary_arrays():
            total_spectra = len(self.parser.coordinates)
            with tqdm(total=total_spectra, desc='Processing spectra', unit='spectrum') as pbar:
                for idx, (x, y, _) in enumerate(self.parser.coordinates):
                    _, intensity_array = self.parser.getspectrum(idx)

                    # Ensure intensity is float64
                    intensity_array = intensity_array.astype(np.float64)

                    # Directly store intensities (1-to-1 mapping)
                    self.zarr_manager.fast_intensities[:, 0, y - 1, x - 1] = intensity_array

                    pbar.update(1)

                self.zarr_manager.copy_to_main_arrays()
