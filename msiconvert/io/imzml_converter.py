from typing import List, Tuple, Callable
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
        """
        Initialize the BaseImzMLConverter.

        Parameters:
        -----------
        root : zarr.Group
            The root Zarr group.
        name : str
            The name of the converter.
        parser : PyImzMLParser
            The ImzML parser.
        """
        super().__init__(root, name)
        self.parser = parser

    def add_base_metadata(self) -> None:
        """
        Add imzML-specific metadata to the root group.
        """
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

    def _process_spectra(self, process_fn: Callable[[int, int, int, any], None]) -> None:
        """
        Iterate over all spectra and process each one using the provided function.

        Parameters:
        -----------
        process_fn : Callable[[int, int, int, any], None]
            A function that accepts the index, x-coordinate, y-coordinate, and the spectrum (tuple of m/z and intensity arrays),
            and performs the necessary mapping/storage logic.
        """
        total_spectra = len(self.parser.coordinates)
        with tqdm(total=total_spectra, desc='Processing spectra', unit='spectrum') as pbar:
            for idx, (x, y, _) in enumerate(self.parser.coordinates):
                try:
                    spectrum = self.parser.getspectrum(idx)
                    process_fn(idx, x, y, spectrum)
                except Exception as err:
                    print(f"Error processing spectrum {idx} at pixel ({x}, {y}): {err}")
                pbar.update(1)



    def run(self) -> None:
        """
        Ensure the ibd file is closed after conversion.
        """
        try:
            super().run()
        finally:
            self.parser.m.close()

@register_converter('imzml_processed')
class ProcessedImzMLConvertor(BaseImzMLConverter):
    def get_labels(self) -> List[str]:
        """
        Get the labels for the processed imzML converter.

        Returns:
        --------
        List[str]: A list of labels.
        """
        return ['common_mass_axis']

    def get_common_mass_axis(self) -> None:
        """
        Create a common mass axis for all spectra.
        """
        print('Creating common mass axis for processed imzML')

        all_mz_values = np.concatenate([self.parser.getspectrum(idx)[0] for idx, _ in enumerate(self.parser.coordinates)])
        self.common_mass_axis = np.unique(all_mz_values)
        print(f'Common mass axis length: {self.common_mass_axis.shape[0]}')

    def get_intensity_array_shape(self) -> Tuple[int, int, int, int]:
        """
        Get the shape of the intensity array.

        Returns:
        --------
        Tuple[int, int, int, int]: The shape of the intensity array.
        """
        return (
            self.common_mass_axis.shape[0],
            1,
            self.parser.imzmldict['max count of pixels y'],
            self.parser.imzmldict['max count of pixels x'],
        )

    def create_zarr_arrays(self) -> None:
        """
        Create the Zarr arrays for storing the data.
        """
        self.zarr_manager = ZarrManager(self.root, self.parser)
        self.zarr_manager.create_arrays(
            self.get_intensity_array_shape,
            self.common_mass_axis,
        )

    def _process_spectrum_processed(self, idx: int, x: int, y: int, spectrum: any) -> None:
        """
        Process a single spectrum for a processed imzML file by mapping its m/z values
        to indices in the common mass axis and storing the corresponding intensities.
        
        Parameters:
        -----------
        idx : int
            Spectrum index.
        x : int
            x-coordinate of the pixel.
        y : int
            y-coordinate of the pixel.
        spectrum : any
            A tuple containing (mz_array, intensity_array) for the pixel.
        """
        mz_array, intensity_array = spectrum
        # Ensure intensities are float64
        intensity_array = intensity_array.astype(np.float64)
        
        # Map the pixel's m/z values to indices in the common mass axis.
        mz_indices = np.searchsorted(self.common_mass_axis, mz_array)
        # Optionally, you could check that mz_indices are within expected bounds.
        if np.any(mz_indices >= self.common_mass_axis.shape[0]):
            raise IndexError(f"Spectrum {idx}: found index out of bounds in common mass axis")

        # Assign the pixel's intensities to the appropriate positions in the temporary array.
        self.zarr_manager.temporary_intensities[mz_indices, 0, y - 1, x - 1] = intensity_array



    def read_binary_data(self) -> None:
        with self.zarr_manager.temporary_arrays():
            self._process_spectra(self._process_spectrum_processed)
            self.zarr_manager.copy_array(self.zarr_manager.temporary_intensities, self.zarr_manager.intensities)



@register_converter('imzml_continuous')
class ContinuousImzMLConvertor(BaseImzMLConverter):
    def get_labels(self) -> List[str]:
        """
        Get the labels for the continuous imzML converter.

        Returns:
        --------
        List[str]: A list of labels.
        """
        return ['common_mass_axis']

    def get_common_mass_axis(self) -> None:
        """
        Extract the common mass axis from the continuous imzML file.
        """
        print('Extracting common mass axis for continuous imzML')

        # Read the m/z values from the first pixel (all pixels have the same m/z values)
        self.common_mass_axis = self.parser.getspectrum(0)[0]
        print(f'Common mass axis length: {self.common_mass_axis.shape[0]}')

    def get_intensity_array_shape(self) -> Tuple[int, int, int, int]:
        """
        Get the shape of the intensity array.

        Returns:
        --------
        Tuple[int, int, int, int]: The shape of the intensity array.
        """
        return (
            self.common_mass_axis.shape[0],  # Number of m/z channels
            1,
            self.parser.imzmldict['max count of pixels y'],
            self.parser.imzmldict['max count of pixels x'],
        )

    def create_zarr_arrays(self) -> None:
        """
        Create the Zarr arrays for storing the data.
        """
        self.zarr_manager = ZarrManager(self.root, self.parser)
        self.zarr_manager.create_arrays(
            self.get_intensity_array_shape,
            self.common_mass_axis,
        )

    def _process_spectrum_continuous(self, idx: int, x: int, y: int, spectrum: any) -> None:
        _, intensity_array = spectrum
        intensity_array = intensity_array.astype(np.float64)

        if intensity_array.shape[0] != self.common_mass_axis.shape[0]:
            raise ValueError(
                f"Spectrum {idx}: intensity length {intensity_array.shape[0]} does not match "
                f"common mass axis length {self.common_mass_axis.shape[0]}"
            )

        self.zarr_manager.temporary_intensities[:, 0, y - 1, x - 1] = intensity_array


    def read_binary_data(self) -> None:
        with self.zarr_manager.temporary_arrays():
            self._process_spectra(self._process_spectrum_continuous)
            self.zarr_manager.copy_array(self.zarr_manager.temporary_intensities, self.zarr_manager.intensities)


