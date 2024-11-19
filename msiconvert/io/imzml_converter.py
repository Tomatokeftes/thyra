from typing import List
import zarr
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser
from .base_converter import BaseMSIConverter
from .registry import register_converter
from ..utils.zarr_manager import ZarrManager, SHAPE


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
        with self.zarr_manager.temporary_arrays():
            for idx, (x, y, _) in enumerate(self.parser.coordinates):
                length = self.parser.mzLengths[idx]
                self.zarr_manager.lengths[0, 0, y - 1, x - 1] = length
                spectra = self.parser.getspectrum(idx)
                self.zarr_manager.fast_mzs[:length, 0, y - 1, x - 1] = spectra[0]
                self.zarr_manager.fast_intensities[:length, 0, y - 1, x - 1] = spectra[1]
            self.zarr_manager.copy_to_main_arrays()

@register_converter('imzml_continuous')
class ContinuousImzMLConvertor(BaseImzMLConverter):
    pass