import abc
import contextlib
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import zarr

from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser

from ..utils.temp_store import single_temp_store
from .utils import get_imzml_pair

# Set a byte size threshold for copying arrays
_DISK_COPY_THRESHOLD = 8 * 10**9

# Type alias for shape dimensions
SHAPE = Tuple[int, int, int, int]


@contextlib.contextmanager
def load_parser(imzml: Path, ibd: Path):
    """
    Load a parser object from pyimzml.

    Parameters:
    -----------
    imzml : Path
        Path to the imzML file.
    ibd : Path
        Path to the ibd file.

    Yields:
    -------
    PyImzMLParser
        An instance of the PyImzMLParser.
    """
    with open(ibd, mode="rb") as ibd_file:
        yield PyImzMLParser(
            filename=str(imzml),
            parse_lib="lxml",
            ibd_file=ibd_file  # the ibd file must be opened manually
        )


def copy_array(source: zarr.Array, destination: zarr.Array) -> None:
    """
    Copy an array; ragged arrays not supported.

    Parameters:
    -----------
    source : zarr.Array
        The source array to copy from.
    destination : zarr.Array
        The destination array to copy to.
    """
    if source.nbytes <= _DISK_COPY_THRESHOLD:
        # Load all data in memory then write at once for speed
        destination[:] = source[:]
    else:
        # Chunk-by-chunk loading for smaller memory footprint
        destination[:] = source


class _BaseImzMLConvertor(abc.ABC):
    """Abstract base class to handle both processed and continuous imzML using polymorphism."""

    def __init__(self, root: zarr.Group, name: str, parser: PyImzMLParser) -> None:
        """
        Initialize the base converter.

        Parameters:
        -----------
        root : zarr.Group
            The root Zarr group.
        name : str
            The name of the dataset.
        parser : PyImzMLParser
            The parser instance for reading imzML data.
        """
        self.root = root
        self.name = name
        self.parser = parser

    @abc.abstractmethod
    def get_labels(self) -> List[str]:
        """Return the list of labels associated with the image."""

    def add_base_metadata(self) -> None:
        """
        Add OME-Zarr compliant metadata to the root group.

        This method adds metadata to the root group of the Zarr file to ensure
        compliance with the OME-Zarr specification. It sets the 'multiscales' 
        attribute with version, name, datasets, axes, and type information. 
        Additionally, it adds 'imzml' metadata including the source filename 
        and a unique identifier (UUID). Finally, it creates a 'labels' group 
        and assigns label metadata to it.

        Attributes:
            self.root.attrs['multiscales'] (list): Metadata for multiscale 
                datasets including version, name, datasets, axes, and type.
            self.root.attrs['imzml'] (dict): Metadata for the imzML source 
                including the source filename and UUID.
            self.root.create_group('labels').attrs['labels'] (list): Label 
                metadata obtained from the get_labels method.
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

    @abc.abstractmethod
    def create_zarr_arrays(self):
        """Generate empty arrays inside the root group."""

    @abc.abstractmethod
    def read_binary_data(self) -> None:
        """Fill in the arrays with data from the ibd file."""

    def run(self) -> None:
        """Primary method to add metadata, create arrays, and read binary data."""
        self.add_base_metadata()
        self.create_zarr_arrays()
        self.read_binary_data()


class _ContinuousImzMLConvertor(_BaseImzMLConvertor):
    def get_labels(self) -> List[str]:
        return ['mzs/0']

    def get_intensity_shape(self) -> SHAPE:
        return (
            self.parser.mzLengths[0],
            1,
            self.parser.imzmldict['max count of pixels y'],
            self.parser.imzmldict['max count of pixels x'],
        )

    def create_zarr_arrays(self):
        intensities = self.root.zeros(
            '0',
            shape=self.get_intensity_shape(),
            dtype=self.parser.intensityPrecision,
        )
        intensities.attrs['_ARRAY_DIMENSIONS'] = _get_xarray_axes(self.root)
        self.root.zeros(
            'labels/mzs/0',
            shape=self.get_intensity_shape(),
            dtype=self.parser.mzPrecision,
            compressor=None,
        )

    def read_binary_data(self) -> None:
        intensities = self.root[0]
        mzs = self.root['labels/mzs/0']
        with single_temp_store() as fast_store:
            fast_intensities = zarr.group(fast_store).zeros(
                '0',
                shape=intensities.shape,
                dtype=intensities.dtype,
                chunks=(-1, 1, 1, 1),
                compressor=None,
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
            copy_array(fast_intensities, intensities)


class _ProcessedImzMLConvertor(_BaseImzMLConvertor):
    def get_labels(self) -> List[str]:
        return ['mzs/0', 'lengths/0']

    def get_intensity_shape(self) -> SHAPE:
        return (
            max(self.parser.mzLengths),
            1,
            self.parser.imzmldict['max count of pixels y'],
            self.parser.imzmldict['max count of pixels x'],
        )

    def create_zarr_arrays(self):
        intensities = self.root.zeros(
            '0',
            shape=self.get_intensity_shape(),
            dtype=self.parser.intensityPrecision,
        )
        intensities.attrs['_ARRAY_DIMENSIONS'] = _get_xarray_axes(self.root)
        self.root.zeros(
            'labels/mzs/0',
            shape=self.get_intensity_shape(),
            dtype=self.parser.mzPrecision,
            compressor=None,
        )
        # Adjust the shape of lengths array
        self.root.zeros(
            'labels/lengths/0',
            shape=(1, 1, self.parser.imzmldict['max count of pixels y'], self.parser.imzmldict['max count of pixels x']),
            dtype=np.uint32,
            compressor=None,
        )

    def read_binary_data(self) -> None:
        intensities = self.root[0]
        mzs = self.root['labels/mzs/0']
        lengths = self.root['labels/lengths/0']

        with single_temp_store() as fast_store:
            fast_group = zarr.group(fast_store)
            fast_intensities = fast_group.zeros(
                '0',
                shape=intensities.shape,
                dtype=intensities.dtype,
                chunks=(-1, 1, 1, 1),
                compressor=None,
            )
            fast_mzs = fast_group.zeros(
                'mzs',
                shape=mzs.shape,
                dtype=mzs.dtype,
                chunks=(-1, 1, 1, 1),
                compressor=None,
            )
            for idx, (x, y, _) in enumerate(self.parser.coordinates):
                length = self.parser.mzLengths[idx]
                # Adjusted indexing for lengths array
                lengths[0, 0, y - 1, x - 1] = length
                spectra = self.parser.getspectrum(idx)
                fast_mzs[:length, 0, y - 1, x - 1] = spectra[0]
                fast_intensities[:length, 0, y - 1, x - 1] = spectra[1]
            copy_array(fast_intensities, intensities)
            copy_array(fast_mzs, mzs)



def convert_to_store(name: str, source_dir: Path, dest_store: zarr.DirectoryStore, imzml_filename: str, ibd_filename: str) -> None:
    """Convert a specific imzML file pair to a Zarr group."""
    pair = get_imzml_pair(source_dir, imzml_filename, ibd_filename)

    if pair is None:
        raise ValueError("The specified imzML and ibd files were not found in the directory.")

    with load_parser(*pair) as parser:
        is_continuous = "continuous" in parser.metadata.file_description.param_by_name
        is_processed = "processed" in parser.metadata.file_description.param_by_name

        if is_continuous == is_processed:
            raise ValueError("Invalid file mode, expected either 'continuous' or 'processed'.")

        root = zarr.group(store=dest_store)

        if is_continuous:
            _ContinuousImzMLConvertor(root, name, parser).run()
        else:
            _ProcessedImzMLConvertor(root, name, parser).run()
            
    print(f"File identified as {'processed' if is_processed else 'continuous'}")


def _get_xarray_axes(root: zarr.Group) -> List[str]:
    """Return axes metadata for Xarray compatibility."""
    return root.attrs['multiscales'][0]['axes']

class ImzMLToZarrConvertor:
    """Standalone converter class for converting ImzML to Zarr format."""

    def __init__(self, imzml_file: Path, ibd_file: Path):
        """
        Initialize the converter with the exact paths to the imzML and ibd files.
        
        Parameters:
        -----------
        imzml_file : Path
            Path to the specific imzML file.
        ibd_file : Path
            Path to the specific ibd file associated with the imzML file.
        """
        self.imzml_file = imzml_file
        self.ibd_file = ibd_file

    def convert(self, dest_path: Path) -> bool:
        """
        Convert the imzML data to Zarr format at the specified destination path.

        Parameters:
        -----------
        dest_path : Path
            Path where the Zarr output should be stored.

        Returns:
        --------
        bool
            True if the conversion was successful, False otherwise.
        """
        if dest_path.exists():
            logging.error(f"Destination {dest_path} already exists.")
            return False

        with contextlib.ExitStack() as stack:
            # Use filename stem (without suffix) as the Zarr root name
            name = dest_path.stem
            dest_store = zarr.DirectoryStore(dest_path)

            # Automatically remove the destination if conversion fails
            stack.callback(dest_store.rmdir)

            try:
                # Execute the actual conversion with specific filenames
                convert_to_store(name, self.imzml_file.parent, dest_store, 
                                 self.imzml_file.name, self.ibd_file.name)
            except (ValueError, KeyError) as error:
                logging.error("Conversion error", exc_info=error)
                return False
            except Exception as error:
                # Handle unexpected errors
                logging.error("Unexpected exception during conversion", exc_info=error)
                return False

            # Confirm conversion success by removing cleanup callback
            stack.pop_all()
            return True