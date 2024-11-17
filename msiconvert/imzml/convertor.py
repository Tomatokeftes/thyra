import abc
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import zarr
from numcodecs import Blosc

from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser
import dask.array as da
from dask.diagnostics import ProgressBar

from ..utils.temp_store import single_temp_store

# Set a byte size threshold for copying arrays
_DISK_COPY_THRESHOLD = 8 * 10**9

# Type alias for shape dimensions
SHAPE = Tuple[int, int, int, int]

class MSIToZarrConvertor:
    """Converter class for converting MSI data to Zarr format."""

    def __init__(self, input_path: Path, output_path: Path):
        """
        Initialize the converter with the input path and output path.

        Parameters:
        -----------
        input_path : Path
            Path to the input file (imzML or Bruker .d directory) or directory containing data files.
        output_path : Path
            Path where the Zarr output should be stored.
        """
        self.input_path = input_path
        self.output_path = output_path

    def convert(self) -> bool:
        if self.output_path.exists():
            logging.error(f"Destination {self.output_path} already exists.")
            return False

        try:
            converter = self._get_converter()
            converter.run()
            zarr.consolidate_metadata(str(self.output_path))
            return True
        except Exception as error:
            logging.error("Conversion failed", exc_info=error)
            return False

    def _get_converter(self) -> '_BaseMSIConvertor':
        """Determine the format and return the appropriate converter instance."""
        if self._is_imzml():
            return self._create_imzml_converter()
        elif self._is_bruker():
            return self._create_bruker_converter()
        else:
            raise ValueError("Unsupported file format or invalid input path.")

    def _is_imzml(self) -> bool:
        """Check if the input path ends with '.imzML'."""
        return self.input_path.suffix.lower() == '.imzml'

    def _is_bruker(self) -> bool:
        """Check if the input path ends with '.d'."""
        return self.input_path.suffix.lower() == '.d'

    def _create_bruker_converter(self) -> '_BaseMSIConvertor':
        """Create a Bruker converter instance."""
        if not self.input_path.is_dir():
            raise ValueError("Expected a directory for Bruker data.")

        # Initialize Bruker data reader with self.input_path
        # For now, raise NotImplementedError since Bruker converter is not implemented
        raise NotImplementedError("Bruker conversion is not yet implemented.")


    def _create_imzml_converter(self) -> '_BaseImzMLConvertor':
        """Create an imzML converter instance."""
        imzml_file = self.input_path
        ibd_file = imzml_file.with_suffix('.ibd')

        if not ibd_file.exists():
            raise ValueError(f"Corresponding .ibd file not found for {imzml_file}")

        # Initialize the parser
        ibd_file_handle = open(ibd_file, mode="rb")
        try:
            parser = PyImzMLParser(
                filename=str(imzml_file),
                parse_lib="lxml",
                ibd_file=ibd_file_handle
            )

            # Determine if continuous or processed
            is_continuous = "continuous" in parser.metadata.file_description.param_by_name
            is_processed = "processed" in parser.metadata.file_description.param_by_name

            if is_continuous == is_processed:
                raise ValueError("Invalid file mode, expected either 'continuous' or 'processed'.")

            # Use filename stem as dataset name
            name = self.output_path.stem
            root = zarr.group(store=zarr.DirectoryStore(self.output_path))

            if is_continuous:
                converter = _ContinuousImzMLConvertor(root, name, parser)
            else:
                converter = _ProcessedImzMLConvertor(root, name, parser)
            return converter
        except Exception:
            ibd_file_handle.close()
            raise

class _BaseMSIConvertor(abc.ABC):
    """Abstract base class to handle MSI data conversion using polymorphism."""

    def __init__(self, root: zarr.Group, name: str) -> None:
        self.root = root
        self.name = name

    @abc.abstractmethod
    def get_labels(self) -> List[str]:
        """Return the list of labels associated with the image."""

    @abc.abstractmethod
    def add_base_metadata(self) -> None:
        """Add format-specific metadata to the root group."""

    @abc.abstractmethod
    def create_zarr_arrays(self):
        """Generate empty arrays inside the root group."""

    @abc.abstractmethod
    def read_binary_data(self) -> None:
        """Fill in the arrays with data from the source file."""

    def run(self) -> None:
        """Primary method to add metadata, create arrays, and read binary data."""
        self.add_base_metadata()
        self.create_zarr_arrays()
        self.read_binary_data()

class _BaseImzMLConvertor(_BaseMSIConvertor):
    """Base class for imzML-specific converters."""

    def __init__(self, root: zarr.Group, name: str, parser: PyImzMLParser) -> None:
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
        """Primary method to add metadata, create arrays, and read binary data."""
        try:
            self.add_base_metadata()
            self.create_zarr_arrays()
            self.read_binary_data()
        finally:
            # Ensure the ibd file is closed after conversion
            self.parser.m.close()


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

class _BrukerConvertor(_BaseMSIConvertor):
    """Converter class for Bruker format."""

    def __init__(self, root: zarr.Group, name: str, bruker_data) -> None:
        super().__init__(root, name)
        self.bruker_data = bruker_data

    def add_base_metadata(self) -> None:
        """Add Bruker-specific metadata to the root group."""
        self.root.attrs['multiscales'] = [{
            'version': '0.4',
            'name': self.name,
            'datasets': [{'path': '0'}],
            'axes': ['c', 'z', 'y', 'x'],
            'type': 'none',
        }]
        self.root.attrs['bruker'] = {
            'source': self.bruker_data.source_file,
            # Additional Bruker metadata
        }
        self.root.create_group('labels').attrs['labels'] = self.get_labels()

    def get_labels(self) -> List[str]:
        # Implement labels for Bruker data
        pass

    def create_zarr_arrays(self):
        # Implement array creation for Bruker data
        pass

    def read_binary_data(self) -> None:
        # Implement data reading for Bruker data
        pass