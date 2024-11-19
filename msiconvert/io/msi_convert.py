from pathlib import Path
import abc
import logging
from typing import List, Tuple
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser
import zarr

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

    def _get_converter(self) -> 'BaseMSIConvertor':
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

    def _create_bruker_converter(self) -> 'BaseMSIConvertor':
        """Create a Bruker converter instance."""
        if not self.input_path.is_dir():
            raise ValueError("Expected a directory for Bruker data.")

        # Initialize Bruker data reader with self.input_path
        # For now, raise NotImplementedError since Bruker converter is not implemented
        raise NotImplementedError("Bruker conversion is not yet implemented.")


    def _create_imzml_converter(self) -> 'BaseImzMLConvertor':
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
                from msiconvert.io.imzml_convertor import ContinuousImzMLConvertor
                converter = ContinuousImzMLConvertor(root, name, parser)
            else:
                from msiconvert.io.imzml_convertor import ProcessedImzMLConvertor
                converter = ProcessedImzMLConvertor(root, name, parser)
            return converter
        except Exception:
            ibd_file_handle.close()
            raise

class BaseMSIConvertor(abc.ABC):
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
