from pathlib import Path
import abc
import logging
from typing import List, Tuple
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser
import zarr
from .registry import get_converter_class
from .base_converter import BaseMSIConverter

class MSIToZarrConverter:
    """Converter class for converting MSI data to Zarr format."""

    def __init__(self, input_path: Path, output_path: Path):
        self.input_path = input_path
        self.output_path = output_path
        self.root = None
        self.name = None
        self.converter_kwargs = {}

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

    def _get_converter(self) -> BaseMSIConverter:
        """Determine the format and return the appropriate converter instance."""
        format_name = self._determine_format()
        converter_class = get_converter_class(format_name)
        converter = converter_class(self.root, self.name, **self.converter_kwargs)
        return converter

    def _determine_format(self) -> str:
        """Determine the format of the input data and set up necessary parameters."""
        if self.input_path.suffix.lower() == '.imzml':
            return self._determine_imzml_format()
        elif self.input_path.suffix.lower() == '.d':
            return 'bruker'
        else:
            raise ValueError("Unsupported file format or invalid input path.")

    def _determine_imzml_format(self) -> str:
        """Determine if the imzML file is continuous or processed."""
        from pyimzml.ImzMLParser import ImzMLParser

        imzml_file = self.input_path
        ibd_file = imzml_file.with_suffix('.ibd')

        if not ibd_file.exists():
            raise ValueError(f"Corresponding .ibd file not found for {imzml_file}")

        ibd_file_handle = open(ibd_file, mode="rb")
        try:
            parser = ImzMLParser(
                filename=str(imzml_file),
                parse_lib="lxml",
                ibd_file=ibd_file_handle
            )

            is_continuous = "continuous" in parser.metadata.file_description.param_by_name
            is_processed = "processed" in parser.metadata.file_description.param_by_name

            if is_continuous == is_processed:
                raise ValueError("Invalid file mode, expected either 'continuous' or 'processed'.")

            self.name = self.output_path.stem
            self.root = zarr.group(store=zarr.DirectoryStore(self.output_path))
            self.converter_kwargs = {'parser': parser}

            if is_continuous:
                return 'imzml_continuous'
            else:
                return 'imzml_processed'
        except Exception:
            ibd_file_handle.close()
            raise

