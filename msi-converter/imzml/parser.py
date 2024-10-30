from __future__ import annotations
import warnings
import numpy as np
from pyimzml.ImzMLParser import ImzMLParser as PyImzMLParser

_REMOVE_WARNINGS = False

class ImageChannel:
    """Represents a single channel within an image."""
    def __init__(self, index: int, color: str = None, suggested_name: str = None):
        self.index = index
        self.color = color
        self.suggested_name = suggested_name

class ImageMetadata:
    """Structure to hold the core image metadata."""
    def __init__(self):
        self.width = None
        self.height = None
        self.depth = 1
        self.duration = 1
        self.n_channels = None
        self.n_channels_per_read = None
        self.n_distinct_channels = None
        self.pixel_type = None
        self.significant_bits = None
        self.channels = []

    def set_channel(self, channel: ImageChannel):
        self.channels.append(channel)

class MetadataStore:
    """Structure to hold raw metadata in a key-value format."""
    def __init__(self):
        self.store = {}

    def add(self, key, value):
        self.store[key] = value

class ImzMLParser:
    """Standalone Parser for ImzML."""
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._parser = None

    def get_parser(self) -> PyImzMLParser:
        """Returns the PyImzMLParser instance for the given file."""
        if self._parser is None:
            if _REMOVE_WARNINGS:
                warnings.filterwarnings('ignore', message=r'.*Accession IMS.*')
                warnings.filterwarnings('ignore', message=r'.*Accession MS.*')

            self._parser = PyImzMLParser(
                self.file_path,
                parse_lib='lxml',
                ibd_file=None,
                include_spectra_metadata=None,
            )
        return self._parser

    def parse_main_metadata(self) -> ImageMetadata:
        """Parse the essential metadata for the ImzML file."""
        parser = self.get_parser()

        # Check for continuous or processed mode
        is_continuous = 'continuous' in parser.metadata.file_description.param_by_name
        is_processed = 'processed' in parser.metadata.file_description.param_by_name
        if is_continuous == is_processed:
            raise ValueError("Invalid file mode, expected either 'continuous' or 'processed'")

        metadata = ImageMetadata()
        metadata.width = parser.imzmldict['max count of pixels x']
        metadata.height = parser.imzmldict['max count of pixels y']
        metadata.depth = 1
        metadata.duration = 1

        # Set channels based on mode
        if is_continuous:
            metadata.n_channels = parser.mzLengths[0]
        else:
            metadata.n_channels = max(parser.mzLengths)

        metadata.n_channels_per_read = metadata.n_channels
        metadata.n_distinct_channels = metadata.n_channels
        metadata.pixel_type = np.dtype(parser.intensityPrecision)
        metadata.significant_bits = metadata.pixel_type.itemsize * 8

        # Initialize channels
        for channel in range(metadata.n_channels):
            metadata.set_channel(ImageChannel(channel))

        return metadata

    def parse_raw_metadata(self) -> MetadataStore:
        """Return all raw metadata in a dictionary format."""
        parser = self.get_parser()
        raw_metadata = MetadataStore()
        raw_metadata.add("width", parser.imzmldict.get('max count of pixels x'))
        raw_metadata.add("height", parser.imzmldict.get('max count of pixels y'))
        raw_metadata.add("pixel_type", parser.intensityPrecision)
        raw_metadata.add("mz_precision", parser.mzPrecision)
        return raw_metadata
