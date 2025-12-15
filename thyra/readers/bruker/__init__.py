"""Bruker MSI reader implementations.

This package provides readers for Bruker MSI data formats:
- timsTOF: TSF/TDF data via SDK (BrukerReader)
- FlexImaging: MALDI-TOF data via pure Python (FlexImagingReader)

Organization:
- timstof/: timsTOF reader and SDK integration
- fleximaging/: FlexImaging reader (pure Python)

Common functionality is provided by BrukerBaseMSIReader and
BrukerFolderStructure for folder analysis.
"""

from ...utils.bruker_exceptions import (
    BrukerReaderError,
    DataError,
    FileFormatError,
    SDKError,
)
from .base_bruker_reader import BrukerBaseMSIReader

# Import readers from submodules to trigger registration
from .fleximaging.fleximaging_reader import FlexImagingReader
from .folder_structure import BrukerFolderInfo, BrukerFolderStructure, BrukerFormat
from .timstof.timstof_reader import BrukerReader

__all__ = [
    # Base classes
    "BrukerBaseMSIReader",
    "BrukerFolderStructure",
    "BrukerFolderInfo",
    "BrukerFormat",
    # Readers
    "BrukerReader",
    "FlexImagingReader",
    # Exceptions
    "BrukerReaderError",
    "DataError",
    "FileFormatError",
    "SDKError",
]
