"""Bruker MSI reader implementations.

This package provides readers for Bruker MSI data formats:
- timsTOF: TSF/TDF data via SDK (BrukerReader)
- Rapiflex: MALDI-TOF data via pure Python (RapiflexReader)

Organization:
- timstof/: timsTOF reader and SDK integration
- rapiflex/: Rapiflex reader (pure Python)

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
from .folder_structure import BrukerFolderInfo, BrukerFolderStructure, BrukerFormat

# Import readers from submodules to trigger registration
from .rapiflex import RapiflexReader
from .timstof.timstof_reader import BrukerReader

__all__ = [
    # Base classes
    "BrukerBaseMSIReader",
    "BrukerFolderStructure",
    "BrukerFolderInfo",
    "BrukerFormat",
    # Readers
    "BrukerReader",
    "RapiflexReader",
    # Exceptions
    "BrukerReaderError",
    "DataError",
    "FileFormatError",
    "SDKError",
]
