# thyra/readers/__init__.py
"""MSI data readers for various instrument formats.

This package provides reader implementations for different MSI data formats:
- ImzML: Open format for MSI data
- Bruker: timsTOF and FlexImaging data

Reader organization:
- bruker/: All Bruker formats (timsTOF, FlexImaging)
- imzml/: ImzML format reader
"""

# Import readers to trigger registration
from . import bruker, imzml  # noqa: F401
