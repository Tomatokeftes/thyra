# thyra/readers/__init__.py
"""MSI data readers for various instrument formats.

This package provides reader implementations for different MSI data formats:
- ImzML: Open format for MSI data
- Bruker: timsTOF and Rapiflex data

Reader organization:
- bruker/: All Bruker formats (timsTOF, Rapiflex)
- imzml/: ImzML format reader
"""

# Import readers to trigger registration
from . import bruker, imzml  # noqa: F401
