# thyra/readers/__init__.py
"""MSI data readers for various instrument formats.

This package provides reader implementations for different MSI data formats:
- ImzML: Open format for MSI data
- Bruker: timsTOF data via SDK
- FlexImaging: Bruker MALDI-TOF data (rapifleX, autoflex, etc.)
"""

# Import readers to trigger registration
from . import bruker, fleximaging, imzml_reader  # noqa: F401
