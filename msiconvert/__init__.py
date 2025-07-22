"""
MSIConverter - Convert Mass Spectrometry Imaging data to SpatialData/Zarr format.

This package provides tools for converting MSI data from various formats (ImzML, Bruker)
into the modern SpatialData/Zarr format with automatic pixel size detection.
"""

__version__ = "1.8.2"

# Import readers to trigger format detector registration
from . import readers  # This triggers the format detector registrations
from .convert import convert_msi

# Import key components - avoid wildcard imports
try:
    from .converters.spatialdata_converter import SpatialDataConverter
except ImportError:
    # SpatialData dependencies not available
    SpatialDataConverter = None

# Expose main API
__all__ = [
    "__version__",
    "convert_msi",
    "SpatialDataConverter",
]
