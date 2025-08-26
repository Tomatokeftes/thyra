"""
MSIConverter - Convert Mass Spectrometry Imaging data to SpatialData/Zarr
format.

This package provides tools for converting MSI data from various formats
(ImzML, Bruker) into the modern SpatialData/Zarr format with automatic
pixel size detection.
"""

# Suppress known warnings from dependencies
import warnings

# Import readers and converters to trigger registration
from . import converters  # This triggers converter registrations  # noqa: F401
from . import readers  # This triggers reader registrations  # noqa: F401
from .convert import convert_msi

warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings("ignore", category=FutureWarning, module="spatialdata")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numba")
warnings.filterwarnings("ignore", category=UserWarning, module="xarray_schema")
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)

__version__ = "1.8.4"

# Import key components - avoid wildcard imports
try:
    from .converters.spatialdata.converter import SpatialDataConverter
except ImportError:
    # SpatialData dependencies not available
    SpatialDataConverter = None

# Expose main API
__all__ = [
    "__version__",
    "convert_msi",
    "SpatialDataConverter",
]
