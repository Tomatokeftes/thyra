# thyra/readers/waters/__init__.py
"""Waters MSI reader implementation.

This package provides the reader for Waters .raw format using
MassLynx native libraries (MassLynxRaw + MLReader). Supports
Waters mass spectrometry imaging data with spatial laser coordinates.

Requires the native libraries to be present in the lib/ subdirectory
or accessible via the WATERS_SDK_PATH environment variable.
"""

from .waters_reader import WatersReader

__all__ = [
    "WatersReader",
]
