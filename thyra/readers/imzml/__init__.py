# thyra/readers/imzml/__init__.py
"""ImzML MSI reader implementation.

This package provides the reader for ImzML format, the open standard
for mass spectrometry imaging data exchange.
"""

from .imzml_reader import ImzMLReader

__all__ = [
    "ImzMLReader",
]
