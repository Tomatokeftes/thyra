__version__ = "0.1.0" 

from .convert import convert_msi
# Import readers to ensure detectors are registered
from .readers import imzml_reader, bruker_reader
# Import converters to ensure they are registered
from .converters import spatialdata_converter, lightweight_converter, anndata_converter