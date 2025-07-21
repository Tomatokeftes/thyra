__version__ = "1.8.1"

from .convert import convert_msi

# Import converters to ensure they are registered
from .converters import *

# Import readers to ensure detectors are registered
from .readers import bruker, imzml_reader
