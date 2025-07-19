__version__ = "1.5.0" 

from .convert import convert_msi
# Import readers to ensure detectors are registered
from .readers import imzml_reader, bruker
# Import converters to ensure they are registered
from .converters import *