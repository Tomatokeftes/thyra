# msiconvert/core/base_converter.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from .base_reader import BaseMSIReader

class BaseMSIConverter(ABC):
    """Abstract base class for converting MSI data to target formats."""
    
    def __init__(self, reader: BaseMSIReader, output_path: Path, **kwargs):
        self.reader = reader
        self.output_path = output_path
        self.options = kwargs
    
    @abstractmethod
    def convert(self) -> bool:
        """
        Convert the MSI data to the target format.
        
        Returns:
        --------
        bool: True if conversion was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Add metadata to the output file."""
        pass