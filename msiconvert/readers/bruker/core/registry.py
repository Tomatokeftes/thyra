"""
Reader registry for automatic reader discovery and registration.

This module provides the registry system that allows readers to be automatically
discovered and used based on data format detection.
"""

from typing import Dict, Type, Any, Optional
from pathlib import Path
import logging

from .base_reader import BaseMSIReader
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Global registry of readers
_READER_REGISTRY: Dict[str, Type[BaseMSIReader]] = {}


def register_reader(format_name: str):
    """
    Decorator to register a reader class for a specific format.
    
    Args:
        format_name: String identifier for the format (e.g., 'bruker', 'imzml')
        
    Returns:
        Decorator function
        
    Example:
        @register_reader('bruker')
        class BrukerReader(BaseMSIReader):
            pass
    """
    def decorator(reader_class: Type[BaseMSIReader]):
        if not issubclass(reader_class, BaseMSIReader):
            raise ConfigurationError(f"Reader {reader_class.__name__} must inherit from BaseMSIReader")
        
        _READER_REGISTRY[format_name] = reader_class
        logger.debug(f"Registered reader {reader_class.__name__} for format '{format_name}'")
        return reader_class
    
    return decorator


def get_reader(format_name: str) -> Type[BaseMSIReader]:
    """
    Get a reader class by format name.
    
    Args:
        format_name: String identifier for the format
        
    Returns:
        Reader class for the specified format
        
    Raises:
        ConfigurationError: If no reader is registered for the format
    """
    if format_name not in _READER_REGISTRY:
        available_formats = list(_READER_REGISTRY.keys())
        raise ConfigurationError(
            f"No reader registered for format '{format_name}'. "
            f"Available formats: {available_formats}"
        )
    
    return _READER_REGISTRY[format_name]


def detect_format(data_path: Path) -> Optional[str]:
    """
    Automatically detect the data format based on file structure.
    
    Args:
        data_path: Path to the data file or directory
        
    Returns:
        String identifier of the detected format, or None if unknown
    """
    data_path = Path(data_path)
    
    # Check for Bruker formats
    if data_path.is_dir() and data_path.suffix == '.d':
        tsf_file = data_path / "analysis.tsf"
        tdf_file = data_path / "analysis.tdf"
        
        if tsf_file.exists() or tdf_file.exists():
            return 'bruker'
    
    # Check for imzML format
    if data_path.is_file() and data_path.suffix.lower() == '.imzml':
        return 'imzml'
    
    return None


def get_reader_for_path(data_path: Path, **kwargs) -> BaseMSIReader:
    """
    Automatically detect format and create appropriate reader.
    
    Args:
        data_path: Path to the data file or directory
        **kwargs: Additional arguments to pass to the reader constructor
        
    Returns:
        Initialized reader instance
        
    Raises:
        ConfigurationError: If format cannot be detected or no reader available
    """
    format_name = detect_format(data_path)
    
    if format_name is None:
        raise ConfigurationError(f"Cannot detect format for path: {data_path}")
    
    reader_class = get_reader(format_name)
    return reader_class(data_path, **kwargs)


def list_registered_readers() -> Dict[str, Type[BaseMSIReader]]:
    """
    Get a copy of all registered readers.
    
    Returns:
        Dictionary mapping format names to reader classes
    """
    return _READER_REGISTRY.copy()