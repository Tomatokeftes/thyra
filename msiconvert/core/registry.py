# msiconvert/core/registry.py
import logging
from pathlib import Path
from typing import Callable, Dict, Type

from .base_converter import BaseMSIConverter
from .base_reader import BaseMSIReader

# Registries for readers and converters
reader_registry: Dict[str, Type[BaseMSIReader]] = {}
converter_registry: Dict[str, Type[BaseMSIConverter]] = {}

# Registry for format detection
format_detectors: Dict[str, Callable[[Path], bool]] = {}


def register_reader(format_name: str):
    """Decorator to register a reader class for a specific format."""

    def decorator(cls: Type[BaseMSIReader]):
        if format_name in reader_registry:
            logging.warning(f"Overwriting existing reader for format '{format_name}'")
        reader_registry[format_name] = cls
        logging.info(f"Registered reader {cls.__name__} for format '{format_name}'")
        return cls

    return decorator


def register_converter(format_name: str):
    """Decorator to register a converter class for a specific format."""

    def decorator(cls: Type[BaseMSIConverter]):
        if format_name in converter_registry:
            logging.warning(
                f"Overwriting existing converter for format '{format_name}'"
            )
        converter_registry[format_name] = cls
        logging.info(f"Registered converter {cls.__name__} for format '{format_name}'")
        return cls

    return decorator


def register_format_detector(format_name: str):
    """Decorator to register a function that detects a specific format."""

    def decorator(func: Callable[[Path], bool]):
        format_detectors[format_name] = func
        return func

    return decorator


def get_reader_class(format_name: str) -> Type[BaseMSIReader]:
    """Get reader class for the specified format."""
    try:
        return reader_registry[format_name]
    except KeyError:
        raise ValueError(f"No reader registered for format '{format_name}'")


def get_converter_class(format_name: str) -> Type[BaseMSIConverter]:
    """Get converter class for the specified format."""
    try:
        return converter_registry[format_name]
    except KeyError:
        raise ValueError(f"No converter registered for format '{format_name}'")


def detect_format(input_path: Path) -> str:
    """
    Detect the format of the input data.

    Args:
        input_path: Path to input file or directory

    Returns:
        Detected format name

    Raises:
        ValueError: If format could not be detected
    """
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")

    logging.info(f"Attempting to detect format for: {input_path}")
    logging.debug(f"File exists: {input_path.exists()}")
    logging.debug(f"Is file: {input_path.is_file()}")
    logging.debug(f"Suffix: {input_path.suffix.lower()}")

    for format_name, detector in format_detectors.items():
        logging.debug(f"Checking detector for format: {format_name}")
        try:
            result = detector(input_path)
            logging.debug(f"  Result: {result}")
            if result:
                return format_name
        except Exception as e:
            logging.warning(f"Error in format detector for {format_name}: {e}")

    # If we get here, no format was detected
    supported_formats = ", ".join(format_detectors.keys())
    raise ValueError(
        f"Unable to detect format for: {input_path}. "
        f"Supported formats are: {supported_formats}"
    )
