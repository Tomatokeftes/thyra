# thyra/core/registry.py
import logging
from pathlib import Path
from threading import RLock
from typing import Dict, Type

from .base_converter import BaseMSIConverter
from .base_reader import BaseMSIReader


class MSIRegistry:
    """Minimal thread-safe registry with extension-based format detection."""

    def __init__(self):
        """Initialize the MSI registry."""
        self._lock = RLock()
        self._readers: Dict[str, Type[BaseMSIReader]] = {}
        self._converters: Dict[str, Type[BaseMSIConverter]] = {}
        # Simple extension mapping - no complex detection needed!
        self._extension_to_format = {".imzml": "imzml", ".d": "bruker"}

    def register_reader(
        self, format_name: str, reader_class: Type[BaseMSIReader]
    ) -> None:
        """Register reader class."""
        with self._lock:
            self._readers[format_name] = reader_class
            logging.info(
                f"Registered reader {reader_class.__name__} for format "
                f"'{format_name}'"
            )

    def register_converter(
        self, format_name: str, converter_class: Type[BaseMSIConverter]
    ) -> None:
        """Register converter class."""
        with self._lock:
            self._converters[format_name] = converter_class
            logging.info(
                f"Registered converter {converter_class.__name__} for format "
                f"'{format_name}'"
            )

    def detect_format(self, input_path: Path) -> str:
        """Ultra-fast format detection via file extension."""
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        extension = input_path.suffix.lower()
        format_name = self._extension_to_format.get(extension)

        if not format_name:
            available = ", ".join(self._extension_to_format.keys())
            raise ValueError(
                f"Unsupported file extension '{extension}'. Supported: " f"{available}"
            )

        # Minimal validation
        if format_name == "imzml":
            ibd_path = input_path.with_suffix(".ibd")
            if not ibd_path.exists():
                raise ValueError(
                    f"ImzML file requires corresponding .ibd file: {ibd_path}"
                )
        elif format_name == "bruker":
            if not input_path.is_dir():
                raise ValueError(
                    f"Bruker format requires .d directory, got file: " f"{input_path}"
                )
            if (
                not (input_path / "analysis.tsf").exists()
                and not (input_path / "analysis.tdf").exists()
            ):
                raise ValueError(
                    f"Bruker .d directory missing analysis files: {input_path}"
                )

        return format_name

    def get_reader_class(self, format_name: str) -> Type[BaseMSIReader]:
        """Get reader class."""
        with self._lock:
            if format_name not in self._readers:
                available = list(self._readers.keys())
                raise ValueError(
                    f"No reader for format '{format_name}'. Available: " f"{available}"
                )
            return self._readers[format_name]

    def get_converter_class(self, format_name: str) -> Type[BaseMSIConverter]:
        """Get converter class."""
        with self._lock:
            if format_name not in self._converters:
                available = list(self._converters.keys())
                raise ValueError(
                    f"No converter for format '{format_name}'. Available: "
                    f"{available}"
                )
            return self._converters[format_name]


# Global registry instance
_registry = MSIRegistry()


# Simple public interface
def detect_format(input_path: Path) -> str:
    """Detect MSI format from input path.

    Args:
        input_path: Path to MSI data file or directory

    Returns:
        Format name ('imzml' or 'bruker')
    """
    return _registry.detect_format(input_path)


def get_reader_class(format_name: str) -> Type[BaseMSIReader]:
    """Get reader class for format.

    Args:
        format_name: MSI format name

    Returns:
        Reader class for the format
    """
    return _registry.get_reader_class(format_name)


def get_converter_class(format_name: str) -> Type[BaseMSIConverter]:
    """Get converter class for format.

    Args:
        format_name: MSI format name

    Returns:
        Converter class for the format
    """
    return _registry.get_converter_class(format_name)


def register_reader(format_name: str):
    """Decorator for reader registration."""

    def decorator(cls: Type[BaseMSIReader]):
        _registry.register_reader(format_name, cls)
        return cls

    return decorator


def register_converter(format_name: str):
    """Decorator for converter registration."""

    def decorator(cls: Type[BaseMSIConverter]):
        _registry.register_converter(format_name, cls)
        return cls

    return decorator
