# thyra/core/registry.py
import logging
from pathlib import Path
from threading import RLock
from typing import Dict, Type

from .base_converter import BaseMSIConverter
from .base_reader import BaseMSIReader


class MSIRegistry:
    """Thread-safe registry with format detection for MSI data."""

    def __init__(self):
        """Initialize the MSI registry."""
        self._lock = RLock()
        self._readers: Dict[str, Type[BaseMSIReader]] = {}
        self._converters: Dict[str, Type[BaseMSIConverter]] = {}
        # Extension mapping for file-based formats
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

    def _is_fleximaging_folder(self, path: Path) -> bool:
        """Check if path is a FlexImaging data folder.

        FlexImaging folders contain:
        - *.dat file (spectral data)
        - *_poslog.txt file (coordinates)
        - *_info.txt file (metadata)

        Args:
            path: Path to check

        Returns:
            True if path is a FlexImaging folder
        """
        if not path.is_dir():
            return False

        has_dat = bool(list(path.glob("*.dat")))
        has_poslog = bool(list(path.glob("*_poslog.txt")))
        has_info = bool(list(path.glob("*_info.txt")))

        return has_dat and has_poslog and has_info

    def detect_format(self, input_path: Path) -> str:
        """Detect MSI format from input path.

        Supports:
        - .imzml files (ImzML format)
        - .d directories (Bruker timsTOF)
        - Folders with .dat + _poslog.txt (Bruker FlexImaging)
        """
        if not input_path.exists():
            raise ValueError(f"Input path does not exist: {input_path}")

        # Check extension-based formats first
        extension = input_path.suffix.lower()
        format_name = self._extension_to_format.get(extension)

        # If no extension match and it's a directory, check for FlexImaging
        if not format_name and input_path.is_dir():
            if self._is_fleximaging_folder(input_path):
                format_name = "fleximaging"

        if not format_name:
            available = list(self._extension_to_format.keys()) + [
                "folder (FlexImaging)"
            ]
            raise ValueError(
                f"Unsupported format for '{input_path}'. "
                f"Supported: {', '.join(available)}"
            )

        # Format-specific validation
        if format_name == "imzml":
            ibd_path = input_path.with_suffix(".ibd")
            if not ibd_path.exists():
                raise ValueError(
                    f"ImzML file requires corresponding .ibd file: {ibd_path}"
                )
        elif format_name == "bruker":
            if not input_path.is_dir():
                raise ValueError(
                    f"Bruker format requires .d directory, got file: {input_path}"
                )
            if (
                not (input_path / "analysis.tsf").exists()
                and not (input_path / "analysis.tdf").exists()
            ):
                raise ValueError(
                    f"Bruker .d directory missing analysis files: {input_path}"
                )
        elif format_name == "fleximaging":
            # Already validated in _is_fleximaging_folder
            pass

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
        Format name ('imzml', 'bruker', or 'fleximaging')
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
