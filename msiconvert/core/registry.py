# msiconvert/core/registry.py
from typing import Dict, Type, Callable
from pathlib import Path
from .base_reader import BaseMSIReader
from .base_converter import BaseMSIConverter

# Registries for readers and converters
reader_registry: Dict[str, Type[BaseMSIReader]] = {}
converter_registry: Dict[str, Type[BaseMSIConverter]] = {}

# Registry for format detection
format_detectors: Dict[str, Callable[[Path], bool]] = {}

def register_reader(format_name: str):
    """Decorator to register a reader class for a specific format."""
    def decorator(cls: Type[BaseMSIReader]):
        reader_registry[format_name] = cls
        return cls
    return decorator

def register_converter(format_name: str):
    """Decorator to register a converter class for a specific format."""
    def decorator(cls: Type[BaseMSIConverter]):
        converter_registry[format_name] = cls
        return cls
    return decorator

def register_format_detector(format_name: str):
    """Decorator to register a function that detects a specific format."""
    def decorator(func: Callable[[Path], bool]):
        print(f"Registering format detector for: {format_name}")
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
    """Detect the format of the input data."""
    print(f"Attempting to detect format for: {input_path}")
    print(f"File exists: {input_path.exists()}")
    print(f"Is file: {input_path.is_file()}")
    print(f"Suffix: {input_path.suffix.lower()}")
    
    for format_name, detector in format_detectors.items():
        print(f"Checking detector for format: {format_name}")
        result = detector(input_path)
        print(f"  Result: {result}")
        if result:
            return format_name
    
    raise ValueError(f"Unable to detect format for: {input_path}")
