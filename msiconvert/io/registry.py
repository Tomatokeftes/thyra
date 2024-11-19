from typing import Dict, Type
from .base_converter import BaseMSIConverter

converter_registry: Dict[str, Type[BaseMSIConverter]] = {}

def register_converter(format_name: str):
    def decorator(cls: Type[BaseMSIConverter]):
        converter_registry[format_name] = cls
        return cls
    return decorator

def get_converter_class(format_name: str) -> Type[BaseMSIConverter]:
    try:
        return converter_registry[format_name]
    except KeyError:
        raise ValueError(f"No converter registered for format '{format_name}'")
