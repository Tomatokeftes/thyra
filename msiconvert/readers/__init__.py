# msiconvert/readers/__init__.py
from pathlib import Path
from ..core.registry import register_format_detector
from . import imzml_reader, bruker_reader  # Import readers explicitly

@register_format_detector('imzml')
def detect_imzml(input_path: Path) -> bool:
    """Detect imzML format."""
    print(f"Testing imzML format for: {input_path}")
    return (input_path.suffix.lower() == '.imzml' and 
            input_path.is_file() and 
            input_path.exists())

@register_format_detector('bruker')
def detect_bruker(input_path: Path) -> bool:
    """Detect Bruker format."""
    return (input_path.suffix.lower() == '.d' and 
            input_path.is_dir() and 
            input_path.exists() and 
            (input_path / 'analysis.tsf').exists())