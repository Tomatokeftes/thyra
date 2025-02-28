# msiconvert/convert.py (updated)
from pathlib import Path
import logging
from typing import Optional, Dict, Any

from .core.registry import detect_format, get_reader_class, get_converter_class

def convert_msi(
    input_path: str,
    output_path: str,
    format_type: str = "spatialdata",
    dataset_id: str = "msi_dataset",
    pixel_size_um: float = 1.0,
    handle_3d: bool = False,
    **kwargs
) -> bool:
    """
    Convert MSI data to the specified format.
    
    Parameters:
    -----------
    input_path : str
        Path to the input MSI file or directory.
    output_path : str
        Path for the output file.
    format_type : str
        Output format type: "spatialdata" or "lightweight".
    dataset_id : str
        Identifier for the dataset.
    pixel_size_um : float
        Size of each pixel in micrometers.
    handle_3d : bool
        Whether to process as 3D data or as 2D slices.
    **kwargs : dict
        Additional arguments for specific converters.
    
    Returns:
    --------
    bool: True if conversion was successful, False otherwise.
    """
    input_path = Path(input_path).resolve()  # Use resolve() to get absolute path
    output_path = Path(output_path).resolve()
    
    print(f"Processing input file: {input_path}")
    
    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return False
        
    if output_path.exists():
        logging.error(f"Destination {output_path} already exists.")
        return False
    
    try:
        # Detect input format
        input_format = detect_format(input_path)
        
        # Create reader
        reader_class = get_reader_class(input_format)
        reader = reader_class(input_path)
        
        # Create converter
        converter_class = get_converter_class(format_type.lower())
        converter = converter_class(
            reader, 
            output_path,
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            handle_3d=handle_3d,
            **kwargs
        )
        
        # Run conversion
        return converter.convert()
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        return False