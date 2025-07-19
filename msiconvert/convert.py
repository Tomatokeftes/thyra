# msiconvert/convert.py
from pathlib import Path
import logging
import traceback
import warnings
# from cryptography.utils import CryptographyDeprecationWarning

from .core.registry import detect_format, get_reader_class, get_converter_class

warnings.filterwarnings(
    "ignore", 
    message=r"Accession IMS:1000046.*",  # or just "ignore" all UserWarning from that module
    category=UserWarning,
    module=r"pyimzml.ontology.ontology"
)

# warnings.filterwarnings(
#     "ignore", 
#     category=CryptographyDeprecationWarning
# )


def convert_msi(
    input_path: str,
    output_path: str,
    format_type: str = "spatialdata",
    dataset_id: str = "msi_dataset",
    pixel_size_um: float = 1.0,
    handle_3d: bool = False,
    **kwargs
) -> bool:
    """Convert MSI data to the specified format with enhanced error handling."""
    
    # Input validation
    if not input_path or not isinstance(input_path, (str, Path)):
        logging.error("Input path must be a valid string or Path object")
        return False
    
    if not output_path or not isinstance(output_path, (str, Path)):
        logging.error("Output path must be a valid string or Path object")
        return False
    
    if not isinstance(format_type, str) or not format_type.strip():
        logging.error("Format type must be a non-empty string")
        return False
    
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        logging.error("Dataset ID must be a non-empty string")
        return False
    
    if not isinstance(pixel_size_um, (int, float)) or pixel_size_um <= 0:
        logging.error("Pixel size must be a positive number")
        return False
    
    if not isinstance(handle_3d, bool):
        logging.error("handle_3d must be a boolean value")
        return False
    
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    logging.info(f"Processing input file: {input_path}")
    
    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return False
        
    if output_path.exists():
        logging.error(f"Destination {output_path} already exists.")
        return False
    
    try:
        # Detect input format
        input_format = detect_format(input_path)
        logging.info(f"Detected format: {input_format}")
        
        # Create reader
        reader_class = get_reader_class(input_format)
        logging.info(f"Using reader: {reader_class.__name__}")
        reader = reader_class(input_path)
        
        # Create converter
        converter_class = get_converter_class(format_type.lower())
        logging.info(f"Using converter: {converter_class.__name__}")
        converter = converter_class(
            reader, 
            output_path,
            dataset_id=dataset_id,
            pixel_size_um=pixel_size_um,
            handle_3d=handle_3d,
            **kwargs
        )
        
        # Run conversion
        logging.info("Starting conversion...")
        result = converter.convert()
        logging.info(f"Conversion {'completed successfully' if result else 'failed'}")
        return result
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        # Log detailed traceback for debugging
        logging.debug(f"Detailed traceback:\n{traceback.format_exc()}")
        return False