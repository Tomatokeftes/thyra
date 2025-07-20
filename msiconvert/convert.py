# msiconvert/convert.py
import logging
import traceback
import warnings
from pathlib import Path

from .core.registry import detect_format, get_converter_class, get_reader_class

# from cryptography.utils import CryptographyDeprecationWarning


warnings.filterwarnings(
    "ignore",
    message=r"Accession IMS:1000046.*",  # or just "ignore" all UserWarning from that module
    category=UserWarning,
    module=r"pyimzml.ontology.ontology",
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
    pixel_size_um: float = None,
    handle_3d: bool = False,
    **kwargs,
) -> bool:
    """Convert MSI data to the specified format with enhanced error handling and automatic pixel size detection."""

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

    if pixel_size_um is not None and (
        not isinstance(pixel_size_um, (int, float)) or pixel_size_um <= 0
    ):
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

        # Handle automatic pixel size detection if not provided
        final_pixel_size = pixel_size_um
        pixel_size_detection_info = None

        if pixel_size_um is None:
            logging.info("Attempting automatic pixel size detection...")
            detected_pixel_size = reader.get_pixel_size()
            if detected_pixel_size is not None:
                final_pixel_size = detected_pixel_size[
                    0
                ]  # Use X size (assuming square pixels)
                logging.info(
                    f"✓ Automatically detected pixel size: {detected_pixel_size[0]:.1f} x {detected_pixel_size[1]:.1f} μm"
                )

                # Create pixel size detection provenance metadata
                pixel_size_detection_info = {
                    "method": "automatic",
                    "detected_x_um": float(detected_pixel_size[0]),
                    "detected_y_um": float(detected_pixel_size[1]),
                    "source_format": input_format,
                    "detection_successful": True,
                    "note": "Pixel size automatically detected from source metadata and applied to coordinate systems",
                }
            else:
                logging.error(
                    "✗ Could not automatically detect pixel size from metadata"
                )
                logging.error(
                    "Please specify --pixel-size manually or ensure the input file contains pixel size metadata"
                )
                return False
        else:
            # Manual pixel size was provided
            pixel_size_detection_info = {
                "method": "manual",
                "source_format": input_format,
                "detection_successful": False,
                "note": "Pixel size manually specified via --pixel-size parameter and applied to coordinate systems",
            }

        # Create converter
        converter_class = get_converter_class(format_type.lower())
        logging.info(f"Using converter: {converter_class.__name__}")
        converter = converter_class(
            reader,
            output_path,
            dataset_id=dataset_id,
            pixel_size_um=final_pixel_size,
            handle_3d=handle_3d,
            pixel_size_detection_info=pixel_size_detection_info,
            **kwargs,
        )

        # Run conversion
        logging.info("Starting conversion...")
        result = converter.convert()
        logging.info(f"Conversion {'completed successfully' if result else 'failed'}")
        return result
    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        # Log detailed traceback for debugging
        logging.error(f"Detailed traceback:\n{traceback.format_exc()}")
        return False
