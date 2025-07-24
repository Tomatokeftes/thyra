# msiconvert/__main__.py
import argparse
import logging
import sys
from pathlib import Path

from msiconvert.convert import convert_msi
from msiconvert.core.registry import detect_format, get_reader_class
from msiconvert.utils.data_processors import optimize_zarr_chunks
from msiconvert.utils.logging_config import setup_logging


def prompt_for_pixel_size(detected_size=None):
    """
    Interactively prompt user for pixel size.

    Args:
        detected_size: Optional tuple of (x_size, y_size) if detection succeeded

    Returns:
        float: Pixel size in micrometers
    """
    if detected_size is not None:
        print(
            f"\nOK Automatically detected pixel size: {detected_size[0]:.1f} x {detected_size[1]:.1f} um"
        )
        while True:
            response = input("Use detected pixel size? [Y/n]: ").strip().lower()
            if response in ["", "y", "yes"]:
                return detected_size[0]  # Use X size (assuming square pixels)
            elif response in ["n", "no"]:
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

    # Manual input
    while True:
        try:
            print("\nPixel size could not be automatically detected.")
            pixel_size_input = input(
                "Please enter pixel size in micrometers (um): "
            ).strip()
            pixel_size = float(pixel_size_input)
            if pixel_size <= 0:
                print("Error: Pixel size must be positive. Please try again.")
                continue
            return pixel_size
        except ValueError:
            print("Error: Please enter a valid number. Please try again.")
        except KeyboardInterrupt:
            print("\nConversion cancelled by user.")
            sys.exit(1)


def detect_pixel_size_interactive(reader, input_format):
    """
    Try automatic detection and fall back to interactive prompt if needed.

    Args:
        reader: MSI reader instance
        input_format: Format of the input file

    Returns:
        tuple: (pixel_size, detection_info_dict, essential_metadata)
    """
    logging.info("Attempting automatic pixel size detection...")
    essential_metadata = reader.get_essential_metadata()
    detected_pixel_size = essential_metadata.pixel_size

    if detected_pixel_size is not None:
        logging.info(
            f"OK Automatically detected pixel size: {detected_pixel_size[0]:.1f} x {detected_pixel_size[1]:.1f} um"
        )
        selected_pixel_size = prompt_for_pixel_size(detected_pixel_size)

        # If user selected the detected size, create automatic detection info
        from msiconvert.config import PIXEL_SIZE_TOLERANCE

        if (
            abs(selected_pixel_size - detected_pixel_size[0]) < PIXEL_SIZE_TOLERANCE
        ):  # Used detected size
            detection_info = {
                "method": "automatic_interactive",
                "detected_x_um": float(detected_pixel_size[0]),
                "detected_y_um": float(detected_pixel_size[1]),
                "source_format": input_format,
                "detection_successful": True,
                "note": "Pixel size automatically detected from source metadata and confirmed by user",
            }
        else:  # User entered different value
            detection_info = {
                "method": "manual_override",
                "detected_x_um": float(detected_pixel_size[0]),
                "detected_y_um": float(detected_pixel_size[1]),
                "user_specified_um": float(selected_pixel_size),
                "source_format": input_format,
                "detection_successful": True,
                "note": "Pixel size was auto-detected but user specified a different value",
            }

        return selected_pixel_size, detection_info, essential_metadata
    else:
        logging.warning("Could not automatically detect pixel size from metadata")
        selected_pixel_size = prompt_for_pixel_size(None)

        detection_info = {
            "method": "manual",
            "source_format": input_format,
            "detection_successful": False,
            "note": "Pixel size could not be auto-detected, manually specified by user",
        }

        return selected_pixel_size, detection_info, essential_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Convert MSI data to SpatialData format"
    )

    parser.add_argument("input", help="Path to input MSI file or directory")
    parser.add_argument("output", help="Path for output file")
    parser.add_argument(
        "--format",
        choices=["spatialdata"],
        default="spatialdata",
        help="Output format type: spatialdata (full SpatialData format)",
    )
    parser.add_argument(
        "--dataset-id", default="msi_dataset", help="Identifier for the dataset"
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Size of each pixel in micrometers (if not specified, automatic detection will be attempted)",
    )
    parser.add_argument(
        "--handle-3d",
        action="store_true",
        help="Process as 3D data (default: treat as 2D slices)",
    )
    parser.add_argument(
        "--optimize-chunks",
        action="store_true",
        help="Optimize Zarr chunks after conversion",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument("--log-file", default=None, help="Path to the log file")

    args = parser.parse_args()

    # Input validation - check early to give better error messages
    if args.pixel_size is not None and args.pixel_size <= 0:
        parser.error("Pixel size must be positive (got: {})".format(args.pixel_size))

    if not args.dataset_id.strip():
        parser.error("Dataset ID cannot be empty")

    # Validate input path exists and is accessible
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    # Check for common input format issues
    if input_path.is_file() and input_path.suffix.lower() == ".imzml":
        ibd_path = input_path.with_suffix(".ibd")
        if not ibd_path.exists():
            parser.error(
                f"ImzML file requires corresponding .ibd file, but not found: {ibd_path}"
            )
    elif input_path.is_dir() and input_path.suffix.lower() == ".d":
        if (
            not (input_path / "analysis.tsf").exists()
            and not (input_path / "analysis.tdf").exists()
        ):
            parser.error(
                f"Bruker .d directory requires analysis.tsf or analysis.tdf file: {input_path}"
            )

    # Check output path conflicts
    output_path = Path(args.output)
    if output_path.exists():
        parser.error(f"Output path already exists: {output_path}")

    # Configure logging
    setup_logging(log_level=getattr(logging, args.log_level), log_file=args.log_file)

    # Determine pixel size (auto-detect or use provided value) and prepare reader
    final_pixel_size = args.pixel_size
    reader = None
    detection_info_override = None
    essential_metadata_override = None

    if args.pixel_size is None:
        # For actual conversion, use interactive detection with reader reuse
        try:
            # Detect format and create reader for pixel size detection
            input_path = Path(args.input).resolve()
            input_format = detect_format(input_path)
            reader_class = get_reader_class(input_format)
            reader = reader_class(input_path)

            # Get pixel size interactively
            (
                final_pixel_size,
                detection_info,
                essential_metadata,
            ) = detect_pixel_size_interactive(reader, input_format)
            # Keep reader alive for conversion - don't close here!

            logging.info(f"Using pixel size: {final_pixel_size} um")
            detection_info_override = detection_info
            essential_metadata_override = essential_metadata
        except Exception as e:
            logging.error(f"Error during pixel size detection: {e}")
            logging.error("Conversion aborted.")
            # Clean up reader on error
            if reader and hasattr(reader, "close"):
                reader.close()
            return

    # Convert MSI data - pass reader for optimization and metadata if available
    try:
        success = convert_msi(
            args.input,
            args.output,
            format_type=args.format,
            dataset_id=args.dataset_id,
            pixel_size_um=final_pixel_size,
            handle_3d=args.handle_3d,
            pixel_size_detection_info_override=detection_info_override,
            essential_metadata=essential_metadata_override,
            reader=reader,  # Pass existing reader for optimization
        )
    finally:
        # Ensure reader is closed after conversion
        if reader and hasattr(reader, "close"):
            reader.close()

    if success and args.optimize_chunks:
        # Optimize chunks for better performance
        if args.format == "spatialdata":
            # For SpatialData format, optimize the table's X array
            optimize_zarr_chunks(args.output, f"tables/{args.dataset_id}/X")

    if success:
        logging.info(
            f"Conversion completed successfully. Output stored at {args.output}"
        )
    else:
        logging.error("Conversion failed.")


if __name__ == "__main__":
    main()
