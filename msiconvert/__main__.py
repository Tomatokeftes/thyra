# msiconvert/__main__.py
import argparse
import logging
from pathlib import Path
from msiconvert.convert import convert_msi
from msiconvert.utils.data_processors import optimize_zarr_chunks
from msiconvert.utils.logging_config import setup_logging
from msiconvert.core.registry import detect_format, get_reader_class


def dry_run_conversion(input_path, output_path, format_type="spatialdata", 
                      dataset_id="msi_dataset", pixel_size_um=1.0, handle_3d=False):
    """Simulate conversion process without writing files."""
    logging.info("=== DRY RUN MODE ===")
    
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    
    # Check if input exists
    if not input_path.exists():
        logging.error(f"Input path does not exist: {input_path}")
        return False
    
    # Check if output would overwrite
    if output_path.exists():
        logging.warning(f"Output path already exists and would be overwritten: {output_path}")
    
    try:
        # Detect format
        input_format = detect_format(input_path)
        logging.info(f"Detected input format: {input_format}")
        
        # Create reader to get metadata
        reader_class = get_reader_class(input_format)
        logging.info(f"Would use reader: {reader_class.__name__}")
        
        reader = reader_class(input_path)
        
        # Get basic metadata
        logging.info(f"Dataset dimensions: {reader.shape}")
        logging.info(f"Number of spectra: {reader.n_spectra}")
        logging.info(f"Mass range: {reader.mass_range}")
        
        # Estimate output size (very rough)
        estimated_size_mb = (reader.n_spectra * len(reader.mass_range) * 4) / (1024 * 1024)  # 4 bytes per float32
        logging.info(f"Estimated output size: ~{estimated_size_mb:.1f} MB")
        
        # Show conversion parameters
        logging.info(f"Output format: {format_type}")
        logging.info(f"Dataset ID: {dataset_id}")
        logging.info(f"Pixel size: {pixel_size_um} μm")
        logging.info(f"3D handling: {handle_3d}")
        logging.info(f"Output path: {output_path}")
        
        logging.info("=== DRY RUN COMPLETED SUCCESSFULLY ===")
        return True
        
    except Exception as e:
        logging.error(f"Dry run failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert MSI data to SpatialData format')
    
    parser.add_argument('input', help='Path to input MSI file or directory')
    parser.add_argument('output', help='Path for output file')
    parser.add_argument(
        '--format', 
        choices=['spatialdata'], 
        default='spatialdata',
        help='Output format type: spatialdata (full SpatialData format)'
    )
    parser.add_argument(
        '--dataset-id',
        default='msi_dataset',
        help='Identifier for the dataset'
    )
    parser.add_argument(
        '--pixel-size',
        type=float,
        default=1.0,
        help='Size of each pixel in micrometers'
    )
    parser.add_argument(
        '--handle-3d',
        action='store_true',
        help='Process as 3D data (default: treat as 2D slices)'
    )
    parser.add_argument(
        '--optimize-chunks',
        action='store_true',
        help='Optimize Zarr chunks after conversion'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    parser.add_argument(
        '--log-file',
        default=None,
        help='Path to the log file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate conversion process without writing output files'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    setup_logging(log_level=getattr(logging, args.log_level), log_file=args.log_file)
    
    # Handle dry-run mode
    if args.dry_run:
        success = dry_run_conversion(
            args.input,
            args.output,
            format_type=args.format,
            dataset_id=args.dataset_id,
            pixel_size_um=args.pixel_size,
            handle_3d=args.handle_3d
        )
    else:
        # Convert MSI data
        success = convert_msi(
            args.input,
            args.output,
            format_type=args.format,
            dataset_id=args.dataset_id,
            pixel_size_um=args.pixel_size,
            handle_3d=args.handle_3d
        )
    
    if success and args.optimize_chunks and not args.dry_run:
        # Optimize chunks for better performance
        if args.format == 'spatialdata':
            # For SpatialData format, optimize the table's X array
            optimize_zarr_chunks(args.output, f'tables/{args.dataset_id}/X')
    
    if success:
        if args.dry_run:
            logging.info("Dry run completed successfully. No files were written.")
        else:
            logging.info(f"Conversion completed successfully. Output stored at {args.output}")
    else:
        if args.dry_run:
            logging.error("Dry run failed.")
        else:
            logging.error("Conversion failed.")

if __name__ == '__main__':
    main()