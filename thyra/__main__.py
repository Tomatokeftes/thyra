# thyra/__main__.py
import argparse
import logging
from pathlib import Path

from thyra.convert import convert_msi
from thyra.utils.data_processors import optimize_zarr_chunks
from thyra.utils.logging_config import setup_logging


def _create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
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
        "--dataset-id",
        default="msi_dataset",
        help="Identifier for the dataset",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in micrometers. If not specified, automatic "
        "detection from metadata will be attempted. Required if fails.",
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

    # Resampling arguments
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Enable mass axis resampling/harmonization",
    )
    parser.add_argument(
        "--resample-method",
        choices=["auto", "nearest_neighbor", "tic_preserving"],
        default="auto",
        help="Resampling method: auto (detect from metadata), "
        "nearest_neighbor (centroid data), tic_preserving (profile data)",
    )
    parser.add_argument(
        "--resample-bins",
        type=int,
        default=5000,
        help="Number of bins for resampled mass axis (default: 5000). "
        "Mutually exclusive with --resample-width-at-mz.",
    )
    parser.add_argument(
        "--resample-min-mz",
        type=float,
        default=None,
        help="Minimum m/z for resampled axis (default: auto-detect from data)",
    )
    parser.add_argument(
        "--resample-max-mz",
        type=float,
        default=None,
        help="Maximum m/z for resampled axis (default: auto-detect from data)",
    )
    parser.add_argument(
        "--resample-width-at-mz",
        type=float,
        default=None,
        help="Mass width (in Da) at reference m/z for physics-based binning. "
        "Default: 0.005 Da at m/z 1000. Mutually exclusive with --resample-bins.",
    )
    parser.add_argument(
        "--resample-reference-mz",
        type=float,
        default=1000.0,
        help="Reference m/z for width specification (default: 1000.0). "
        "Used with --resample-width-at-mz.",
    )

    return parser


def _validate_basic_arguments(parser: argparse.ArgumentParser, args) -> None:
    """Validate basic arguments."""
    if args.pixel_size is not None and args.pixel_size <= 0:
        parser.error("Pixel size must be positive (got: {})".format(args.pixel_size))

    if not args.dataset_id.strip():
        parser.error("Dataset ID cannot be empty")


def _validate_resampling_bins(parser: argparse.ArgumentParser, args) -> None:
    """Validate resampling bin arguments."""
    if args.resample_bins <= 0:
        parser.error(
            "Number of resampling bins must be positive (got: {})".format(
                args.resample_bins
            )
        )


def _validate_resampling_ranges(parser: argparse.ArgumentParser, args) -> None:
    """Validate resampling m/z range arguments."""
    if args.resample_min_mz is not None and args.resample_min_mz <= 0:
        parser.error(
            "Minimum m/z must be positive (got: {})".format(args.resample_min_mz)
        )

    if args.resample_max_mz is not None and args.resample_max_mz <= 0:
        parser.error(
            "Maximum m/z must be positive (got: {})".format(args.resample_max_mz)
        )

    if (
        args.resample_min_mz is not None
        and args.resample_max_mz is not None
        and args.resample_min_mz >= args.resample_max_mz
    ):
        parser.error("Minimum m/z must be less than maximum m/z")


def _validate_resampling_mutual_exclusivity(
    parser: argparse.ArgumentParser, args
) -> None:
    """Validate mutual exclusivity of resampling parameters."""
    if args.resample_bins != 5000 and args.resample_width_at_mz is not None:
        parser.error(
            "--resample-bins and --resample-width-at-mz are mutually exclusive. "
            "Use either --resample-bins for fixed bin count or --resample-width-at-mz "
            "for physics-based binning with target resolution."
        )


def _validate_resampling_width_params(parser: argparse.ArgumentParser, args) -> None:
    """Validate width-based resampling parameters."""
    if args.resample_width_at_mz is not None and args.resample_width_at_mz <= 0:
        parser.error(
            "Mass width must be positive (got: {})".format(args.resample_width_at_mz)
        )

    if args.resample_reference_mz <= 0:
        parser.error(
            "Reference m/z must be positive (got: {})".format(
                args.resample_reference_mz
            )
        )


def _validate_arguments(parser: argparse.ArgumentParser, args) -> None:
    """Validate command line arguments."""
    _validate_basic_arguments(parser, args)
    _validate_resampling_bins(parser, args)
    _validate_resampling_ranges(parser, args)
    _validate_resampling_mutual_exclusivity(parser, args)
    _validate_resampling_width_params(parser, args)


def _check_imzml_requirements(
    parser: argparse.ArgumentParser, input_path: Path
) -> None:
    """Check ImzML format requirements."""
    ibd_path = input_path.with_suffix(".ibd")
    if not ibd_path.exists():
        parser.error(
            f"ImzML file requires corresponding .ibd file, "
            f"but not found: {ibd_path}"
        )


def _check_bruker_requirements(
    parser: argparse.ArgumentParser, input_path: Path
) -> None:
    """Check Bruker format requirements."""
    if (
        not (input_path / "analysis.tsf").exists()
        and not (input_path / "analysis.tdf").exists()
    ):
        parser.error(
            f"Bruker .d directory requires analysis.tsf or analysis.tdf file: "
            f"{input_path}"
        )


def _validate_input_path(parser: argparse.ArgumentParser, input_path: Path) -> None:
    """Validate input path and format requirements."""
    if not input_path.exists():
        parser.error(f"Input path does not exist: {input_path}")

    if input_path.is_file() and input_path.suffix.lower() == ".imzml":
        _check_imzml_requirements(parser, input_path)
    elif input_path.is_dir() and input_path.suffix.lower() == ".d":
        _check_bruker_requirements(parser, input_path)


def _validate_output_path(parser: argparse.ArgumentParser, output_path: Path) -> None:
    """Validate output path."""
    if output_path.exists():
        parser.error(f"Output path already exists: {output_path}")


def _perform_conversion(args) -> bool:
    """Perform the MSI data conversion."""
    # Build resampling config if enabled
    resampling_config = None
    if args.resample:
        resampling_config = {
            "method": args.resample_method,
            "target_bins": args.resample_bins,
            "min_mz": args.resample_min_mz,
            "max_mz": args.resample_max_mz,
            "width_at_mz": args.resample_width_at_mz,
            "reference_mz": args.resample_reference_mz,
        }

    return convert_msi(
        args.input,
        args.output,
        format_type=args.format,
        dataset_id=args.dataset_id,
        pixel_size_um=args.pixel_size,
        handle_3d=args.handle_3d,
        resampling_config=resampling_config,
    )


def _optimize_output(args) -> None:
    """Optimize output chunks if requested."""
    if args.format == "spatialdata":
        optimize_zarr_chunks(args.output, f"tables/{args.dataset_id}/X")


def main() -> None:
    """Main entry point for the CLI."""
    parser = _create_argument_parser()
    args = parser.parse_args()

    # Validate arguments and paths
    _validate_arguments(parser, args)
    input_path = Path(args.input)
    output_path = Path(args.output)
    _validate_input_path(parser, input_path)
    _validate_output_path(parser, output_path)

    # Configure logging
    setup_logging(log_level=getattr(logging, args.log_level), log_file=args.log_file)

    # Perform conversion
    success = _perform_conversion(args)

    # Optimize chunks if requested and conversion succeeded
    if success and args.optimize_chunks:
        _optimize_output(args)

    # Log final result
    if success:
        logging.info(
            f"Conversion completed successfully. Output stored at " f"{args.output}"
        )
    else:
        logging.error("Conversion failed.")


if __name__ == "__main__":
    main()
