# thyra/__main__.py

# Configure dependencies to suppress warnings BEFORE any imports
import logging  # noqa: E402
import os  # noqa: E402
import sqlite3  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import click  # noqa: E402

from thyra.convert import convert_msi  # noqa: E402
from thyra.utils.data_processors import optimize_zarr_chunks  # noqa: E402
from thyra.utils.logging_config import setup_logging  # noqa: E402

# Configure Dask to use new query planning (silences legacy DataFrame warning)
os.environ["DASK_DATAFRAME__QUERY_PLANNING"] = "True"

# Suppress dependency warnings at the earliest possible moment
warnings.filterwarnings("ignore", category=FutureWarning, module="dask")
warnings.filterwarnings("ignore", category=UserWarning, module="xarray_schema")
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="The legacy Dask DataFrame implementation is deprecated",
    category=FutureWarning,
)


def _get_calibration_states(bruker_path: Path) -> list[dict]:
    """Read calibration states from calibration.sqlite.

    Args:
        bruker_path: Path to Bruker .d directory

    Returns:
        List of calibration state dictionaries with id, datetime, and version info
    """
    cal_file = bruker_path / "calibration.sqlite"
    if not cal_file.exists():
        return []

    try:
        conn = sqlite3.connect(str(cal_file))
        cursor = conn.cursor()

        # Query calibration states
        cursor.execute(
            """
            SELECT cs.Id, ci.DateTime
            FROM CalibrationState cs
            LEFT JOIN CalibrationInfo ci ON cs.Id = ci.StateId
            ORDER BY cs.Id
            """
        )

        states = []
        for row in cursor.fetchall():
            state_id, datetime_str = row
            states.append(
                {
                    "id": state_id,
                    "datetime": datetime_str or "Unknown",
                    "version": state_id,
                }
            )

        conn.close()
        return states

    except Exception:
        return []


@click.command()
@click.argument("input", type=click.Path(exists=True, path_type=Path))
@click.argument("output", type=click.Path(path_type=Path))
# Basic conversion options
@click.option(
    "--format",
    type=click.Choice(["spatialdata"]),
    default="spatialdata",
    help="Output format type: spatialdata (full SpatialData format)",
)
@click.option(
    "--dataset-id",
    default="msi_dataset",
    help="Identifier for the dataset",
)
@click.option(
    "--pixel-size",
    type=float,
    default=None,
    help="Pixel size in micrometers. If not specified, automatic detection "
    "from metadata will be attempted.",
)
@click.option(
    "--handle-3d",
    is_flag=True,
    help="Process as 3D data (default: treat as 2D slices)",
)
@click.option(
    "--optimize-chunks",
    is_flag=True,
    help="Optimize Zarr chunks after conversion",
)
# Logging options
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Set the logging level",
)
@click.option(
    "--log-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to the log file",
)
# Bruker calibration options
@click.option(
    "--use-recalibrated/--no-recalibrated",
    default=True,
    help="Use recalibrated state (default: True)",
)
@click.option(
    "--interactive-calibration",
    is_flag=True,
    help="Interactively select Bruker calibration state",
)
# Resampling options
@click.option(
    "--resample",
    is_flag=True,
    help="Enable mass axis resampling/harmonization",
)
@click.option(
    "--resample-method",
    type=click.Choice(["auto", "nearest_neighbor", "tic_preserving"]),
    default="auto",
    help="Resampling method: auto (detect from metadata), "
    "nearest_neighbor (centroid data), tic_preserving (profile data)",
)
@click.option(
    "--resample-bins",
    type=int,
    default=None,
    help="Number of bins for resampled mass axis. "
    "If not specified, uses physics-based width calculation (5 mDa @ m/z 1000). "
    "Mutually exclusive with --resample-width-at-mz.",
)
@click.option(
    "--resample-min-mz",
    type=float,
    default=None,
    help="Minimum m/z for resampled axis (default: auto-detect from data)",
)
@click.option(
    "--resample-max-mz",
    type=float,
    default=None,
    help="Maximum m/z for resampled axis (default: auto-detect from data)",
)
@click.option(
    "--resample-width-at-mz",
    type=float,
    default=None,
    help="Mass width (in Da) at reference m/z for physics-based binning. "
    "Default: 0.005 Da at m/z 1000. Mutually exclusive with --resample-bins.",
)
@click.option(
    "--resample-reference-mz",
    type=float,
    default=1000.0,
    help="Reference m/z for width specification (default: 1000.0). "
    "Used with --resample-width-at-mz.",
)
@click.option(
    "--mass-axis-type",
    type=click.Choice(
        ["auto", "constant", "linear_tof", "reflector_tof", "orbitrap", "fticr"]
    ),
    default="auto",
    help="Mass axis spacing type: auto (detect from metadata), "
    "constant (uniform spacing), linear_tof (sqrt spacing), "
    "reflector_tof (logarithmic spacing), orbitrap (1/sqrt spacing), "
    "fticr (quadratic spacing). Only used with --resample.",
)
def main(
    input: Path,
    output: Path,
    format: str,
    dataset_id: str,
    pixel_size: Optional[float],
    handle_3d: bool,
    optimize_chunks: bool,
    log_level: str,
    log_file: Optional[Path],
    use_recalibrated: bool,
    interactive_calibration: bool,
    resample: bool,
    resample_method: str,
    resample_bins: Optional[int],
    resample_min_mz: Optional[float],
    resample_max_mz: Optional[float],
    resample_width_at_mz: Optional[float],
    resample_reference_mz: float,
    mass_axis_type: str,
):
    """Convert MSI data to SpatialData format.

    INPUT: Path to input MSI file or directory
    OUTPUT: Path for output file
    """
    # Validate basic arguments
    if pixel_size is not None and pixel_size <= 0:
        raise click.BadParameter("Pixel size must be positive", param_hint="pixel_size")

    if not dataset_id.strip():
        raise click.BadParameter("Dataset ID cannot be empty", param_hint="dataset_id")

    # Validate resampling arguments
    if resample_bins is not None and resample_bins <= 0:
        raise click.BadParameter(
            "Number of resampling bins must be positive", param_hint="resample_bins"
        )

    if resample_min_mz is not None and resample_min_mz <= 0:
        raise click.BadParameter(
            "Minimum m/z must be positive", param_hint="resample_min_mz"
        )

    if resample_max_mz is not None and resample_max_mz <= 0:
        raise click.BadParameter(
            "Maximum m/z must be positive", param_hint="resample_max_mz"
        )

    if (
        resample_min_mz is not None
        and resample_max_mz is not None
        and resample_min_mz >= resample_max_mz
    ):
        raise click.BadParameter("Minimum m/z must be less than maximum m/z")

    if resample_bins is not None and resample_width_at_mz is not None:
        raise click.BadParameter(
            "--resample-bins and --resample-width-at-mz are mutually exclusive"
        )

    if resample_width_at_mz is not None and resample_width_at_mz <= 0:
        raise click.BadParameter(
            "Mass width must be positive", param_hint="resample_width_at_mz"
        )

    if resample_reference_mz <= 0:
        raise click.BadParameter(
            "Reference m/z must be positive", param_hint="resample_reference_mz"
        )

    # Validate input path
    if not input.exists():
        raise click.BadParameter(f"Input path does not exist: {input}")

    if input.is_file() and input.suffix.lower() == ".imzml":
        ibd_path = input.with_suffix(".ibd")
        if not ibd_path.exists():
            raise click.BadParameter(
                f"ImzML file requires corresponding .ibd file, but not found: {ibd_path}"
            )
    elif input.is_dir() and input.suffix.lower() == ".d":
        if not (input / "analysis.tsf").exists() and not (
            input / "analysis.tdf"
        ).exists():
            raise click.BadParameter(
                f"Bruker .d directory requires analysis.tsf or analysis.tdf file: {input}"
            )

    # Validate output path
    if output.exists():
        raise click.BadParameter(f"Output path already exists: {output}")

    # Configure logging
    setup_logging(log_level=getattr(logging, log_level), log_file=log_file)

    # Handle interactive calibration selection for Bruker datasets
    # Note: Currently informational only - specific state selection not yet implemented
    if interactive_calibration and input.is_dir() and input.suffix.lower() == ".d":
        states = _get_calibration_states(input)
        if states:
            click.echo("\nCalibration information:")
            for state in states:
                is_active = state["id"] == max(s["id"] for s in states)
                active_marker = " (active/will be used)" if is_active else ""
                recal_info = (
                    f" - recalibrated {state['version'] - 1} times"
                    if state["version"] > 1
                    else ""
                )
                click.echo(
                    f"  State {state['id']}: {state['datetime']}{recal_info}{active_marker}"
                )
            if use_recalibrated:
                click.echo(
                    f"\nUsing active calibration state (State {max(s['id'] for s in states)})"
                )
            else:
                click.echo("\nUsing original calibration (--no-recalibrated flag set)")

    # Build resampling config if enabled
    resampling_config = None
    if resample:
        resampling_config = {
            "method": resample_method,
            "axis_type": mass_axis_type,
            "target_bins": resample_bins,
            "min_mz": resample_min_mz,
            "max_mz": resample_max_mz,
            "width_at_mz": resample_width_at_mz,
            "reference_mz": resample_reference_mz,
        }

    # Perform conversion
    success = convert_msi(
        str(input),
        str(output),
        format_type=format,
        dataset_id=dataset_id,
        pixel_size_um=pixel_size,
        handle_3d=handle_3d,
        resampling_config=resampling_config,
    )

    # Optimize chunks if requested and conversion succeeded
    if success and optimize_chunks:
        if format == "spatialdata":
            optimize_zarr_chunks(str(output), f"tables/{dataset_id}/X")

    # Log final result
    if success:
        logging.info(f"Conversion completed successfully. Output stored at {output}")
    else:
        logging.error("Conversion failed.")


if __name__ == "__main__":
    main()
