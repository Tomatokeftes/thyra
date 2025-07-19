# msiconvert/readers/__init__.py (updated)
import logging
from pathlib import Path

from ..core.registry import register_format_detector


@register_format_detector("imzml")
def detect_imzml(input_path: Path) -> bool:
    """Detect imzML format."""
    logging.debug(f"Testing imzML format for: {input_path}")

    # Check if file exists, is a file, and has .imzml extension
    if not input_path.is_file() or not input_path.exists():
        return False

    if input_path.suffix.lower() != ".imzml":
        return False

    # Additional check: .ibd file should exist
    ibd_path = input_path.with_suffix(".ibd")
    if not ibd_path.exists():
        logging.warning(
            f"imzML file found, but corresponding .ibd file missing: {ibd_path}"
        )
        return False

    return True


@register_format_detector("bruker")
def detect_bruker(input_path: Path) -> bool:
    """Detect Bruker format."""
    logging.debug(f"Testing Bruker format for: {input_path}")

    # Check if directory exists and has .d extension
    if not input_path.is_dir() or not input_path.exists():
        return False

    if input_path.suffix.lower() != ".d":
        return False

    # Check for analysis.tsf or analysis.tdf
    tsf_path = input_path / "analysis.tsf"
    tdf_path = input_path / "analysis.tdf"

    if tsf_path.exists():
        logging.debug(f"Found Bruker TSF file: {tsf_path}")
        return True

    if tdf_path.exists():
        logging.debug(f"Found Bruker TDF file: {tdf_path}")
        return True

    return False
