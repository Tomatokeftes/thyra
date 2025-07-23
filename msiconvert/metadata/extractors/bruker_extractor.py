# msiconvert/metadata/extractors/bruker_extractor.py
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ...core.base_extractor import MetadataExtractor
from ..types import ComprehensiveMetadata, EssentialMetadata

logger = logging.getLogger(__name__)


class BrukerMetadataExtractor(MetadataExtractor):
    """Bruker-specific metadata extractor with optimized single-query extraction."""

    def __init__(self, conn: sqlite3.Connection, data_path: Path):
        """
        Initialize Bruker metadata extractor.

        Args:
            conn: Active SQLite database connection
            data_path: Path to the Bruker .d directory
        """
        super().__init__(conn)
        self.conn = conn
        self.data_path = data_path

    def _extract_essential_impl(self) -> EssentialMetadata:
        """Extract essential metadata with single optimized query."""
        cursor = self.conn.cursor()

        # Single comprehensive query for all essential data
        essential_query = """
        SELECT
            BeamScanSizeX, BeamScanSizeY, SpotSize,
            MIN(SpotXPos), MAX(SpotXPos),
            MIN(SpotYPos), MAX(SpotYPos),
            COUNT(*) as frame_count,
            MIN(MzAcqRangeLower), MAX(MzAcqRangeUpper)
        FROM MaldiFrameLaserInfo
        """

        try:
            cursor.execute(essential_query)
            result = cursor.fetchone()

            if not result:
                raise ValueError("No data found in MaldiFrameLaserInfo table")

            (
                beam_x,
                beam_y,
                spot_size,
                min_x,
                max_x,
                min_y,
                max_y,
                frame_count,
                min_mass,
                max_mass,
            ) = result

            # Calculate dimensions and bounds
            dimensions = self._calculate_dimensions_from_coords(
                min_x, max_x, min_y, max_y
            )
            coordinate_bounds = (
                float(min_x) if min_x is not None else 0.0,
                float(max_x) if max_x is not None else 0.0,
                float(min_y) if min_y is not None else 0.0,
                float(max_y) if max_y is not None else 0.0,
            )

            # Extract pixel size
            pixel_size = None
            if beam_x is not None and beam_y is not None:
                pixel_size = (float(beam_x), float(beam_y))

            # Mass range
            mass_range = (
                float(min_mass) if min_mass is not None else 0.0,
                float(max_mass) if max_mass is not None else 1000.0,
            )

            # Frame count
            n_spectra = int(frame_count) if frame_count is not None else 0

            # Memory estimation
            estimated_memory = self._estimate_memory_from_frames(n_spectra)

            return EssentialMetadata(
                dimensions=dimensions,
                coordinate_bounds=coordinate_bounds,
                mass_range=mass_range,
                pixel_size=pixel_size,
                n_spectra=n_spectra,
                estimated_memory_gb=estimated_memory,
                source_path=str(self.data_path),
            )

        except sqlite3.OperationalError as e:
            logger.error(f"SQL error extracting essential metadata: {e}")
            raise ValueError(
                f"Failed to extract essential metadata from Bruker database: {e}"
            )
        except Exception as e:
            logger.error(f"Unexpected error extracting essential metadata: {e}")
            raise

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        """Extract comprehensive metadata with additional database queries."""
        essential = self.get_essential()

        return ComprehensiveMetadata(
            essential=essential,
            format_specific=self._extract_bruker_specific(),
            acquisition_params=self._extract_acquisition_params(),
            instrument_info=self._extract_instrument_info(),
            raw_metadata=self._extract_global_metadata(),
        )

    def _calculate_dimensions_from_coords(
        self,
        min_x: Optional[float],
        max_x: Optional[float],
        min_y: Optional[float],
        max_y: Optional[float],
    ) -> Tuple[int, int, int]:
        """Calculate dataset dimensions from coordinate bounds."""
        if any(coord is None for coord in [min_x, max_x, min_y, max_y]):
            return (0, 0, 1)  # Default for problematic data

        # Bruker coordinates are typically in position units
        # Calculate grid dimensions assuming integer grid positions
        x_range = int(max_x - min_x) + 1 if max_x > min_x else 1
        y_range = int(max_y - min_y) + 1 if max_y > min_y else 1

        return (max(1, x_range), max(1, y_range), 1)  # Assume 2D data (z=1)

    def _estimate_memory_from_frames(self, frame_count: int) -> float:
        """Estimate memory usage from frame count."""
        if frame_count <= 0:
            return 0.0

        # Rough estimate for Bruker data:
        # - Average ~2000 peaks per frame
        # - 8 bytes per float64 value
        # - mz + intensity arrays
        avg_peaks_per_frame = 2000
        bytes_per_value = 8
        estimated_bytes = frame_count * avg_peaks_per_frame * 2 * bytes_per_value

        return estimated_bytes / (1024**3)  # Convert to GB

    def _extract_bruker_specific(self) -> Dict[str, Any]:
        """Extract Bruker format-specific metadata."""
        format_specific = {
            "data_format": "bruker_tdf" if self._is_tdf_format() else "bruker_tsf",
            "database_path": str(self.data_path / "analysis.tsf"),
            "is_maldi": self._is_maldi_dataset(),
        }

        # Add file type detection
        if (self.data_path / "analysis.tdf").exists():
            format_specific["binary_file"] = str(self.data_path / "analysis.tdf")
        elif (self.data_path / "analysis.tsf").exists():
            format_specific["binary_file"] = str(self.data_path / "analysis.tsf")

        return format_specific

    def _extract_acquisition_params(self) -> Dict[str, Any]:
        """Extract acquisition parameters from database."""
        params = {}
        cursor = self.conn.cursor()

        # Extract laser parameters if available
        try:
            cursor.execute(
                """
                SELECT DISTINCT LaserPower, LaserFrequency, BeamScanSizeX, BeamScanSizeY, SpotSize
                FROM MaldiFrameLaserInfo
                LIMIT 1
            """
            )
            result = cursor.fetchone()

            if result:
                laser_power, laser_freq, beam_x, beam_y, spot_size = result
                if laser_power is not None:
                    params["laser_power"] = laser_power
                if laser_freq is not None:
                    params["laser_frequency"] = laser_freq
                if beam_x is not None:
                    params["beam_scan_size_x"] = beam_x
                if beam_y is not None:
                    params["beam_scan_size_y"] = beam_y
                if spot_size is not None:
                    params["laser_spot_size"] = spot_size

        except sqlite3.OperationalError:
            logger.debug("Could not extract laser parameters")

        # Extract timing parameters
        try:
            cursor.execute(
                "SELECT Value FROM GlobalMetadata WHERE Key = 'AcquisitionDateTime'"
            )
            result = cursor.fetchone()
            if result:
                params["acquisition_datetime"] = result[0]
        except sqlite3.OperationalError:
            pass

        return params

    def _extract_instrument_info(self) -> Dict[str, Any]:
        """Extract instrument information from global metadata."""
        instrument = {}
        cursor = self.conn.cursor()

        # Common instrument metadata keys
        instrument_keys = [
            ("InstrumentName", "instrument_name"),
            ("InstrumentSerialNumber", "instrument_serial_number"),
            ("InstrumentModel", "instrument_model"),
            ("SoftwareVersion", "software_version"),
            ("MzCalibrationMode", "mz_calibration_mode"),
        ]

        try:
            for db_key, result_key in instrument_keys:
                cursor.execute(
                    "SELECT Value FROM GlobalMetadata WHERE Key = ?", (db_key,)
                )
                result = cursor.fetchone()
                if result:
                    instrument[result_key] = result[0]
        except sqlite3.OperationalError:
            logger.debug("Could not extract instrument metadata")

        return instrument

    def _extract_global_metadata(self) -> Dict[str, Any]:
        """Extract all global metadata from database."""
        raw_metadata = {}
        cursor = self.conn.cursor()

        try:
            cursor.execute("SELECT Key, Value FROM GlobalMetadata")
            for key, value in cursor.fetchall():
                raw_metadata[key] = value

        except sqlite3.OperationalError:
            logger.debug("GlobalMetadata table not found or accessible")

        return raw_metadata

    def _is_tdf_format(self) -> bool:
        """Check if this is TDF format (vs TSF)."""
        return (self.data_path / "analysis.tdf").exists()

    def _is_maldi_dataset(self) -> bool:
        """Check if this is a MALDI dataset by checking for laser info."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM MaldiFrameLaserInfo")
            result = cursor.fetchone()
            return result and result[0] > 0
        except sqlite3.OperationalError:
            return False
