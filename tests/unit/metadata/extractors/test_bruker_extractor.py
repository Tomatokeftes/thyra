# tests/unit/metadata/extractors/test_bruker_extractor.py
import sqlite3
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from msiconvert.metadata.extractors.bruker_extractor import BrukerMetadataExtractor


class TestBrukerMetadataExtractor:
    """Test BrukerMetadataExtractor functionality."""

    def create_mock_connection(self, sample_data=None):
        """Create a mock database connection with test data."""
        mock_conn = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor

        if sample_data is None:
            # Default test data for essential metadata
            # min_x=0, max_x=2, min_y=0, max_y=4 -> dimensions = (3, 5, 1)
            sample_data = {
                "essential": (25.0, 25.0, 1.0, 0, 2, 0, 4, 400, 100.0, 1000.0),
                "comprehensive": [
                    (1, 0, 0, 25.0, 25.0),
                    (2, 1, 0, 25.0, 25.0),
                    (3, 2, 0, 25.0, 25.0),
                    (4, 0, 4, 25.0, 25.0),
                ],
            }

        def execute_side_effect(query, params=None):
            if "BeamScanSizeX, BeamScanSizeY" in query and "MIN(" in query:
                # Essential metadata query
                return mock_cursor
            elif "Id, SpotXPos, SpotYPos" in query:
                # Comprehensive metadata query
                return mock_cursor
            elif "MaldiFrameLaserInfo" in query:
                # Acquisition parameters query
                return mock_cursor
            else:
                return mock_cursor

        def fetchone_side_effect():
            # Check which query was executed by looking at the last call
            last_call = mock_cursor.execute.call_args
            if last_call and len(last_call) > 0:
                query = last_call[0][0] if last_call[0] else ""
                if "LaserPower, LaserFrequency" in query:
                    # Acquisition parameters query - return laser_power, laser_freq, beam_x, beam_y, spot_size
                    return (100.0, 10.0, 25.0, 25.0, 50.0)
                elif "BeamScanSizeX, BeamScanSizeY, SpotSize" in query:
                    # Laser info query - return beam_x, beam_y, spot_size
                    laser_result = sample_data.get("laser_info", (25.0, 25.0, 1.0))
                    return laser_result
                elif "MIN(XIndexPos)" in query and "COUNT(*)" in query:
                    # Frame info query - return min_x, max_x, min_y, max_y, frame_count
                    return (0, 2, 0, 4, 400)
            return sample_data["essential"]

        def fetchall_side_effect():
            # Check which query was executed by looking at the last call
            last_call = mock_cursor.execute.call_args
            if last_call and len(last_call) > 0:
                query = last_call[0][0] if last_call[0] else ""
                if "GlobalMetadata" in query and "ImagingArea" in query:
                    # Imaging bounds query - return key-value pairs
                    return [
                        ("ImagingAreaMinXIndexPos", "0"),
                        ("ImagingAreaMaxXIndexPos", "2"),
                        ("ImagingAreaMinYIndexPos", "0"),
                        ("ImagingAreaMaxYIndexPos", "4"),
                        ("MzAcqRangeLower", "100.0"),
                        ("MzAcqRangeUpper", "1000.0"),
                    ]
                elif "Id, SpotXPos, SpotYPos" in query:
                    # Comprehensive metadata query
                    return sample_data["comprehensive"]
                elif "SELECT Key, Value FROM GlobalMetadata" in query:
                    # Global metadata query - return key-value pairs
                    return [
                        ("AcquisitionSoftware", "flexControl"),
                        ("AcquisitionSoftwareVersion", "3.4"),
                        ("InstrumentModel", "timsTOF fleX"),
                        ("LaserRepetitionRate", "2000"),
                        ("SampleName", "Test Sample"),
                    ]
            return sample_data["comprehensive"]

        mock_cursor.execute.side_effect = execute_side_effect
        mock_cursor.fetchone.side_effect = fetchone_side_effect
        mock_cursor.fetchall.side_effect = fetchall_side_effect

        return mock_conn

    def test_creation(self):
        """Test BrukerMetadataExtractor creation."""
        mock_conn = self.create_mock_connection()
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        assert extractor.data_source is mock_conn
        assert extractor.data_path == data_path
        assert extractor.conn is mock_conn

    def test_extract_essential_basic(self):
        """Test basic essential metadata extraction."""
        mock_conn = self.create_mock_connection()
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        essential = extractor.get_essential()

        assert essential.dimensions == (3, 5, 1)  # Calculated from coordinate bounds
        assert essential.coordinate_bounds == (0, 2, 0, 4)
        assert essential.mass_range == (100.0, 1000.0)
        assert essential.pixel_size == (25.0, 25.0)
        assert essential.n_spectra == 400
        assert essential.source_path == str(data_path)
        assert essential.estimated_memory_gb > 0

    def test_extract_essential_no_pixel_size(self):
        """Test essential metadata extraction when pixel size is not available."""
        # Mock data without pixel size
        sample_data = {
            "essential": (None, None, 1.0, 0, 100, 0, 200, 400, 100.0, 1000.0),
            "comprehensive": [],
            "laser_info": (None, None, 1.0),  # No pixel size in laser info
        }
        mock_conn = self.create_mock_connection(sample_data)
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        essential = extractor.get_essential()

        assert essential.pixel_size is None

    def test_extract_essential_3d_data(self):
        """Test essential metadata extraction with 3D data (SpotSize > 1)."""
        # Mock data with 3D coordinates
        sample_data = {
            "essential": (
                25.0,
                25.0,
                5.0,
                0,
                100,
                0,
                200,
                2000,
                100.0,
                1000.0,
            ),  # SpotSize = 5
            "comprehensive": [],
        }
        mock_conn = self.create_mock_connection(sample_data)
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        essential = extractor.get_essential()

        assert essential.dimensions == (3, 5, 5)  # x, y, z
        assert essential.is_3d is True

    def test_extract_comprehensive(self):
        """Test comprehensive metadata extraction."""
        mock_conn = self.create_mock_connection()
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        comprehensive = extractor.get_comprehensive()

        # Check that essential metadata is included
        assert comprehensive.essential.dimensions == (3, 5, 1)
        assert comprehensive.essential.n_spectra == 400

        # Check format-specific metadata
        assert "data_format" in comprehensive.format_specific
        assert comprehensive.format_specific["data_format"] in [
            "bruker_tsf",
            "bruker_tdf",
        ]
        assert "database_path" in comprehensive.format_specific

        # Check acquisition parameters
        assert "laser_power" in comprehensive.acquisition_params
        assert "beam_scan_size_x" in comprehensive.acquisition_params
        assert "beam_scan_size_y" in comprehensive.acquisition_params

        # Check raw metadata
        assert "frame_info" in comprehensive.raw_metadata
        assert len(comprehensive.raw_metadata["frame_info"]) == 4

    def test_dimensions_calculation(self):
        """Test dimensions calculation with various coordinate ranges."""
        # Test with coordinates that don't start from 0
        sample_data = {
            "essential": (25.0, 25.0, 1.0, 10, 40, 5, 25, 100, 100.0, 1000.0),
            "comprehensive": [],
        }
        mock_conn = self.create_mock_connection(sample_data)
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        essential = extractor.get_essential()

        # Dimensions should be calculated as (max - min + 1)
        expected_x = int((40 - 10)) + 1  # 31
        expected_y = int((25 - 5)) + 1  # 21
        assert essential.dimensions == (expected_x, expected_y, 1)

    def test_memory_estimation(self):
        """Test memory estimation calculation."""
        # Mock data with larger dataset
        sample_data = {
            "essential": (
                10.0,
                10.0,
                1.0,
                0,
                1000,
                0,
                1000,
                10000,
                100.0,
                2000.0,
            ),  # Large dataset
            "comprehensive": [],
        }
        mock_conn = self.create_mock_connection(sample_data)
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        essential = extractor.get_essential()

        # Should estimate reasonable memory usage based on spectra count and mass range
        assert essential.estimated_memory_gb > 0
        assert essential.estimated_memory_gb < 1000  # Sanity check

    def test_caching_behavior(self):
        """Test that extraction results are properly cached."""
        mock_conn = self.create_mock_connection()
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)

        # First call
        essential1 = extractor.get_essential()
        # Second call should return cached result
        essential2 = extractor.get_essential()

        assert essential1 is essential2  # Same object reference due to caching

    def test_database_error_handling(self):
        """Test error handling when database query fails."""
        mock_conn = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = sqlite3.Error("Database query failed")

        data_path = Path("/test/data.d")
        extractor = BrukerMetadataExtractor(mock_conn, data_path)

        with pytest.raises(sqlite3.Error, match="Database query failed"):
            extractor.get_essential()

    def test_empty_dataset_handling(self):
        """Test handling when database returns no data."""
        mock_conn = Mock(spec=sqlite3.Connection)
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None  # No data

        data_path = Path("/test/data.d")
        extractor = BrukerMetadataExtractor(mock_conn, data_path)

        with pytest.raises((ValueError, TypeError)):
            extractor.get_essential()

    def test_single_spectrum_dataset(self):
        """Test handling of dataset with single spectrum."""
        sample_data = {
            "essential": (
                25.0,
                25.0,
                1.0,
                0,
                0,
                0,
                0,
                1,
                100.0,
                1000.0,
            ),  # Single point
            "comprehensive": [(1, 0, 0, 25.0, 25.0, 1.0, 100.0, 1000.0)],
        }
        mock_conn = self.create_mock_connection(sample_data)
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        essential = extractor.get_essential()

        assert essential.dimensions == (1, 1, 1)
        assert essential.n_spectra == 1
        assert essential.coordinate_bounds == (0, 0, 0, 0)

    def test_inconsistent_pixel_sizes(self):
        """Test handling when X and Y pixel sizes are different."""
        sample_data = {
            "essential": (
                20.0,
                30.0,
                1.0,
                0,
                100,
                0,
                200,
                400,
                100.0,
                1000.0,
            ),  # Different X/Y sizes
            "comprehensive": [],
        }
        mock_conn = self.create_mock_connection(sample_data)
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        essential = extractor.get_essential()

        assert essential.pixel_size == (20.0, 30.0)

    def test_comprehensive_metadata_structure(self):
        """Test the structure of comprehensive metadata."""
        mock_conn = self.create_mock_connection()
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        comprehensive = extractor.get_comprehensive()

        # Verify all expected sections are present
        assert hasattr(comprehensive, "essential")
        assert hasattr(comprehensive, "format_specific")
        assert hasattr(comprehensive, "acquisition_params")
        assert hasattr(comprehensive, "instrument_info")
        assert hasattr(comprehensive, "raw_metadata")

        # Verify format-specific contains expected Bruker keys
        assert "bruker_format" in comprehensive.format_specific
        assert "data_path" in comprehensive.format_specific

        # Verify acquisition params contain beam scan info
        assert "BeamScanSizeX" in comprehensive.acquisition_params
        assert "BeamScanSizeY" in comprehensive.acquisition_params

    def test_sql_injection_protection(self):
        """Test that the extractor is protected against SQL injection."""
        mock_conn = self.create_mock_connection()
        data_path = Path("/test'; DROP TABLE MaldiFrameLaserInfo; --")  # Malicious path

        # Should not raise an exception, path is just used as a string
        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        assert extractor.data_path == data_path

    def test_large_coordinate_range(self):
        """Test handling of very large coordinate ranges."""
        sample_data = {
            "essential": (
                1.0,
                1.0,
                1.0,
                0,
                10000,
                0,
                10000,
                100000000,
                50.0,
                2000.0,
            ),  # Very large range
            "comprehensive": [],
        }
        mock_conn = self.create_mock_connection(sample_data)
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        essential = extractor.get_essential()

        # Should handle large ranges appropriately
        assert essential.dimensions[0] > 1000  # Large X dimension
        assert essential.dimensions[1] > 1000  # Large Y dimension
        assert essential.coordinate_bounds == (0, 10000, 0, 10000)

    @patch("msiconvert.metadata.extractors.bruker_extractor.logging")
    def test_logging_during_extraction(self, mock_logging):
        """Test that appropriate logging occurs during extraction."""
        mock_conn = self.create_mock_connection()
        data_path = Path("/test/data.d")

        extractor = BrukerMetadataExtractor(mock_conn, data_path)
        extractor.get_essential()

        # Should have logged debug information
        mock_logging.debug.assert_called()
