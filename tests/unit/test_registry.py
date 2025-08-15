"""
Tests for the simplified format registry system.
"""

import pytest

from msiconvert.core.base_converter import BaseMSIConverter
from msiconvert.core.base_reader import BaseMSIReader
from msiconvert.core.registry import (
    _registry,
    detect_format,
    get_converter_class,
    get_reader_class,
    register_converter,
    register_reader,
)


class TestRegistry:
    """Test the registry functionality."""

    def setup_method(self):
        """Set up each test by clearing the registries."""
        # Store original registry values to restore later
        with _registry._lock:
            self.original_readers = _registry._readers.copy()
            self.original_converters = _registry._converters.copy()

            # Clear registries for testing
            _registry._readers.clear()
            _registry._converters.clear()

    def teardown_method(self):
        """Restore original registry values after each test."""
        with _registry._lock:
            _registry._readers.clear()
            _registry._readers.update(self.original_readers)

            _registry._converters.clear()
            _registry._converters.update(self.original_converters)

    def test_register_reader(self):
        """Test registering a reader class."""

        # Create a test reader class
        class TestReader(BaseMSIReader):
            def get_metadata(self):
                pass

            def get_dimensions(self):
                pass

            def get_common_mass_axis(self):
                pass

            def iter_spectra(self, batch_size=None):
                pass

            def close(self):
                pass

        # Register it
        register_reader("test_format")(TestReader)

        # Check if it was properly registered
        with _registry._lock:
            assert "test_format" in _registry._readers
            assert _registry._readers["test_format"] == TestReader

        # Test getting the reader class
        retrieved_class = get_reader_class("test_format")
        assert retrieved_class == TestReader

    def test_register_converter(self):
        """Test registering a converter class."""

        # Create a test converter class
        class TestConverter(BaseMSIConverter):
            def _create_data_structures(self):
                pass

            def _save_output(self, data_structures):
                pass

        # Register it
        register_converter("test_format")(TestConverter)

        # Check if it was properly registered
        with _registry._lock:
            assert "test_format" in _registry._converters
            assert _registry._converters["test_format"] == TestConverter

        # Test getting the converter class
        retrieved_class = get_converter_class("test_format")
        assert retrieved_class == TestConverter

    def test_detect_format_imzml(self, tmp_path):
        """Test ImzML format detection via extension."""
        # Create test files
        imzml_file = tmp_path / "test.imzml"
        ibd_file = tmp_path / "test.ibd"
        imzml_file.touch()
        ibd_file.touch()

        assert detect_format(imzml_file) == "imzml"

    def test_detect_format_bruker(self, tmp_path):
        """Test Bruker format detection via extension."""
        # Create test directory
        bruker_dir = tmp_path / "test.d"
        bruker_dir.mkdir()
        (bruker_dir / "analysis.tsf").touch()

        assert detect_format(bruker_dir) == "bruker"

    def test_detect_format_bruker_tdf(self, tmp_path):
        """Test Bruker format detection with .tdf file."""
        # Create test directory
        bruker_dir = tmp_path / "test.d"
        bruker_dir.mkdir()
        (bruker_dir / "analysis.tdf").touch()

        assert detect_format(bruker_dir) == "bruker"

    def test_unsupported_extension(self, tmp_path):
        """Test error for unsupported extension."""
        unknown_file = tmp_path / "test.xyz"
        unknown_file.touch()

        with pytest.raises(ValueError, match="Unsupported file extension"):
            detect_format(unknown_file)

    def test_missing_ibd_file(self, tmp_path):
        """Test error for ImzML without .ibd file."""
        imzml_file = tmp_path / "test.imzml"
        imzml_file.touch()

        with pytest.raises(ValueError, match="requires corresponding .ibd file"):
            detect_format(imzml_file)

    def test_bruker_missing_analysis_files(self, tmp_path):
        """Test error for Bruker .d directory without analysis files."""
        bruker_dir = tmp_path / "test.d"
        bruker_dir.mkdir()

        with pytest.raises(ValueError, match="missing analysis files"):
            detect_format(bruker_dir)

    def test_bruker_not_directory(self, tmp_path):
        """Test error for .d file instead of directory."""
        fake_bruker = tmp_path / "test.d"
        fake_bruker.touch()  # Create as file, not directory

        with pytest.raises(ValueError, match="requires .d directory"):
            detect_format(fake_bruker)

    def test_nonexistent_path(self, tmp_path):
        """Test error for non-existent path."""
        nonexistent = tmp_path / "nonexistent.imzml"

        with pytest.raises(ValueError, match="Input path does not exist"):
            detect_format(nonexistent)

    def test_get_nonexistent_reader(self):
        """Test getting a non-existent reader class."""
        with pytest.raises(ValueError, match="No reader for format"):
            get_reader_class("nonexistent_format")

    def test_get_nonexistent_converter(self):
        """Test getting a non-existent converter class."""
        with pytest.raises(ValueError, match="No converter for format"):
            get_converter_class("nonexistent_format")
