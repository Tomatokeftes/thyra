"""
Tests for the format registry system.
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from msiconvert.core.base_converter import BaseMSIConverter
from msiconvert.core.base_reader import BaseMSIReader
from msiconvert.core.registry import (
    converter_registry,
    detect_format,
    format_detectors,
    get_converter_class,
    get_reader_class,
    reader_registry,
    register_converter,
    register_format_detector,
    register_reader,
)


class TestRegistry:
    """Test the registry functionality."""

    def setup_method(self):
        """Set up each test by clearing the registries."""
        # Store original registry values to restore later
        self.original_reader_registry = reader_registry.copy()
        self.original_converter_registry = converter_registry.copy()
        self.original_format_detectors = format_detectors.copy()

        # Clear registries for testing
        reader_registry.clear()
        converter_registry.clear()
        format_detectors.clear()

    def teardown_method(self):
        """Restore original registry values after each test."""
        reader_registry.clear()
        reader_registry.update(self.original_reader_registry)

        converter_registry.clear()
        converter_registry.update(self.original_converter_registry)

        format_detectors.clear()
        format_detectors.update(self.original_format_detectors)

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
        assert "test_format" in reader_registry
        assert reader_registry["test_format"] == TestReader

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
        assert "test_format" in converter_registry
        assert converter_registry["test_format"] == TestConverter

        # Test getting the converter class
        retrieved_class = get_converter_class("test_format")
        assert retrieved_class == TestConverter

    def test_register_format_detector(self):
        """Test registering a format detector function."""

        # Create a test detector function
        def test_detector(path):
            return path.name == "test.file"

        # Register it
        register_format_detector("test_format")(test_detector)

        # Check if it was properly registered
        assert "test_format" in format_detectors
        assert format_detectors["test_format"] == test_detector

    def test_detect_format(self):
        """Test format detection."""

        # Register some test detectors
        @register_format_detector("format_a")
        def detect_a(path):
            return path.name == "test_a.file"

        @register_format_detector("format_b")
        def detect_b(path):
            return path.name == "test_b.file"

        # Test detection with patching
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ):
            path_a = Path("test_a.file")
            path_b = Path("test_b.file")
            path_c = Path("test_c.file")

            assert detect_format(path_a) == "format_a"
            assert detect_format(path_b) == "format_b"

            # Test with unknown format
            with pytest.raises(ValueError):
                detect_format(path_c)

    def test_get_nonexistent_reader(self):
        """Test getting a non-existent reader class."""
        with pytest.raises(ValueError):
            get_reader_class("nonexistent_format")

    def test_get_nonexistent_converter(self):
        """Test getting a non-existent converter class."""
        with pytest.raises(ValueError):
            get_converter_class("nonexistent_format")
