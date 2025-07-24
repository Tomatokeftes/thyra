# tests/unit/metadata/extractors/test_imzml_extractor.py
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from msiconvert.metadata.extractors.imzml_extractor import ImzMLMetadataExtractor


class TestImzMLMetadataExtractor:
    """Test ImzMLMetadataExtractor functionality."""

    def create_mock_parser(
        self, coordinates=None, mzs_list=None, intensities_list=None
    ):
        """Create a mock pyimzML parser for testing."""
        mock_parser = Mock()

        # Default test data
        if coordinates is None:
            coordinates = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)]
        if mzs_list is None:
            mzs_list = [
                [100.0, 200.0, 300.0],
                [100.0, 150.0, 250.0],
                [120.0, 200.0, 280.0],
                [110.0, 180.0, 290.0],
            ]
        if intensities_list is None:
            intensities_list = [
                [1000.0, 2000.0, 1500.0],
                [800.0, 1200.0, 1800.0],
                [900.0, 1600.0, 1300.0],
                [1100.0, 1400.0, 1700.0],
            ]

        mock_parser.coordinates = coordinates
        mock_parser.getspectrum.side_effect = lambda i: (
            mzs_list[i],
            intensities_list[i],
        )

        # Mock metadata extraction
        mock_parser.metadata = Mock()
        mock_parser.metadata.find.return_value = []

        # Mock imzmldict as an empty dictionary by default
        mock_parser.imzmldict = {}

        return mock_parser

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_creation(self, mock_imzml_parser_class):
        """Test ImzMLMetadataExtractor creation."""
        mock_parser = self.create_mock_parser()
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))
        assert extractor.data_source is mock_parser
        assert extractor.imzml_path == Path("/test/path.imzML")

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_extract_essential_basic(self, mock_imzml_parser_class):
        """Test basic essential metadata extraction."""
        mock_parser = self.create_mock_parser()
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))
        essential = extractor.get_essential()

        assert essential.dimensions == (2, 2, 1)  # 2x2 grid, 1 z-slice
        assert essential.coordinate_bounds == (1, 2, 1, 2)  # min_x, max_x, min_y, max_y
        assert essential.mass_range == (100.0, 300.0)  # Full m/z range
        assert essential.n_spectra == 4
        assert essential.source_path == str(Path("/test/path.imzML"))
        assert essential.estimated_memory_gb > 0

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_extract_essential_with_pixel_size(self, mock_imzml_parser_class):
        """Test essential metadata extraction with pixel size detection."""
        mock_parser = self.create_mock_parser()

        # Set pixel size in imzmldict (primary detection method)
        mock_parser.imzmldict = {
            "pixel size x": 25.0,
            "pixel size y": 25.0,
        }

        # Also mock XML-based pixel size metadata as fallback
        mock_x_param = Mock()
        mock_x_param.get.return_value = 25.0
        mock_y_param = Mock()
        mock_y_param.get.return_value = 25.0

        mock_parser.metadata.find.side_effect = lambda accession: (
            [mock_x_param]
            if accession == "IMS:1000046"
            else [mock_y_param]
            if accession == "IMS:1000047"
            else []
        )

        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))
        essential = extractor.get_essential()

        assert essential.pixel_size == (25.0, 25.0)

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_extract_essential_no_pixel_size(self, mock_imzml_parser_class):
        """Test essential metadata extraction without pixel size."""
        mock_parser = self.create_mock_parser()
        mock_parser.metadata.find.return_value = []  # No pixel size metadata
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))
        essential = extractor.get_essential()

        assert essential.pixel_size is None

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_extract_essential_3d_data(self, mock_imzml_parser_class):
        """Test essential metadata extraction with 3D data."""
        coordinates_3d = [
            (1, 1, 1),
            (2, 1, 1),
            (1, 2, 1),
            (2, 2, 1),  # z=1
            (1, 1, 2),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2),  # z=2
        ]
        mock_parser = self.create_mock_parser(coordinates=coordinates_3d)
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))
        essential = extractor.get_essential()

        assert essential.dimensions == (2, 2, 2)  # 2x2x2 grid
        assert essential.n_spectra == 8
        assert essential.is_3d is True

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_extract_comprehensive(self, mock_imzml_parser_class):
        """Test comprehensive metadata extraction."""
        mock_parser = self.create_mock_parser()

        # Mock additional metadata
        mock_metadata_elem = Mock()
        mock_metadata_elem.attrib = {
            "accession": "MS:1000031",
            "name": "instrument model",
            "value": "test_instrument",
        }
        mock_parser.metadata.iter.return_value = [mock_metadata_elem]

        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))
        comprehensive = extractor.get_comprehensive()

        # Check that essential metadata is included
        assert comprehensive.essential.dimensions == (2, 2, 1)
        assert comprehensive.essential.n_spectra == 4

        # Check format-specific metadata
        assert "imzml_version" in comprehensive.format_specific
        assert "scan_settings" in comprehensive.format_specific

        # Check that raw metadata includes instrument info
        assert comprehensive.raw_metadata is not None

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_mass_range_calculation(self, mock_imzml_parser_class):
        """Test mass range calculation from spectra sampling."""
        # Create test data with varied m/z ranges
        mzs_varied = [
            [50.0, 100.0, 150.0],  # Min: 50
            [80.0, 200.0, 400.0],  # Max: 400
            [60.0, 120.0, 300.0],
            [70.0, 180.0, 350.0],
        ]
        mock_parser = self.create_mock_parser(mzs_list=mzs_varied)
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))
        essential = extractor.get_essential()

        assert essential.mass_range == (50.0, 400.0)

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_memory_estimation(self, mock_imzml_parser_class):
        """Test memory estimation calculation."""
        # Create larger dataset for meaningful memory estimation
        large_coordinates = [
            (x, y, 1) for x in range(1, 11) for y in range(1, 11)
        ]  # 10x10 grid
        large_mzs = [
            [100.0 + i for i in range(1000)] for _ in range(100)
        ]  # 1000 m/z values per spectrum
        large_intensities = [[1000.0] * 1000 for _ in range(100)]

        mock_parser = self.create_mock_parser(
            coordinates=large_coordinates,
            mzs_list=large_mzs,
            intensities_list=large_intensities,
        )
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))
        essential = extractor.get_essential()

        # Should estimate some reasonable memory usage
        assert essential.estimated_memory_gb > 0
        assert essential.estimated_memory_gb < 100  # Sanity check

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_coordinate_bounds_calculation(self, mock_imzml_parser_class):
        """Test coordinate bounds calculation."""
        # Test with non-standard coordinate ranges
        custom_coordinates = [(5, 10, 1), (15, 10, 1), (5, 20, 1), (15, 20, 1)]
        mock_parser = self.create_mock_parser(coordinates=custom_coordinates)
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))
        essential = extractor.get_essential()

        assert essential.coordinate_bounds == (
            5,
            15,
            10,
            20,
        )  # min_x, max_x, min_y, max_y

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_caching_behavior(self, mock_imzml_parser_class):
        """Test that extraction results are properly cached."""
        mock_parser = self.create_mock_parser()
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))

        # First call
        essential1 = extractor.get_essential()
        # Second call should return cached result
        essential2 = extractor.get_essential()

        assert essential1 is essential2  # Same object reference due to caching

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_error_handling_parser_failure(self, mock_imzml_parser_class):
        """Test error handling when parser initialization fails."""
        # Create a parser that will fail when accessing coordinates
        mock_parser = Mock()
        mock_parser.coordinates = None  # This will cause an error
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))

        with pytest.raises(Exception):
            extractor.get_essential()

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_empty_dataset_handling(self, mock_imzml_parser_class):
        """Test handling of empty datasets."""
        mock_parser = self.create_mock_parser(
            coordinates=[], mzs_list=[], intensities_list=[]
        )
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))

        with pytest.raises((ValueError, IndexError)):
            extractor.get_essential()

    @patch("msiconvert.metadata.extractors.imzml_extractor.ImzMLParser")
    def test_single_spectrum_dataset(self, mock_imzml_parser_class):
        """Test handling of single spectrum dataset."""
        single_coordinates = [(1, 1, 1)]
        single_mzs = [[100.0, 200.0, 300.0]]
        single_intensities = [[1000.0, 2000.0, 1500.0]]

        mock_parser = self.create_mock_parser(
            coordinates=single_coordinates,
            mzs_list=single_mzs,
            intensities_list=single_intensities,
        )
        mock_imzml_parser_class.return_value = mock_parser

        extractor = ImzMLMetadataExtractor(mock_parser, Path("/test/path.imzML"))
        essential = extractor.get_essential()

        assert essential.dimensions == (1, 1, 1)
        assert essential.n_spectra == 1
        assert essential.coordinate_bounds == (1, 1, 1, 1)
