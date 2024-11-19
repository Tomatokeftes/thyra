import pytest
from pathlib import Path
from msiconvert.io.parser import ImzMLParser, ImageChannel

@pytest.fixture
def imzml_file(tmp_path):
    # Assume a sample imzML file exists at `tests/data/sample.imzML`
    imzml_file_path = Path("tests/data/test_continuous.imzML")
    if not imzml_file_path.exists():
        pytest.skip("Sample imzML file not found")
    return imzml_file_path

def test_parser_initialization(imzml_file):
    """Test the initialization of the ImzMLParser."""
    parser = ImzMLParser(str(imzml_file))
    assert parser.file_path == str(imzml_file)

def test_main_metadata_parsing(imzml_file):
    """Test parsing of main metadata."""
    parser = ImzMLParser(str(imzml_file))
    metadata = parser.parse_main_metadata()

    assert metadata.width > 0
    assert metadata.height > 0
    assert metadata.n_channels > 0
    assert metadata.pixel_type is not None
    assert metadata.significant_bits > 0

def test_channel_initialization(imzml_file):
    """Test initialization of channels in main metadata."""
    parser = ImzMLParser(str(imzml_file))
    metadata = parser.parse_main_metadata()
    assert len(metadata.channels) == metadata.n_channels
    assert all(isinstance(channel, ImageChannel) for channel in metadata.channels)

def test_raw_metadata_parsing(imzml_file):
    """Test parsing of raw metadata."""
    parser = ImzMLParser(str(imzml_file))
    raw_metadata = parser.parse_raw_metadata()

    assert raw_metadata.store["width"] > 0
    assert raw_metadata.store["height"] > 0
    assert raw_metadata.store["pixel_type"] is not None
    assert raw_metadata.store["mz_precision"] is not None
