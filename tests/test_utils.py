import pytest
from pathlib import Path
from msiconvert.imzml.utils import get_imzml_pair  # Update with the correct module path

@pytest.fixture
def sample_directory(tmp_path):
    """Fixture to create a temporary directory with sample files."""
    imzml_file = tmp_path / "sample.imzML"
    ibd_file = tmp_path / "sample.ibd"
    non_matching_file = tmp_path / "non_matching.imzML"

    # Create files
    imzml_file.touch()
    ibd_file.touch()
    non_matching_file.touch()
    
    return tmp_path

def test_get_imzml_pair_existing_pair(sample_directory):
    """Test get_imzml_pair correctly finds an existing imzML/ibd file pair."""
    imzml_filename = "sample.imzML"
    ibd_filename = "sample.ibd"
    result = get_imzml_pair(sample_directory, imzml_filename, ibd_filename)
    
    assert result is not None
    assert result[0] == sample_directory / imzml_filename
    assert result[1] == sample_directory / ibd_filename

def test_get_imzml_pair_missing_imzml(sample_directory):
    """Test get_imzml_pair returns None if the imzML file is missing."""
    ibd_filename = "sample.ibd"
    imzml_filename = "missing.imzML"  # Non-existent imzML filename
    result = get_imzml_pair(sample_directory, imzml_filename, ibd_filename)
    
    assert result is None

def test_get_imzml_pair_missing_ibd(sample_directory):
    """Test get_imzml_pair returns None if the ibd file is missing."""
    imzml_filename = "sample.imzML"
    ibd_filename = "missing.ibd"  # Non-existent ibd filename
    result = get_imzml_pair(sample_directory, imzml_filename, ibd_filename)
    
    assert result is None

def test_get_imzml_pair_no_matching_files(sample_directory):
    """Test get_imzml_pair returns None if both imzML and ibd files are missing."""
    imzml_filename = "non_existent.imzML"
    ibd_filename = "non_existent.ibd"
    result = get_imzml_pair(sample_directory, imzml_filename, ibd_filename)
    
    assert result is None

def test_get_imzml_pair_non_matching_file(sample_directory):
    """Test get_imzml_pair ignores non-matching imzML files."""
    imzml_filename = "sample.imzML"
    ibd_filename = "sample.ibd"
    non_matching_imzml = sample_directory / "non_matching.imzML"
    result = get_imzml_pair(sample_directory, imzml_filename, ibd_filename)
    
    assert result is not None
    assert non_matching_imzml not in result
