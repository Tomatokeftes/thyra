import pytest
from pathlib import Path
from msiconvert.io.format import ImzMLFormat, NotImplementedClass

def test_imzml_format_initialization():
    """Test ImzMLFormat initialization and basic properties."""
    test_path = Path("/path/to/test.imzml")
    imzml_format = ImzMLFormat(test_path)
    assert imzml_format._path == test_path
    assert imzml_format._enabled

def test_get_name():
    """Test that the format returns the correct name."""
    assert ImzMLFormat.get_name() == "ImzML"

def test_is_spatial():
    """Test that the format correctly indicates spatial status."""
    assert ImzMLFormat.is_spatial() is False

def test_is_spectral():
    """Test that the format correctly indicates spectral status."""
    assert ImzMLFormat.is_spectral() is False

def test_is_writable():
    """Test that the format indicates it is not writable."""
    assert ImzMLFormat.is_writable() is False

def test_need_conversion():
    """Test that need_conversion indicates conversion is required."""
    imzml_format = ImzMLFormat(Path("tests/data/test_continuous.imzML"))
    assert imzml_format.need_conversion is True

def test_not_implemented_class():
    """Test that NotImplementedClass raises errors on access."""
    with pytest.raises(NotImplementedError):
        instance = NotImplementedClass()
        instance.some_attribute = "value"
    with pytest.raises(NotImplementedError):
        instance = NotImplementedClass()
        _ = instance.some_attribute
