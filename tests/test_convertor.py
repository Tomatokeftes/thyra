import pytest
import zarr
import numpy as np
from pathlib import Path
from collections import defaultdict
from msiconvert.imzml.convertor import convert_to_store, _get_xarray_axes, ImzMLToZarrConvertor

@pytest.fixture
def continuous_imzml_files():
    """Fixture pointing to a real continuous imzML file."""
    imzml_path = Path("tests/data/test_continuous.imzML")
    ibd_path = Path("tests/data/test_continuous.ibd")
    
    if not imzml_path.exists() or not ibd_path.exists():
        pytest.skip("Continuous imzML sample files not found.")
    
    return imzml_path, ibd_path

@pytest.fixture
def processed_imzml_files():
    """Fixture pointing to a real processed imzML file."""
    imzml_path = Path(r"C:\Users\tvisv\Downloads\MSIConverter\tests\data\test_processed.imzML")
    ibd_path = Path(r"C:\Users\tvisv\Downloads\MSIConverter\tests\data\test_processed.ibd")
    
    if not imzml_path.exists() or not ibd_path.exists():
        pytest.skip("Processed imzML sample files not found.")
    
    return imzml_path, ibd_path

def test_convert_continuous_to_zarr(continuous_imzml_files, tmp_path):
    """Test conversion of a real continuous imzML file to Zarr format."""
    imzml_path, _ = continuous_imzml_files
    zarr_dest = tmp_path / "continuous_output.zarr"
    
    # Perform the conversion
    convert_to_store("test_continuous", imzml_path.parent, zarr_dest, imzml_filename="test_continuous.imzML", ibd_filename="test_continuous.ibd")

    # Open Zarr store and check structure
    zarr_root = zarr.open(zarr_dest, mode="r")
    assert "0" in zarr_root
    assert "labels/mzs/0" in zarr_root

    # Check metadata and shapes
    assert zarr_root.attrs["multiscales"][0]["name"] == "test_continuous"
    assert _get_xarray_axes(zarr_root) == ['c', 'z', 'y', 'x']
    
    # Assertions based on expected values
    assert zarr_root["0"].shape == (4, 1, 14, 26)  # (m/z values, z, y, x)
    
    # Extract unique non-zero m/z values and compare
    mz_values = np.unique(zarr_root["labels/mzs/0"][:][zarr_root["labels/mzs/0"][:] > 0])
    assert mz_values.tolist() == [100, 200, 300, 400]  # m/z values

    # Confirm non-zero data in intensity array
    assert np.sum(zarr_root["0"][:]) > 0
    assert zarr_root["0"][:, 0, :, :].shape == (4, 14, 26)  # Shape of average mass spectrum

def test_convert_processed_to_zarr(processed_imzml_files, tmp_path, capsys):
    """Test conversion of a real processed imzML file to Zarr format."""
    imzml_path, _ = processed_imzml_files
    zarr_dest = tmp_path / "processed_output.zarr"
    
    # Perform the conversion with exact filenames
    convert_to_store("test_processed", imzml_path.parent, zarr_dest, 
                     imzml_filename="test_processed.imzML", ibd_filename="test_processed.ibd")
    
    # Open Zarr store and verify structure and metadata
    zarr_root = zarr.open(zarr_dest, mode="r")
    assert all(key in zarr_root for key in ["0", "labels/mzs/0", "labels/lengths/0"])

    # Verify metadata and dimensions
    assert zarr_root.attrs["multiscales"][0]["name"] == "test_processed"
    assert _get_xarray_axes(zarr_root) == ['c', 'z', 'y', 'x']
    assert zarr_root["0"].shape == (1, 1, 14, 26)

    # Expected intensity sums for each m/z
    expected_intensity_sums = {
        100: 50 * 31,
        200: 60 * 27,
        300: 70 * 21,
        400: 90 * 11
    }
    
    # Fetch arrays
    lengths = zarr_root["labels/lengths/0"]
    mzs = zarr_root["labels/mzs/0"]
    intensities = zarr_root["0"]

    # Sum intensities per m/z across all pixels
    intensity_sums = defaultdict(int)
    for y in range(lengths.shape[2]):
        for x in range(lengths.shape[3]):
            # Use boolean indexing for non-zero m/z values up to `lengths[y, x]`
            pixel_length = lengths[0, 0, y, x]
            valid_mzs = mzs[:pixel_length, 0, y, x]
            valid_intensities = intensities[:pixel_length, 0, y, x]
            
            for mz, intensity in zip(valid_mzs, valid_intensities):
                intensity_sums[mz] += intensity

    # Check that calculated intensity sums match expected values
    for mz, expected_sum in expected_intensity_sums.items():
        assert intensity_sums[mz] == expected_sum, f"Mismatch for m/z {mz}: expected {expected_sum}, got {intensity_sums[mz]}"

def test_continuous_imzml_to_zarr_conversion(continuous_imzml_files, tmp_path):
    # Set up paths to the specific test imzML and ibd files
    imzml_path, ibd_path = continuous_imzml_files
    dest_path = tmp_path / "output.zarr"
    
    # Initialize converter with the exact file paths
    converter = ImzMLToZarrConvertor(imzml_path, ibd_path)
    success = converter.convert(dest_path)

    # Print diagnostic information
    print(f"Conversion success: {success}")
    print(f"Destination Zarr path exists: {dest_path.exists()}")

    # Assert that the conversion was successful and output Zarr store exists
    assert success, "Conversion failed"
    assert dest_path.exists(), "Destination Zarr store does not exist"

def test_processed_imzml_to_zarr_conversion(processed_imzml_files, tmp_path):
    # Set up paths to the specific test imzML and ibd files
    imzml_path, ibd_path = processed_imzml_files
    dest_path = tmp_path / "output.zarr"
    
    # Initialize converter with the exact file paths
    converter = ImzMLToZarrConvertor(imzml_path, ibd_path)
    success = converter.convert(dest_path)

    # Print diagnostic information
    print(f"Conversion success: {success}")
    print(f"Destination Zarr path exists: {dest_path.exists()}")

    # Check if Zarr store exists and contains expected arrays
    if success and dest_path.exists():
        zarr_root = zarr.open(dest_path, mode="r")
        assert "0" in zarr_root, "Intensity data array '0' missing"
        assert "labels/mzs/0" in zarr_root, "m/z labels missing"
        assert "labels/lengths/0" in zarr_root, "Lengths array missing"
    
    # Assert that the conversion was successful
    assert success, "Conversion failed"
    assert dest_path.exists(), "Destination Zarr store does not exist"
