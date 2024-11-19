import pytest
import zarr
import numpy as np
from pathlib import Path
from collections import defaultdict

# Adjust the import to reflect the new module structure
from msiconvert.io.convertor import _get_xarray_axes, MSIToZarrConvertor

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
    imzml_path = Path("tests/data/test_processed.imzML")
    ibd_path = Path("tests/data/test_processed.ibd")
    
    if not imzml_path.exists() or not ibd_path.exists():
        pytest.skip("Processed imzML sample files not found.")
    
    return imzml_path, ibd_path

def test_convert_continuous_to_zarr(continuous_imzml_files, tmp_path):
    """Test conversion of a real continuous imzML file to Zarr format."""
    imzml_path, ibd_path = continuous_imzml_files
    zarr_dest = tmp_path / "continuous_output.zarr"
    
    # Initialize converter with the input directory and output path
    converter = MSIToZarrConvertor(imzml_path, zarr_dest)
    success = converter.convert()

    # Check that conversion was successful
    assert success, "Conversion failed"

    # Open Zarr store and check structure
    zarr_root = zarr.open(zarr_dest, mode="r")
    assert "0" in zarr_root, "Intensity data array '0' missing"
    assert "labels/mzs/0" in zarr_root, "m/z labels missing"

    # Check metadata and shapes
    assert zarr_root.attrs["multiscales"][0]["name"] == zarr_dest.stem  # Should be 'continuous_output'
    assert _get_xarray_axes(zarr_root) == ['c', 'z', 'y', 'x']
    
    # Assertions based on expected values
    assert zarr_root["0"].shape == (4, 1, 14, 26)  # (m/z values, z, y, x)
    
    # Extract unique non-zero m/z values and compare
    mz_values = np.unique(zarr_root["labels/mzs/0"][:][zarr_root["labels/mzs/0"][:] > 0])
    assert mz_values.tolist() == [100, 200, 300, 400], f"Unexpected m/z values: {mz_values.tolist()}"

    # Confirm non-zero data in intensity array
    assert np.sum(zarr_root["0"][:]) > 0, "Intensity array sum is zero"
    assert zarr_root["0"][:, 0, :, :].shape == (4, 14, 26), "Unexpected shape of average mass spectrum"

def test_convert_processed_to_zarr(processed_imzml_files, tmp_path):
    """Test conversion of a real processed imzML file to Zarr format."""
    imzml_path, ibd_path = processed_imzml_files
    zarr_dest = tmp_path / "processed_output.zarr"
    
    # Initialize converter with the input directory and output path
    converter = MSIToZarrConvertor(imzml_path, zarr_dest)
    success = converter.convert()
    
    # Check that conversion was successful
    assert success, "Conversion failed"

    # Open Zarr store and verify structure and metadata
    zarr_root = zarr.open(zarr_dest, mode="r")
    expected_keys = ["0", "labels/mzs/0", "labels/lengths/0"]
    for key in expected_keys:
        assert key in zarr_root, f"{key} missing in Zarr store"

    # Verify metadata and dimensions
    assert zarr_root.attrs["multiscales"][0]["name"] == zarr_dest.stem  # Should be 'processed_output'
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
            # Get the valid length for this pixel
            pixel_length = lengths[0, 0, y, x]
            # Skip pixels with zero length
            if pixel_length == 0:
                continue
            # Get the valid m/z and intensity values
            valid_mzs = mzs[:pixel_length, 0, y, x]
            valid_intensities = intensities[:pixel_length, 0, y, x]
            # Sum intensities per m/z
            for mz, intensity in zip(valid_mzs, valid_intensities):
                intensity_sums[mz] += intensity

    # Check that calculated intensity sums match expected values
    for mz, expected_sum in expected_intensity_sums.items():
        assert intensity_sums[mz] == expected_sum, f"Mismatch for m/z {mz}: expected {expected_sum}, got {intensity_sums[mz]}"

def test_continuous_imzml_to_zarr_conversion(continuous_imzml_files, tmp_path):
    """Test conversion of continuous imzML file using MSIToZarrConvertor."""
    imzml_path, ibd_path = continuous_imzml_files
    dest_path = tmp_path / "output_continuous.zarr"
    
    # Initialize converter with the input directory and output path
    converter = MSIToZarrConvertor(imzml_path, dest_path)
    success = converter.convert()

    # Print diagnostic information
    print(f"Conversion success: {success}")
    print(f"Destination Zarr path exists: {dest_path.exists()}")

    # Assert that the conversion was successful and output Zarr store exists
    assert success, "Conversion failed"
    assert dest_path.exists(), "Destination Zarr store does not exist"

def test_processed_imzml_to_zarr_conversion(processed_imzml_files, tmp_path):
    """Test conversion of processed imzML file using MSIToZarrConvertor."""
    imzml_path, ibd_path = processed_imzml_files
    dest_path = tmp_path / "output_processed.zarr"
    
    # Initialize converter with the input directory and output path
    converter = MSIToZarrConvertor(imzml_path, dest_path)
    success = converter.convert()

    # Print diagnostic information
    print(f"Conversion success: {success}")
    print(f"Destination Zarr path exists: {dest_path.exists()}")

    # Check if Zarr store exists and contains expected arrays
    assert success, "Conversion failed"
    assert dest_path.exists(), "Destination Zarr store does not exist"
    zarr_root = zarr.open(dest_path, mode="r")
    expected_keys = ["0", "labels/mzs/0", "labels/lengths/0"]
    for key in expected_keys:
        assert key in zarr_root, f"{key} missing in Zarr store"
