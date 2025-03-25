"""
Tests for the imzML reader.
"""
import pytest
import numpy as np
from pathlib import Path
import os

from msiconvert.readers.imzml_reader import ImzMLReader


class TestImzMLReader:
    """Test the ImzML reader functionality."""
    
    def test_initialization(self, create_minimal_imzml):
        """Test initializing the reader with a valid file."""
        imzml_path, _, _, _ = create_minimal_imzml
        
        # Test initialization with path as string
        reader1 = ImzMLReader(str(imzml_path))
        assert hasattr(reader1, 'parser')
        
        # Test initialization with path as Path
        reader2 = ImzMLReader(imzml_path)
        assert hasattr(reader2, 'parser')
        
        # Clean up
        reader1.close()
        reader2.close()
    
    def test_missing_ibd(self, temp_dir):
        """Test error when .ibd file is missing."""
        # Create imzML file without ibd
        imzml_path = temp_dir / "missing.imzML"
        with open(imzml_path, "w") as f:
            f.write("dummy content")
        
        with pytest.raises(ValueError):
            ImzMLReader(imzml_path)
    
    def test_get_metadata(self, create_minimal_imzml):
        """Test getting metadata from imzML file."""
        imzml_path, _, _, _ = create_minimal_imzml
        
        reader = ImzMLReader(imzml_path)
        metadata = reader.get_metadata()
        
        assert "source" in metadata
        assert str(imzml_path) == metadata["source"]
        assert "file_mode" in metadata
        
        reader.close()
    
    def test_get_dimensions(self, create_minimal_imzml):
        """Test getting dimensions from imzML file."""
        imzml_path, _, _, _ = create_minimal_imzml
        
        reader = ImzMLReader(imzml_path)
        dimensions = reader.get_dimensions()
        
        # Our test imzML has a 2x2 grid
        assert len(dimensions) == 3  # (x, y, z)
        assert dimensions[0] == 2  # 2 pixels in x
        assert dimensions[1] == 2  # 2 pixels in y
        assert dimensions[2] == 1  # 1 plane in z
        
        reader.close()
    
    def test_get_common_mass_axis(self, create_minimal_imzml):
        """Test getting common mass axis from imzML file."""
        imzml_path, _, mzs, _ = create_minimal_imzml
        
        reader = ImzMLReader(imzml_path)
        mass_axis = reader.get_common_mass_axis()
        
        # Check that we got a valid mass axis
        assert len(mass_axis) > 0
        
        # The values should match our input mzs for a 'processed' imzML
        np.testing.assert_allclose(mass_axis, mzs)
        
        reader.close()
    
    def test_iter_spectra(self, create_minimal_imzml):
        """Test iterating through spectra."""
        imzml_path, _, mzs, intensities = create_minimal_imzml
        
        reader = ImzMLReader(imzml_path)
        
        # Count spectra and check data
        count = 0
        for coords, spectrum_indices, spectrum_intensities in reader.iter_spectra():
            # Check coordinates format
            assert len(coords) == 3
            x, y, z = coords
            assert x >= 0 and y >= 0 and z >= 0
            
            # Check that indices are valid and intensities match
            assert len(spectrum_indices) > 0
            assert len(spectrum_intensities) > 0
            
            # The test seems to be expecting actual mz values, not indices
            # So let's get the actual mz values from the common mass axis
            common_axis = reader.get_common_mass_axis()
            spectrum_mzs = common_axis[spectrum_indices]
            
            # This is the problematic line in the test - we need mz values
            np.testing.assert_allclose(spectrum_mzs, mzs)
            
            count += 1
        
        # We should have 4 spectra (2x2 grid)
        assert count == 4
        
        reader.close()
    
    def test_iter_and_reconstruct(self, create_minimal_imzml):
        """Test iterating through spectra and reconstructing full data."""
        imzml_path, _, mzs, _ = create_minimal_imzml
        
        reader = ImzMLReader(imzml_path)
        
        # Get common mass axis
        common_axis = reader.get_common_mass_axis()
        
        # Manually collect data similar to what the former 'read' method would do
        coordinates = []
        intensities = []
        
        # Iterate through all spectra
        for coords, spectrum_mzs, spectrum_intensities in reader.iter_spectra():
            coordinates.append(coords)
            
            # In a real application, you might need to map these to the common axis
            # For the test, we'll just collect the data
            intensities.append(spectrum_intensities)
        
        # We should have 4 spectra (2x2 grid)
        assert len(coordinates) == 4
        assert len(intensities) == 4
        
        # Get dimensions
        dimensions = reader.get_dimensions()
        assert dimensions[0] == 2  # width
        assert dimensions[1] == 2  # height
        
        reader.close()
    
    def test_close(self, create_minimal_imzml):
        """Test closing the reader."""
        imzml_path, _, _, _ = create_minimal_imzml
        
        reader = ImzMLReader(imzml_path)
        # Close should work without errors
        reader.close()