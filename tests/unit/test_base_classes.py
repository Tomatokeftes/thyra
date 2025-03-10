"""
Tests for the base reader and converter classes.
"""
import pytest
import numpy as np
from pathlib import Path

from msiconvert.core.base_reader import BaseMSIReader
from msiconvert.core.base_converter import BaseMSIConverter


class TestBaseMSIReader:
    """Test the base MSI reader abstract class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        # Subclassing without implementing methods should fail
        with pytest.raises(TypeError):
            class IncompleteReader(BaseMSIReader):
                pass
            
            reader = IncompleteReader()
    
    def test_implementation(self):
        """Test that implementing all required methods works."""
        # Define a minimal implementation
        class MinimalReader(BaseMSIReader):
            def get_metadata(self):
                return {"test": "metadata"}
                
            def get_dimensions(self):
                return (1, 1, 1)
                
            def get_common_mass_axis(self):
                return np.array([100.0, 200.0])
                
            def iter_spectra(self):
                yield ((0, 0, 0), np.array([100.0]), np.array([1.0]))
                
            def close(self):
                pass
        
        # Should instantiate without errors
        reader = MinimalReader()
        
        # Check method functionality
        assert reader.get_metadata() == {"test": "metadata"}
        assert reader.get_dimensions() == (1, 1, 1)
        np.testing.assert_array_equal(reader.get_common_mass_axis(), np.array([100.0, 200.0]))
        
        # Test iterator
        coords, mzs, intensities = next(reader.iter_spectra())
        assert coords == (0, 0, 0)
        np.testing.assert_array_equal(mzs, np.array([100.0]))
        np.testing.assert_array_equal(intensities, np.array([1.0]))


class TestBaseMSIConverter:
    """Test the base MSI converter abstract class."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are properly defined."""
        # Mock reader for testing
        class MockReader(BaseMSIReader):
            def get_metadata(self): return {}
            def get_dimensions(self): return (1, 1, 1)
            def get_common_mass_axis(self): return np.array([])
            def iter_spectra(self): yield ((0, 0, 0), np.array([]), np.array([]))
            def close(self): pass
        
        # Subclassing without implementing methods should fail
        with pytest.raises(TypeError):
            class IncompleteConverter(BaseMSIConverter):
                pass
            
            converter = IncompleteConverter(MockReader(), "test.out")
    
    def test_utility_methods(self):
        """Test the utility methods in the base converter."""
        # Create a minimal implementation for testing
        class MinimalConverter(BaseMSIConverter):
            def _create_data_structures(self):
                return {}
                
            def _save_output(self, data_structures):
                return True
        
        # Create a mock reader
        class MockReader(BaseMSIReader):
            def get_metadata(self): 
                return {"source": "test"}
                
            def get_dimensions(self): 
                return (2, 2, 1)
                
            def get_common_mass_axis(self): 
                return np.array([100.0, 200.0, 300.0])
                
            def iter_spectra(self):
                for x in range(2):
                    for y in range(2):
                        yield ((x, y, 0), 
                               np.array([100.0, 300.0]), 
                               np.array([1.0, 2.0]))
                
            def close(self): 
                pass
        
        # Create the converter
        converter = MinimalConverter(MockReader(), Path("test.out"))
        
        # Initialize for testing utility methods
        converter._initialize_conversion()
        
        # Test coordinate to index conversion
        assert converter._get_pixel_index(0, 0, 0) == 0
        assert converter._get_pixel_index(1, 0, 0) == 1
        assert converter._get_pixel_index(0, 1, 0) == 2
        assert converter._get_pixel_index(1, 1, 0) == 3
        
        # Test mass mapping
        mz_indices = converter._map_mass_to_indices(np.array([100.0, 300.0]))
        np.testing.assert_array_equal(mz_indices, np.array([0, 2]))
        
        # Test sparse matrix creation
        sparse_matrix = converter._create_sparse_matrix()
        assert sparse_matrix.shape == (4, 3)  # 4 pixels, 3 mass values
        
        # Test adding to sparse matrix
        converter._add_to_sparse_matrix(sparse_matrix, 0, np.array([0, 2]), np.array([1.0, 2.0]))
        assert sparse_matrix[0, 0] == 1.0
        assert sparse_matrix[0, 2] == 2.0
        
        # Test coordinates dataframe
        coords_df = converter._create_coordinates_dataframe()
        assert len(coords_df) == 4
        assert "x" in coords_df.columns
        assert "y" in coords_df.columns
        assert "z" in coords_df.columns
        
        # Test mass dataframe
        mass_df = converter._create_mass_dataframe()
        assert len(mass_df) == 3
        assert "mz" in mass_df.columns