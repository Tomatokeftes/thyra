"""
Integration tests for converting Bruker files to various formats.
These tests will be skipped if Bruker dependencies are not available.
"""
import pytest
import os
import sys
import numpy as np
from pathlib import Path
import shutil
import ctypes
from unittest.mock import patch, MagicMock

from msiconvert.convert import convert_msi


# Skip all tests if Bruker DLL/shared library is not available
bruker_dll_available = (
    (sys.platform.startswith("win32") and Path("C:/timsdata.dll").exists()) or
    (sys.platform.startswith("linux") and Path("/usr/lib/libtimsdata.so").exists())
)

pytestmark = pytest.mark.skipif(
    not bruker_dll_available,
    reason="Bruker DLL/shared library not available"
)


class TestBrukerConversion:
    """Test the end-to-end conversion of Bruker files."""
    
    @pytest.fixture
    def mock_bruker_data_dir(self, mock_bruker_data):
        """Return the mock Bruker data directory."""
        return mock_bruker_data
    
    @patch('msiconvert.readers.bruker_reader.cdll')
    @patch('msiconvert.readers.bruker_reader.sqlite3')
    def test_detect_bruker_format(self, mock_sqlite3, mock_cdll, mock_bruker_data_dir):
        """Test that the Bruker format is correctly detected."""
        from msiconvert.core.registry import detect_format
        
        # Expected to be detected as 'bruker'
        detected_format = detect_format(mock_bruker_data_dir)
        assert detected_format == "bruker"
    
    @patch('msiconvert.readers.bruker_reader.cdll')
    @patch('msiconvert.readers.bruker_reader.sqlite3')
    def test_convert_to_anndata(self, mock_sqlite3, mock_cdll, mock_bruker_data_dir, temp_dir):
        """Test converting Bruker to AnnData format."""
        # Set up mocks for Bruker reader
        mock_dll = MagicMock()
        mock_cdll.LoadLibrary.return_value = mock_dll
        mock_dll.tsf_open.return_value = 123  # Non-zero handle
        
        # Mock SQLite connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock query results for a 2x2 grid
        mock_cursor.fetchone.return_value = (4,)  # 4 frames
        mock_cursor.fetchall.return_value = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 grid
        
        # Mock the spectrum data
        x = 100
        indices = np.arange(x)
        intensities = np.random.rand(x) * 100
        mzs = np.linspace(100, 1000, x)
        
        def mock_read_line_spectrum(self, frame_id):
            return indices, intensities
            
        def mock_index_to_mz(self, frame_id, indices):
            return mzs
        
        # Apply patches
        with patch('msiconvert.readers.bruker_reader.BrukerReader.read_line_spectrum', mock_read_line_spectrum):
            with patch('msiconvert.readers.bruker_reader.BrukerReader.index_to_mz', mock_index_to_mz):
                # Set output path
                output_path = temp_dir / "output_bruker.h5ad"
                
                # Run conversion
                result = convert_msi(
                    str(mock_bruker_data_dir),
                    str(output_path),
                    format_type="anndata",
                    dataset_id="test_bruker",
                    pixel_size_um=2.0
                )
                
                # Check result
                assert result is True
                assert output_path.exists()
                
                # Further validation would require loading the file
                # But we'll skip that due to mock limitations
    
    @patch('msiconvert.readers.bruker_reader.cdll')
    @patch('msiconvert.readers.bruker_reader.sqlite3')
    def test_convert_to_lightweight(self, mock_sqlite3, mock_cdll, mock_bruker_data_dir, temp_dir):
        """Test converting Bruker to lightweight format."""
        # Set up mocks for Bruker reader
        mock_dll = MagicMock()
        mock_cdll.LoadLibrary.return_value = mock_dll
        mock_dll.tsf_open.return_value = 123  # Non-zero handle
        
        # Mock SQLite connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock query results for a 2x2 grid
        mock_cursor.fetchone.return_value = (4,)  # 4 frames
        mock_cursor.fetchall.return_value = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 grid
        
        # Mock the spectrum data
        x = 100
        indices = np.arange(x)
        intensities = np.random.rand(x) * 100
        mzs = np.linspace(100, 1000, x)
        
        def mock_read_line_spectrum(self, frame_id):
            return indices, intensities
            
        def mock_index_to_mz(self, frame_id, indices):
            return mzs
        
        # Apply patches
        with patch('msiconvert.readers.bruker_reader.BrukerReader.read_line_spectrum', mock_read_line_spectrum):
            with patch('msiconvert.readers.bruker_reader.BrukerReader.index_to_mz', mock_index_to_mz):
                # Set output path
                output_path = temp_dir / "output_bruker.zarr"
                
                # Run conversion
                result = convert_msi(
                    str(mock_bruker_data_dir),
                    str(output_path),
                    format_type="lightweight",
                    dataset_id="test_bruker",
                    pixel_size_um=2.0
                )
                
                # Check result
                assert result is True
                assert output_path.exists()
                
                # Further validation would require loading the file
                # But we'll skip that due to mock limitations
    
    @pytest.mark.skipif(not pytest.importorskip("spatialdata", reason="SpatialData not installed"),
                      reason="SpatialData not installed")
    @patch('msiconvert.readers.bruker_reader.cdll')
    @patch('msiconvert.readers.bruker_reader.sqlite3')
    def test_convert_to_spatialdata(self, mock_sqlite3, mock_cdll, mock_bruker_data_dir, temp_dir):
        """Test converting Bruker to SpatialData format."""
        # Skip if SpatialData is not available
        spatialdata = pytest.importorskip("spatialdata")
        
        # Set up mocks for Bruker reader
        mock_dll = MagicMock()
        mock_cdll.LoadLibrary.return_value = mock_dll
        mock_dll.tsf_open.return_value = 123  # Non-zero handle
        
        # Mock SQLite connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock query results for a 2x2 grid
        mock_cursor.fetchone.return_value = (4,)  # 4 frames
        mock_cursor.fetchall.return_value = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 grid
        
        # Mock the spectrum data
        x = 100
        indices = np.arange(x)
        intensities = np.random.rand(x) * 100
        mzs = np.linspace(100, 1000, x)
        
        def mock_read_line_spectrum(self, frame_id):
            return indices, intensities
            
        def mock_index_to_mz(self, frame_id, indices):
            return mzs
        
        # Apply patches
        with patch('msiconvert.readers.bruker_reader.BrukerReader.read_line_spectrum', mock_read_line_spectrum):
            with patch('msiconvert.readers.bruker_reader.BrukerReader.index_to_mz', mock_index_to_mz):
                # Set output path
                output_path = temp_dir / "output_bruker_spatial.zarr"
                
                # Run conversion
                result = convert_msi(
                    str(mock_bruker_data_dir),
                    str(output_path),
                    format_type="spatialdata",
                    dataset_id="test_bruker",
                    pixel_size_um=2.0
                )
                
                # Check result
                assert result is True
                assert output_path.exists()
                
                # Further validation would require loading the file
                # But we'll skip that due to mock limitations