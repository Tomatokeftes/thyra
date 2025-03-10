"""
Tests for the Bruker reader.
Note: Full testing requires actual Bruker files and the timsdata DLL.
These tests focus on structure and mocking.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

from msiconvert.readers.bruker_reader import BrukerReader


class TestBrukerReaderStructure:
    """Test the structure and interface of the Bruker reader."""
    
    def test_class_registration(self):
        """Test that the BrukerReader class is properly registered."""
        from msiconvert.core.registry import reader_registry
        assert "bruker" in reader_registry
        assert reader_registry["bruker"] == BrukerReader
    
    def test_interface_implementation(self):
        """Test that the BrukerReader implements the BaseMSIReader interface."""
        from msiconvert.core.base_reader import BaseMSIReader
        assert issubclass(BrukerReader, BaseMSIReader)
        
        # Check that it implements all required methods
        required_methods = [
            'get_metadata',
            'get_dimensions',
            'get_common_mass_axis',
            'iter_spectra',
            'close'
        ]
        
        for method in required_methods:
            assert hasattr(BrukerReader, method)


@pytest.mark.skipif(not Path("C:/timsdata.dll").exists() and 
                    not Path("/usr/lib/libtimsdata.so").exists(),
                    reason="Bruker DLL/shared library not available")
class TestBrukerReaderWithMocks:
    """Test Bruker reader functionality using mocks."""
    
    @patch('msiconvert.readers.bruker_reader.cdll')
    @patch('msiconvert.readers.bruker_reader.sqlite3')
    def test_initialization(self, mock_sqlite3, mock_cdll, mock_bruker_data):
        """Test initialization with mocked dependencies."""
        # Setup mocks
        mock_dll = MagicMock()
        mock_cdll.LoadLibrary.return_value = mock_dll
        mock_dll.tsf_open.return_value = 123  # Non-zero handle
        
        # Mock SQLite connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock query results
        mock_cursor.fetchone.return_value = (10,)  # 10 frames
        mock_cursor.fetchall.return_value = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 grid
        
        # Create reader
        reader = BrukerReader(mock_bruker_data)
        
        # Check initialization
        mock_cdll.LoadLibrary.assert_called_once()
        mock_dll.tsf_open.assert_called_once()
        mock_sqlite3.connect.assert_called_once()
        
        # Check internal state
        assert reader.handle == 123
        assert reader.frame_count == 10
        assert len(reader.frame_positions) == 4
        
        # Clean up
        reader.close()
    
    @patch('msiconvert.readers.bruker_reader.cdll')
    @patch('msiconvert.readers.bruker_reader.sqlite3')
    def test_get_metadata(self, mock_sqlite3, mock_cdll, mock_bruker_data):
        """Test getting metadata with mocked dependencies."""
        # Setup mocks
        mock_dll = MagicMock()
        mock_cdll.LoadLibrary.return_value = mock_dll
        mock_dll.tsf_open.return_value = 123
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (10,)
        mock_cursor.fetchall.return_value = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        # Create reader
        reader = BrukerReader(mock_bruker_data)
        
        # Get metadata
        metadata = reader.get_metadata()
        
        # Check metadata
        assert "source" in metadata
        assert str(mock_bruker_data) == metadata["source"]
        assert metadata["frame_count"] == 10
        
        # Clean up
        reader.close()
    
    @patch('msiconvert.readers.bruker_reader.cdll')
    @patch('msiconvert.readers.bruker_reader.sqlite3')
    def test_get_dimensions(self, mock_sqlite3, mock_cdll, mock_bruker_data):
        """Test getting dimensions with mocked dependencies."""
        # Setup mocks
        mock_dll = MagicMock()
        mock_cdll.LoadLibrary.return_value = mock_dll
        mock_dll.tsf_open.return_value = 123
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (10,)
        mock_cursor.fetchall.return_value = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        # Create reader
        reader = BrukerReader(mock_bruker_data)
        
        # Get dimensions
        dimensions = reader.get_dimensions()
        
        # Check dimensions (should be 2x2x1 from our mock data)
        assert len(dimensions) == 3
        assert dimensions[0] == 2
        assert dimensions[1] == 2
        assert dimensions[2] == 1
        
        # Clean up
        reader.close()
    
    @patch('msiconvert.readers.bruker_reader.cdll')
    @patch('msiconvert.readers.bruker_reader.sqlite3')
    def test_close(self, mock_sqlite3, mock_cdll, mock_bruker_data):
        """Test closing the reader with mocked dependencies."""
        # Setup mocks
        mock_dll = MagicMock()
        mock_cdll.LoadLibrary.return_value = mock_dll
        mock_dll.tsf_open.return_value = 123
        
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_sqlite3.connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = (10,)
        mock_cursor.fetchall.return_value = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        # Create reader
        reader = BrukerReader(mock_bruker_data)
        
        # Close reader
        reader.close()
        
        # Check close calls
        mock_dll.tsf_close.assert_called_once_with(123)
        mock_conn.close.assert_called_once()
        
        # Check internal state
        assert reader.handle is None
        assert reader.conn is None