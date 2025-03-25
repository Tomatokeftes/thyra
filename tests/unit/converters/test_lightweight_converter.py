"""
Tests for the Lightweight converter.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import zarr

from msiconvert.converters.lightweight_converter import LightweightConverter


class TestLightweightConverter:
    """Test the Lightweight converter functionality."""
    
    def test_initialization(self, mock_reader, temp_dir):
        """Test converter initialization."""
        output_path = temp_dir / "test_output.zarr"
        
        # Initialize converter
        converter = LightweightConverter(
            mock_reader,
            output_path,
            dataset_id="test_dataset",
            pixel_size_um=2.5,
            compression_level=7
        )
        
        # Check initialization
        assert converter.reader == mock_reader
        assert converter.output_path == output_path
        assert converter.dataset_id == "test_dataset"
        assert converter.pixel_size_um == 2.5
        assert converter.compression_level == 7
        assert converter.root is None
    
    @patch('msiconvert.converters.lightweight_converter.zarr')
    def test_create_data_structures(self, mock_zarr, mock_reader, temp_dir):
        """Test creating zarr data structures."""
        output_path = temp_dir / "test_output.zarr"
        
        # Mock zarr functionality
        mock_root = MagicMock()
        mock_zarr.open.return_value = mock_root
        mock_root.create_group.return_value = MagicMock()
        mock_compressor = MagicMock()
        mock_zarr.Blosc.return_value = mock_compressor
        
        # Initialize converter
        converter = LightweightConverter(mock_reader, output_path)
        converter._initialize_conversion()
        
        # Create data structures
        data_structures = converter._create_data_structures()
        
        # Check zarr initialization
        mock_zarr.open.assert_called_once_with(str(output_path), mode='w')
        
        # Check that arrays were created
        assert mock_root.array.call_count >= 2  # At least mass_values and coordinates
        assert mock_root.create_group.call_count >= 1  # sparse_data group
        assert mock_root.create_dataset.call_count >= 2  # data and indices datasets
        
        # Check returned data structures
        assert "data_buffer" in data_structures
        assert "indices_buffer" in data_structures
        assert "current_size" in data_structures
        assert data_structures["current_size"] == 0
    
    @patch('msiconvert.converters.lightweight_converter.zarr')
    def test_add_metadata(self, mock_zarr, mock_reader, temp_dir):
        """Test adding metadata to zarr store."""
        output_path = temp_dir / "test_output.zarr"
        
        # Mock zarr functionality
        mock_root = MagicMock()
        mock_zarr.open.return_value = mock_root
        mock_root.attrs = {}
        
        # Initialize converter
        converter = LightweightConverter(mock_reader, output_path, dataset_id="test_dataset", pixel_size_um=2.0)
        converter.root = mock_root
        
        # Add metadata
        metadata = {"source": "test", "instrument": "test_instrument"}
        converter.add_metadata(metadata)
        
        # Check metadata
        assert mock_root.attrs["metadata"] == metadata
        assert mock_root.attrs["dataset_id"] == "test_dataset"
        assert mock_root.attrs["pixel_size_um"] == 2.0
    
    @patch('msiconvert.converters.lightweight_converter.zarr')
    def test_create_arrays(self, mock_zarr, mock_reader, temp_dir):
        """Test creating zarr arrays."""
        output_path = temp_dir / "test_output.zarr"
        
        # Mock zarr functionality
        mock_root = MagicMock()
        mock_zarr.open.return_value = mock_root
        mock_compressor = MagicMock()
        mock_zarr.Blosc.return_value = mock_compressor
        
        # Initialize converter
        converter = LightweightConverter(mock_reader, output_path)
        converter._initialize_conversion()
        converter.root = mock_root
        
        # Create arrays
        converter._create_arrays()
        
        # Check array creation
        mock_root.array.assert_called()  # Called for mass_values and coordinates
        mock_root.create_group.assert_called_once_with('sparse_data')
        mock_root.create_dataset.assert_called()  # Called for data and indices
    
    @patch('msiconvert.converters.lightweight_converter.zarr')
    def test_process_single_spectrum(self, mock_zarr, mock_reader, temp_dir):
        """Test processing a single spectrum."""
        output_path = temp_dir / "test_output.zarr"
        
        # Mock zarr functionality
        mock_root = MagicMock()
        mock_zarr.open.return_value = mock_root
        mock_compressor = MagicMock()
        mock_zarr.Blosc.return_value = mock_compressor
        
        # Initialize converter
        converter = LightweightConverter(mock_reader, output_path)
        converter._initialize_conversion()
        converter._buffer_size = 1000  # Set a large buffer size to avoid flushing
        
        # Create data structures
        data_structures = {
            "data_buffer": [],
            "indices_buffer": [],
            "current_size": 0
        }
        
        # Process a test spectrum
        mzs = np.array([200.0, 500.0])  # Example m/z values
        intensities = np.array([100.0, 200.0])  # Example intensities
        converter._process_single_spectrum(data_structures, (1, 1, 0), mzs, intensities)
        
        # Check that data was added to buffers
        assert len(data_structures["data_buffer"]) == 2
        assert len(data_structures["indices_buffer"]) == 2
        
        # Check values
        pixel_idx = converter._get_pixel_index(1, 1, 0)
        mz_indices = converter._map_mass_to_indices(mzs)
        
        assert data_structures["data_buffer"][0] == 100.0
        assert data_structures["data_buffer"][1] == 200.0
        assert data_structures["indices_buffer"][0] == [pixel_idx, mz_indices[0]]
        assert data_structures["indices_buffer"][1] == [pixel_idx, mz_indices[1]]
    
    @patch('msiconvert.converters.lightweight_converter.zarr')
    def test_flush_buffer(self, mock_zarr, mock_reader, temp_dir):
        """Test flushing data buffers to zarr arrays."""
        output_path = temp_dir / "test_output.zarr"
        
        # Mock zarr functionality
        mock_root = MagicMock()
        mock_zarr.open.return_value = mock_root
        
        # Mock arrays
        mock_data_array = MagicMock()
        mock_indices_array = MagicMock()
        mock_root.__getitem__.side_effect = lambda key: {
            'sparse_data/data': mock_data_array,
            'sparse_data/indices': mock_indices_array
        }[key]
        
        # Initialize converter
        converter = LightweightConverter(mock_reader, output_path)
        converter._initialize_conversion()
        converter.root = mock_root
        
        # Create test data
        data_buffer = [100.0, 200.0, 300.0]
        indices_buffer = [[0, 10], [0, 20], [1, 30]]
        current_size = 5  # Pretend we already have 5 elements
        
        # Flush buffer
        new_size = converter._flush_buffer(data_buffer, indices_buffer, current_size)
        
        # Check resize operation
        mock_data_array.resize.assert_called_once_with(8)  # 5 + 3 = 8
        mock_indices_array.resize.assert_called_once_with((8, 2))
        
        # Check data assignment
        mock_data_array.__setitem__.assert_called_once()
        mock_indices_array.__setitem__.assert_called_once()
        
        # Check returned size
        assert new_size == 8
    
    @patch('msiconvert.converters.lightweight_converter.zarr')
    def test_finalize_data(self, mock_zarr, mock_reader, temp_dir):
        """Test finalizing data structures."""
        output_path = temp_dir / "test_output.zarr"
        
        # Mock zarr functionality
        mock_root = MagicMock()
        mock_zarr.open.return_value = mock_root
        
        # Initialize converter
        converter = LightweightConverter(mock_reader, output_path)
        converter._initialize_conversion()
        converter.root = mock_root
        
        # Create a mock flush_buffer method
        converter._flush_buffer = MagicMock(return_value=10)
        
        # Create data structures with some data
        data_structures = {
            "data_buffer": [100.0, 200.0],
            "indices_buffer": [[0, 10], [0, 20]],
            "current_size": 5
        }
        
        # Finalize data
        converter._finalize_data(data_structures)
        
        # Check that flush_buffer was called
        converter._flush_buffer.assert_called_once_with(
            data_structures["data_buffer"],
            data_structures["indices_buffer"],
            data_structures["current_size"]
        )
    
    @patch('msiconvert.converters.lightweight_converter.zarr')
    def test_save_output(self, mock_zarr, mock_reader, temp_dir):
        """Test saving output."""
        output_path = temp_dir / "test_output.zarr"
        
        # Mock zarr functionality
        mock_root = MagicMock()
        mock_zarr.open.return_value = mock_root
        
        # Initialize converter
        converter = LightweightConverter(mock_reader, output_path)
        converter.root = mock_root
        
        # Save output
        result = converter._save_output({})
        
        # Check that output was saved successfully
        assert result is True
        
        # Note: consolidate_metadata is no longer called in the implementation
    
    @patch('msiconvert.converters.lightweight_converter.zarr')
    def test_convert_end_to_end(self, mock_zarr, mock_reader, temp_dir):
        """Test the full conversion process."""
        output_path = temp_dir / "test_output.zarr"
        
        # Mock zarr functionality
        mock_root = MagicMock()
        mock_zarr.open.return_value = mock_root
        mock_compressor = MagicMock()
        mock_zarr.Blosc.return_value = mock_compressor
        mock_zarr.consolidate_metadata = MagicMock()
        
        # Mock array access
        mock_data_array = MagicMock()
        mock_indices_array = MagicMock()
        mock_root.__getitem__.side_effect = lambda key: {
            'sparse_data/data': mock_data_array,
            'sparse_data/indices': mock_indices_array
        }.get(key, MagicMock())
        
        # Initialize converter
        converter = LightweightConverter(mock_reader, output_path)
        
        # Run conversion
        result = converter.convert()
        
        # Check result
        assert result is True
        
        # Check that zarr store was opened
        mock_zarr.open.assert_called_once_with(str(output_path), mode='w')
        
        # Check that metadata was consolidated
        mock_zarr.consolidate_metadata.assert_called_once()