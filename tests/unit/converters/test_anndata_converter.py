"""
Tests for the AnnData converter.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import sparse
import anndata

from msiconvert.converters.anndata_converter import AnnDataConverter


class TestAnnDataConverter:
    """Test the AnnData converter functionality."""
    
    def test_initialization(self, mock_reader, temp_dir):
        """Test converter initialization."""
        output_path = temp_dir / "test_output.h5ad"
        
        # Initialize converter
        converter = AnnDataConverter(
            mock_reader,
            output_path,
            dataset_id="test_dataset",
            pixel_size_um=2.5
        )
        
        # Check initialization
        assert converter.reader == mock_reader
        assert converter.output_path == output_path
        assert converter.dataset_id == "test_dataset"
        assert converter.pixel_size_um == 2.5
        assert converter.compression_level == 5  # Default value
    
    def test_create_data_structures(self, mock_reader, temp_dir):
        """Test creating data structures."""
        output_path = temp_dir / "test_output.h5ad"
        
        # Initialize converter and prepare for conversion
        converter = AnnDataConverter(mock_reader, output_path)
        converter._initialize_conversion()
        
        # Create data structures
        data_structures = converter._create_data_structures()
        
        # Check data structures
        assert "sparse_data" in data_structures
        assert "obs_df" in data_structures
        assert "var_df" in data_structures
        
        # Check sparse matrix
        assert isinstance(data_structures["sparse_data"], sparse.lil_matrix)
        assert data_structures["sparse_data"].shape == (9, 100)  # 3x3 grid, 100 m/z values
        
        # Check observation dataframe
        assert isinstance(data_structures["obs_df"], pd.DataFrame)
        assert len(data_structures["obs_df"]) == 9  # 3x3 grid
        
        # Check variable dataframe
        assert isinstance(data_structures["var_df"], pd.DataFrame)
        assert len(data_structures["var_df"]) == 100  # 100 m/z values
    
    def test_process_spectrum(self, mock_reader, temp_dir):
        """Test processing a single spectrum."""
        output_path = temp_dir / "test_output.h5ad"
        
        # Initialize converter and prepare for conversion
        converter = AnnDataConverter(mock_reader, output_path)
        converter._initialize_conversion()
        
        # Create data structures
        data_structures = converter._create_data_structures()
        
        # Process a test spectrum
        mzs = np.array([200.0, 500.0])  # Example m/z values
        intensities = np.array([100.0, 200.0])  # Example intensities
        converter._process_single_spectrum(data_structures, (1, 1, 0), mzs, intensities)
        
        # Get the pixel index
        pixel_idx = converter._get_pixel_index(1, 1, 0)
        
        # Map m/z values to indices
        mz_indices = converter._map_mass_to_indices(mzs)
        
        # Check that data was added to the sparse matrix
        assert data_structures["sparse_data"][pixel_idx, mz_indices[0]] == 100.0
        assert data_structures["sparse_data"][pixel_idx, mz_indices[1]] == 200.0
    
    def test_finalize_data(self, mock_reader, temp_dir):
        """Test finalizing data structures."""
        output_path = temp_dir / "test_output.h5ad"
        
        # Initialize converter and prepare for conversion
        converter = AnnDataConverter(mock_reader, output_path)
        converter._initialize_conversion()
        
        # Create data structures
        data_structures = converter._create_data_structures()
        
        # Process a test spectrum
        mzs = np.array([200.0, 500.0])
        intensities = np.array([100.0, 200.0])
        converter._process_single_spectrum(data_structures, (1, 1, 0), mzs, intensities)
        
        # Finalize data
        converter._finalize_data(data_structures)
        
        # Check that sparse matrix was converted to CSR format
        assert isinstance(data_structures["sparse_data"], sparse.csr_matrix)
    
    def test_add_metadata(self, mock_reader, temp_dir):
        """Test adding metadata to AnnData object."""
        output_path = temp_dir / "test_output.h5ad"
        
        # Initialize converter
        converter = AnnDataConverter(mock_reader, output_path, dataset_id="test_dataset", pixel_size_um=2.0)
        converter._initialize_conversion()
        
        # Create a simple AnnData object
        adata = anndata.AnnData(
            X=sparse.csr_matrix((3, 3)),
            obs=pd.DataFrame(index=["p1", "p2", "p3"]),
            var=pd.DataFrame(index=["m1", "m2", "m3"])
        )
        
        # Add metadata
        converter.add_metadata(adata)
        
        # Check metadata
        assert "metadata" in adata.uns
        assert adata.uns["dataset_id"] == "test_dataset"
        assert adata.uns["pixel_size_um"] == 2.0
    
    @pytest.mark.parametrize("use_zarr", [True, False])
    def test_save_anndata(self, mock_reader, temp_dir, use_zarr, monkeypatch):
        """Test saving AnnData with different backends."""
        output_path = temp_dir / "test_output.h5ad"
        
        # Initialize converter
        converter = AnnDataConverter(mock_reader, output_path)
        converter._initialize_conversion()
        
        # Create a mock AnnData class
        class MockAnnData:
            def __init__(self, X, obs, var):
                self.X = X
                self.obs = obs
                self.var = var
                self.uns = {}
                # Set n_obs and n_vars as properties
                self._n_obs = 10
                self._n_vars = 10
                if use_zarr:
                    self._n_obs = 100000
                    self._n_vars = 10000
                    
            @property
            def n_obs(self):
                return self._n_obs
                
            @property
            def n_vars(self):
                return self._n_vars
                
            def write_h5ad(self, path):
                pass
                
            def write_zarr(self, store, chunks=None):
                pass
        
        # Replace AnnData with our mock
        monkeypatch.setattr("msiconvert.converters.anndata_converter.AnnData", MockAnnData)
        
        # Create a simple AnnData object
        adata = MockAnnData(
            X=sparse.csr_matrix((10, 10)),
            obs=pd.DataFrame(index=[f"p{i}" for i in range(10)]),
            var=pd.DataFrame(index=[f"m{i}" for i in range(10)])
        )
        
        # Save the data
        converter._save_anndata(adata)
    
    def test_convert_end_to_end(self, mock_reader, temp_dir, monkeypatch):
        """Test the full conversion process."""
        output_path = temp_dir / "test_output.h5ad"
        
        # Create a mock for saving to avoid actual file writing
        class MockAnnData:
            def __init__(self, X, obs, var):
                self.X = X
                self.obs = obs
                self.var = var
                self.uns = {}
                self.n_obs = obs.shape[0]
                self.n_vars = var.shape[0]
            
            def write_h5ad(self, path):
                pass
        
        # Patch AnnData to use our mock
        monkeypatch.setattr("msiconvert.converters.anndata_converter.AnnData", MockAnnData)
        
        # Initialize converter
        converter = AnnDataConverter(mock_reader, output_path)
        
        # Run conversion
        result = converter.convert()
        
        # Check result
        assert result is True