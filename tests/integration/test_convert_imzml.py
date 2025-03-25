"""
Integration tests for converting imzML files to various formats.
"""
import pytest
import os
import numpy as np
from pathlib import Path
import shutil
import anndata
import zarr

from msiconvert.convert import convert_msi


class TestImzMLConversion:
    """Test the end-to-end conversion of imzML files."""
    
    def test_convert_to_anndata(self, create_minimal_imzml, temp_dir):
        """Test converting imzML to AnnData format."""
        # Get test data
        imzml_path, _, mzs, _ = create_minimal_imzml
        output_path = temp_dir / "output.h5ad"
        
        # Run conversion
        result = convert_msi(
            str(imzml_path),
            str(output_path),
            format_type="anndata",
            dataset_id="test_dataset",
            pixel_size_um=2.0
        )
        
        # Check result
        assert result is True
        assert output_path.exists()
        
        # Verify the output file by loading it
        try:
            adata = anndata.read_h5ad(output_path)
            
            # Check basic structure
            assert adata.n_obs == 4  # 2x2 grid = 4 pixels
            assert adata.n_vars == len(mzs)  # Should match number of m/z values
            
            # Check metadata
            assert "metadata" in adata.uns
            assert adata.uns["dataset_id"] == "test_dataset"
            assert adata.uns["pixel_size_um"] == 2.0
            
            # Check coordinates
            assert "x" in adata.obs.columns
            assert "y" in adata.obs.columns
            
        except Exception as e:
            pytest.fail(f"Failed to load generated AnnData file: {e}")
    
    def test_convert_to_lightweight(self, create_minimal_imzml, temp_dir):
        """Test converting imzML to lightweight format."""
        # Get test data
        imzml_path, _, mzs, _ = create_minimal_imzml
        output_path = temp_dir / "output.zarr"
        
        # Run conversion
        result = convert_msi(
            str(imzml_path),
            str(output_path),
            format_type="lightweight",
            dataset_id="test_dataset",
            pixel_size_um=2.0
        )
        
        # Check result
        assert result is True
        assert output_path.exists()
        assert output_path.is_dir()  # Zarr stores are directories
        
        # Verify the output file by loading it
        try:
            z = zarr.open(str(output_path), mode='r')
            
            # Check arrays exist
            assert "mass_values" in z
            assert "coordinates" in z
            assert "sparse_data" in z
            assert "data" in z["sparse_data"]
            assert "indices" in z["sparse_data"]
            
            # Check array shapes
            assert len(z["mass_values"]) == len(mzs)  # Should match number of m/z values
            assert len(z["coordinates"]) == 4  # 2x2 grid = 4 pixels
            
            # Check metadata
            assert "metadata" in z.attrs
            assert "dataset_id" in z.attrs
            assert z.attrs["dataset_id"] == "test_dataset"
            assert "pixel_size_um" in z.attrs
            assert z.attrs["pixel_size_um"] == 2.0
            
        except Exception as e:
            pytest.fail(f"Failed to load generated Zarr store: {e}")
    
    @pytest.mark.skipif(not pytest.importorskip("spatialdata", reason="SpatialData not installed"),
                      reason="SpatialData not installed")
    def test_convert_to_spatialdata(self, create_minimal_imzml, temp_dir):
        """Test converting imzML to SpatialData format."""
        # Skip if SpatialData is not available
        spatialdata = pytest.importorskip("spatialdata")
        
        # Get test data
        imzml_path, _, mzs, _ = create_minimal_imzml
        output_path = temp_dir / "output.zarr"
        
        # Run conversion
        result = convert_msi(
            str(imzml_path),
            str(output_path),
            format_type="spatialdata",
            dataset_id="test_dataset",
            pixel_size_um=2.0
        )
        
        # Check result
        assert result is True
        assert output_path.exists()
        assert output_path.is_dir()  # Zarr stores are directories
        
        # Verify the output file by loading it
        try:
            sdata = spatialdata.SpatialData.read(str(output_path))
            
            # Check structure
            assert len(sdata.tables) == 1
            assert "test_dataset" in sdata.tables
            assert len(sdata.shapes) == 1
            
            # Get the table
            table = sdata.tables["test_dataset"]
            
            # Check table structure
            assert table.n_obs == 4  # 2x2 grid = 4 pixels
            assert table.n_vars == len(mzs)  # Should match number of m/z values
            
            # Check spatial coordinates are now in the obs dataframe
            assert "spatial_x" in table.obs.columns
            assert "spatial_y" in table.obs.columns
            
        except Exception as e:
            pytest.fail(f"Failed to load generated SpatialData file: {e}")

    
    def test_convert_nonexistent_file(self, temp_dir):
        """Test error handling with nonexistent input file."""
        # Create nonexistent path
        nonexistent_path = temp_dir / "nonexistent.imzML"
        output_path = temp_dir / "output.h5ad"
        
        # Run conversion
        result = convert_msi(
            str(nonexistent_path),
            str(output_path),
            format_type="anndata"
        )
        
        # Check result
        assert result is False
        assert not output_path.exists()
    
    def test_conversion_with_existing_output(self, create_minimal_imzml, temp_dir):
        """Test error handling when output file already exists."""
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "existing_output.h5ad"
        
        # Create the output file
        with open(output_path, "w") as f:
            f.write("existing file")
        
        # Run conversion
        result = convert_msi(
            str(imzml_path),
            str(output_path),
            format_type="anndata"
        )
        
        # Check result
        assert result is False
        
        # Verify file wasn't overwritten
        with open(output_path, "r") as f:
            content = f.read()
        assert content == "existing file"
    
    def test_convert_with_invalid_format(self, create_minimal_imzml, temp_dir):
        """Test error handling with invalid format type."""
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "output.h5ad"
        
        # Run conversion with invalid format
        result = convert_msi(
            str(imzml_path),
            str(output_path),
            format_type="invalid_format"
        )
        
        # Check result
        assert result is False
        assert not output_path.exists()
    
    def test_convert_with_optimize_chunks(self, create_minimal_imzml, temp_dir, monkeypatch):
        """Test conversion with chunk optimization."""
        # Mock the optimize_zarr_chunks function
        mock_called = False
        
        def mock_optimize(zarr_path, array_path):
            nonlocal mock_called
            mock_called = True
            return True
            
        monkeypatch.setattr("msiconvert.__main__.optimize_zarr_chunks", mock_optimize)
        
        # Get test data
        imzml_path, _, _, _ = create_minimal_imzml
        output_path = temp_dir / "output.zarr"
        
        # Create a mock main function
        from msiconvert.__main__ import main
        import sys
        
        # Save original argv
        original_argv = sys.argv.copy()
        
        try:
            # Set argv for the test
            sys.argv = [
                "msiconvert",
                str(imzml_path),
                str(output_path),
                "--format", "lightweight",
                "--optimize-chunks"
            ]
            
            # Call main function, catching SystemExit
            try:
                main()
                success = True
            except SystemExit as e:
                success = e.code == 0
            
            # Check results
            assert success
            assert mock_called
            assert output_path.exists()
            
        finally:
            # Restore original argv
            sys.argv = original_argv