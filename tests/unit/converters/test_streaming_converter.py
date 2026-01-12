# tests/unit/converters/test_streaming_converter.py
"""Tests for the streaming SpatialData converter."""

import tempfile
from pathlib import Path
from typing import Generator, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from thyra.converters.spatialdata.streaming_converter import (
    SPATIALDATA_AVAILABLE,
    StreamingSpatialDataConverter,
)


class MockEssentialMetadata:
    """Mock essential metadata for testing."""

    def __init__(
        self,
        dimensions: Tuple[int, int, int] = (10, 10, 1),
        n_spectra: int = 100,
        mass_range: Tuple[float, float] = (100.0, 1000.0),
        pixel_size: Optional[Tuple[float, float]] = (20.0, 20.0),
    ):
        self.dimensions = dimensions
        self.n_spectra = n_spectra
        self.mass_range = mass_range
        self.pixel_size = pixel_size
        self.coordinate_bounds = (0, dimensions[0], 0, dimensions[1])
        self.estimated_memory_gb = 0.1
        self.source_path = Path("/mock/path.imzML")
        self.total_peaks = n_spectra * 500  # Estimate


class MockMSIReader:
    """Mock MSI reader for testing the streaming converter."""

    def __init__(
        self,
        dimensions: Tuple[int, int, int] = (10, 10, 1),
        peaks_per_spectrum: int = 500,
        mass_range: Tuple[float, float] = (100.0, 1000.0),
    ):
        self.dimensions = dimensions
        self.peaks_per_spectrum = peaks_per_spectrum
        self.mass_range = mass_range
        self.n_spectra = dimensions[0] * dimensions[1] * dimensions[2]

        # Generate consistent random data
        np.random.seed(42)

    def get_essential_metadata(self) -> MockEssentialMetadata:
        """Return mock essential metadata."""
        return MockEssentialMetadata(
            dimensions=self.dimensions,
            n_spectra=self.n_spectra,
            mass_range=self.mass_range,
        )

    def get_comprehensive_metadata(self) -> MagicMock:
        """Return mock comprehensive metadata."""
        mock = MagicMock()
        mock.format_specific = {}
        mock.acquisition_params = {}
        mock.instrument_info = {}
        mock.raw_metadata = {}
        mock.essential = self.get_essential_metadata()
        return mock

    def iter_spectra(
        self,
    ) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """Yield mock spectra for all pixels."""
        n_x, n_y, n_z = self.dimensions
        min_mz, max_mz = self.mass_range

        for z in range(n_z):
            for y in range(n_y):
                for x in range(n_x):
                    # Generate random m/z values within range
                    mzs = np.sort(
                        np.random.uniform(min_mz, max_mz, self.peaks_per_spectrum)
                    )
                    # Generate random intensities
                    intensities = np.random.exponential(1000, self.peaks_per_spectrum)

                    yield (x, y, z), mzs, intensities

    def get_optical_image_paths(self):
        """Return empty list (no optical images)."""
        return []

    def close(self):
        """Mock close method."""
        pass


@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
class TestStreamingSpatialDataConverter:
    """Tests for StreamingSpatialDataConverter."""

    def test_converter_initialization(self):
        """Test that the converter initializes correctly."""
        reader = MockMSIReader()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.zarr"

            converter = StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="test_dataset",
                pixel_size_um=20.0,
                chunk_size=50,
                target_bins=1000,
            )

            assert converter._chunk_size == 50
            assert converter._target_bins == 1000
            assert converter.dataset_id == "test_dataset"

    def test_basic_conversion(self):
        """Test basic conversion with small synthetic dataset."""
        reader = MockMSIReader(
            dimensions=(5, 5, 1),  # 25 pixels
            peaks_per_spectrum=100,
            mass_range=(200.0, 800.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.zarr"

            converter = StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="test_dataset",
                pixel_size_um=20.0,
                chunk_size=10,  # Small chunks for testing
                target_bins=500,  # Small number of bins for speed
            )

            # Run conversion
            success = converter.convert()
            assert success, "Conversion should succeed"

            # Check output exists
            assert output_path.exists(), "Output path should exist"

            # Try to load the output
            from spatialdata import SpatialData

            sdata = SpatialData.read(str(output_path))

            # Verify structure
            assert len(sdata.tables) > 0, "Should have tables"
            assert len(sdata.shapes) > 0, "Should have shapes"
            assert len(sdata.images) > 0, "Should have images (TIC)"

            # Check table content
            table_key = list(sdata.tables.keys())[0]
            adata = sdata.tables[table_key]

            assert adata.shape[0] == 25, "Should have 25 pixels (5x5)"
            assert adata.shape[1] == 500, "Should have 500 mass bins"

    def test_chunked_processing(self):
        """Test that chunked processing produces correct output."""
        reader = MockMSIReader(
            dimensions=(10, 10, 1),  # 100 pixels
            peaks_per_spectrum=200,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_chunked.zarr"

            converter = StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="chunked_test",
                chunk_size=25,  # Process in 4 chunks
                target_bins=1000,
            )

            success = converter.convert()
            assert success

            # Verify all pixels are present
            from spatialdata import SpatialData

            sdata = SpatialData.read(str(output_path))
            table_key = list(sdata.tables.keys())[0]
            adata = sdata.tables[table_key]

            assert adata.shape[0] == 100, "All 100 pixels should be present"

    def test_tic_calculation(self):
        """Test that TIC values are calculated correctly."""
        reader = MockMSIReader(
            dimensions=(3, 3, 1),  # 9 pixels
            peaks_per_spectrum=50,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_tic.zarr"

            converter = StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="tic_test",
                chunk_size=5,
                target_bins=100,
            )

            success = converter.convert()
            assert success

            from spatialdata import SpatialData

            sdata = SpatialData.read(str(output_path))

            # Check TIC image exists
            image_key = [k for k in sdata.images.keys() if "tic" in k][0]
            tic_image = sdata.images[image_key]

            # TIC should be 3x3
            assert tic_image.shape[-2:] == (3, 3)

            # TIC values should be positive (sum of intensities)
            tic_values = tic_image.values
            assert np.all(tic_values >= 0), "TIC values should be non-negative"

    def test_resampling(self):
        """Test that spectra are correctly resampled to common mass axis."""
        reader = MockMSIReader(
            dimensions=(2, 2, 1),
            peaks_per_spectrum=100,
            mass_range=(100.0, 500.0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_resample.zarr"

            converter = StreamingSpatialDataConverter(
                reader=reader,
                output_path=output_path,
                dataset_id="resample_test",
                target_bins=200,  # Specific number of bins
            )

            success = converter.convert()
            assert success

            from spatialdata import SpatialData

            sdata = SpatialData.read(str(output_path))
            table_key = list(sdata.tables.keys())[0]
            adata = sdata.tables[table_key]

            # Check mass axis has correct number of bins
            assert adata.shape[1] == 200, "Should have 200 mass bins"

            # Check m/z values in var
            mz_values = adata.var["mz"].values
            assert len(mz_values) == 200
            assert mz_values[0] >= 100.0
            assert mz_values[-1] <= 500.0


@pytest.mark.skipif(
    not SPATIALDATA_AVAILABLE,
    reason="SpatialData dependencies not available",
)
def test_streaming_converter_memory_efficiency():
    """Test that streaming converter uses bounded memory."""
    # This is a basic check - actual memory profiling would need
    # more sophisticated tools

    reader = MockMSIReader(
        dimensions=(20, 20, 1),  # 400 pixels
        peaks_per_spectrum=500,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_memory.zarr"

        converter = StreamingSpatialDataConverter(
            reader=reader,
            output_path=output_path,
            chunk_size=50,  # Small chunks
            target_bins=1000,
        )

        # Should complete without memory issues
        success = converter.convert()
        assert success
