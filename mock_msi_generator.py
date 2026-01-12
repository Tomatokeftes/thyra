"""Generate mock MSI data for testing streaming SpatialData conversion.

This script creates synthetic MSI-like data without needing real datasets,
making it easy to test and iterate on memory-efficient conversion approaches.

Usage:
    poetry run python mock_msi_generator.py [--size small|medium|large|huge]

Sizes:
    small  : 100x100 pixels, ~10k spectra (quick test, <1s)
    medium : 500x500 pixels, ~250k spectra (realistic, ~5s)
    large  : 1000x1000 pixels, ~1M spectra (stress test, ~30s)
    huge   : 2000x2000 pixels, ~4M spectra (memory test, ~2min)

The mock data simulates processed MSI data with:
- Sparse spectra (typical ~50-200 peaks per pixel)
- Realistic m/z range (100-2000 Da)
- Gaussian peak shapes with noise
"""

import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import numpy as np


@dataclass
class MockMSIConfig:
    """Configuration for mock MSI data generation."""

    n_x: int = 100
    n_y: int = 100
    n_z: int = 1
    mz_min: float = 100.0
    mz_max: float = 2000.0
    n_mz_bins: int = 50000  # Common mass axis bins
    peaks_per_spectrum: Tuple[int, int] = (50, 200)  # min, max peaks
    intensity_range: Tuple[float, float] = (100.0, 10000.0)
    noise_level: float = 0.1
    sparsity: float = 0.0  # Fraction of empty pixels (0-1)
    seed: Optional[int] = 42


PRESETS = {
    "small": MockMSIConfig(n_x=100, n_y=100, n_mz_bins=30000),
    "medium": MockMSIConfig(n_x=500, n_y=500, n_mz_bins=50000),
    "large": MockMSIConfig(n_x=1000, n_y=1000, n_mz_bins=100000),
    "huge": MockMSIConfig(n_x=2000, n_y=2000, n_mz_bins=200000),
}


class MockMSIReader:
    """Mock MSI reader that generates synthetic data on-the-fly.

    This reader mimics the interface of ImzMLReader but generates
    random MSI-like data, useful for testing conversion pipelines
    without needing real data files.
    """

    def __init__(self, config: MockMSIConfig):
        """Initialize the mock MSI reader with the given configuration."""
        self.config = config
        self._rng = np.random.default_rng(config.seed)

        # Build common mass axis
        self._common_mass_axis = np.linspace(
            config.mz_min, config.mz_max, config.n_mz_bins
        )

        # Simulate some "real" peaks that appear in multiple spectra
        # These are like biomarker peaks
        n_common_peaks = 20
        self._common_peak_indices = self._rng.choice(
            config.n_mz_bins, size=n_common_peaks, replace=False
        )

        # Pre-generate which pixels are empty (for sparse datasets)
        self._n_pixels = config.n_x * config.n_y * config.n_z
        self._empty_pixels = set()
        if config.sparsity > 0:
            n_empty = int(self._n_pixels * config.sparsity)
            self._empty_pixels = set(
                self._rng.choice(self._n_pixels, size=n_empty, replace=False)
            )

        # Mock data path for compatibility
        self.data_path = Path("mock_msi_data")

    def get_essential_metadata(self):
        """Return metadata matching the EssentialMetadata interface."""

        class MockMetadata:
            """Mock metadata object for testing."""

            dimensions: Tuple[int, int, int]
            n_spectra: int
            mass_range: Tuple[float, float]
            spectrum_type: str
            pixel_size: Tuple[float, float]
            coordinate_bounds: dict
            estimated_memory_gb: float

        meta = MockMetadata()
        meta.dimensions = (self.config.n_x, self.config.n_y, self.config.n_z)
        meta.n_spectra = self._n_pixels - len(self._empty_pixels)
        meta.mass_range = (self.config.mz_min, self.config.mz_max)
        meta.spectrum_type = "processed"
        meta.pixel_size = (10.0, 10.0)  # (x, y) pixel size in um
        meta.coordinate_bounds = {
            "x": (0, self.config.n_x - 1),
            "y": (0, self.config.n_y - 1),
            "z": (0, self.config.n_z - 1),
        }
        meta.estimated_memory_gb = (
            self._n_pixels * 150 * 8 / (1024**3)
        )  # rough estimate
        return meta

    def scan_mass_range(self):
        """Scan mass range - returns min/max m/z."""
        return self.config.mz_min, self.config.mz_max

    def iter_spectra(
        self, batch_size: int = 1000
    ) -> Iterator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray]]:
        """Iterate through synthetic spectra.

        Yields:
            Tuple of (coords, mz_values, intensities) for each pixel
        """
        cfg = self.config

        for z in range(cfg.n_z):
            for y in range(cfg.n_y):
                for x in range(cfg.n_x):
                    pixel_idx = z * (cfg.n_x * cfg.n_y) + y * cfg.n_x + x

                    # Skip empty pixels
                    if pixel_idx in self._empty_pixels:
                        continue

                    # Generate spectrum for this pixel
                    mz_indices, intensities = self._generate_spectrum()
                    mz_values = self._common_mass_axis[mz_indices]

                    yield (x, y, z), mz_values, intensities

    def _generate_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single synthetic spectrum."""
        cfg = self.config

        # Random number of peaks
        n_peaks = self._rng.integers(
            cfg.peaks_per_spectrum[0], cfg.peaks_per_spectrum[1]
        )

        # Include some common peaks (biomarkers)
        n_common = min(len(self._common_peak_indices), n_peaks // 3)
        common_indices = self._rng.choice(
            self._common_peak_indices, size=n_common, replace=False
        )

        # Random peaks
        n_random = n_peaks - n_common
        random_indices = self._rng.choice(cfg.n_mz_bins, size=n_random, replace=False)

        # Combine and sort
        all_indices = np.unique(np.concatenate([common_indices, random_indices]))

        # Generate intensities with log-normal distribution
        intensities = self._rng.lognormal(
            mean=np.log(1000), sigma=1.5, size=len(all_indices)
        )

        # Clip to range
        intensities = np.clip(
            intensities, cfg.intensity_range[0], cfg.intensity_range[1]
        )

        # Add noise
        if cfg.noise_level > 0:
            noise = self._rng.normal(0, cfg.noise_level * intensities)
            intensities = np.maximum(intensities + noise, 0)

        return all_indices.astype(np.int32), intensities.astype(np.float64)

    def get_common_mass_axis(self) -> np.ndarray:
        """Return the common mass axis."""
        return self._common_mass_axis


def test_streaming_conversion(config: MockMSIConfig, output_dir: Path):
    """Test the streaming converter with mock data."""
    import tracemalloc

    from thyra.converters.spatialdata.streaming_converter import (
        StreamingSpatialDataConverter,
    )

    print("\nMock data config:")
    print(f"  Dimensions: {config.n_x} x {config.n_y} x {config.n_z}")
    print(f"  Total pixels: {config.n_x * config.n_y * config.n_z:,}")
    print(f"  M/z bins: {config.n_mz_bins:,}")
    print(f"  Peaks per spectrum: {config.peaks_per_spectrum}")
    print(f"  Sparsity: {config.sparsity * 100:.1f}%")

    # Create mock reader
    reader = MockMSIReader(config)

    output_path = output_dir / "mock_streaming.zarr"

    # Start memory tracking
    tracemalloc.start()

    print("\nConverting with streaming method...")
    start_time = time.time()

    converter = StreamingSpatialDataConverter(
        reader=reader,
        output_path=output_path,
        dataset_id="mock",
        pixel_size_um=10.0,
        zero_copy=True,
        chunk_size=5000,
    )

    # Pre-set the mass axis and dimensions to skip initialization
    # (mock reader doesn't support full initialization flow)
    converter._common_mass_axis = reader.get_common_mass_axis()
    converter._dimensions = (config.n_x, config.n_y, config.n_z)

    # Call the direct zarr method directly
    try:
        converter._stream_write_direct_zarr()
        success = True
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        success = False

    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"\nConversion: {'SUCCESS' if success else 'FAILED'}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Peak memory: {peak / (1024 * 1024):.1f} MB")

    if success:
        # Verify output
        print("\nVerifying output...")

        import zarr

        store = zarr.open_group(str(output_path), mode="r")

        if "tables" in store:
            table_name = list(store["tables"].keys())[0]
            X = store[f"tables/{table_name}/X"]
            shape = X.attrs.get("shape", [0, 0])
            indptr = X["indptr"][-1]

            print(f"  Shape: {shape[0]:,} x {shape[1]:,}")
            print(f"  Total NNZ: {indptr:,}")

            # Estimate expected NNZ
            n_spectra = config.n_x * config.n_y - len(reader._empty_pixels)
            avg_peaks = sum(config.peaks_per_spectrum) / 2
            expected_nnz = int(n_spectra * avg_peaks)

            print(f"  Expected NNZ (approx): {expected_nnz:,}")

        # Test SpatialData reading
        print("\nTesting SpatialData.read()...")
        try:
            from spatialdata import SpatialData

            sdata = SpatialData.read(str(output_path), selection=("tables",))
            table = list(sdata.tables.values())[0]
            print("  SpatialData read: SUCCESS")
            print(f"  Table shape: {table.X.shape}")
            print(f"  Table NNZ: {table.X.nnz:,}")
        except Exception as e:
            print(f"  SpatialData read: FAILED - {e}")

    return output_path


def main():
    """Run the mock MSI data generator test."""
    print("=" * 70)
    print("  MOCK MSI DATA GENERATOR FOR STREAMING CONVERSION TEST")
    print("=" * 70)

    # Get size from command line
    size = "small"
    for arg in sys.argv[1:]:
        if arg.startswith("--size="):
            size = arg.split("=")[1]
        elif arg in PRESETS:
            size = arg

    if size not in PRESETS:
        print(f"Unknown size: {size}")
        print(f"Available: {list(PRESETS.keys())}")
        return

    config = PRESETS[size]
    print(f"\nUsing preset: {size}")

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="mock_msi_"))
    print(f"Output directory: {temp_dir}")

    # Run test
    output_path = test_streaming_conversion(config, temp_dir)

    print("\n" + "=" * 70)
    print(f"Output saved to: {output_path}")
    print(f'To clean up: rmdir /s /q "{temp_dir}"')


if __name__ == "__main__":
    main()
