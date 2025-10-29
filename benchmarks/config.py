"""
Benchmark configuration and dataset registry.

This module defines all datasets and benchmark configurations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class DatasetConfig:
    """Configuration for a benchmark dataset."""

    name: str
    path: Path
    format_type: str  # 'imzml' or 'bruker'
    description: str
    resampling_config: Optional[Dict[str, Any]] = None

    @property
    def zarr_path(self) -> Path:
        """Get path for converted zarr output."""
        suffix = ""
        if self.resampling_config:
            if "target_bins" in self.resampling_config.get("params", {}):
                bins = self.resampling_config["params"]["target_bins"]
                suffix = f"_{bins}bins"
        return Path("benchmarks/converted") / f"{self.name}{suffix}.zarr"


# Dataset registry - using real test data
# NOTE: Using only Xenium dataset for fair comparison across formats
# (same dataset in different formats: ImzML, Bruker .d, and SpatialData/Zarr)
DATASETS = {
    # Xenium dataset (18GB) - for publication-quality benchmarks
    # ImzML version was resampled by SCILS
    # Pre-converted Zarr exists at benchmarks/converted/xenium.zarr
    "xenium": DatasetConfig(
        name="xenium",
        path=Path("test_data/20240826_xenium_0041899.imzML"),
        format_type="imzml",
        description="Xenium 18GB dataset (ImzML resampled by SCILS)",
    ),
}


# Benchmark configurations
class BenchmarkConfig:
    """Global benchmark configuration."""

    # Directories
    CONVERTED_DIR = Path("benchmarks/converted")
    RESULTS_DIR = Path("benchmarks/results")
    PLOTS_DIR = Path("benchmarks/plots")

    # Access pattern configurations
    N_RANDOM_PIXELS = 100
    N_ION_IMAGES = 10
    ROI_SIZE = (50, 50)  # pixels

    # M/Z query ranges (Da)
    MZ_RANGES = [
        (100, 110),  # Small range
        (200, 250),  # Medium range
        (500, 600),  # Large range
        (100, 500),  # Very large range
    ]

    # Specific m/z values for ion image extraction
    TARGET_MZ_VALUES = [150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    MZ_TOLERANCE = 0.1  # Da

    # Parallel processing
    WORKER_COUNTS = [1, 2, 4, 8]

    # Statistical rigor - number of repeated runs for each benchmark
    N_BENCHMARK_RUNS = 1  # Set to 1 for testing, 5 for production

    # Plot styling
    PLOT_DPI = 300
    PLOT_STYLE = {
        "imzml": "#A23B72",
        "bruker": "#F18F01",
        "spatialdata": "#2E86AB",
        "speedup": "#06A77D",
    }

    @classmethod
    def setup_directories(cls):
        """Create all necessary directories."""
        cls.CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
        cls.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        cls.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
