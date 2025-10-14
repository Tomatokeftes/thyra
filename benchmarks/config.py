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
DATASETS = {
    # ImzML test data
    "pea_imzml": DatasetConfig(
        name="pea_imzml",
        path=Path("test_data/pea.imzML"),
        format_type="imzml",
        description="PEA ImzML dataset",
    ),
    # 'bellini_imzml': DatasetConfig(
    #     name='bellini_imzml',
    #     path=Path('test_data/bellini.imzML'),
    #     format_type='imzml',
    #     description='Bellini ImzML dataset'
    # ),
    # Bruker data - raw (no resampling, preserves all unique m/z ~78M bins)
    "bruker_pea_raw": DatasetConfig(
        name="bruker_pea_raw",
        path=Path("test_data/20231109_PEA_NEDC.d"),
        format_type="bruker",
        description="Bruker PEA raw data (no interpolation, ~78M m/z bins)",
        resampling_config=None,  # No resampling - preserves all unique m/z
    ),
    # Bruker data - resampled to 300k bins (industry standard)
    "bruker_pea_300k": DatasetConfig(
        name="bruker_pea_300k",
        path=Path("test_data/20231109_PEA_NEDC.d"),
        format_type="bruker",
        description="Bruker PEA resampled to 300k bins",
        resampling_config={
            "method": "bin_width_at_mz",
            "params": {"target_bins": 300000},
        },
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
