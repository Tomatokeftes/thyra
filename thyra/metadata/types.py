# thyra/metadata/core/metadata_types.py
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EssentialMetadata:
    """Critical metadata for processing decisions and interpolation setup."""

    # (x, y, z) grid dimensions
    dimensions: Tuple[int, int, int]
    # min_x, max_x, min_y, max_y
    coordinate_bounds: Tuple[float, float, float, float]
    # (min_mass, max_mass)
    mass_range: Tuple[float, float]
    # (x_size, y_size) in micrometers
    pixel_size: Optional[Tuple[float, float]]
    # Total number of spectra
    n_spectra: int
    # Total number of peaks across all spectra (for COO allocation)
    total_peaks: int
    # Memory usage estimate
    estimated_memory_gb: float
    # Path to source data
    source_path: str
    # (x_offset, y_offset, z_offset) for raw coordinate normalization
    coordinate_offsets: Optional[Tuple[int, int, int]] = None
    # Spectrum type for resampling decisions (e.g., "centroid spectrum")
    spectrum_type: Optional[str] = None
    # Per-pixel peak counts for CSR indptr construction (streaming converter)
    # Array of size n_pixels where arr[pixel_idx] = peak_count
    # pixel_idx = z * (n_x * n_y) + y * n_x + x
    peak_counts_per_pixel: Optional[NDArray[np.int32]] = None

    @property
    def has_pixel_size(self) -> bool:
        """Check if pixel size information is available."""
        return self.pixel_size is not None

    @property
    def is_3d(self) -> bool:
        """Check if dataset is 3D (z > 1)."""
        return self.dimensions[2] > 1


@dataclass
class ComprehensiveMetadata:
    """Complete metadata including format-specific details."""

    essential: EssentialMetadata
    format_specific: Dict[str, Any]  # Format-specific metadata (ImzML/Bruker)
    acquisition_params: Dict[str, Any]  # Acquisition parameters
    instrument_info: Dict[str, Any]  # Instrument information
    raw_metadata: Dict[str, Any]  # Original format metadata (unprocessed)

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Convenience access to dimensions from essential metadata."""
        return self.essential.dimensions

    @property
    def pixel_size(self) -> Optional[Tuple[float, float]]:
        """Convenience access to pixel size from essential metadata."""
        return self.essential.pixel_size

    @property
    def coordinate_bounds(self) -> Tuple[float, float, float, float]:
        """Convenience access to coordinate bounds from essential metadata."""
        return self.essential.coordinate_bounds
