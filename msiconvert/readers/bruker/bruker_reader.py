"""
Ultimate Bruker reader implementation combining best features from all implementations.

This module provides a high-performance, memory-efficient reader for Bruker TSF/TDF
data formats with lazy loading, intelligent caching, and comprehensive error handling.
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ...core.base_extractor import MetadataExtractor
from ...core.base_reader import BaseMSIReader
from ...core.registry import register_reader
from ...metadata.extractors.bruker_extractor import BrukerMetadataExtractor
from ...utils.bruker_exceptions import DataError, FileFormatError, SDKError
from .sdk.dll_manager import DLLManager
from .sdk.sdk_functions import SDKFunctions
from .utils.batch_processor import BatchProcessor
from .utils.coordinate_cache import CoordinateCache
from .utils.mass_axis_builder import MassAxisBuilder
from .utils.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


@register_reader("bruker")
class BrukerReader(BaseMSIReader):
    """
    Ultimate Bruker reader combining the best features from multiple implementations.

    Features:
    - Lazy loading with intelligent caching
    - Memory-efficient batch processing
    - Robust SDK integration with fallback mechanisms
    - Comprehensive error handling and recovery
    - Progress tracking and performance monitoring
    - Compatible with spatialdata_converter.py interface
    """

    def __init__(
        self,
        data_path: Path,
        use_recalibrated_state: bool = False,
        cache_coordinates: bool = True,
        memory_limit_gb: Optional[float] = None,
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None,
        **kwargs,
    ):
        """
        Initialize the Ultimate Bruker reader.

        Args:
            data_path: Path to Bruker .d directory
            use_recalibrated_state: Whether to use recalibrated data
            cache_coordinates: Whether to cache coordinates for performance
            memory_limit_gb: Optional memory limit in GB
            batch_size: Optional batch size for processing
            progress_callback: Optional callback for progress updates
            **kwargs: Additional arguments
        """
        super().__init__(data_path, **kwargs)
        self.use_recalibrated_state = use_recalibrated_state
        self.progress_callback = progress_callback

        # Validate and setup paths
        self._validate_data_path()
        self._detect_file_type()

        # Initialize components
        self._setup_components(cache_coordinates, memory_limit_gb, batch_size)

        # Initialize SDK and connections
        self._initialize_sdk()
        self._initialize_database()

        # Cached properties (lazy loaded)
        self._common_mass_axis: Optional[np.ndarray] = None
        self._frame_count: Optional[int] = None

        # Performance tracking
        self._performance_stats = {
            "total_spectra_read": 0,
            "total_read_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        logger.info(
            f"Initialized Ultimate BrukerReader for {self.file_type.upper()} data at {data_path}"
        )

    def _validate_data_path(self) -> None:
        """Validate the data path and check for required files."""
        if not self.data_path.exists():
            raise FileFormatError(f"Data path does not exist: {self.data_path}")

        if not self.data_path.is_dir():
            raise FileFormatError(f"Data path must be a directory: {self.data_path}")

        if not self.data_path.suffix == ".d":
            raise FileFormatError(f"Expected .d directory, got: {self.data_path}")

    def _detect_file_type(self) -> None:
        """Detect whether this is TSF or TDF data."""
        tsf_path = self.data_path / "analysis.tsf"
        tdf_path = self.data_path / "analysis.tdf"

        if tsf_path.exists():
            self.file_type = "tsf"
            self.db_path = tsf_path
        elif tdf_path.exists():
            self.file_type = "tdf"
            self.db_path = tdf_path
        else:
            raise FileFormatError(
                f"No analysis.tsf or analysis.tdf found in {self.data_path}"
            )

        logger.debug(f"Detected file type: {self.file_type.upper()}")

    def _setup_components(
        self,
        cache_coordinates: bool,
        memory_limit_gb: Optional[float],
        batch_size: Optional[int],
    ) -> None:
        """Setup all utility components."""
        # Memory manager
        self.memory_manager = MemoryManager(
            memory_limit_gb=memory_limit_gb, buffer_pool_size=20
        )

        # Coordinate cache
        self.coordinate_cache = CoordinateCache(
            db_path=self.db_path, preload_all=cache_coordinates
        )

        # Mass axis builder
        self.mass_axis_builder = MassAxisBuilder(
            strategy="auto",
            memory_limit_mb=memory_limit_gb * 1024 if memory_limit_gb else 1024,
            progress_callback=self.progress_callback,
        )

        # Batch processor
        self.batch_processor = BatchProcessor(
            target_memory_mb=512,
            min_batch_size=10,
            max_batch_size=batch_size or 100,
            progress_callback=self.progress_callback,
        )
        # Disable batch processor progress bars to avoid double progress display
        self.batch_processor._quiet_mode = True

    def _initialize_sdk(self) -> None:
        """Initialize the Bruker SDK with error handling."""
        try:
            # Initialize DLL manager
            self.dll_manager = DLLManager(
                data_directory=self.data_path, force_reload=False
            )

            # Initialize SDK functions
            self.sdk = SDKFunctions(self.dll_manager, self.file_type)

            # Open the data file
            self.handle = self.sdk.open_file(
                str(self.data_path), self.use_recalibrated_state
            )

            logger.info(f"Successfully initialized {self.file_type.upper()} SDK")

        except Exception as e:
            logger.error(f"Failed to initialize SDK: {e}")
            raise SDKError(f"Failed to initialize Bruker SDK: {e}") from e

    def _initialize_database(self) -> None:
        """Initialize database connection with optimizations."""
        try:
            self.conn = sqlite3.connect(str(self.db_path))

            # Apply SQLite optimizations
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = NORMAL")
            self.conn.execute("PRAGMA cache_size = 10000")
            self.conn.execute("PRAGMA temp_store = MEMORY")

            logger.debug("Initialized database connection with optimizations")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DataError(f"Failed to open database: {e}") from e

    def _create_metadata_extractor(self) -> MetadataExtractor:
        """Create Bruker metadata extractor."""
        if not hasattr(self, "conn") or self.conn is None:
            raise ValueError("Database connection not available")
        return BrukerMetadataExtractor(self.conn, self.data_path)

    def get_common_mass_axis(self) -> np.ndarray:
        """
        Return the common mass axis composed of all unique m/z values.

        Returns:
            Array of unique m/z values in ascending order
        """
        if self._common_mass_axis is None:
            self._common_mass_axis = self._build_common_mass_axis()

        return self._common_mass_axis

    def _build_common_mass_axis(self) -> np.ndarray:
        """Build the common mass axis using the optimized builder."""
        logger.info("Building common mass axis")

        # Create iterator for mass axis building
        def mz_iterator():
            for coords, mzs, intensities in self._iter_spectra_raw():
                yield coords, mzs, intensities

        frame_count = self._get_frame_count()

        # Build mass axis
        mass_axis = self.mass_axis_builder.build_from_spectra_iterator(
            mz_iterator(), total_spectra=frame_count
        )

        if len(mass_axis) == 0:
            logger.warning("No m/z values found in dataset")
            return np.array([])

        logger.info(f"Built common mass axis with {len(mass_axis)} unique m/z values")
        return mass_axis

    def iter_spectra(
        self, batch_size: Optional[int] = None
    ) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """
        Iterate through all spectra with optional batching.

        Args:
            batch_size: Optional batch size for processing

        Yields:
            Tuples of (coordinates, mz_array, intensity_array)
        """
        if batch_size is None:
            # Use memory-efficient iteration
            yield from self._iter_spectra_raw()
        else:
            # Use batch processing
            yield from self._iter_spectra_batched(batch_size)

    def _iter_spectra_raw(
        self,
    ) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """Raw spectrum iteration without batching."""
        frame_count = self._get_frame_count()

        # Setup progress tracking
        with tqdm(
            total=frame_count,
            desc="Reading spectra",
            unit="spectrum",
            disable=True,  # Disable to avoid double progress with converter
        ) as pbar:
            for frame_id in range(1, frame_count + 1):
                try:
                    # Get coordinates
                    coords = self.coordinate_cache.get_coordinate(frame_id)
                    if coords is None:
                        logger.warning(f"No coordinates found for frame {frame_id}")
                        pbar.update(1)
                        continue

                    # Read spectrum
                    start_time = time.time()
                    mzs, intensities = self.sdk.read_spectrum(self.handle, frame_id)
                    read_time = (time.time() - start_time) * 1000

                    # Update performance stats
                    self._performance_stats["total_spectra_read"] += 1
                    self._performance_stats["total_read_time_ms"] += read_time

                    if mzs.size > 0 and intensities.size > 0:
                        yield coords, mzs, intensities

                    pbar.update(1)

                    # Progress callback
                    if self.progress_callback:
                        self.progress_callback(frame_id, frame_count)

                    # Memory management
                    if frame_id % 100 == 0:  # Check every 100 spectra
                        if not self.memory_manager.check_memory_limit():
                            logger.warning("Memory limit approached, optimizing")
                            self.memory_manager.optimize_memory()

                except Exception as e:
                    logger.warning(f"Error reading spectrum for frame {frame_id}: {e}")
                    pbar.update(1)
                    continue

    def _iter_spectra_batched(
        self, batch_size: int
    ) -> Generator[Tuple[Tuple[int, int, int], np.ndarray, np.ndarray], None, None]:
        """Batched spectrum iteration for better memory management."""
        frame_count = self._get_frame_count()

        def batch_processor_func(frame_ids, batch_info):
            """Process a batch of frame IDs."""
            results = []

            # Get coordinates for all frames in batch
            coords_dict = self.coordinate_cache.get_coordinates_batch(frame_ids)

            for frame_id in frame_ids:
                try:
                    coords = coords_dict.get(frame_id)
                    if coords is None:
                        continue

                    # Read spectrum
                    mzs, intensities = self.sdk.read_spectrum(self.handle, frame_id)

                    if mzs.size > 0 and intensities.size > 0:
                        results.append((coords, mzs, intensities))

                except Exception as e:
                    logger.warning(f"Error reading spectrum for frame {frame_id}: {e}")
                    continue

            return results

        # Create frame ID iterator
        frame_ids = list(range(1, frame_count + 1))

        # Process in batches
        for batch_results in self.batch_processor.process_spectrum_batches(
            iter(frame_ids),  # Pass individual frame IDs
            frame_count,  # Total number of frames
            batch_processor_func,
            batch_size=batch_size,  # Use the requested batch size
        ):
            for spectrum_data in batch_results:
                yield spectrum_data

    def _get_frame_count(self) -> int:
        """Get the total number of frames."""
        if self._frame_count is None:
            self._frame_count = self.coordinate_cache.get_frame_count()

        return self._frame_count

    def close(self) -> None:
        """Close all resources and connections."""
        logger.info("Closing Bruker reader")

        try:
            # Close SDK handle
            if hasattr(self, "handle") and self.handle:
                self.sdk.close_file(self.handle)
                self.handle = None

            # Close database connection
            if hasattr(self, "conn") and self.conn:
                self.conn.close()
                self.conn = None

            # Clear caches
            if hasattr(self, "coordinate_cache"):
                self.coordinate_cache.clear_cache()

            if hasattr(self, "memory_manager"):
                self.memory_manager.buffer_pool.clear()

            logger.info("Successfully closed all resources")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Get spatial dimensions (pixel grid) from metadata extractor.

        Note: This is the spatial pixel grid, not the mass axis dimensions.
        Mass axis interpolation to common m/z values is handled during conversion.

        Returns:
            Tuple of (x_pixels, y_pixels, z_pixels) spatial dimensions
        """
        essential_metadata = self.get_essential_metadata()
        return essential_metadata.dimensions

    @property
    def mass_range(self) -> Tuple[float, float]:
        """
        Get mass range from metadata extractor.

        Note: This is the acquisition mass range, not the final interpolated axis.
        The actual common mass axis for interpolation is built from all unique m/z values.

        Returns:
            Tuple of (min_mass, max_mass) in m/z units
        """
        essential_metadata = self.get_essential_metadata()
        return essential_metadata.mass_range

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {
            "reader_stats": self._performance_stats.copy(),
            "memory_stats": self.memory_manager.get_stats(),
            "batch_stats": self.batch_processor.get_stats(),
            "mass_axis_stats": self.mass_axis_builder.get_stats(),
            "coordinate_stats": self.coordinate_cache.get_coverage_stats(),
        }

        # Calculate derived metrics
        if self._performance_stats["total_spectra_read"] > 0:
            stats["average_read_time_ms"] = (
                self._performance_stats["total_read_time_ms"]
                / self._performance_stats["total_spectra_read"]
            )

        return stats

    def optimize_performance(self) -> None:
        """Optimize performance by cleaning caches and managing memory."""
        logger.info("Optimizing reader performance")

        # Memory optimization
        self.memory_manager.optimize_memory()

        # Coordinate cache optimization
        self.coordinate_cache.optimize_cache(keep_recent=1000)

        # Reset statistics for fresh measurements
        self.batch_processor.reset_stats()
        self.mass_axis_builder.reset_stats()

        logger.info("Performance optimization complete")

    def __repr__(self) -> str:
        """String representation of the reader."""
        return (
            f"BrukerReader(path={self.data_path}, "
            f"type={self.file_type.upper()}, "
            f"frames={self._get_frame_count()})"
        )

    @property
    def n_spectra(self) -> int:
        """
        Return the total number of spectra in the dataset.

        Returns:
            Total number of frames (efficient implementation using cached frame count)
        """
        return self._get_frame_count()

    def __del__(self) -> None:
        """Destructor to ensure cleanup."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during destruction
