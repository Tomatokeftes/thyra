"""Memory management utilities for efficient buffer handling.

This module provides memory management capabilities including buffer
pooling, memory monitoring, and efficient array operations to handle
large datasets.
"""

import gc
import logging
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional psutil import for memory monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

logger = logging.getLogger(__name__)


class BufferPool:
    """Thread-safe buffer pool for reusing numpy arrays.

    This reduces memory allocation overhead by reusing buffers of common
    sizes throughout the reading process.
    """

    def __init__(self, max_buffers_per_size: int = 10):
        """Initialize the buffer pool.

        Args:
            max_buffers_per_size: Maximum number of buffers to keep per size
        """
        self.max_buffers_per_size = max_buffers_per_size
        self._float64_buffers: Dict[int, List[np.ndarray]] = {}
        self._float32_buffers: Dict[int, List[np.ndarray]] = {}
        self._uint32_buffers: Dict[int, List[np.ndarray]] = {}
        self._lock = Lock()

    def get_float64_buffer(self, size: int) -> np.ndarray:
        """Get a float64 buffer of the specified size.

        Args:
            size: Required buffer size

        Returns:
            numpy array buffer
        """
        with self._lock:
            buffers = self._float64_buffers.get(size, [])
            if buffers:
                buffer = buffers.pop()
                logger.debug(f"Reused float64 buffer of size {size}")
                return buffer
            else:
                buffer = np.empty(size, dtype=np.float64)
                logger.debug(f"Created new float64 buffer of size {size}")
                return buffer

    def get_float32_buffer(self, size: int) -> np.ndarray:
        """Get a float32 buffer of the specified size.

        Args:
            size: Required buffer size

        Returns:
            numpy array buffer
        """
        with self._lock:
            buffers = self._float32_buffers.get(size, [])
            if buffers:
                buffer = buffers.pop()
                logger.debug(f"Reused float32 buffer of size {size}")
                return buffer
            else:
                buffer = np.empty(size, dtype=np.float32)
                logger.debug(f"Created new float32 buffer of size {size}")
                return buffer

    def get_uint32_buffer(self, size: int) -> np.ndarray:
        """Get a uint32 buffer of the specified size.

        Args:
            size: Required buffer size

        Returns:
            numpy array buffer
        """
        with self._lock:
            buffers = self._uint32_buffers.get(size, [])
            if buffers:
                buffer = buffers.pop()
                logger.debug(f"Reused uint32 buffer of size {size}")
                return buffer
            else:
                buffer = np.empty(size, dtype=np.uint32)
                logger.debug(f"Created new uint32 buffer of size {size}")
                return buffer

    def return_float64_buffer(self, buffer: np.ndarray) -> None:
        """Return a float64 buffer to the pool.

        Args:
            buffer: Buffer to return to the pool
        """
        if buffer.dtype != np.float64:
            return

        size = buffer.size
        with self._lock:
            if size not in self._float64_buffers:
                self._float64_buffers[size] = []

            buffers = self._float64_buffers[size]
            if len(buffers) < self.max_buffers_per_size:
                buffers.append(buffer)
                logger.debug(f"Returned float64 buffer of size {size} to pool")

    def return_float32_buffer(self, buffer: np.ndarray) -> None:
        """Return a float32 buffer to the pool."""
        if buffer.dtype != np.float32:
            return

        size = buffer.size
        with self._lock:
            if size not in self._float32_buffers:
                self._float32_buffers[size] = []

            buffers = self._float32_buffers[size]
            if len(buffers) < self.max_buffers_per_size:
                buffers.append(buffer)
                logger.debug(f"Returned float32 buffer of size {size} to pool")

    def return_uint32_buffer(self, buffer: np.ndarray) -> None:
        """Return a uint32 buffer to the pool."""
        if buffer.dtype != np.uint32:
            return

        size = buffer.size
        with self._lock:
            if size not in self._uint32_buffers:
                self._uint32_buffers[size] = []

            buffers = self._uint32_buffers[size]
            if len(buffers) < self.max_buffers_per_size:
                buffers.append(buffer)
                logger.debug(f"Returned uint32 buffer of size {size} to pool")

    def clear(self) -> None:
        """Clear all buffers from the pool."""
        with self._lock:
            self._float64_buffers.clear()
            self._float32_buffers.clear()
            self._uint32_buffers.clear()
            logger.info("Cleared all buffers from pool")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the buffer pool."""
        with self._lock:
            return {
                "float64_sizes": list(self._float64_buffers.keys()),
                "float64_counts": {k: len(v) for k, v in self._float64_buffers.items()},
                "float32_sizes": list(self._float32_buffers.keys()),
                "float32_counts": {k: len(v) for k, v in self._float32_buffers.items()},
                "uint32_sizes": list(self._uint32_buffers.keys()),
                "uint32_counts": {k: len(v) for k, v in self._uint32_buffers.items()},
            }


class MemoryManager:
    """Comprehensive memory management for the Bruker reader.

    This class provides memory monitoring, buffer management, and memory
    optimization strategies for large dataset processing.
    """

    def __init__(
        self,
        memory_limit_gb: Optional[float] = None,
        buffer_pool_size: int = 10,
    ):
        """Initialize the memory manager.

        Args:
            memory_limit_gb: Optional memory limit in GB
            buffer_pool_size: Maximum buffers per size in the pool
        """
        self.memory_limit_gb = memory_limit_gb
        self.buffer_pool = BufferPool(buffer_pool_size)
        if PSUTIL_AVAILABLE:
            self._process = psutil.Process()
        else:
            self._process = None

        # Memory monitoring
        self._peak_memory_mb = 0.0
        self._total_allocations = 0

        logger.info(f"Initialized MemoryManager with limit: {memory_limit_gb}GB")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics.

        Returns:
            Dictionary with memory usage information
        """
        if not PSUTIL_AVAILABLE or self._process is None:
            # Return basic fallback values when psutil is not available
            return {
                "rss_mb": 0.0,
                "vms_mb": 0.0,
                "peak_mb": self._peak_memory_mb,
                "system_available_gb": 4.0,  # Assume 4GB available
                "system_percent": 50.0,
            }

        memory_info = self._process.memory_info()
        system_memory = psutil.virtual_memory()

        rss_mb = memory_info.rss / 1024 / 1024
        vms_mb = memory_info.vms / 1024 / 1024

        # Update peak memory
        if rss_mb > self._peak_memory_mb:
            self._peak_memory_mb = rss_mb

        return {
            "rss_mb": rss_mb,
            "vms_mb": vms_mb,
            "peak_mb": self._peak_memory_mb,
            "system_available_gb": system_memory.available / 1024 / 1024 / 1024,
            "system_percent": system_memory.percent,
        }

    def check_memory_limit(self) -> bool:
        """Check if we're approaching memory limits.

        Returns:
            True if memory usage is within limits, False otherwise
        """
        if self.memory_limit_gb is None:
            return True

        usage = self.get_memory_usage()
        current_gb = usage["rss_mb"] / 1024

        if current_gb > self.memory_limit_gb:
            logger.warning(
                f"Memory limit exceeded: {current_gb:.2f}GB > {self.memory_limit_gb}GB"
            )
            return False

        return True

    def optimize_memory(self) -> None:
        """Perform memory optimization operations.

        This includes garbage collection and buffer pool cleanup.
        """
        logger.debug("Performing memory optimization")

        # Clear buffer pool if memory pressure is high
        usage = self.get_memory_usage()
        if usage["system_percent"] > 80:  # System memory usage > 80%
            logger.info("High system memory usage detected, clearing buffer pool")
            self.buffer_pool.clear()

        # Force garbage collection
        gc.collect()

        logger.debug("Memory optimization complete")

    def allocate_spectrum_buffers(
        self, estimated_peaks: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate buffers for spectrum data with intelligent sizing.

        Args:
            estimated_peaks: Estimated number of peaks in the spectrum

        Returns:
            Tuple of (mz_buffer, intensity_buffer)
        """
        # Add some padding to estimated size
        buffer_size = max(1024, int(estimated_peaks * 1.2))

        mz_buffer = self.buffer_pool.get_float64_buffer(buffer_size)
        intensity_buffer = self.buffer_pool.get_float32_buffer(buffer_size)

        self._total_allocations += 1

        return mz_buffer, intensity_buffer

    def return_spectrum_buffers(
        self, mz_buffer: np.ndarray, intensity_buffer: np.ndarray
    ) -> None:
        """Return spectrum buffers to the pool.

        Args:
            mz_buffer: m/z buffer to return
            intensity_buffer: Intensity buffer to return
        """
        self.buffer_pool.return_float64_buffer(mz_buffer)
        self.buffer_pool.return_float32_buffer(intensity_buffer)

    def estimate_memory_requirements(
        self, n_spectra: int, avg_peaks_per_spectrum: int, n_unique_masses: int
    ) -> Dict[str, float]:
        """Estimate memory requirements for dataset processing.

        Args:
            n_spectra: Number of spectra in the dataset
            avg_peaks_per_spectrum: Average peaks per spectrum
            n_unique_masses: Number of unique m/z values

        Returns:
            Dictionary with memory estimates in MB
        """
        # Raw data memory (assuming temporary storage)
        raw_data_mb = (
            (n_spectra * avg_peaks_per_spectrum * (8 + 4)) / 1024 / 1024
        )  # 8 bytes for m/z, 4 for intensity

        # Common mass axis memory
        mass_axis_mb = n_unique_masses * 8 / 1024 / 1024

        # Sparse matrix memory (estimated)
        # Assume 10% fill rate for sparse matrix
        sparse_matrix_mb = (n_spectra * n_unique_masses * 0.1 * 4) / 1024 / 1024

        # Coordinate data
        coordinates_mb = n_spectra * 3 * 4 / 1024 / 1024  # 3 coords * 4 bytes each

        # Buffer overhead (estimated)
        buffer_overhead_mb = max(100, raw_data_mb * 0.1)  # 10% overhead, min 100MB

        total_mb = (
            raw_data_mb
            + mass_axis_mb
            + sparse_matrix_mb
            + coordinates_mb
            + buffer_overhead_mb
        )

        return {
            "raw_data_mb": raw_data_mb,
            "mass_axis_mb": mass_axis_mb,
            "sparse_matrix_mb": sparse_matrix_mb,
            "coordinates_mb": coordinates_mb,
            "buffer_overhead_mb": buffer_overhead_mb,
            "total_mb": total_mb,
            "total_gb": total_mb / 1024,
        }

    def suggest_batch_size(
        self,
        total_spectra: int,
        avg_peaks_per_spectrum: int,
        target_memory_mb: float = 512,
    ) -> int:
        """Suggest an optimal batch size based on memory constraints.

        Args:
            total_spectra: Total number of spectra
            avg_peaks_per_spectrum: Average peaks per spectrum
            target_memory_mb: Target memory usage per batch in MB

        Returns:
            Suggested batch size
        """
        # Estimate memory per spectrum
        memory_per_spectrum_mb = (
            (avg_peaks_per_spectrum * 12) / 1024 / 1024
        )  # 12 bytes per peak

        # Calculate batch size to stay within target memory
        batch_size = max(1, int(target_memory_mb / memory_per_spectrum_mb))

        # Ensure reasonable bounds
        batch_size = min(batch_size, total_spectra, 1000)  # Max 1000 spectra per batch
        batch_size = max(batch_size, 10)  # Min 10 spectra per batch

        logger.info(
            f"Suggested batch size: {batch_size} (target: {target_memory_mb}MB)"
        )
        return batch_size

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory manager statistics."""
        memory_usage = self.get_memory_usage()
        buffer_stats = self.buffer_pool.get_stats()

        return {
            "memory_usage": memory_usage,
            "buffer_pool": buffer_stats,
            "total_allocations": self._total_allocations,
            "memory_limit_gb": self.memory_limit_gb,
        }
