"""
Batch processing utilities for efficient handling of large datasets.

This module provides utilities for processing large datasets in manageable
batches with progress tracking and memory management.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BatchInfo:
    """Information about a processing batch."""

    batch_id: int
    start_index: int
    end_index: int
    size: int
    estimated_memory_mb: float


class BatchProcessor:
    """
    Efficient batch processor for large spectrum datasets.

    This class provides intelligent batching strategies based on memory
    constraints and processing requirements.
    """

    def __init__(
        self,
        target_memory_mb: float = 512,
        min_batch_size: int = 10,
        max_batch_size: int = 1000,
        progress_callback: Optional[Callable] = None,
    ):
        """
        Initialize the batch processor.

        Args:
            target_memory_mb: Target memory usage per batch
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            progress_callback: Optional callback for progress updates
        """
        self.target_memory_mb = target_memory_mb
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.progress_callback = progress_callback

        # Statistics
        self._stats = {
            "total_batches": 0,
            "total_items_processed": 0,
            "average_batch_size": 0.0,
            "average_processing_time_ms": 0.0,
            "peak_memory_mb": 0.0,
        }

        logger.info(
            f"Initialized BatchProcessor (target: {target_memory_mb}MB, "
            f"batch size: {min_batch_size}-{max_batch_size})"
        )

    def calculate_optimal_batch_size(
        self, total_items: int, avg_item_size_bytes: float
    ) -> int:
        """
        Calculate optimal batch size based on memory constraints.

        Args:
            total_items: Total number of items to process
            avg_item_size_bytes: Average size per item in bytes

        Returns:
            Optimal batch size
        """
        # Calculate memory per item in MB
        item_size_mb = avg_item_size_bytes / 1024 / 1024

        # Calculate batch size to stay within target memory
        if item_size_mb > 0:
            batch_size = int(self.target_memory_mb / item_size_mb)
        else:
            batch_size = self.max_batch_size

        # Apply bounds
        batch_size = max(self.min_batch_size, min(batch_size, self.max_batch_size))
        batch_size = min(batch_size, total_items)

        logger.debug(
            f"Calculated optimal batch size: {batch_size} "
            f"(item size: {item_size_mb:.3f}MB)"
        )

        return batch_size

    def create_batches(
        self, total_items: int, batch_size: Optional[int] = None
    ) -> List[BatchInfo]:
        """
        Create batch information for processing.

        Args:
            total_items: Total number of items to process
            batch_size: Optional fixed batch size

        Returns:
            List of BatchInfo objects
        """
        if batch_size is None:
            # Use a reasonable default if no size provided
            batch_size = min(
                self.max_batch_size, max(self.min_batch_size, total_items // 10)
            )

        batches = []
        batch_id = 0

        for start_idx in range(0, total_items, batch_size):
            end_idx = min(start_idx + batch_size, total_items)
            actual_size = end_idx - start_idx

            batch_info = BatchInfo(
                batch_id=batch_id,
                start_index=start_idx,
                end_index=end_idx,
                size=actual_size,
                estimated_memory_mb=0.0,  # Will be updated during processing
            )

            batches.append(batch_info)
            batch_id += 1

        logger.info(
            f"Created {len(batches)} batches for {total_items} items "
            f"(avg size: {total_items / len(batches):.1f})"
        )

        return batches

    def process_spectrum_batches(
        self,
        spectrum_iterator: Iterator,
        total_spectra: int,
        processor_func: Callable,
        batch_size: Optional[int] = None,
    ) -> Iterator[Any]:
        """
        Process spectra in batches with progress tracking.

        Args:
            spectrum_iterator: Iterator yielding spectrum data
            total_spectra: Total number of spectra
            processor_func: Function to process each batch
            batch_size: Optional batch size

        Yields:
            Results from processor_func for each batch
        """
        if batch_size is None:
            # Estimate batch size based on typical spectrum size
            avg_spectrum_size = 1000 * 12  # 1000 peaks * 12 bytes per peak
            batch_size = self.calculate_optimal_batch_size(
                total_spectra, avg_spectrum_size
            )

        batches = self.create_batches(total_spectra, batch_size)

        # Setup progress tracking
        pbar = tqdm(
            total=total_spectra, 
            desc="Processing batches", 
            unit="spectrum",
            disable=getattr(self, "_quiet_mode", False)
        )

        try:
            current_batch = []
            current_batch_idx = 0
            spectrum_count = 0

            for spectrum_data in spectrum_iterator:
                current_batch.append(spectrum_data)
                spectrum_count += 1

                # Check if batch is complete
                if len(current_batch) >= batch_size or spectrum_count >= total_spectra:
                    # Process the batch
                    if current_batch_idx < len(batches):
                        batch_info = batches[current_batch_idx]
                        batch_info.size = len(current_batch)
                    else:
                        batch_info = BatchInfo(
                            batch_id=current_batch_idx,
                            start_index=spectrum_count - len(current_batch),
                            end_index=spectrum_count,
                            size=len(current_batch),
                            estimated_memory_mb=0.0,
                        )

                    # Process batch
                    result = processor_func(current_batch, batch_info)
                    yield result

                    # Update statistics
                    self._stats["total_batches"] += 1
                    self._stats["total_items_processed"] += len(current_batch)

                    # Update progress
                    pbar.update(len(current_batch))

                    # Progress callback
                    if self.progress_callback:
                        self.progress_callback(spectrum_count, total_spectra)

                    # Reset for next batch
                    current_batch = []
                    current_batch_idx += 1

                # Break if we've processed all spectra
                if spectrum_count >= total_spectra:
                    break

        finally:
            pbar.close()

        # Update final statistics
        if self._stats["total_batches"] > 0:
            self._stats["average_batch_size"] = (
                self._stats["total_items_processed"] / self._stats["total_batches"]
            )

    def adaptive_batch_processing(
        self,
        spectrum_iterator: Iterator,
        total_spectra: int,
        processor_func: Callable,
        initial_batch_size: Optional[int] = None,
    ) -> Iterator[Any]:
        """
        Adaptive batch processing that adjusts batch size based on performance.

        Args:
            spectrum_iterator: Iterator yielding spectrum data
            total_spectra: Total number of spectra
            processor_func: Function to process each batch
            initial_batch_size: Initial batch size

        Yields:
            Results from processor_func for each batch
        """
        import time

        if initial_batch_size is None:
            batch_size = min(50, max(10, total_spectra // 100))
        else:
            batch_size = initial_batch_size

        pbar = tqdm(
            total=total_spectra, 
            desc="Adaptive processing", 
            unit="spectrum",
            disable=getattr(self, "_quiet_mode", False)
        )

        try:
            current_batch = []
            spectrum_count = 0
            batch_times = []

            for spectrum_data in spectrum_iterator:
                current_batch.append(spectrum_data)
                spectrum_count += 1

                # Process batch when it reaches target size
                if len(current_batch) >= batch_size or spectrum_count >= total_spectra:
                    # Time the batch processing
                    start_time = time.time()

                    batch_info = BatchInfo(
                        batch_id=len(batch_times),
                        start_index=spectrum_count - len(current_batch),
                        end_index=spectrum_count,
                        size=len(current_batch),
                        estimated_memory_mb=0.0,
                    )

                    result = processor_func(current_batch, batch_info)
                    yield result

                    # Record timing
                    batch_time = time.time() - start_time
                    batch_times.append(batch_time)

                    # Adaptive batch size adjustment
                    if len(batch_times) >= 3:  # Adjust after a few batches
                        avg_time = np.mean(batch_times[-3:])

                        # Adjust batch size based on performance
                        if avg_time < 0.5:  # Fast processing, increase batch size
                            batch_size = min(batch_size + 10, self.max_batch_size)
                        elif avg_time > 2.0:  # Slow processing, decrease batch size
                            batch_size = max(batch_size - 10, self.min_batch_size)

                        logger.debug(
                            f"Adjusted batch size to {batch_size} "
                            f"(avg time: {avg_time:.2f}s)"
                        )

                    # Update progress
                    pbar.update(len(current_batch))

                    # Progress callback
                    if self.progress_callback:
                        self.progress_callback(spectrum_count, total_spectra)

                    # Reset for next batch
                    current_batch = []

                if spectrum_count >= total_spectra:
                    break

        finally:
            pbar.close()

    def process_with_memory_monitoring(
        self, items: List[Any], processor_func: Callable, memory_limit_mb: float = None
    ) -> Iterator[Any]:
        """
        Process items with memory monitoring and adaptive batch sizing.

        Args:
            items: List of items to process
            processor_func: Function to process each batch
            memory_limit_mb: Optional memory limit override

        Yields:
            Results from processor_func for each batch
        """
        try:
            import psutil

            PSUTIL_AVAILABLE = True
        except ImportError:
            PSUTIL_AVAILABLE = False
            psutil = None

        if memory_limit_mb is None:
            memory_limit_mb = self.target_memory_mb

        if PSUTIL_AVAILABLE:
            process = psutil.Process()
        else:
            process = None
        total_items = len(items)
        batch_size = self.min_batch_size
        processed = 0

        pbar = tqdm(
            total=total_items, 
            desc="Memory-aware processing", 
            unit="item",
            disable=getattr(self, "_quiet_mode", False)
        )

        try:
            while processed < total_items:
                # Check current memory usage
                if PSUTIL_AVAILABLE and process:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                else:
                    memory_mb = 100.0  # Fallback value

                # Adjust batch size based on memory usage
                if memory_mb > memory_limit_mb * 0.8:  # 80% of limit
                    batch_size = max(self.min_batch_size, batch_size // 2)
                    logger.warning(
                        f"High memory usage ({memory_mb:.1f}MB), "
                        f"reducing batch size to {batch_size}"
                    )
                elif memory_mb < memory_limit_mb * 0.4:  # 40% of limit
                    batch_size = min(self.max_batch_size, batch_size * 2)

                # Create batch
                end_idx = min(processed + batch_size, total_items)
                batch = items[processed:end_idx]

                if not batch:
                    break

                # Process batch
                batch_info = BatchInfo(
                    batch_id=processed // batch_size,
                    start_index=processed,
                    end_index=end_idx,
                    size=len(batch),
                    estimated_memory_mb=memory_mb,
                )

                result = processor_func(batch, batch_info)
                yield result

                # Update statistics
                processed += len(batch)
                pbar.update(len(batch))

                # Update peak memory
                if memory_mb > self._stats["peak_memory_mb"]:
                    self._stats["peak_memory_mb"] = memory_mb

        finally:
            pbar.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._stats = {
            "total_batches": 0,
            "total_items_processed": 0,
            "average_batch_size": 0.0,
            "average_processing_time_ms": 0.0,
            "peak_memory_mb": 0.0,
        }
