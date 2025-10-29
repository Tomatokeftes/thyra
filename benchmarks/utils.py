"""
Utility functions for benchmarking.
"""

import gc
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psutil

try:
    import tracemalloc

    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False


def get_directory_size(path: Path) -> int:
    """
    Get total size of a directory or file in bytes.

    For ImzML, includes both .imzML and .ibd files.
    For Bruker .d, includes all files in the directory.
    """
    total = 0
    if path.is_file():
        total = path.stat().st_size
        # Check for companion .ibd file
        if path.suffix.lower() == ".imzml":
            ibd_path = path.with_suffix(".ibd")
            if ibd_path.exists():
                total += ibd_path.stat().st_size
    elif path.is_dir():
        for item in path.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except (PermissionError, FileNotFoundError):
                    pass
    return total


def bytes_to_mb(bytes_size: int) -> float:
    """Convert bytes to megabytes."""
    return bytes_size / (1024**2)


def bytes_to_gb(bytes_size: int) -> float:
    """Convert bytes to gigabytes."""
    return bytes_size / (1024**3)


@contextmanager
def benchmark_context(name: str, verbose: bool = True, track_memory: bool = False):
    """
    Context manager for timing and memory profiling.

    Args:
        name: Description of the operation being benchmarked
        verbose: Print progress messages
        track_memory: Use tracemalloc for accurate memory tracking (slower)

    Usage:
        with benchmark_context("Operation") as metrics:
            # do work
            pass
        print(metrics['time_sec'])

    Note:
        Memory tracking using RSS can show negative values for short operations
        due to garbage collection. For accurate memory tracking of short
        operations, use track_memory=True (uses tracemalloc, but slower).
    """
    if verbose:
        print(f"  Running: {name}...", end=" ", flush=True)

    gc.collect()
    process = psutil.Process()

    metrics = {}
    start_time = time.time()

    # Memory tracking setup
    mem_before_rss = process.memory_info().rss / (1024**2)

    if track_memory and TRACEMALLOC_AVAILABLE:
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

    try:
        yield metrics
    finally:
        metrics["time_sec"] = time.time() - start_time

        # Get final memory
        mem_after_rss = process.memory_info().rss / (1024**2)
        mem_delta = mem_after_rss - mem_before_rss

        # Store memory metrics
        metrics["mem_before_mb"] = mem_before_rss
        metrics["mem_after_mb"] = mem_after_rss
        metrics["mem_delta_mb"] = mem_delta

        # For backwards compatibility, keep mem_used_mb
        # But use absolute value to avoid confusing negative values
        metrics["mem_used_mb"] = abs(mem_delta)

        # Tracemalloc gives more accurate delta for Python allocations
        if track_memory and TRACEMALLOC_AVAILABLE:
            snapshot_after = tracemalloc.take_snapshot()
            top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
            total_diff = sum(stat.size_diff for stat in top_stats)
            metrics["mem_python_delta_mb"] = total_diff / (1024**2)
            tracemalloc.stop()
        else:
            metrics["mem_python_delta_mb"] = None

        if verbose:
            time_str = f"{metrics['time_sec']:.3f}s"
            # Show actual delta (can be negative)
            if mem_delta >= 0:
                mem_str = f"mem: +{mem_delta:.1f}MB"
            else:
                mem_str = f"mem: {mem_delta:.1f}MB (GC)"
            print(f"{time_str} ({mem_str})")


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


def format_size(bytes_size: int) -> str:
    """Format size in human-readable format."""
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024**2:
        return f"{bytes_size/1024:.1f}KB"
    elif bytes_size < 1024**3:
        return f"{bytes_size/(1024**2):.1f}MB"
    else:
        return f"{bytes_size/(1024**3):.2f}GB"


def calculate_speedup(baseline_time: float, new_time: float) -> float:
    """Calculate speedup factor."""
    if new_time == 0:
        return float("inf")
    return baseline_time / new_time


def save_metrics(metrics: Dict[str, Any], output_path: Path):
    """Save metrics to JSON file."""
    import json

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to: {output_path}")


def load_metrics(input_path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    import json

    with open(input_path, "r") as f:
        return json.load(f)


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate statistical measures for repeated benchmark runs.

    Args:
        values: List of measurements from repeated runs

    Returns:
        Dictionary with mean, std, min, max, median, and coefficient of variation
    """
    if not values:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "cv": 0.0,
            "n": 0,
        }

    arr = np.array(values)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = (std_val / mean_val * 100) if mean_val > 0 else 0.0

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "cv": round(cv, 2),  # Coefficient of variation (%)
        "n": len(values),
    }


def aggregate_repeated_metrics(
    runs: List[Dict[str, Any]], metric_keys: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from multiple benchmark runs.

    Args:
        runs: List of metric dictionaries from repeated runs
        metric_keys: Keys to aggregate (e.g., ['time_sec', 'memory_mb'])

    Returns:
        Dictionary mapping metric names to their statistics
    """
    aggregated = {}

    for key in metric_keys:
        values = []
        for run in runs:
            if key in run and run[key] is not None:
                values.append(run[key])

        if values:
            aggregated[key] = calculate_statistics(values)

    return aggregated


@contextmanager
def repeated_benchmark_context(name: str, n_runs: int = 5, verbose: bool = True):
    """
    Context manager for running benchmarks multiple times and collecting statistics.

    Usage:
        with repeated_benchmark_context("Operation", n_runs=5) as runs:
            for run_id in range(n_runs):
                # Perform operation
                runs.append({'time_sec': 1.23, 'memory_mb': 456})

        # After context, runs contains statistics
        print(runs.get_statistics('time_sec'))
    """
    if verbose:
        print(f"  Running: {name} ({n_runs} runs)...", flush=True)

    class BenchmarkRuns:
        def __init__(self):
            self.runs: List[Dict[str, Any]] = []

        def append(self, metrics: Dict[str, Any]):
            self.runs.append(metrics)

        def get_statistics(self, key: str) -> Dict[str, float]:
            """Get statistics for a specific metric."""
            values = [r[key] for r in self.runs if key in r and r[key] is not None]
            return calculate_statistics(values)

        def get_all_statistics(self, keys: List[str]) -> Dict[str, Dict[str, float]]:
            """Get statistics for multiple metrics."""
            return aggregate_repeated_metrics(self.runs, keys)

    runs = BenchmarkRuns()
    try:
        yield runs
    finally:
        if verbose and runs.runs:
            print(f"    Completed {len(runs.runs)} runs")


def format_metric_with_uncertainty(mean: float, std: float, unit: str = "") -> str:
    """
    Format a metric with uncertainty for display.

    Args:
        mean: Mean value
        std: Standard deviation
        unit: Optional unit string

    Returns:
        Formatted string like "1.23 ± 0.04 s"
    """
    if std == 0:
        return f"{mean:.3f}{unit}"
    return f"{mean:.3f} ± {std:.3f}{unit}"
