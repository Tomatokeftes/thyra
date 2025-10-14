"""
Utility functions for benchmarking.
"""

import gc
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict

import psutil


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
def benchmark_context(name: str, verbose: bool = True):
    """
    Context manager for timing and memory profiling.

    Usage:
        with benchmark_context("Operation") as metrics:
            # do work
            pass
        print(metrics['time_sec'])
    """
    if verbose:
        print(f"  Running: {name}...", end=" ", flush=True)

    gc.collect()
    process = psutil.Process()

    metrics = {}
    metrics["mem_before_mb"] = process.memory_info().rss / (1024**2)
    start_time = time.time()

    try:
        yield metrics
    finally:
        metrics["time_sec"] = time.time() - start_time
        metrics["mem_after_mb"] = process.memory_info().rss / (1024**2)
        metrics["mem_used_mb"] = metrics["mem_after_mb"] - metrics["mem_before_mb"]

        if verbose:
            print(f"{metrics['time_sec']:.3f}s (mem: {metrics['mem_used_mb']:.1f}MB)")


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
