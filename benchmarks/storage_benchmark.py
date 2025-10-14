"""
Storage efficiency benchmarks.

Compares file sizes, compression ratios, and conversion performance
between original formats (ImzML, Bruker) and SpatialData/Zarr.
"""

import gc
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil
from config import BenchmarkConfig, DatasetConfig
from utils import bytes_to_gb, bytes_to_mb, format_size, format_time, get_directory_size

from thyra import convert_msi


class StorageBenchmark:
    """Benchmark storage efficiency and conversion performance."""

    def __init__(self):
        BenchmarkConfig.setup_directories()
        self.results: List[Dict[str, Any]] = []

    def benchmark_conversion(self, dataset: DatasetConfig) -> Dict[str, Any]:
        """
        Benchmark dataset conversion and measure storage metrics.

        Returns metrics including:
        - Input/output file sizes
        - Compression ratio
        - Conversion time and throughput
        - Memory usage
        """
        print(f"\n{'='*70}")
        print(f"STORAGE BENCHMARK: {dataset.name}")
        print(f"{'='*70}")
        print(f"Input: {dataset.path}")
        print(f"Format: {dataset.format_type}")
        if dataset.resampling_config:
            print(f"Resampling: {dataset.resampling_config}")
        print(f"{'='*70}\n")

        # Check if already converted
        output_path = dataset.zarr_path
        if output_path.exists():
            print(f"[SKIP] Already converted: {output_path}")
            return self._measure_existing_conversion(dataset, output_path)

        # Measure input size
        input_size_bytes = get_directory_size(dataset.path)
        input_size_mb = bytes_to_mb(input_size_bytes)

        print(f"Input size: {format_size(input_size_bytes)}")

        # Setup memory profiling
        process = psutil.Process()
        mem_before_mb = process.memory_info().rss / (1024**2)

        # Perform conversion
        gc.collect()
        start_time = time.time()

        try:
            success = convert_msi(
                input_path=str(dataset.path),
                output_path=str(output_path),
                pixel_size_x=1.0,
                pixel_size_y=1.0,
                resampling_config=dataset.resampling_config,
            )

            if not success:
                raise RuntimeError("Conversion failed")

        except Exception as e:
            print(f"[ERROR] Conversion failed: {e}")
            return None

        conversion_time_sec = time.time() - start_time
        mem_after_mb = process.memory_info().rss / (1024**2)
        mem_used_mb = mem_after_mb - mem_before_mb

        # Measure output size
        output_size_bytes = get_directory_size(output_path)
        output_size_mb = bytes_to_mb(output_size_bytes)

        # Calculate metrics
        compression_ratio = (
            input_size_bytes / output_size_bytes if output_size_bytes > 0 else 0
        )
        throughput_mb_per_sec = (
            input_size_mb / conversion_time_sec if conversion_time_sec > 0 else 0
        )

        metrics = {
            "dataset": dataset.name,
            "source_format": dataset.format_type,
            "description": dataset.description,
            "resampling": dataset.resampling_config,
            "input_size_bytes": input_size_bytes,
            "input_size_mb": round(input_size_mb, 2),
            "output_size_bytes": output_size_bytes,
            "output_size_mb": round(output_size_mb, 2),
            "compression_ratio": round(compression_ratio, 2),
            "conversion_time_sec": round(conversion_time_sec, 2),
            "throughput_mb_per_sec": round(throughput_mb_per_sec, 2),
            "peak_memory_mb": round(mem_used_mb, 2),
        }

        print(f"\n[SUCCESS] Conversion complete:")
        print(f"  Output size: {format_size(output_size_bytes)}")
        print(f"  Compression: {compression_ratio:.2f}x")
        print(f"  Time: {format_time(conversion_time_sec)}")
        print(f"  Throughput: {throughput_mb_per_sec:.1f} MB/s")
        print(f"  Peak memory: {mem_used_mb:.1f} MB")

        self.results.append(metrics)
        return metrics

    def _measure_existing_conversion(
        self, dataset: DatasetConfig, output_path: Path
    ) -> Dict[str, Any]:
        """Measure metrics for already-converted dataset."""
        input_size_bytes = get_directory_size(dataset.path)
        output_size_bytes = get_directory_size(output_path)

        compression_ratio = (
            input_size_bytes / output_size_bytes if output_size_bytes > 0 else 0
        )

        metrics = {
            "dataset": dataset.name,
            "source_format": dataset.format_type,
            "description": dataset.description,
            "resampling": dataset.resampling_config,
            "input_size_bytes": input_size_bytes,
            "input_size_mb": round(bytes_to_mb(input_size_bytes), 2),
            "output_size_bytes": output_size_bytes,
            "output_size_mb": round(bytes_to_mb(output_size_bytes), 2),
            "compression_ratio": round(compression_ratio, 2),
            "conversion_time_sec": None,  # Not measured
            "throughput_mb_per_sec": None,
            "peak_memory_mb": None,
        }

        print(f"  Input: {format_size(input_size_bytes)}")
        print(f"  Output: {format_size(output_size_bytes)}")
        print(f"  Compression: {compression_ratio:.2f}x")

        self.results.append(metrics)
        return metrics

    def benchmark_multiple_datasets(
        self, datasets: List[DatasetConfig]
    ) -> List[Dict[str, Any]]:
        """Benchmark multiple datasets."""
        print("\n" + "=" * 70)
        print("STORAGE EFFICIENCY BENCHMARK SUITE")
        print("=" * 70)

        for dataset in datasets:
            if not dataset.path.exists():
                print(f"\n[SKIP] Dataset not found: {dataset.path}")
                continue

            self.benchmark_conversion(dataset)

        return self.results

    def print_summary(self):
        """Print summary table of all results."""
        if not self.results:
            print("\nNo results to summarize.")
            return

        print("\n" + "=" * 80)
        print("STORAGE BENCHMARK SUMMARY")
        print("=" * 80)
        print(
            f"\n{'Dataset':<25} {'Input':<12} {'Output':<12} {'Compression':<12} {'Time':<10}"
        )
        print("-" * 80)

        for r in self.results:
            input_str = format_size(r["input_size_bytes"])
            output_str = format_size(r["output_size_bytes"])
            comp_str = f"{r['compression_ratio']:.2f}x"
            time_str = (
                format_time(r["conversion_time_sec"])
                if r["conversion_time_sec"]
                else "N/A"
            )

            print(
                f"{r['dataset']:<25} {input_str:<12} {output_str:<12} {comp_str:<12} {time_str:<10}"
            )

    def save_results(self):
        """Save results to JSON."""
        from utils import save_metrics

        output_path = BenchmarkConfig.RESULTS_DIR / "storage_benchmark.json"
        save_metrics(self.results, output_path)


def main():
    """Run storage benchmarks on available datasets."""
    from config import DATASETS

    benchmark = StorageBenchmark()

    # Get all available datasets
    available_datasets = [ds for ds in DATASETS.values() if ds.path.exists()]

    if not available_datasets:
        print("No datasets found. Please add data to test_data/ directory.")
        return

    # Run benchmarks
    benchmark.benchmark_multiple_datasets(available_datasets)
    benchmark.print_summary()
    benchmark.save_results()


if __name__ == "__main__":
    main()
