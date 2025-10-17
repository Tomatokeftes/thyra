"""
Main benchmark runner script.

Runs all benchmarks in the correct order and generates visualizations.
"""

import argparse
import sys
from pathlib import Path

from config import DATASETS, BenchmarkConfig


def run_storage_benchmarks():
    """Run storage efficiency benchmarks."""
    print("\n" + "=" * 70)
    print("RUNNING STORAGE BENCHMARKS")
    print("=" * 70)

    from storage_benchmark import StorageBenchmark

    benchmark = StorageBenchmark()
    available_datasets = [ds for ds in DATASETS.values() if ds.path.exists()]

    if not available_datasets:
        print("[SKIP] No datasets found")
        return False

    benchmark.benchmark_multiple_datasets(available_datasets)
    benchmark.print_summary()
    benchmark.save_results()

    return True


def run_spatial_access_benchmarks():
    """Run spatial access pattern benchmarks."""
    print("\n" + "=" * 70)
    print("RUNNING SPATIAL ACCESS BENCHMARKS")
    print("=" * 70)

    from spatial_access_benchmark import SpatialAccessBenchmark

    benchmark = SpatialAccessBenchmark()
    available_datasets = [ds for ds in DATASETS.values() if ds.path.exists()]

    if not available_datasets:
        print("[SKIP] No datasets found")
        return False

    for dataset in available_datasets:
        benchmark.compare_dataset(dataset)

    benchmark.print_summary()
    benchmark.save_results()

    return True


def run_spectral_access_benchmarks():
    """Run spectral (m/z) access pattern benchmarks."""
    print("\n" + "=" * 70)
    print("RUNNING SPECTRAL ACCESS BENCHMARKS")
    print("=" * 70)

    from spectral_access_benchmark import SpectralAccessBenchmark

    benchmark = SpectralAccessBenchmark()
    available_datasets = [ds for ds in DATASETS.values() if ds.path.exists()]

    if not available_datasets:
        print("[SKIP] No datasets found")
        return False

    for dataset in available_datasets:
        benchmark.compare_dataset(dataset)

    benchmark.print_summary()
    benchmark.save_results()

    return True


def run_parallel_benchmarks():
    """Run parallel processing benchmarks."""
    print("\n" + "=" * 70)
    print("RUNNING PARALLEL PROCESSING BENCHMARKS")
    print("=" * 70)

    from parallel_benchmark import ParallelBenchmark

    benchmark = ParallelBenchmark()
    available_datasets = [
        ds for ds in DATASETS.values() if ds.path.exists() and ds.zarr_path.exists()
    ]

    if not available_datasets:
        print("[SKIP] No converted datasets found. Run other benchmarks first.")
        return False

    for dataset in available_datasets:
        benchmark.benchmark_dataset_scalability(dataset)

    benchmark.print_summary()
    benchmark.save_results()

    return True


def run_bruker_interpolation_benchmark():
    """Run Bruker interpolation comparison benchmark."""
    print("\n" + "=" * 70)
    print("RUNNING BRUKER INTERPOLATION BENCHMARK")
    print("=" * 70)

    from bruker_interpolation_benchmark import BrukerInterpolationBenchmark

    # Find raw and resampled Bruker datasets
    raw_dataset = None
    resampled_dataset = None

    for ds in DATASETS.values():
        if ds.format_type == "bruker":
            if ds.resampling_config is None:
                raw_dataset = ds
            elif (
                ds.resampling_config
                and "target_bins" in ds.resampling_config.get("params", {})
                and ds.resampling_config["params"]["target_bins"] == 300000
            ):
                resampled_dataset = ds

    if not raw_dataset or not resampled_dataset:
        print("[SKIP] Bruker raw and resampled datasets not configured")
        return False

    if not raw_dataset.path.exists():
        print(f"[SKIP] Bruker data not found: {raw_dataset.path}")
        return False

    benchmark = BrukerInterpolationBenchmark()
    benchmark.compare_datasets(raw_dataset, resampled_dataset)
    benchmark.print_summary()
    benchmark.save_results()

    return True


def generate_visualizations():
    """Generate all plots."""
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    from visualize import BenchmarkVisualizer

    visualizer = BenchmarkVisualizer()
    visualizer.create_all_plots()

    return True


def main():
    """Run all benchmarks and generate visualizations."""
    parser = argparse.ArgumentParser(description="Run Thyra benchmarks for paper")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=["storage", "spatial", "spectral", "parallel", "bruker", "all"],
        default=["all"],
        help="Which benchmarks to run",
    )
    parser.add_argument(
        "--skip-viz", action="store_true", help="Skip visualization generation"
    )

    args = parser.parse_args()

    # Setup directories
    BenchmarkConfig.setup_directories()

    print("\n" + "=" * 70)
    print("THYRA BENCHMARK SUITE")
    print("=" * 70)
    print("\nComprehensive benchmarks comparing:")
    print("  - ImzML vs SpatialData/Zarr")
    print("  - Bruker .d vs SpatialData/Zarr")
    print("  - Bruker raw vs resampled (300k bins)")
    print("\nBenchmark categories:")
    print("  1. Storage efficiency")
    print("  2. Spatial access patterns")
    print("  3. Spectral (m/z) access patterns")
    print("  4. Parallel processing scalability")
    print("  5. Bruker interpolation comparison")
    print("=" * 70)

    benchmarks_to_run = args.benchmarks
    if "all" in benchmarks_to_run:
        benchmarks_to_run = ["storage", "spatial", "spectral", "parallel", "bruker"]

    success_count = 0
    total_count = 0

    # Run selected benchmarks
    if "storage" in benchmarks_to_run:
        total_count += 1
        if run_storage_benchmarks():
            success_count += 1

    if "spatial" in benchmarks_to_run:
        total_count += 1
        if run_spatial_access_benchmarks():
            success_count += 1

    if "spectral" in benchmarks_to_run:
        total_count += 1
        if run_spectral_access_benchmarks():
            success_count += 1

    if "parallel" in benchmarks_to_run:
        total_count += 1
        if run_parallel_benchmarks():
            success_count += 1

    if "bruker" in benchmarks_to_run:
        total_count += 1
        if run_bruker_interpolation_benchmark():
            success_count += 1

    # Generate visualizations
    if not args.skip_viz:
        if generate_visualizations():
            print("\nVisualizations generated successfully")

    # Final summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUITE COMPLETE")
    print("=" * 70)
    print(f"\nCompleted: {success_count}/{total_count} benchmark categories")
    print(f"\nResults saved to: {BenchmarkConfig.RESULTS_DIR}")
    print(f"Plots saved to: {BenchmarkConfig.PLOTS_DIR}")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
