"""
Spatial access pattern benchmarks.

Tests access patterns commonly used in MSI workflows:
- Full dataset iteration (sequential)
- Region of Interest (ROI) extraction
- Single pixel access (random)
- Spatial slicing (rows, columns)
"""

import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import spatialdata as sd
from config import BenchmarkConfig, DatasetConfig
from utils import benchmark_context


class SpatialAccessBenchmark:
    """Benchmark spatial access patterns."""

    def __init__(self):
        BenchmarkConfig.setup_directories()
        self.results: List[Dict[str, Any]] = []

    def benchmark_imzml_access(self, dataset: DatasetConfig) -> Dict[str, float]:
        """Benchmark spatial access patterns for ImzML format."""
        print(f"\n{'='*70}")
        print(f"SPATIAL ACCESS: {dataset.name} (ImzML)")
        print(f"{'='*70}\n")

        from pyimzml.ImzMLParser import ImzMLParser

        metrics = {}

        # Sequential access - all spectra
        with benchmark_context("Sequential access (all spectra)") as m:
            parser = ImzMLParser(str(dataset.path))
            count = 0
            for idx, (x, y, z) in enumerate(parser.coordinates):
                mzs, intensities = parser.getspectrum(idx)
                count += 1
            metrics["sequential_access_sec"] = m["time_sec"]
            metrics["sequential_access_count"] = count

        # ROI access
        del parser
        with benchmark_context(
            f"ROI access (first {BenchmarkConfig.N_RANDOM_PIXELS} spectra)"
        ) as m:
            parser = ImzMLParser(str(dataset.path))
            n_pixels = min(BenchmarkConfig.N_RANDOM_PIXELS, len(parser.coordinates))
            for idx in range(n_pixels):
                mzs, intensities = parser.getspectrum(idx)
            metrics["roi_access_sec"] = m["time_sec"]
            metrics["roi_access_count"] = n_pixels

        # Random access
        del parser
        with benchmark_context(
            f"Random pixel access ({BenchmarkConfig.N_RANDOM_PIXELS} pixels)"
        ) as m:
            parser = ImzMLParser(str(dataset.path))
            random.seed(42)
            n_pixels = min(BenchmarkConfig.N_RANDOM_PIXELS, len(parser.coordinates))
            random_indices = random.sample(range(len(parser.coordinates)), n_pixels)
            for idx in random_indices:
                mzs, intensities = parser.getspectrum(idx)
            metrics["random_access_sec"] = m["time_sec"]
            metrics["random_access_count"] = n_pixels

        del parser
        return metrics

    def benchmark_bruker_access(self, dataset: DatasetConfig) -> Dict[str, float]:
        """Benchmark spatial access patterns for Bruker .d format."""
        print(f"\n{'='*70}")
        print(f"SPATIAL ACCESS: {dataset.name} (Bruker)")
        print(f"{'='*70}\n")

        from thyra.readers.bruker.bruker_reader import BrukerReader

        metrics = {}

        # Sequential access - all spectra
        with benchmark_context("Sequential access (all spectra)") as m:
            reader = BrukerReader(str(dataset.path))
            count = 0
            for coords, mzs, intensities in reader.iter_spectra():
                count += 1
            metrics["sequential_access_sec"] = m["time_sec"]
            metrics["sequential_access_count"] = count

        # ROI access - first N spectra
        del reader
        with benchmark_context(
            f"ROI access (first {BenchmarkConfig.N_RANDOM_PIXELS} spectra)"
        ) as m:
            reader = BrukerReader(str(dataset.path))
            count = 0
            for coords, mzs, intensities in reader.iter_spectra():
                count += 1
                if count >= BenchmarkConfig.N_RANDOM_PIXELS:
                    break
            metrics["roi_access_sec"] = m["time_sec"]
            metrics["roi_access_count"] = count

        # Random access simulation (Bruker doesn't support true random access)
        # We iterate and only process specific indices
        del reader
        with benchmark_context(f"Random pixel access (simulated)") as m:
            reader = BrukerReader(str(dataset.path))
            random.seed(42)
            total_spectra = reader.n_spectra
            n_pixels = min(BenchmarkConfig.N_RANDOM_PIXELS, total_spectra)
            # Sample from first 1000 for efficiency
            random_indices = set(
                random.sample(range(min(1000, total_spectra)), n_pixels)
            )
            count = 0
            accessed = 0
            for coords, mzs, intensities in reader.iter_spectra():
                if count in random_indices:
                    accessed += 1
                count += 1
                if count >= min(1000, total_spectra):
                    break
            metrics["random_access_sec"] = m["time_sec"]
            metrics["random_access_count"] = accessed

        del reader
        return metrics

    def benchmark_spatialdata_access(self, dataset: DatasetConfig) -> Dict[str, float]:
        """Benchmark spatial access patterns for SpatialData/Zarr format."""
        print(f"\n{'='*70}")
        print(f"SPATIAL ACCESS: {dataset.name} (SpatialData/Zarr)")
        print(f"{'='*70}\n")

        zarr_path = dataset.zarr_path

        if not zarr_path.exists():
            print(f"[SKIP] Zarr file not found: {zarr_path}")
            return {}

        metrics = {}

        # Sequential access - full dataset
        with benchmark_context("Sequential access (full dataset)") as m:
            sdata = sd.read_zarr(str(zarr_path))
            msi_data = list(sdata.images.values())[0]
            # Force computation
            _ = msi_data.compute()
            n_pixels = msi_data.shape[1] * msi_data.shape[2]
            metrics["sequential_access_sec"] = m["time_sec"]
            metrics["sequential_access_count"] = n_pixels
            del sdata, msi_data

        # ROI access
        with benchmark_context(f"ROI access ({BenchmarkConfig.ROI_SIZE})") as m:
            sdata = sd.read_zarr(str(zarr_path))
            msi_data = list(sdata.images.values())[0]

            # Extract a spatial ROI
            roi_y = min(BenchmarkConfig.ROI_SIZE[0], msi_data.shape[1])
            roi_x = min(BenchmarkConfig.ROI_SIZE[1], msi_data.shape[2])

            roi_data = msi_data[:, :roi_y, :roi_x].compute()
            n_pixels = roi_y * roi_x
            metrics["roi_access_sec"] = m["time_sec"]
            metrics["roi_access_count"] = n_pixels
            del sdata, msi_data, roi_data

        # Random pixel access
        with benchmark_context(
            f"Random pixel access ({BenchmarkConfig.N_RANDOM_PIXELS} pixels)"
        ) as m:
            sdata = sd.read_zarr(str(zarr_path))
            msi_data = list(sdata.images.values())[0]

            random.seed(42)
            max_y = msi_data.shape[1]
            max_x = msi_data.shape[2]
            n_pixels = min(BenchmarkConfig.N_RANDOM_PIXELS, max_y * max_x)

            random_coords = [
                (random.randint(0, max_y - 1), random.randint(0, max_x - 1))
                for _ in range(n_pixels)
            ]

            for y, x in random_coords:
                _ = msi_data[:, y, x].compute()

            metrics["random_access_sec"] = m["time_sec"]
            metrics["random_access_count"] = n_pixels
            del sdata, msi_data

        return metrics

    def compare_dataset(self, dataset: DatasetConfig) -> Dict[str, Any]:
        """Compare spatial access patterns across all formats for a dataset."""
        result = {"dataset": dataset.name, "description": dataset.description}

        # Benchmark original format
        if dataset.format_type == "imzml":
            orig_metrics = self.benchmark_imzml_access(dataset)
            result["imzml"] = orig_metrics
        elif dataset.format_type == "bruker":
            orig_metrics = self.benchmark_bruker_access(dataset)
            result["bruker"] = orig_metrics

        # Benchmark SpatialData
        sd_metrics = self.benchmark_spatialdata_access(dataset)
        result["spatialdata"] = sd_metrics

        # Calculate speedups
        if orig_metrics and sd_metrics:
            result["speedups"] = {}
            for key in ["sequential_access_sec", "roi_access_sec", "random_access_sec"]:
                if key in orig_metrics and key in sd_metrics:
                    orig_time = orig_metrics[key]
                    sd_time = sd_metrics[key]
                    if sd_time > 0:
                        result["speedups"][key] = round(orig_time / sd_time, 2)

        self.results.append(result)
        return result

    def print_summary(self):
        """Print summary of spatial access benchmarks."""
        if not self.results:
            print("\nNo results to summarize.")
            return

        print("\n" + "=" * 80)
        print("SPATIAL ACCESS BENCHMARK SUMMARY")
        print("=" * 80)

        for result in self.results:
            print(f"\nDataset: {result['dataset']}")
            print("-" * 80)

            # Print original format metrics
            orig_key = "imzml" if "imzml" in result else "bruker"
            if orig_key in result:
                print(f"\n{orig_key.upper()} (original):")
                for key, val in result[orig_key].items():
                    if "sec" in key:
                        print(f"  {key}: {val:.3f}s")
                    elif "count" in key:
                        print(f"  {key}: {val}")

            # Print SpatialData metrics
            if "spatialdata" in result:
                print("\nSpatialData/Zarr:")
                for key, val in result["spatialdata"].items():
                    if "sec" in key:
                        print(f"  {key}: {val:.3f}s")
                    elif "count" in key:
                        print(f"  {key}: {val}")

            # Print speedups
            if "speedups" in result:
                print("\nSpeedups (SpatialData vs original):")
                for key, speedup in result["speedups"].items():
                    operation = key.replace("_sec", "").replace("_", " ").title()
                    print(f"  {operation}: {speedup:.2f}x")

    def save_results(self):
        """Save results to JSON."""
        from utils import save_metrics

        output_path = BenchmarkConfig.RESULTS_DIR / "spatial_access_benchmark.json"
        save_metrics(self.results, output_path)


def main():
    """Run spatial access benchmarks."""
    from config import DATASETS

    benchmark = SpatialAccessBenchmark()

    # Get available datasets
    available_datasets = [ds for ds in DATASETS.values() if ds.path.exists()]

    if not available_datasets:
        print("No datasets found.")
        return

    print("\n" + "=" * 70)
    print("SPATIAL ACCESS PATTERN BENCHMARK SUITE")
    print("=" * 70)

    for dataset in available_datasets:
        benchmark.compare_dataset(dataset)

    benchmark.print_summary()
    benchmark.save_results()


if __name__ == "__main__":
    main()
