"""
Parallel processing benchmarks.

Tests Dask-based parallel computation capabilities of SpatialData/Zarr:
- Parallel chunk reading
- Concurrent ion image extraction
- Parallel normalization operations
- Scalability with different worker counts
"""

import gc
import time
from pathlib import Path
from typing import Any, Dict, List

import dask.array as da
import numpy as np
import spatialdata as sd
from config import BenchmarkConfig, DatasetConfig
from dask.distributed import Client, LocalCluster
from utils import benchmark_context


class ParallelBenchmark:
    """Benchmark parallel processing capabilities."""

    def __init__(self):
        BenchmarkConfig.setup_directories()
        self.results: List[Dict[str, Any]] = []

    def benchmark_parallel_ion_images(
        self, dataset: DatasetConfig, n_workers: int
    ) -> Dict[str, float]:
        """
        Benchmark parallel ion image extraction.

        Extract multiple ion images concurrently using Dask.
        """
        zarr_path = dataset.zarr_path

        if not zarr_path.exists():
            return {}

        print(f"\n  Workers: {n_workers}")

        # Setup Dask cluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=True,
            memory_limit="2GB",
        )
        client = Client(cluster)

        try:
            with benchmark_context(
                f"Extract {BenchmarkConfig.N_ION_IMAGES} ion images"
            ) as m:
                sdata = sd.read_zarr(str(zarr_path))
                msi_data = list(sdata.images.values())[0]
                mz_coords = msi_data.coords["c"].values

                # Extract multiple ion images in parallel
                ion_images = []
                target_mz_list = BenchmarkConfig.TARGET_MZ_VALUES[
                    : BenchmarkConfig.N_ION_IMAGES
                ]

                for target_mz in target_mz_list:
                    mz_mask = (
                        np.abs(mz_coords - target_mz) <= BenchmarkConfig.MZ_TOLERANCE
                    )
                    mz_indices = np.where(mz_mask)[0]

                    if len(mz_indices) > 0:
                        # Lazy computation - queued for parallel execution
                        ion_image = msi_data[mz_indices, :, :].sum(axis=0)
                        ion_images.append(ion_image)

                # Compute all in parallel
                if ion_images:
                    computed = da.compute(*ion_images)

                time_sec = m["time_sec"]
                del sdata, msi_data

        finally:
            client.close()
            cluster.close()

        return {
            "n_workers": n_workers,
            "time_sec": time_sec,
            "n_images": len(target_mz_list),
        }

    def benchmark_parallel_normalization(
        self, dataset: DatasetConfig, n_workers: int
    ) -> Dict[str, float]:
        """
        Benchmark parallel TIC normalization.

        Compute Total Ion Count normalization across spatial dimensions.
        """
        zarr_path = dataset.zarr_path

        if not zarr_path.exists():
            return {}

        # Setup Dask cluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=True,
            memory_limit="2GB",
        )
        client = Client(cluster)

        try:
            with benchmark_context("TIC normalization") as m:
                sdata = sd.read_zarr(str(zarr_path))
                msi_data = list(sdata.images.values())[0]

                # Compute TIC for each pixel (sum across m/z axis)
                tic = msi_data.sum(axis=0)

                # Normalize by TIC
                normalized = msi_data / (tic + 1e-10)

                # Force computation
                _ = normalized.compute()

                time_sec = m["time_sec"]
                del sdata, msi_data, tic, normalized

        finally:
            client.close()
            cluster.close()

        return {"n_workers": n_workers, "time_sec": time_sec}

    def benchmark_parallel_mz_slicing(
        self, dataset: DatasetConfig, n_workers: int
    ) -> Dict[str, float]:
        """
        Benchmark parallel m/z range extraction.

        Extract multiple m/z ranges concurrently.
        """
        zarr_path = dataset.zarr_path

        if not zarr_path.exists():
            return {}

        # Setup Dask cluster
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=True,
            memory_limit="2GB",
        )
        client = Client(cluster)

        try:
            with benchmark_context(
                f"Extract {len(BenchmarkConfig.MZ_RANGES)} m/z ranges"
            ) as m:
                sdata = sd.read_zarr(str(zarr_path))
                msi_data = list(sdata.images.values())[0]
                mz_coords = msi_data.coords["c"].values

                # Extract multiple m/z ranges in parallel
                mz_slices = []

                for mz_min, mz_max in BenchmarkConfig.MZ_RANGES:
                    mz_mask = (mz_coords >= mz_min) & (mz_coords <= mz_max)
                    mz_indices = np.where(mz_mask)[0]

                    if len(mz_indices) > 0:
                        mz_slice = msi_data[mz_indices, :, :].sum(axis=0)
                        mz_slices.append(mz_slice)

                # Compute all in parallel
                if mz_slices:
                    computed = da.compute(*mz_slices)

                time_sec = m["time_sec"]
                del sdata, msi_data

        finally:
            client.close()
            cluster.close()

        return {
            "n_workers": n_workers,
            "time_sec": time_sec,
            "n_ranges": len(BenchmarkConfig.MZ_RANGES),
        }

    def benchmark_dataset_scalability(self, dataset: DatasetConfig) -> Dict[str, Any]:
        """
        Benchmark scalability across different worker counts.

        Tests how well SpatialData/Zarr scales with parallel workers.
        """
        print(f"\n{'='*70}")
        print(f"PARALLEL SCALABILITY: {dataset.name}")
        print(f"{'='*70}\n")

        zarr_path = dataset.zarr_path

        if not zarr_path.exists():
            print(f"[SKIP] Zarr file not found: {zarr_path}")
            return {}

        result = {
            "dataset": dataset.name,
            "description": dataset.description,
            "ion_images": [],
            "normalization": [],
            "mz_slicing": [],
        }

        # Test each worker count
        for n_workers in BenchmarkConfig.WORKER_COUNTS:
            print(f"\nTesting with {n_workers} worker(s):")
            print("-" * 70)

            # Ion image extraction
            print("  Ion image extraction:")
            ion_metrics = self.benchmark_parallel_ion_images(dataset, n_workers)
            if ion_metrics:
                result["ion_images"].append(ion_metrics)

            gc.collect()
            time.sleep(1)  # Let system stabilize

            # Normalization
            print("  TIC normalization:")
            norm_metrics = self.benchmark_parallel_normalization(dataset, n_workers)
            if norm_metrics:
                result["normalization"].append(norm_metrics)

            gc.collect()
            time.sleep(1)

            # M/z slicing
            print("  M/z range extraction:")
            mz_metrics = self.benchmark_parallel_mz_slicing(dataset, n_workers)
            if mz_metrics:
                result["mz_slicing"].append(mz_metrics)

            gc.collect()
            time.sleep(1)

        # Calculate speedups relative to single worker
        if result["ion_images"]:
            baseline_time = result["ion_images"][0]["time_sec"]
            result["ion_images_speedups"] = [
                {
                    "n_workers": m["n_workers"],
                    "speedup": round(baseline_time / m["time_sec"], 2),
                }
                for m in result["ion_images"]
            ]

        if result["normalization"]:
            baseline_time = result["normalization"][0]["time_sec"]
            result["normalization_speedups"] = [
                {
                    "n_workers": m["n_workers"],
                    "speedup": round(baseline_time / m["time_sec"], 2),
                }
                for m in result["normalization"]
            ]

        if result["mz_slicing"]:
            baseline_time = result["mz_slicing"][0]["time_sec"]
            result["mz_slicing_speedups"] = [
                {
                    "n_workers": m["n_workers"],
                    "speedup": round(baseline_time / m["time_sec"], 2),
                }
                for m in result["mz_slicing"]
            ]

        self.results.append(result)
        return result

    def print_summary(self):
        """Print summary of parallel benchmarks."""
        if not self.results:
            print("\nNo results to summarize.")
            return

        print("\n" + "=" * 80)
        print("PARALLEL PROCESSING BENCHMARK SUMMARY")
        print("=" * 80)

        for result in self.results:
            print(f"\nDataset: {result['dataset']}")
            print("-" * 80)

            # Ion image extraction scalability
            if "ion_images_speedups" in result:
                print("\nIon Image Extraction Speedups:")
                for item in result["ion_images_speedups"]:
                    print(f"  {item['n_workers']} worker(s): {item['speedup']:.2f}x")

            # Normalization scalability
            if "normalization_speedups" in result:
                print("\nTIC Normalization Speedups:")
                for item in result["normalization_speedups"]:
                    print(f"  {item['n_workers']} worker(s): {item['speedup']:.2f}x")

            # M/z slicing scalability
            if "mz_slicing_speedups" in result:
                print("\nM/z Range Extraction Speedups:")
                for item in result["mz_slicing_speedups"]:
                    print(f"  {item['n_workers']} worker(s): {item['speedup']:.2f}x")

    def save_results(self):
        """Save results to JSON."""
        from utils import save_metrics

        output_path = BenchmarkConfig.RESULTS_DIR / "parallel_benchmark.json"
        save_metrics(self.results, output_path)


def main():
    """Run parallel processing benchmarks."""
    from config import DATASETS

    benchmark = ParallelBenchmark()

    # Get available datasets (only those already converted to Zarr)
    available_datasets = [
        ds for ds in DATASETS.values() if ds.path.exists() and ds.zarr_path.exists()
    ]

    if not available_datasets:
        print("No converted datasets found. Run conversion benchmarks first.")
        return

    print("\n" + "=" * 70)
    print("PARALLEL PROCESSING BENCHMARK SUITE")
    print("=" * 70)
    print("\nDemonstrates Dask-based parallel computation with SpatialData/Zarr")
    print(f"Testing with worker counts: {BenchmarkConfig.WORKER_COUNTS}")

    for dataset in available_datasets:
        benchmark.benchmark_dataset_scalability(dataset)

    benchmark.print_summary()
    benchmark.save_results()


if __name__ == "__main__":
    main()
