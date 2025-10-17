"""
Spectral (m/z) access pattern benchmarks.

Tests m/z-based queries - the key advantage of SpatialData/Zarr:
- Single m/z ion image extraction
- M/z range queries
- Multiple ion image extraction
- Spectral slicing
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import spatialdata as sd
from config import BenchmarkConfig, DatasetConfig
from utils import benchmark_context


class SpectralAccessBenchmark:
    """Benchmark spectral (m/z) access patterns."""

    def __init__(self):
        BenchmarkConfig.setup_directories()
        self.results: List[Dict[str, Any]] = []

    def benchmark_imzml_mz_queries(self, dataset: DatasetConfig) -> Dict[str, Any]:
        """Benchmark m/z queries on ImzML format."""
        print(f"\n{'='*70}")
        print(f"SPECTRAL ACCESS: {dataset.name} (ImzML)")
        print(f"{'='*70}\n")

        from pyimzml.ImzMLParser import ImzMLParser

        metrics = {"mz_ranges": {}, "ion_images": {}}

        # M/z range queries
        for mz_min, mz_max in BenchmarkConfig.MZ_RANGES:
            with benchmark_context(f"M/z range {mz_min}-{mz_max} Da") as m:
                parser = ImzMLParser(str(dataset.path))
                ion_image = []

                for idx, (x, y, z) in enumerate(parser.coordinates):
                    mzs, intensities = parser.getspectrum(idx)
                    mask = (mzs >= mz_min) & (mzs <= mz_max)
                    if np.any(mask):
                        ion_image.append(intensities[mask].sum())
                    else:
                        ion_image.append(0.0)

                metrics["mz_ranges"][f"{mz_min}-{mz_max}"] = m["time_sec"]
                del parser

        # Multiple ion image extraction
        n_ions = min(
            BenchmarkConfig.N_ION_IMAGES, len(BenchmarkConfig.TARGET_MZ_VALUES)
        )
        target_mz_list = BenchmarkConfig.TARGET_MZ_VALUES[:n_ions]

        with benchmark_context(f"Extract {n_ions} ion images") as m:
            parser = ImzMLParser(str(dataset.path))

            # For each target m/z, iterate all pixels
            for target_mz in target_mz_list:
                ion_image = []
                for idx, (x, y, z) in enumerate(parser.coordinates):
                    mzs, intensities = parser.getspectrum(idx)
                    mask = np.abs(mzs - target_mz) <= BenchmarkConfig.MZ_TOLERANCE
                    if np.any(mask):
                        ion_image.append(intensities[mask].sum())
                    else:
                        ion_image.append(0.0)

            metrics["ion_images"]["total_time_sec"] = m["time_sec"]
            metrics["ion_images"]["n_images"] = n_ions
            del parser

        return metrics

    def benchmark_bruker_mz_queries(self, dataset: DatasetConfig) -> Dict[str, Any]:
        """Benchmark m/z queries on Bruker format."""
        print(f"\n{'='*70}")
        print(f"SPECTRAL ACCESS: {dataset.name} (Bruker)")
        print(f"{'='*70}\n")

        from thyra.readers.bruker.bruker_reader import BrukerReader

        metrics = {"mz_ranges": {}, "ion_images": {}}

        # M/z range queries
        for mz_min, mz_max in BenchmarkConfig.MZ_RANGES:
            with benchmark_context(f"M/z range {mz_min}-{mz_max} Da") as m:
                reader = BrukerReader(str(dataset.path))
                ion_image = []

                for coords, mzs, intensities in reader.iter_spectra():
                    mask = (mzs >= mz_min) & (mzs <= mz_max)
                    if np.any(mask):
                        ion_image.append(intensities[mask].sum())
                    else:
                        ion_image.append(0.0)

                metrics["mz_ranges"][f"{mz_min}-{mz_max}"] = m["time_sec"]
                del reader

        # Multiple ion image extraction
        n_ions = min(
            BenchmarkConfig.N_ION_IMAGES, len(BenchmarkConfig.TARGET_MZ_VALUES)
        )
        target_mz_list = BenchmarkConfig.TARGET_MZ_VALUES[:n_ions]

        with benchmark_context(f"Extract {n_ions} ion images") as m:
            # For each target m/z, must iterate all pixels
            for target_mz in target_mz_list:
                reader = BrukerReader(str(dataset.path))
                ion_image = []

                for coords, mzs, intensities in reader.iter_spectra():
                    mask = np.abs(mzs - target_mz) <= BenchmarkConfig.MZ_TOLERANCE
                    if np.any(mask):
                        ion_image.append(intensities[mask].sum())
                    else:
                        ion_image.append(0.0)

                del reader

            metrics["ion_images"]["total_time_sec"] = m["time_sec"]
            metrics["ion_images"]["n_images"] = n_ions

        return metrics

    def benchmark_spatialdata_mz_queries(
        self, dataset: DatasetConfig
    ) -> Dict[str, Any]:
        """Benchmark m/z queries on SpatialData/Zarr format."""
        print(f"\n{'='*70}")
        print(f"SPECTRAL ACCESS: {dataset.name} (SpatialData/Zarr)")
        print(f"{'='*70}\n")

        zarr_path = dataset.zarr_path

        if not zarr_path.exists():
            print(f"[SKIP] Zarr file not found: {zarr_path}")
            return {}

        metrics = {"mz_ranges": {}, "ion_images": {}}

        # Load once and reuse
        sdata = sd.read_zarr(str(zarr_path))
        msi_data = list(sdata.images.values())[0]

        # Get m/z coordinates
        mz_coords = msi_data.coords["c"].values

        # M/z range queries
        for mz_min, mz_max in BenchmarkConfig.MZ_RANGES:
            with benchmark_context(f"M/z range {mz_min}-{mz_max} Da") as m:
                # Find m/z indices
                mz_mask = (mz_coords >= mz_min) & (mz_coords <= mz_max)
                mz_indices = np.where(mz_mask)[0]

                if len(mz_indices) > 0:
                    # Direct slice - key advantage!
                    mz_slice = msi_data[mz_indices, :, :]
                    _ = mz_slice.compute()

                metrics["mz_ranges"][f"{mz_min}-{mz_max}"] = m["time_sec"]

        # Multiple ion image extraction
        n_ions = min(
            BenchmarkConfig.N_ION_IMAGES, len(BenchmarkConfig.TARGET_MZ_VALUES)
        )
        target_mz_list = BenchmarkConfig.TARGET_MZ_VALUES[:n_ions]

        with benchmark_context(f"Extract {n_ions} ion images") as m:
            ion_images = []

            for target_mz in target_mz_list:
                # Find closest m/z within tolerance
                mz_mask = np.abs(mz_coords - target_mz) <= BenchmarkConfig.MZ_TOLERANCE
                mz_indices = np.where(mz_mask)[0]

                if len(mz_indices) > 0:
                    # Extract and sum intensities for this m/z
                    ion_image = msi_data[mz_indices, :, :].sum(axis=0).compute()
                    ion_images.append(ion_image)

            metrics["ion_images"]["total_time_sec"] = m["time_sec"]
            metrics["ion_images"]["n_images"] = len(ion_images)

        del sdata, msi_data
        return metrics

    def compare_dataset(self, dataset: DatasetConfig) -> Dict[str, Any]:
        """Compare spectral access patterns across all formats for a dataset."""
        result = {"dataset": dataset.name, "description": dataset.description}

        # Benchmark original format
        if dataset.format_type == "imzml":
            orig_metrics = self.benchmark_imzml_mz_queries(dataset)
            result["imzml"] = orig_metrics
        elif dataset.format_type == "bruker":
            orig_metrics = self.benchmark_bruker_mz_queries(dataset)
            result["bruker"] = orig_metrics

        # Benchmark SpatialData
        sd_metrics = self.benchmark_spatialdata_mz_queries(dataset)
        result["spatialdata"] = sd_metrics

        # Calculate speedups for m/z range queries
        if (
            orig_metrics
            and sd_metrics
            and "mz_ranges" in orig_metrics
            and "mz_ranges" in sd_metrics
        ):
            result["mz_range_speedups"] = {}
            for mz_range in orig_metrics["mz_ranges"]:
                if mz_range in sd_metrics["mz_ranges"]:
                    orig_time = orig_metrics["mz_ranges"][mz_range]
                    sd_time = sd_metrics["mz_ranges"][mz_range]
                    if sd_time > 0:
                        result["mz_range_speedups"][mz_range] = round(
                            orig_time / sd_time, 2
                        )

        # Calculate speedup for ion image extraction
        if (
            orig_metrics
            and sd_metrics
            and "ion_images" in orig_metrics
            and "ion_images" in sd_metrics
        ):
            orig_time = orig_metrics["ion_images"].get("total_time_sec", 0)
            sd_time = sd_metrics["ion_images"].get("total_time_sec", 0)
            if orig_time > 0 and sd_time > 0:
                result["ion_image_speedup"] = round(orig_time / sd_time, 2)

        self.results.append(result)
        return result

    def print_summary(self):
        """Print summary of spectral access benchmarks."""
        if not self.results:
            print("\nNo results to summarize.")
            return

        print("\n" + "=" * 80)
        print("SPECTRAL ACCESS BENCHMARK SUMMARY")
        print("=" * 80)

        for result in self.results:
            print(f"\nDataset: {result['dataset']}")
            print("-" * 80)

            # Print m/z range query speedups
            if "mz_range_speedups" in result:
                print("\nM/z Range Query Speedups (SpatialData vs original):")
                for mz_range, speedup in result["mz_range_speedups"].items():
                    print(f"  {mz_range} Da: {speedup:.2f}x faster")

            # Print ion image extraction speedup
            if "ion_image_speedup" in result:
                print(
                    f"\nIon Image Extraction Speedup: {result['ion_image_speedup']:.2f}x faster"
                )
                n_images = result["spatialdata"]["ion_images"].get("n_images", 0)
                print(f"  ({n_images} ion images extracted)")

    def save_results(self):
        """Save results to JSON."""
        from utils import save_metrics

        output_path = BenchmarkConfig.RESULTS_DIR / "spectral_access_benchmark.json"
        save_metrics(self.results, output_path)


def main():
    """Run spectral access benchmarks."""
    from config import DATASETS

    benchmark = SpectralAccessBenchmark()

    # Get available datasets
    available_datasets = [ds for ds in DATASETS.values() if ds.path.exists()]

    if not available_datasets:
        print("No datasets found.")
        return

    print("\n" + "=" * 70)
    print("SPECTRAL ACCESS PATTERN BENCHMARK SUITE")
    print("=" * 70)
    print("\nThis benchmark demonstrates SpatialData's key advantage:")
    print("Direct m/z axis slicing vs iterating all pixels")

    for dataset in available_datasets:
        benchmark.compare_dataset(dataset)

    benchmark.print_summary()
    benchmark.save_results()


if __name__ == "__main__":
    main()
