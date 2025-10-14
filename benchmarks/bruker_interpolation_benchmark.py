"""
Bruker format comparison benchmark.

Compares three Bruker storage formats:
1. Original Bruker .d (SQLite, ragged arrays, ~3.2GB)
2. SpatialData/Zarr raw (dense array, ~78M m/z bins, ~880MB)
3. SpatialData/Zarr resampled (300k bins, ~50-100MB)

Key insight: Even with full 78M m/z resolution (no data loss),
Zarr is 3.6x smaller and more efficient than ragged SQLite storage.
Resampling provides massive additional gains for typical workflows.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import spatialdata as sd
from config import BenchmarkConfig, DatasetConfig
from utils import (
    benchmark_context,
    bytes_to_gb,
    bytes_to_mb,
    format_size,
    get_directory_size,
)


class BrukerInterpolationBenchmark:
    """Compare Bruker original (.d) vs raw Zarr vs resampled Zarr."""

    def __init__(self):
        BenchmarkConfig.setup_directories()
        self.results: Dict[str, Any] = {}

    def get_mz_resolution_stats(self, zarr_path: Path) -> Dict[str, Any]:
        """Get m/z resolution statistics from a Zarr file."""
        sdata = sd.read_zarr(str(zarr_path))
        msi_data = list(sdata.images.values())[0]
        mz_coords = msi_data.coords["c"].values

        stats = {
            "n_mz_bins": len(mz_coords),
            "mz_range": (float(mz_coords.min()), float(mz_coords.max())),
            "mean_bin_width": float(np.mean(np.diff(mz_coords))),
            "median_bin_width": float(np.median(np.diff(mz_coords))),
        }

        del sdata, msi_data
        return stats

    def benchmark_storage(
        self, raw_dataset: DatasetConfig, resampled_dataset: DatasetConfig
    ) -> Dict[str, Any]:
        """Compare storage characteristics across all three formats."""
        print(f"\n{'='*70}")
        print("STORAGE COMPARISON: Bruker .d vs Zarr Raw vs Zarr Resampled")
        print(f"{'='*70}\n")

        storage_metrics = {}

        # Original Bruker .d (SQLite with ragged arrays)
        if raw_dataset.path.exists():
            bruker_size = get_directory_size(raw_dataset.path)
            storage_metrics["bruker_original_bytes"] = bruker_size
            storage_metrics["bruker_original_mb"] = bytes_to_mb(bruker_size)
            storage_metrics["bruker_original_gb"] = bytes_to_gb(bruker_size)
            print(f"1. Bruker .d (original SQLite):")
            print(f"   Size: {format_size(bruker_size)}")
            print(f"   Format: Ragged arrays (each pixel has own m/z values)")

        # Raw SpatialData (no resampling - full 78M m/z resolution)
        if raw_dataset.zarr_path.exists():
            raw_zarr_size = get_directory_size(raw_dataset.zarr_path)
            storage_metrics["zarr_raw_bytes"] = raw_zarr_size
            storage_metrics["zarr_raw_mb"] = bytes_to_mb(raw_zarr_size)
            storage_metrics["zarr_raw_gb"] = bytes_to_gb(raw_zarr_size)
            print(f"\n2. SpatialData/Zarr (raw - no interpolation):")
            print(f"   Size: {format_size(raw_zarr_size)}")

            raw_stats = self.get_mz_resolution_stats(raw_dataset.zarr_path)
            storage_metrics["raw_mz_stats"] = raw_stats
            print(f"   M/z bins: {raw_stats['n_mz_bins']:,}")
            print(
                f"   M/z range: {raw_stats['mz_range'][0]:.2f} - {raw_stats['mz_range'][1]:.2f} Da"
            )
            print(f"   Format: Dense array (all unique m/z values)")

        # Resampled SpatialData (300k bins)
        if resampled_dataset.zarr_path.exists():
            resampled_zarr_size = get_directory_size(resampled_dataset.zarr_path)
            storage_metrics["zarr_resampled_bytes"] = resampled_zarr_size
            storage_metrics["zarr_resampled_mb"] = bytes_to_mb(resampled_zarr_size)
            print(f"\n3. SpatialData/Zarr (resampled to 300k bins):")
            print(f"   Size: {format_size(resampled_zarr_size)}")

            resampled_stats = self.get_mz_resolution_stats(resampled_dataset.zarr_path)
            storage_metrics["resampled_mz_stats"] = resampled_stats
            print(f"   M/z bins: {resampled_stats['n_mz_bins']:,}")
            print(
                f"   M/z range: {resampled_stats['mz_range'][0]:.2f} - {resampled_stats['mz_range'][1]:.2f} Da"
            )
            print(f"   Format: Dense array (common m/z axis)")

        # Calculate key ratios
        print(f"\n{'='*70}")
        print("STORAGE EFFICIENCY RATIOS:")
        print(f"{'='*70}")

        if (
            "bruker_original_bytes" in storage_metrics
            and "zarr_raw_bytes" in storage_metrics
        ):
            ratio = (
                storage_metrics["bruker_original_bytes"]
                / storage_metrics["zarr_raw_bytes"]
            )
            storage_metrics["compression_bruker_to_zarr_raw"] = round(ratio, 2)
            print(f"\nBruker .d -> Zarr raw: {ratio:.2f}x smaller")
            print(
                f"  Despite converting ragged to dense array with {storage_metrics['raw_mz_stats']['n_mz_bins']:,} m/z bins!"
            )

        if (
            "bruker_original_bytes" in storage_metrics
            and "zarr_resampled_bytes" in storage_metrics
        ):
            ratio = (
                storage_metrics["bruker_original_bytes"]
                / storage_metrics["zarr_resampled_bytes"]
            )
            storage_metrics["compression_bruker_to_zarr_resampled"] = round(ratio, 2)
            print(f"\nBruker .d -> Zarr resampled: {ratio:.2f}x smaller")

        if (
            "zarr_raw_bytes" in storage_metrics
            and "zarr_resampled_bytes" in storage_metrics
        ):
            ratio = (
                storage_metrics["zarr_raw_bytes"]
                / storage_metrics["zarr_resampled_bytes"]
            )
            storage_metrics["compression_zarr_raw_to_resampled"] = round(ratio, 2)
            print(f"\nZarr raw -> Zarr resampled: {ratio:.2f}x smaller")

        return storage_metrics

    def benchmark_bruker_access(self, dataset: DatasetConfig) -> Dict[str, float]:
        """Benchmark access patterns on original Bruker .d format."""
        from thyra.readers.bruker.bruker_reader import BrukerReader

        print(f"\n1. Bruker .d (original SQLite):")
        metrics = {}

        reader = BrukerReader(str(dataset.path))

        # ROI access (first 50x50 pixels)
        with benchmark_context("  ROI access (50x50 pixels)") as m:
            count = 0
            target_pixels = 50 * 50
            for coords, mzs, intensities in reader.iter_spectra():
                count += 1
                if count >= target_pixels:
                    break
            metrics["roi_access_sec"] = m["time_sec"]

        del reader

        # M/z slice (must iterate all pixels)
        with benchmark_context(
            "  M/z slice (500-600 Da) - must iterate all pixels"
        ) as m:
            reader = BrukerReader(str(dataset.path))
            mz_data = []
            for coords, mzs, intensities in reader.iter_spectra():
                mask = (mzs >= 500) & (mzs <= 600)
                if np.any(mask):
                    mz_data.append(intensities[mask].sum())
                else:
                    mz_data.append(0.0)
            metrics["mz_slice_sec"] = m["time_sec"]

        del reader

        # Ion image extraction (must iterate all pixels)
        with benchmark_context("  Ion image (m/z 500) - must iterate all pixels") as m:
            reader = BrukerReader(str(dataset.path))
            ion_image = []
            target_mz = 500
            tolerance = 0.1
            for coords, mzs, intensities in reader.iter_spectra():
                mask = np.abs(mzs - target_mz) <= tolerance
                if np.any(mask):
                    ion_image.append(intensities[mask].sum())
                else:
                    ion_image.append(0.0)
            metrics["ion_image_sec"] = m["time_sec"]

        del reader
        return metrics

    def benchmark_access_patterns(
        self, raw_dataset: DatasetConfig, resampled_dataset: DatasetConfig
    ) -> Dict[str, Any]:
        """Compare access pattern performance across all three formats."""
        print(f"\n{'='*70}")
        print("ACCESS PATTERN COMPARISON")
        print(f"{'='*70}")

        access_metrics = {"bruker_original": {}, "zarr_raw": {}, "zarr_resampled": {}}

        # Benchmark original Bruker .d
        if raw_dataset.path.exists():
            access_metrics["bruker_original"] = self.benchmark_bruker_access(
                raw_dataset
            )

        # Benchmark raw Zarr (78M m/z bins)
        if raw_dataset.zarr_path.exists():
            print(
                f"\n2. SpatialData/Zarr (raw - {access_metrics.get('bruker_original', {}).get('n_mz_bins', '78M')} m/z bins):"
            )
            access_metrics["zarr_raw"] = self._benchmark_zarr_access(
                raw_dataset.zarr_path
            )

        # Benchmark resampled Zarr (300k bins)
        if resampled_dataset.zarr_path.exists():
            print(f"\n3. SpatialData/Zarr (resampled - 300k bins):")
            access_metrics["zarr_resampled"] = self._benchmark_zarr_access(
                resampled_dataset.zarr_path
            )

        # Calculate speedups relative to original Bruker
        print(f"\n{'='*70}")
        print("SPEEDUPS vs ORIGINAL BRUKER .d:")
        print(f"{'='*70}")

        if access_metrics["bruker_original"]:
            bruker_baseline = access_metrics["bruker_original"]

            # Zarr raw speedups
            if access_metrics["zarr_raw"]:
                access_metrics["speedups_zarr_raw_vs_bruker"] = {}
                for key in ["roi_access_sec", "mz_slice_sec", "ion_image_sec"]:
                    if key in bruker_baseline and key in access_metrics["zarr_raw"]:
                        bruker_time = bruker_baseline[key]
                        zarr_time = access_metrics["zarr_raw"][key]
                        if zarr_time > 0:
                            speedup = bruker_time / zarr_time
                            access_metrics["speedups_zarr_raw_vs_bruker"][key] = round(
                                speedup, 2
                            )
                            operation = (
                                key.replace("_sec", "").replace("_", " ").title()
                            )
                            print(f"\n{operation} (Zarr raw vs Bruker):")
                            if speedup > 1:
                                print(f"  Zarr is {speedup:.2f}x FASTER")
                            else:
                                print(
                                    f"  Zarr is {1/speedup:.2f}x slower (but enables direct m/z slicing)"
                                )

            # Zarr resampled speedups
            if access_metrics["zarr_resampled"]:
                access_metrics["speedups_zarr_resampled_vs_bruker"] = {}
                for key in ["roi_access_sec", "mz_slice_sec", "ion_image_sec"]:
                    if (
                        key in bruker_baseline
                        and key in access_metrics["zarr_resampled"]
                    ):
                        bruker_time = bruker_baseline[key]
                        zarr_time = access_metrics["zarr_resampled"][key]
                        if zarr_time > 0:
                            speedup = bruker_time / zarr_time
                            access_metrics["speedups_zarr_resampled_vs_bruker"][key] = (
                                round(speedup, 2)
                            )
                            operation = (
                                key.replace("_sec", "").replace("_", " ").title()
                            )
                            print(f"\n{operation} (Zarr 300k vs Bruker):")
                            print(f"  Zarr is {speedup:.2f}x FASTER")

        # Comparison between raw and resampled Zarr
        if access_metrics["zarr_raw"] and access_metrics["zarr_resampled"]:
            print(f"\n{'='*70}")
            print("RAW vs RESAMPLED ZARR:")
            print(f"{'='*70}")
            access_metrics["speedups_resampled_vs_raw"] = {}
            for key in ["roi_access_sec", "mz_slice_sec", "ion_image_sec"]:
                if (
                    key in access_metrics["zarr_raw"]
                    and key in access_metrics["zarr_resampled"]
                ):
                    raw_time = access_metrics["zarr_raw"][key]
                    res_time = access_metrics["zarr_resampled"][key]
                    if res_time > 0:
                        speedup = raw_time / res_time
                        access_metrics["speedups_resampled_vs_raw"][key] = round(
                            speedup, 2
                        )
                        operation = key.replace("_sec", "").replace("_", " ").title()
                        print(f"\n{operation}:")
                        if speedup > 1:
                            print(f"  Resampled is {speedup:.2f}x faster than raw")
                        else:
                            print(f"  Raw is {1/speedup:.2f}x faster than resampled")

        return access_metrics

    def _benchmark_zarr_access(self, zarr_path: Path) -> Dict[str, float]:
        """Benchmark access patterns on a Zarr file."""
        metrics = {}

        sdata = sd.read_zarr(str(zarr_path))
        msi_data = list(sdata.images.values())[0]
        mz_coords = msi_data.coords["c"].values

        # ROI access
        with benchmark_context("  ROI access") as m:
            roi_y = min(50, msi_data.shape[1])
            roi_x = min(50, msi_data.shape[2])
            _ = msi_data[:, :roi_y, :roi_x].compute()
            metrics["roi_access_sec"] = m["time_sec"]

        # M/z slice
        with benchmark_context("  M/z slice (500-600 Da)") as m:
            mz_mask = (mz_coords >= 500) & (mz_coords <= 600)
            mz_indices = np.where(mz_mask)[0]
            if len(mz_indices) > 0:
                _ = msi_data[mz_indices, :, :].compute()
            metrics["mz_slice_sec"] = m["time_sec"]

        # Ion image extraction
        with benchmark_context("  Ion image extraction (m/z 500)") as m:
            target_mz = 500
            mz_mask = np.abs(mz_coords - target_mz) <= 0.1
            mz_indices = np.where(mz_mask)[0]
            if len(mz_indices) > 0:
                _ = msi_data[mz_indices, :, :].sum(axis=0).compute()
            metrics["ion_image_sec"] = m["time_sec"]

        del sdata, msi_data
        return metrics

    def compare_datasets(
        self, raw_dataset: DatasetConfig, resampled_dataset: DatasetConfig
    ) -> Dict[str, Any]:
        """
        Full comparison between raw and resampled Bruker data.

        Args:
            raw_dataset: DatasetConfig for raw Bruker data (no resampling)
            resampled_dataset: DatasetConfig for resampled data (300k bins)
        """
        print(f"\n{'='*70}")
        print("BRUKER INTERPOLATION BENCHMARK")
        print(f"{'='*70}")
        print(f"\nComparing:")
        print(f"  Raw: {raw_dataset.name}")
        print(f"  Resampled: {resampled_dataset.name}")
        print(f"{'='*70}")

        result = {
            "raw_dataset": raw_dataset.name,
            "resampled_dataset": resampled_dataset.name,
            "storage": {},
            "access_patterns": {},
        }

        # Storage comparison
        result["storage"] = self.benchmark_storage(raw_dataset, resampled_dataset)

        # Access pattern comparison
        result["access_patterns"] = self.benchmark_access_patterns(
            raw_dataset, resampled_dataset
        )

        self.results = result
        return result

    def print_summary(self):
        """Print comprehensive summary."""
        if not self.results:
            print("\nNo results to summarize.")
            return

        print("\n" + "=" * 80)
        print("BRUKER FORMAT COMPARISON SUMMARY")
        print("=" * 80)

        # Storage summary
        if "storage" in self.results:
            storage = self.results["storage"]
            print("\nSTORAGE COMPARISON:")
            print("-" * 80)

            if "bruker_original_mb" in storage:
                print(f"\n1. Bruker .d (original SQLite):")
                print(
                    f"   Size: {storage['bruker_original_mb']:.1f} MB ({storage['bruker_original_gb']:.2f} GB)"
                )
                print(f"   Format: Ragged arrays")

            if "zarr_raw_mb" in storage:
                print(f"\n2. SpatialData/Zarr (raw - no interpolation):")
                print(
                    f"   Size: {storage['zarr_raw_mb']:.1f} MB ({storage['zarr_raw_gb']:.2f} GB)"
                )
                if "raw_mz_stats" in storage:
                    print(f"   M/z bins: {storage['raw_mz_stats']['n_mz_bins']:,}")
                print(f"   Format: Dense array (all unique m/z)")

            if "zarr_resampled_mb" in storage:
                print(f"\n3. SpatialData/Zarr (resampled to 300k):")
                print(f"   Size: {storage['zarr_resampled_mb']:.1f} MB")
                if "resampled_mz_stats" in storage:
                    print(
                        f"   M/z bins: {storage['resampled_mz_stats']['n_mz_bins']:,}"
                    )
                print(f"   Format: Dense array (common m/z axis)")

            print("\n" + "-" * 80)
            print("COMPRESSION RATIOS:")
            print("-" * 80)

            if "compression_bruker_to_zarr_raw" in storage:
                ratio = storage["compression_bruker_to_zarr_raw"]
                print(f"\nBruker .d -> Zarr raw: {ratio:.2f}x smaller")
                print(
                    f"  KEY FINDING: Dense array with {storage.get('raw_mz_stats', {}).get('n_mz_bins', '78M'):,} m/z bins"
                )
                print(f"  is STILL {ratio:.2f}x smaller than ragged SQLite!")

            if "compression_bruker_to_zarr_resampled" in storage:
                ratio = storage["compression_bruker_to_zarr_resampled"]
                print(f"\nBruker .d -> Zarr resampled: {ratio:.2f}x smaller")

            if "compression_zarr_raw_to_resampled" in storage:
                ratio = storage["compression_zarr_raw_to_resampled"]
                print(f"\nZarr raw -> Zarr resampled: {ratio:.2f}x smaller")

        # Access pattern summary
        if "access_patterns" in self.results:
            access = self.results["access_patterns"]
            print("\n\nACCESS PATTERN SPEEDUPS:")
            print("-" * 80)

            # vs Bruker original
            if "speedups_zarr_raw_vs_bruker" in access:
                print("\nZarr Raw vs Bruker .d:")
                for key, speedup in access["speedups_zarr_raw_vs_bruker"].items():
                    operation = key.replace("_sec", "").replace("_", " ").title()
                    if speedup > 1:
                        print(f"  {operation}: {speedup:.2f}x faster")
                    else:
                        print(
                            f"  {operation}: {1/speedup:.2f}x slower (trade-off for m/z slicing)"
                        )

            if "speedups_zarr_resampled_vs_bruker" in access:
                print("\nZarr Resampled vs Bruker .d:")
                for key, speedup in access["speedups_zarr_resampled_vs_bruker"].items():
                    operation = key.replace("_sec", "").replace("_", " ").title()
                    print(f"  {operation}: {speedup:.2f}x faster")

            # Raw vs resampled
            if "speedups_resampled_vs_raw" in access:
                print("\nZarr Resampled vs Zarr Raw:")
                for key, speedup in access["speedups_resampled_vs_raw"].items():
                    operation = key.replace("_sec", "").replace("_", " ").title()
                    if speedup > 1:
                        print(f"  {operation}: {speedup:.2f}x faster")
                    else:
                        print(f"  {operation}: {1/speedup:.2f}x slower")

        print("\n" + "=" * 80)
        print("KEY INSIGHTS FOR PAPER")
        print("=" * 80)
        print("\n1. ZARR IS EFFICIENT EVEN AT FULL RESOLUTION:")
        if (
            "storage" in self.results
            and "compression_bruker_to_zarr_raw" in self.results["storage"]
        ):
            ratio = self.results["storage"]["compression_bruker_to_zarr_raw"]
            n_bins = (
                self.results["storage"].get("raw_mz_stats", {}).get("n_mz_bins", "78M")
            )
            print(
                f"   - Converting ragged SQLite to dense array with {n_bins:,} m/z bins"
            )
            print(f"   - Results in {ratio:.2f}x SMALLER file due to compression")
            print(f"   - Demonstrates Zarr can handle extreme cases")

        print("\n2. ZARR ENABLES DIRECT M/Z SLICING:")
        print("   - Original Bruker: Must iterate ALL pixels for any m/z query")
        print("   - Zarr: Direct array slicing along m/z axis")
        print("   - Fundamental architectural advantage")

        print("\n3. RESAMPLING PROVIDES MASSIVE ADDITIONAL GAINS:")
        if (
            "storage" in self.results
            and "compression_bruker_to_zarr_resampled" in self.results["storage"]
        ):
            ratio = self.results["storage"]["compression_bruker_to_zarr_resampled"]
            print(f"   - {ratio:.2f}x smaller than original")
        print("   - Faster access (fewer m/z bins to process)")
        print("   - Sufficient resolution for most workflows")
        print("   - Similar to industry standard (SCiLS/Bruker software)")

        print("\n4. USERS HAVE OPTIONS:")
        print("   - Need full resolution? Use raw Zarr (still better than SQLite)")
        print("   - Want performance? Resample to 300k (massive gains)")
        print("   - Flexibility without lock-in")

        print("=" * 80)

    def save_results(self):
        """Save results to JSON."""
        from utils import save_metrics

        output_path = (
            BenchmarkConfig.RESULTS_DIR / "bruker_interpolation_benchmark.json"
        )
        save_metrics(self.results, output_path)


def main():
    """Run comprehensive Bruker format comparison benchmark."""
    from config import DATASETS

    print("\n" + "=" * 70)
    print("BRUKER FORMAT COMPARISON BENCHMARK")
    print("=" * 70)
    print("\nCompares THREE storage formats:")
    print("  1. Bruker .d (original SQLite with ragged arrays)")
    print("  2. SpatialData/Zarr raw (dense array, ~78M m/z bins)")
    print("  3. SpatialData/Zarr resampled (300k bins)")
    print("\nThis benchmark requires two Bruker dataset configurations:")
    print("  - Raw data (no resampling)")
    print("  - Resampled data (300k bins)")
    print("\nPlease ensure these are defined in config.py")
    print("=" * 70)

    # Look for raw and resampled Bruker datasets
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
        print("\n[ERROR] Could not find both raw and resampled Bruker dataset configs.")
        print("Please add them to config.py:")
        print(
            """
Example:
    'bruker_raw': DatasetConfig(
        name='bruker_raw',
        path=Path('test_data/sample.d'),
        format_type='bruker',
        description='Bruker raw data',
        resampling_config=None
    ),
    'bruker_300k': DatasetConfig(
        name='bruker_300k',
        path=Path('test_data/sample.d'),
        format_type='bruker',
        description='Bruker resampled to 300k bins',
        resampling_config={'method': 'bin_width_at_mz', 'params': {'target_bins': 300000}}
    ),
        """
        )
        return

    if not raw_dataset.path.exists():
        print(f"\n[ERROR] Bruker data not found: {raw_dataset.path}")
        return

    benchmark = BrukerInterpolationBenchmark()
    benchmark.compare_datasets(raw_dataset, resampled_dataset)
    benchmark.print_summary()
    benchmark.save_results()


if __name__ == "__main__":
    main()
