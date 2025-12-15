"""Compare ImzML, Bruker .d, and SpatialData/Zarr formats on Xenium dataset.

This benchmark compares the same Xenium dataset in three different formats:
1. ImzML (original, resampled by SCILS)
2. Bruker .d (original, raw data)
3. SpatialData/Zarr (converted from ImzML)

Access patterns tested:
- Random pixel access
- Random m/z range query
- Random ROI extraction
"""

import csv
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import spatialdata as sd
from pyimzml.ImzMLParser import ImzMLParser

from thyra.readers.bruker.timstof.timstof_reader import BrukerReader

# Configuration
N_ROUNDS = 100
RANDOM_SEED = 42

# Paths (relative to project root)
IMZML_PATH = Path(__file__).parent.parent / "test_data/20240826_xenium_0041899.imzML"
BRUKER_PATH = Path(__file__).parent.parent / "test_data/20240826_Xenium_0041899.d"
ZARR_PATH = Path(__file__).parent / "converted/xenium.zarr"
OUTPUT_CSV = Path(__file__).parent / "results/xenium_comparison.csv"


class RandomPixelAccess:
    """Random pixel spectrum access."""

    name = "Random Pixel"

    def generate_choices(self, n_pixels: int, n_mz: int) -> List[Dict]:
        """Generate random pixel indices."""
        random.seed(RANDOM_SEED)
        return [{"pixel_idx": random.randint(0, n_pixels - 1)} for _ in range(N_ROUNDS)]

    def describe_choice(self, choice: Dict) -> str:
        """Describe the choice."""
        return f"pixel_{choice['pixel_idx']}"


class RandomMzRangeAccess:
    """Random m/z range query."""

    name = "Random m/z Range"

    def generate_choices(self, n_pixels: int, n_mz: int) -> List[Dict]:
        """Generate random m/z ranges."""
        random.seed(RANDOM_SEED)
        choices = []
        for _ in range(N_ROUNDS):
            width = random.randint(100, 1000)  # bins
            start = random.randint(0, max(0, n_mz - width))
            choices.append({"mz_start_idx": start, "mz_width_bins": width})
        return choices

    def describe_choice(self, choice: Dict) -> str:
        """Describe the choice."""
        return f"mz_{choice['mz_start_idx']}_{choice['mz_width_bins']}"


class RandomROIAccess:
    """Random region of interest access."""

    name = "Random ROI"

    def generate_choices(self, n_pixels: int, n_mz: int) -> List[Dict]:
        """Generate random ROI coordinates with sizes from 10x10 to 50x50."""
        random.seed(RANDOM_SEED)
        grid_size = int(np.sqrt(n_pixels))
        choices = []
        for _ in range(N_ROUNDS):
            # Random ROI size between 10x10 and 50x50
            roi_size = random.randint(10, 50)
            x_start = random.randint(0, max(0, grid_size - roi_size))
            y_start = random.randint(0, max(0, grid_size - roi_size))
            choices.append(
                {
                    "x_start": x_start,
                    "y_start": y_start,
                    "roi_size": roi_size,
                }
            )
        return choices

    def describe_choice(self, choice: Dict) -> str:
        """Describe the choice."""
        return f"roi_{choice['x_start']}_{choice['y_start']}_{choice['roi_size']}"


def benchmark_imzml() -> List[Dict]:
    """Benchmark ImzML format (processed mode - ragged arrays)."""
    print("\n1. BENCHMARKING IMZML (PROCESSED)")
    print(f"   Path: {IMZML_PATH}")

    # Measure initialization time separately
    init_start = time.perf_counter()
    parser = ImzMLParser(str(IMZML_PATH))
    n_pixels = len(parser.coordinates)
    # Get m/z count from first spectrum
    mzs, _ = parser.getspectrum(0)
    n_mz = len(mzs)
    init_time = time.perf_counter() - init_start

    print(f"   Dimensions: {n_pixels} pixels, {n_mz} m/z bins")
    print(f"   Initialization time: {init_time:.3f} seconds")

    results = []
    patterns = [RandomPixelAccess(), RandomMzRangeAccess(), RandomROIAccess()]

    for pattern in patterns:
        print(f"   Testing {pattern.name}...", end="", flush=True)
        choices = pattern.generate_choices(n_pixels, n_mz)

        for round_idx, choice in enumerate(choices):
            start = time.perf_counter()

            if isinstance(pattern, RandomPixelAccess):
                pixel_idx = choice["pixel_idx"]
                mzs, intensities = parser.getspectrum(pixel_idx)
                data_size = len(mzs)
            elif isinstance(pattern, RandomMzRangeAccess):
                # Ion image extraction: iterate through ALL pixels to get the m/z range
                # This is what users do in practice to create spatial images
                start_mz = choice[
                    "mz_start_idx"
                ]  # This is an index, we need actual m/z value
                width = choice["mz_width_bins"]

                # Iterate through ALL pixels (this is what creating an ion image requires)
                data_size = 0
                for i in range(n_pixels):
                    mzs, intensities = parser.getspectrum(i)
                    # In a real query, we'd filter by m/z range and sum intensities
                    # Here we're measuring the latency of accessing all the data
                    data_size += len(mzs)
            else:  # ROI
                # Extract actual spatial ROI using coordinates
                roi_size = choice["roi_size"]
                x_start = choice["x_start"]
                y_start = choice["y_start"]
                grid_size = int(np.sqrt(n_pixels))

                # Collect pixel indices for the ROI
                roi_pixel_indices = []
                for dy in range(roi_size):
                    for dx in range(roi_size):
                        x = x_start + dx
                        y = y_start + dy
                        if x < grid_size and y < grid_size:
                            pixel_idx = y * grid_size + x
                            if pixel_idx < n_pixels:
                                roi_pixel_indices.append(pixel_idx)

                # Access each pixel in the ROI (random access)
                data_size = 0
                for pixel_idx in roi_pixel_indices:
                    mzs, intensities = parser.getspectrum(pixel_idx)
                    data_size += len(mzs)

            end = time.perf_counter()

            results.append(
                {
                    "format": "ImzML (Processed)",
                    "access_pattern": pattern.name,
                    "round": round_idx,
                    "latency_seconds": end - start,
                    "data_size": data_size,
                }
            )

        print(" Done")

    return results


def benchmark_bruker() -> List[Dict]:
    """Benchmark Bruker .d format."""
    print("\n2. BENCHMARKING BRUKER .d FORMAT")
    print(f"   Path: {BRUKER_PATH}")

    # Measure initialization time separately
    init_start = time.perf_counter()
    reader = BrukerReader(str(BRUKER_PATH))

    # Build coordinate map for direct access (only for Random Pixel pattern)
    # This is fair because we're measuring access time, not initialization
    coord_map = []  # List of (x, y) tuples indexed by pixel_idx
    n_pixels = 0
    n_mz = 0

    for coords, mzs, intensities in reader.iter_spectra():
        coord_map.append((coords[0], coords[1]))  # (x, y)
        if n_mz == 0:
            n_mz = len(mzs)
        n_pixels += 1

    init_time = time.perf_counter() - init_start

    print(f"   Dimensions: {n_pixels} pixels, {n_mz} m/z values (raw)")
    print(f"   Initialization time: {init_time:.3f} seconds")

    results = []
    patterns = [RandomPixelAccess(), RandomMzRangeAccess(), RandomROIAccess()]

    for pattern in patterns:
        print(f"   Testing {pattern.name}...", end="", flush=True)
        choices = pattern.generate_choices(
            n_pixels, 100000
        )  # Use large n_mz for choices

        for round_idx, choice in enumerate(choices):
            start = time.perf_counter()

            if isinstance(pattern, RandomPixelAccess):
                # Use direct coordinate-based access (fair comparison)
                pixel_idx = choice["pixel_idx"]
                x, y = coord_map[pixel_idx]

                # Create reader for this query
                reader = BrukerReader(str(BRUKER_PATH))
                result = reader.get_spectrum_by_coordinates(x, y)

                if result:
                    mzs, intensities = result
                    data_size = len(mzs)
                else:
                    data_size = 0
            elif isinstance(pattern, RandomMzRangeAccess):
                # Ion image extraction: iterate through ALL pixels to get the m/z range
                # This is what users do in practice to create spatial images
                # Create new reader for full iteration
                reader = BrukerReader(str(BRUKER_PATH))
                data_size = 0
                for coords, mzs, intensities in reader.iter_spectra():
                    # In a real query, we'd filter by m/z range and sum intensities
                    # Here we're measuring the latency of accessing all the data
                    data_size += len(mzs)
            else:  # ROI
                # Extract actual spatial ROI using direct coordinate access
                roi_size = choice["roi_size"]
                x_start = choice["x_start"]
                y_start = choice["y_start"]
                grid_size = int(np.sqrt(n_pixels))

                # Collect pixel indices for the ROI and get their coordinates
                roi_coords = []
                for dy in range(roi_size):
                    for dx in range(roi_size):
                        x = x_start + dx
                        y = y_start + dy
                        if x < grid_size and y < grid_size:
                            pixel_idx = y * grid_size + x
                            if pixel_idx < n_pixels:
                                roi_coords.append(coord_map[pixel_idx])

                # Create reader and access each ROI pixel directly by coordinates
                reader = BrukerReader(str(BRUKER_PATH))
                data_size = 0
                for x, y in roi_coords:
                    result = reader.get_spectrum_by_coordinates(x, y)
                    if result:
                        mzs, intensities = result
                        data_size += len(mzs)

            end = time.perf_counter()

            results.append(
                {
                    "format": "Bruker .d",
                    "access_pattern": pattern.name,
                    "round": round_idx,
                    "latency_seconds": end - start,
                    "data_size": data_size,
                }
            )

        print(" Done")

    return results


def benchmark_spatialdata() -> List[Dict]:
    """Benchmark SpatialData/Zarr format."""
    print("\n3. BENCHMARKING SPATIALDATA/ZARR FORMAT")
    print(f"   Path: {ZARR_PATH}")

    # Measure initialization time separately
    init_start = time.perf_counter()
    sdata = sd.read_zarr(str(ZARR_PATH))
    table_key = list(sdata.tables.keys())[0]
    table = sdata.tables[table_key]
    matrix = table.X
    n_pixels = matrix.shape[0]
    n_mz = matrix.shape[1]
    init_time = time.perf_counter() - init_start

    print(f"   Dimensions: {n_pixels} pixels, {n_mz} m/z bins")
    print(f"   Initialization time: {init_time:.3f} seconds")

    results = []
    patterns = [RandomPixelAccess(), RandomMzRangeAccess(), RandomROIAccess()]

    for pattern in patterns:
        print(f"   Testing {pattern.name}...", end="", flush=True)
        choices = pattern.generate_choices(n_pixels, n_mz)

        for round_idx, choice in enumerate(choices):
            start = time.perf_counter()

            if isinstance(pattern, RandomPixelAccess):
                pixel_idx = choice["pixel_idx"]
                spectrum = matrix[pixel_idx, :]
                # Force computation for sparse matrix
                if hasattr(spectrum, "toarray"):
                    _ = spectrum.toarray()
                else:
                    _ = np.asarray(spectrum)
                data_size = (
                    spectrum.nnz if hasattr(spectrum, "nnz") else spectrum.shape[1]
                )
            elif isinstance(pattern, RandomMzRangeAccess):
                # Ion image extraction: slice ALL pixels for the m/z range and sum
                start_idx = choice["mz_start_idx"]
                width = choice["mz_width_bins"]
                end_idx = start_idx + width
                # Slice the m/z columns and sum to get ion image
                ion_image = matrix[:, start_idx:end_idx].sum(axis=1)
                # Force computation - for sparse matrix this returns a matrix, flatten it
                _ = np.asarray(ion_image).ravel()
                data_size = width * n_pixels  # Approximate data accessed
            else:  # ROI
                roi_size_pixels = choice["roi_size"] * choice["roi_size"]
                start_pixel = (
                    choice["x_start"] * int(np.sqrt(n_pixels)) + choice["y_start"]
                )
                end_pixel = min(start_pixel + roi_size_pixels, n_pixels)
                # Extract ROI data and compute a summary (e.g., mean spectrum)
                roi_data = matrix[start_pixel:end_pixel, :]
                mean_spectrum = roi_data.mean(axis=0)
                # Force computation
                _ = np.asarray(mean_spectrum).ravel()
                data_size = (
                    roi_data.nnz
                    if hasattr(roi_data, "nnz")
                    else roi_data.shape[0] * roi_data.shape[1]
                )

            end = time.perf_counter()

            results.append(
                {
                    "format": "SpatialData",
                    "access_pattern": pattern.name,
                    "round": round_idx,
                    "latency_seconds": end - start,
                    "data_size": data_size,
                }
            )

        print(" Done")

    return results


def save_results(results: List[Dict], output_path: Path):
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "format",
                "access_pattern",
                "round",
                "latency_seconds",
                "data_size",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[Dict]):
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Group by format and pattern
    from collections import defaultdict

    groups = defaultdict(list)
    for r in results:
        key = (r["format"], r["access_pattern"])
        groups[key].append(r["latency_seconds"])

    for (fmt, pattern), latencies in sorted(groups.items()):
        median = np.median(latencies)
        mean = np.mean(latencies)
        p95 = np.percentile(latencies, 95)
        print(
            f"{fmt:20s} | {pattern:20s} | Median: {median*1000:8.2f} ms | Mean: {mean*1000:8.2f} ms | P95: {p95*1000:8.2f} ms"
        )


def main():
    """Run Xenium three-format comparison."""
    print("=" * 70)
    print("XENIUM DATASET - THREE FORMAT COMPARISON")
    print("ImzML (Processed) vs Bruker .d vs SpatialData/Zarr")
    print("=" * 70)
    print(f"Rounds per pattern: {N_ROUNDS}")
    print(f"Random seed: {RANDOM_SEED}")

    all_results = []

    # Run benchmarks
    all_results.extend(benchmark_imzml())
    all_results.extend(benchmark_bruker())
    all_results.extend(benchmark_spatialdata())

    # Save and summarize
    save_results(all_results, OUTPUT_CSV)
    print_summary(all_results)

    print("\nTo plot results, run:")
    print(
        f"  python benchmarks/plot_latency_results.py {OUTPUT_CSV} benchmarks/results/xenium_comparison_raincloud.png"
    )


if __name__ == "__main__":
    main()
