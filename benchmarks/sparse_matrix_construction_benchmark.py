"""Benchmark comparing sparse matrix construction approaches.

This script compares different methods for building large sparse matrices
to optimize the conversion process. It simulates realistic MSI data with:
- 1,000,000 pixels
- 400,000 m/z bins
- ~2,000 peaks per pixel (~0.5% density)
"""

import time
import tracemalloc
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class SparseMatrixBenchmark:
    """Benchmark sparse matrix construction approaches."""

    def __init__(
        self, n_pixels: int = 1_000_000, n_mz: int = 400_000, avg_peaks: int = 2000
    ):
        """Initialize benchmark parameters.

        Args:
            n_pixels: Number of pixels (spectra) to simulate
            n_mz: Size of common m/z axis
            avg_peaks: Average number of non-zero peaks per pixel
        """
        self.n_pixels = n_pixels
        self.n_mz = n_mz
        self.avg_peaks = avg_peaks
        self.total_nnz = n_pixels * avg_peaks

        print("=" * 70)
        print("SPARSE MATRIX CONSTRUCTION BENCHMARK")
        print("=" * 70)
        print(f"Dataset parameters:")
        print(f"  Pixels:           {self.n_pixels:,}")
        print(f"  M/Z bins:         {self.n_mz:,}")
        print(f"  Avg peaks/pixel:  {self.avg_peaks:,}")
        print(f"  Est. sparsity:    {self.avg_peaks/self.n_mz*100:.2f}%")
        print(f"  Est. total NNZ:   {self.total_nnz:,} ({self.total_nnz/1e9:.2f}B)")
        print("=" * 70)
        print()

    def generate_synthetic_data(
        self, seed: int = 42
    ) -> Generator[Tuple[int, NDArray[np.int32], NDArray[np.float64]], None, None]:
        """Generate realistic sparse spectra data.

        Args:
            seed: Random seed for reproducibility

        Yields:
            Tuple of (pixel_idx, mz_indices, intensities)
        """
        np.random.seed(seed)

        for pixel_idx in range(self.n_pixels):
            # Poisson-distributed peak count (realistic for mass spec)
            n_peaks = np.random.poisson(self.avg_peaks)
            n_peaks = max(100, min(n_peaks, self.n_mz))  # Clamp to reasonable range

            # Random peak positions (sorted, no duplicates)
            mz_indices = np.random.choice(self.n_mz, n_peaks, replace=False)
            mz_indices = np.sort(mz_indices).astype(np.int32)

            # Exponential intensity distribution (realistic for mass spec)
            intensities = np.random.exponential(1000.0, n_peaks).astype(np.float64)

            yield pixel_idx, mz_indices, intensities

    def benchmark_lil_matrix(self) -> Dict[str, Any]:
        """Benchmark current approach: LIL matrix.

        Returns:
            Dictionary with timing, memory, and result data
        """
        print("[LIL] Benchmarking LIL Matrix (current approach)...")
        print("-" * 70)

        tracemalloc.start()
        start_time = time.time()

        # Create LIL matrix
        lil = sparse.lil_matrix((self.n_pixels, self.n_mz), dtype=np.float64)

        # Track performance at intervals
        last_check = start_time
        rates = []

        # Create progress bar if available
        data_iterator = self.generate_synthetic_data()
        if HAS_TQDM:
            data_iterator = tqdm(
                data_iterator,
                total=self.n_pixels,
                desc="  Building LIL",
                unit=" pixels",
                unit_scale=True,
                ncols=70,
            )

        for pixel_idx, mz_idx, intensities in data_iterator:
            lil[pixel_idx, mz_idx] = intensities

            # Log progress every 100k pixels (for non-tqdm mode)
            if not HAS_TQDM and (pixel_idx + 1) % 100_000 == 0:
                now = time.time()
                interval_time = now - last_check
                interval_rate = 100_000 / interval_time
                elapsed = now - start_time
                overall_rate = (pixel_idx + 1) / elapsed
                percent = ((pixel_idx + 1) / self.n_pixels) * 100
                eta = (
                    (self.n_pixels - (pixel_idx + 1)) / overall_rate
                    if overall_rate > 0
                    else 0
                )
                rates.append((pixel_idx + 1, overall_rate))

                print(
                    f"  [{percent:5.1f}%] {pixel_idx+1:8,} pixels | "
                    f"Rate: {overall_rate:6.0f} spec/s | "
                    f"Elapsed: {elapsed:6.1f}s | "
                    f"ETA: {eta:6.1f}s"
                )
                last_check = now

        # Convert to CSR
        print("  Converting LIL → CSR...")
        convert_start = time.time()
        csr = lil.tocsr()
        convert_time = time.time() - convert_start

        total_time = time.time() - start_time
        peak_mem_bytes, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  [DONE] Conversion completed: {convert_time:.2f}s")
        print(f"  [DONE] Total time: {total_time:.2f}s")
        print(f"  [DONE] Peak memory: {peak_mem_bytes/1e9:.2f} GB")
        print(f"  [DONE] Final NNZ: {csr.nnz:,}")
        print()

        return {
            "method": "LIL Matrix",
            "time_seconds": total_time,
            "convert_time_seconds": convert_time,
            "peak_memory_gb": peak_mem_bytes / 1e9,
            "nnz": csr.nnz,
            "rates": rates,
            "csr_matrix": csr,
        }

    def benchmark_coo_preallocated(self) -> Dict[str, Any]:
        """Benchmark proposed approach: COO with pre-allocated arrays.

        Returns:
            Dictionary with timing, memory, and result data
        """
        print("[COO] Benchmarking COO Pre-allocated (proposed approach)...")
        print("-" * 70)

        tracemalloc.start()
        start_time = time.time()

        # Pre-allocate arrays with 10% buffer for variance
        estimated_nnz = int(self.total_nnz * 1.1)
        print(f"  Pre-allocating for {estimated_nnz:,} non-zeros...")

        rows = np.empty(estimated_nnz, dtype=np.int32)
        cols = np.empty(estimated_nnz, dtype=np.int32)
        data = np.empty(estimated_nnz, dtype=np.float64)

        # Track performance
        current_idx = 0
        last_check = start_time
        rates = []

        # Create progress bar if available
        data_iterator = self.generate_synthetic_data()
        if HAS_TQDM:
            data_iterator = tqdm(
                data_iterator,
                total=self.n_pixels,
                desc="  Building COO",
                unit=" pixels",
                unit_scale=True,
                ncols=70,
            )

        for pixel_idx, mz_idx, intensities in data_iterator:
            n = len(mz_idx)

            # Direct array assignment (no Python list overhead)
            rows[current_idx : current_idx + n] = pixel_idx
            cols[current_idx : current_idx + n] = mz_idx
            data[current_idx : current_idx + n] = intensities
            current_idx += n

            # Log progress every 100k pixels (for non-tqdm mode)
            if not HAS_TQDM and (pixel_idx + 1) % 100_000 == 0:
                now = time.time()
                interval_time = now - last_check
                interval_rate = 100_000 / interval_time
                elapsed = now - start_time
                overall_rate = (pixel_idx + 1) / elapsed
                percent = ((pixel_idx + 1) / self.n_pixels) * 100
                eta = (
                    (self.n_pixels - (pixel_idx + 1)) / overall_rate
                    if overall_rate > 0
                    else 0
                )
                rates.append((pixel_idx + 1, overall_rate))

                print(
                    f"  [{percent:5.1f}%] {pixel_idx+1:8,} pixels | "
                    f"Rate: {overall_rate:6.0f} spec/s | "
                    f"Elapsed: {elapsed:6.1f}s | "
                    f"ETA: {eta:6.1f}s"
                )
                last_check = now

        # Trim arrays to actual size
        print(f"  Trimming arrays (used {current_idx:,} / {estimated_nnz:,})...")
        rows = rows[:current_idx]
        cols = cols[:current_idx]
        data = data[:current_idx]

        # Convert COO → CSR
        print("  Converting COO → CSR...")
        convert_start = time.time()
        coo = sparse.coo_matrix(
            (data, (rows, cols)), shape=(self.n_pixels, self.n_mz), dtype=np.float64
        )
        csr = coo.tocsr()
        convert_time = time.time() - convert_start

        total_time = time.time() - start_time
        peak_mem_bytes, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  [DONE] Conversion completed: {convert_time:.2f}s")
        print(f"  [DONE] Total time: {total_time:.2f}s")
        print(f"  [DONE] Peak memory: {peak_mem_bytes/1e9:.2f} GB")
        print(f"  [DONE] Final NNZ: {csr.nnz:,}")
        print()

        return {
            "method": "COO Pre-allocated",
            "time_seconds": total_time,
            "convert_time_seconds": convert_time,
            "peak_memory_gb": peak_mem_bytes / 1e9,
            "nnz": csr.nnz,
            "rates": rates,
            "csr_matrix": csr,
        }

    def benchmark_batched_lil(self, batch_size: int = 100_000) -> Dict[str, Any]:
        """Benchmark alternative approach: Batched LIL → CSR.

        Args:
            batch_size: Number of pixels per batch

        Returns:
            Dictionary with timing, memory, and result data
        """
        print(f"[BATCH] Benchmarking Batched LIL (batch size: {batch_size:,})...")
        print("-" * 70)

        tracemalloc.start()
        start_time = time.time()

        csr_batches: List[sparse.csr_matrix] = []
        data_gen = self.generate_synthetic_data()

        last_check = start_time
        rates = []
        total_processed = 0

        # Create progress bar if available
        if HAS_TQDM:
            pbar = tqdm(
                total=self.n_pixels,
                desc="  Batched LIL",
                unit=" pixels",
                unit_scale=True,
                ncols=70,
            )

        while total_processed < self.n_pixels:
            batch_end = min(total_processed + batch_size, self.n_pixels)
            batch_pixels = batch_end - total_processed

            # Create batch LIL matrix
            batch_lil = sparse.lil_matrix((batch_pixels, self.n_mz), dtype=np.float64)

            # Fill batch
            for local_idx in range(batch_pixels):
                pixel_idx, mz_idx, intensities = next(data_gen)
                batch_lil[local_idx, mz_idx] = intensities
                total_processed += 1

                if HAS_TQDM:
                    pbar.update(1)

                # Log progress every 100k pixels (for non-tqdm mode)
                if not HAS_TQDM and total_processed % 100_000 == 0:
                    now = time.time()
                    interval_time = now - last_check
                    interval_rate = 100_000 / interval_time
                    elapsed = now - start_time
                    overall_rate = total_processed / elapsed
                    percent = (total_processed / self.n_pixels) * 100
                    eta = (
                        (self.n_pixels - total_processed) / overall_rate
                        if overall_rate > 0
                        else 0
                    )
                    rates.append((total_processed, overall_rate))

                    print(
                        f"  [{percent:5.1f}%] {total_processed:8,} pixels | "
                        f"Rate: {overall_rate:6.0f} spec/s | "
                        f"Elapsed: {elapsed:6.1f}s | "
                        f"ETA: {eta:6.1f}s"
                    )
                    last_check = now

            # Convert batch to CSR and store
            csr_batches.append(batch_lil.tocsr())
            del batch_lil  # Free memory

        if HAS_TQDM:
            pbar.close()

        # Stack all batches
        print("  Stacking CSR batches...")
        stack_start = time.time()
        csr = sparse.vstack(csr_batches, format="csr")
        stack_time = time.time() - stack_start

        total_time = time.time() - start_time
        peak_mem_bytes, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"  ✓ Stacking completed: {stack_time:.2f}s")
        print(f"  ✓ Total time: {total_time:.2f}s")
        print(f"  ✓ Peak memory: {peak_mem_bytes/1e9:.2f} GB")
        print(f"  ✓ Final NNZ: {csr.nnz:,}")
        print()

        return {
            "method": f"Batched LIL ({batch_size//1000}k)",
            "time_seconds": total_time,
            "convert_time_seconds": stack_time,
            "peak_memory_gb": peak_mem_bytes / 1e9,
            "nnz": csr.nnz,
            "rates": rates,
            "csr_matrix": csr,
        }

    def verify_results(self, results: List[Dict[str, Any]]) -> bool:
        """Verify that all methods produce identical CSR matrices.

        Args:
            results: List of benchmark results

        Returns:
            True if all matrices are identical
        """
        print("[VERIFY] Verifying result consistency...")
        print("-" * 70)

        baseline = results[0]["csr_matrix"]
        all_match = True

        for i, result in enumerate(results[1:], 1):
            csr = result["csr_matrix"]

            # Check shape
            if baseline.shape != csr.shape:
                print(
                    f"  [FAIL] {result['method']}: Shape mismatch "
                    f"({baseline.shape} vs {csr.shape})"
                )
                all_match = False
                continue

            # Check NNZ
            if baseline.nnz != csr.nnz:
                print(
                    f"  [WARN] {result['method']}: NNZ mismatch "
                    f"({baseline.nnz:,} vs {csr.nnz:,})"
                )
                all_match = False

            # Check data equality (may have different orderings, so convert to dense)
            if not np.allclose(baseline.toarray(), csr.toarray(), rtol=1e-9):
                print(f"  [FAIL] {result['method']}: Data values differ")
                all_match = False
            else:
                print(f"  [OK] {result['method']}: Identical to baseline")

        print()
        return all_match

    def generate_report(self, results: List[Dict[str, Any]]) -> None:
        """Generate comparison report.

        Args:
            results: List of benchmark results
        """
        print("=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Dataset: {self.n_pixels:,} pixels × {self.n_mz:,} m/z bins")
        print(f"Density: ~{self.avg_peaks/self.n_mz*100:.2f}%")
        print()

        # Tabular results
        print(
            f"{'Method':<25} {'Time (s)':>10} {'Memory (GB)':>12} {'Rate (spec/s)':>15}"
        )
        print("-" * 70)

        baseline_time = results[0]["time_seconds"]

        for r in results:
            rate = self.n_pixels / r["time_seconds"]
            speedup = baseline_time / r["time_seconds"]

            print(
                f"{r['method']:<25} "
                f"{r['time_seconds']:>10.1f} "
                f"{r['peak_memory_gb']:>12.2f} "
                f"{rate:>15,.0f}"
            )

            if speedup != 1.0:
                print(f"  → {speedup:.2f}x speedup vs baseline")

        print("=" * 70)
        print()

        # Decision recommendation
        best = min(results[1:], key=lambda x: x["time_seconds"])
        speedup = baseline_time / best["time_seconds"]

        print("RECOMMENDATION:")
        if speedup >= 2.0:
            print(f"  [PROCEED] Use {best['method']} optimization")
            print(f"     {speedup:.2f}x faster than current approach")
        elif speedup >= 1.5:
            print(f"  [CONSIDER] {best['method']} optimization")
            print(f"     {speedup:.2f}x speedup (moderate improvement)")
        else:
            print(f"  [NOT RECOMMENDED] - speedup too small ({speedup:.2f}x)")

        print()

    def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """Run all benchmarks and generate report.

        Returns:
            List of benchmark results
        """
        results = []

        # Run each benchmark
        results.append(self.benchmark_lil_matrix())
        results.append(self.benchmark_coo_preallocated())
        results.append(self.benchmark_batched_lil(batch_size=100_000))

        # Verify consistency
        if self.verify_results(results):
            print("[OK] All methods produce identical results\n")
        else:
            print("[WARN] WARNING: Results differ between methods!\n")

        # Generate report
        self.generate_report(results)

        return results


def main():
    """Run benchmark with realistic parameters."""
    # Use smaller dataset for quick testing (can be increased for full benchmark)
    benchmark = SparseMatrixBenchmark(
        n_pixels=1_000_000,  # 1M pixels (realistic large dataset)
        n_mz=400_000,  # 400k m/z bins
        avg_peaks=2000,  # ~2k peaks per pixel
    )

    results = benchmark.run_all_benchmarks()

    # Optionally save results to JSON
    try:
        import json
        from pathlib import Path

        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "sparse_construction_benchmark.json"

        # Prepare JSON-serializable results (exclude CSR matrices)
        json_results = []
        for r in results:
            json_r = {k: v for k, v in r.items() if k != "csr_matrix"}
            json_results.append(json_r)

        with open(output_file, "w") as f:
            json.dump(
                {
                    "parameters": {
                        "n_pixels": benchmark.n_pixels,
                        "n_mz": benchmark.n_mz,
                        "avg_peaks": benchmark.avg_peaks,
                    },
                    "results": json_results,
                },
                f,
                indent=2,
            )

        print(f"Results saved to: {output_file}")
    except Exception as e:
        print(f"Could not save results: {e}")


if __name__ == "__main__":
    main()
