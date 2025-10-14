# Sparse Matrix Construction Benchmark

## Purpose

This benchmark compares different approaches for constructing large sparse matrices during MSI data conversion. The goal is to identify the most efficient method to replace the current LIL (List of Lists) matrix approach, which suffers from performance degradation at ~90% completion due to memory fragmentation.

## Problem Statement

Current conversion process:
- Uses `scipy.sparse.lil_matrix` for incremental construction
- Degrades from 1500 spec/sec → 400 spec/sec at 90% completion
- Large datasets (1.5M pixels) take 45+ minutes to convert

Expected improvement with COO (Coordinate) approach:
- Consistent 1500-2000 spec/sec throughout
- Estimated conversion time: ~14 minutes (3x faster)

## Benchmark Overview

### Test Parameters

**Dataset characteristics:**
- **Pixels:** 1,000,000 (realistic large dataset)
- **M/Z bins:** 400,000 (common mass axis size)
- **Avg peaks/pixel:** ~2,000 (0.5% density)
- **Total non-zeros:** ~2 billion values

### Methods Compared

1. **LIL Matrix (Baseline)** - Current production approach
   - Uses `scipy.sparse.lil_matrix`
   - Expected: 25-35 minutes, memory fragmentation visible

2. **COO Pre-allocated (Proposed)** - Optimized approach
   - Pre-allocates 3 numpy arrays (rows, cols, data)
   - Direct array assignment (no Python list overhead)
   - Expected: 8-15 minutes, consistent performance

3. **Batched LIL (Alternative)** - Chunked approach
   - Builds LIL in 100k pixel batches
   - Converts each batch to CSR, then stacks
   - Expected: 15-25 minutes

## How to Run

### On Linux/Mac

```bash
# Install dependencies
poetry install

# Run benchmark
poetry run python benchmarks/sparse_matrix_construction_benchmark.py
```

### On Windows

```bash
# Option 1: Using poetry
poetry run python benchmarks/sparse_matrix_construction_benchmark.py

# Option 2: Using system Python (auto-installs dependencies)
python benchmarks/install_deps_and_run.py
```

### Quick Test (Smaller Dataset)

To run a faster test with fewer pixels, edit the `main()` function in the benchmark script:

```python
benchmark = SparseMatrixBenchmark(
    n_pixels=100_000,   # Reduced from 1M for quick test
    n_mz=400_000,
    avg_peaks=2000,
)
```

## Expected Runtime

- **Full benchmark (1M pixels):** 40-60 minutes total
  - LIL Matrix: 25-35 minutes
  - COO Pre-allocated: 8-15 minutes
  - Batched LIL: 15-25 minutes

- **Quick test (100k pixels):** 4-6 minutes total

## Output

### Console Output

The benchmark prints:
1. Progress updates every 100k pixels
2. Interval and overall throughput (spec/sec)
3. Timing and memory statistics for each method
4. Verification that all methods produce identical results
5. Performance comparison table
6. Recommendation on whether to proceed with optimization

### Example Output

```
======================================================================
SPARSE MATRIX CONSTRUCTION BENCHMARK
======================================================================
Dataset parameters:
  Pixels:           1,000,000
  M/Z bins:         400,000
  Avg peaks/pixel:  2,000
  Est. sparsity:    0.50%
  Est. total NNZ:   2,000,000,000 (2.00B)
======================================================================

[LIL] Benchmarking LIL Matrix (current approach)...
----------------------------------------------------------------------
  100,000 pixels | Interval:  1,500 spec/s | Overall:  1,500 spec/s | Elapsed:   66.7s
  200,000 pixels | Interval:  1,450 spec/s | Overall:  1,475 spec/s | Elapsed:  135.6s
  ...
  900,000 pixels | Interval:    400 spec/s | Overall:    950 spec/s | Elapsed:  947.4s
  [DONE] Conversion completed: 2.15s
  [DONE] Total time: 1,850.32s
  [DONE] Peak memory: 9.45 GB

[COO] Benchmarking COO Pre-allocated (proposed approach)...
----------------------------------------------------------------------
  100,000 pixels | Interval:  1,850 spec/s | Overall:  1,850 spec/s | Elapsed:   54.1s
  ...
  1,000,000 pixels | Interval:  1,820 spec/s | Overall:  1,835 spec/s | Elapsed:  545.0s
  [DONE] Total time: 548.67s
  [DONE] Peak memory: 6.23 GB

======================================================================
BENCHMARK SUMMARY
======================================================================
Method                        Time (s)   Memory (GB)   Rate (spec/s)
----------------------------------------------------------------------
LIL Matrix                    1,850.3        9.45           540
COO Pre-allocated               548.7        6.23         1,823
  → 3.37x speedup vs baseline
Batched LIL (100k)            1,245.8        7.81           803
  → 1.49x speedup vs baseline
======================================================================

RECOMMENDATION:
  [PROCEED] Use COO Pre-allocated optimization
     3.37x faster than current approach
```

### JSON Output

Results are saved to:
```
benchmarks/results/sparse_construction_benchmark.json
```

Contains detailed metrics for analysis and plotting.

## Interpreting Results

### Success Criteria

**Proceed with COO optimization if:**
- ✅ Speedup ≥ 2.0x vs baseline
- ✅ Memory usage similar or lower than LIL
- ✅ All methods produce identical CSR matrices

**Not recommended if:**
- ❌ Speedup < 1.5x (not worth implementation complexity)
- ❌ Significantly higher memory usage
- ❌ Results differ between methods

### What to Look For

1. **Performance degradation:**
   - LIL should show slowing at 700k-900k pixels
   - COO should maintain consistent rate throughout

2. **Memory efficiency:**
   - LIL: ~8-12 GB peak (high due to Python list overhead)
   - COO: ~5-7 GB peak (pure numpy arrays)

3. **Verification:**
   - All methods must produce identical CSR matrices
   - If verification fails, investigation needed before production use

## Next Steps After Benchmark

### If Results are Positive (≥2x speedup)

1. **Review results** on more powerful machine
2. **Implement COO builder** in production code:
   - `thyra/converters/spatialdata/base_spatialdata_converter.py`
   - Add `_COOMatrixBuilder` helper class
   - Replace `_create_sparse_matrix()` logic
   - Update `_add_to_sparse_matrix()` for COO accumulation

3. **Run integration tests** to verify no regressions
4. **Benchmark on real data** (Bruker .d files)
5. **Create pull request** with benchmark results

### If Results are Inconclusive

1. Try different batch sizes for Batched LIL approach
2. Profile to identify specific bottlenecks
3. Consider hybrid approaches (e.g., COO + chunking)
4. Test with different dataset sizes and densities

## Technical Details

### Why LIL is Slow

- Python lists store elements as pointers (2x memory overhead)
- Each integer wrapped in PyObject (additional overhead)
- Dynamic reallocations as lists grow
- Poor cache locality

### Why COO is Fast

- Typed numpy arrays (native C storage)
- No Python object overhead
- Pre-allocated (no reallocations)
- Excellent cache locality
- Zero-copy conversion to CSR

### Memory Layout Comparison

**LIL Matrix:**
```
sparse_matrix[pixel_idx, mz_indices] = intensities
# Internally: Python list of lists with PyObject wrappers
# Memory: ~8-12 GB for 1M × 400k matrix
```

**COO Pre-allocated:**
```
rows[idx:idx+n] = pixel_idx       # int32 array
cols[idx:idx+n] = mz_indices      # int32 array
data[idx:idx+n] = intensities     # float64 array
# Memory: ~5-7 GB for 2B non-zeros
```

## Troubleshooting

### ModuleNotFoundError: No module named 'numpy'

Use the helper script that auto-installs dependencies:
```bash
python benchmarks/install_deps_and_run.py
```

### UnicodeEncodeError on Windows

The benchmark has been updated to use ASCII-safe output (no emoji).
If you still see this error, ensure you're using the latest version.

### Out of Memory

Reduce dataset size in `main()`:
```python
benchmark = SparseMatrixBenchmark(
    n_pixels=100_000,  # Reduce this
    n_mz=400_000,
    avg_peaks=2000,
)
```

### Benchmark Runs Forever

Expected runtime is 40-60 minutes for 1M pixels.
For quick testing, use 100k pixels (4-6 minutes).

## Questions?

See the main repository README or open an issue on GitHub.
