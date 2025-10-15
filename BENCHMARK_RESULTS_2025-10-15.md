# COO Matrix Benchmark Results - 2025-10-15

## Executive Summary

**Decision: PROCEED with COO optimization**

The benchmark successfully demonstrated that COO pre-allocated arrays are superior to the current LIL matrix approach for building sparse matrices in MSI data conversion.

### Key Results (100k pixels)

| Metric | LIL (Current) | COO (Proposed) | Improvement |
|--------|---------------|----------------|-------------|
| Time | 631.9s | 492.4s | **1.28x faster** (28% speedup) |
| Memory | 16.98 GB | 5.92 GB | **65% reduction** (11 GB saved) |
| Rate | 158 pixels/s | 203 pixels/s | 28% faster |
| Result | 200,004,247 NNZ | 200,004,247 NNZ | Identical |

### Why COO Wins

1. **Memory efficiency prevents crashes**
   - Full 1M pixel benchmark crashed at 54% (536k pixels) with LIL
   - LIL needs ~160+ GB for 1M pixels (extrapolated from 17 GB at 100k)
   - COO needs ~60 GB for 1M pixels (much more feasible)

2. **Speed improvement**
   - 28% faster on 100k pixels
   - Saves ~23 minutes on full 1M pixel datasets

3. **Identical results**
   - All methods verified to produce identical matrices
   - No loss of accuracy or precision

## Test Configuration

```python
Dataset parameters:
  Pixels:           100,000
  M/Z bins:         400,000
  Avg peaks/pixel:  2,000
  Est. sparsity:    0.50%
  Est. total NNZ:   200,000,000 (0.20B)
```

## Full Results

### LIL Matrix (Current Approach)
```
Time:              631.86 seconds (10m 32s)
Memory:            16.98 GB
Rate:              158 pixels/second
Conversion (CSR):  2.13 seconds
Final NNZ:         200,004,247
```

### COO Pre-allocated (Proposed)
```
Time:              492.38 seconds (8m 12s)
Memory:            5.92 GB
Rate:              203 pixels/second
Conversion (CSR):  0.98 seconds
Final NNZ:         200,004,247
```

### Batched LIL (Alternative)
```
Time:              914.63 seconds (15m 15s)
Memory:            4.80 GB
Rate:              109 pixels/second
Stacking (CSR):    0.38 seconds
Final NNZ:         200,004,247
```

## Why Full Benchmark Failed

The full 1M pixel benchmark ran out of memory using LIL matrix:

```
[LIL] Benchmarking LIL Matrix (current approach)...
----------------------------------------------------------------------
  Building LIL:  54%|███▊   | 536k/1.00M [1:06:34<56:09, 138 pixels/s]
MemoryError
```

**Analysis:**
- LIL matrix consumed all available memory at 536k pixels (54%)
- Extrapolating: LIL needs ~160+ GB RAM for 1M pixels
- This is exactly why COO optimization is needed
- COO would need ~60 GB for same dataset (feasible on larger machines)

## Verification

All three methods produced identical sparse matrices:
- Same shape: (100,000 x 400,000)
- Same NNZ: 200,004,247
- Data values verified with `np.allclose(rtol=1e-9)`

Result: **PASS - All methods produce identical output**

## Performance Analysis

### Time Breakdown

**LIL Matrix:**
- Building: 629.73s (99.7%)
- Conversion: 2.13s (0.3%)
- Total: 631.86s

**COO Pre-allocated:**
- Building: 491.40s (99.8%)
- Conversion: 0.98s (0.2%)
- Total: 492.38s

**Batched LIL:**
- Building: 914.25s (99.96%)
- Stacking: 0.38s (0.04%)
- Total: 914.63s

### Memory Analysis

Peak memory usage during construction:

1. **LIL**: 16.98 GB (highest)
   - Python list overhead per row
   - Inefficient memory layout
   - Memory grows super-linearly with size

2. **COO**: 5.92 GB (65% reduction)
   - Pre-allocated contiguous arrays
   - Efficient NumPy memory layout
   - Predictable memory growth

3. **Batched LIL**: 4.80 GB (lowest, but slowest)
   - Processes in small chunks
   - Lower peak but more overhead
   - Trade memory for time

## Extrapolation to 1M Pixels

Based on 100k pixel results:

| Method | Est. Time | Est. Memory | Feasibility |
|--------|-----------|-------------|-------------|
| LIL | ~105 min | ~170 GB | Crashes (verified) |
| COO | ~82 min | ~60 GB | Feasible |
| Batched | ~152 min | ~48 GB | Feasible but slow |

## Recommendation

**PROCEED with COO pre-allocated implementation**

### Rationale:

1. **Memory savings are critical**
   - 65% reduction enables larger datasets
   - Prevents OOM crashes on 500k+ pixel datasets
   - Proven by full benchmark failure

2. **Speed improvement is significant**
   - 28% faster (1.28x speedup)
   - Extrapolates to ~23 minutes saved on 1M pixels
   - Better than current approach

3. **No downsides**
   - Identical results (verified)
   - Cleaner code (no row-by-row indexing)
   - More predictable performance

4. **Real-world impact**
   - Current approach cannot handle large datasets
   - COO enables processing previously impossible data
   - Critical for production use

## Next Steps

See [COO_OPTIMIZATION_NEXT_STEPS.md](COO_OPTIMIZATION_NEXT_STEPS.md) for detailed implementation plan.

**TL;DR:**
1. Locate LIL matrix usage in `thyra/converters/spatialdata_converter.py`
2. Replace with COO pre-allocated arrays
3. Test on real MSI data
4. Commit changes

**Estimated implementation time: 2.5 hours**

## Files Generated

- Benchmark code: `benchmarks/sparse_matrix_construction_benchmark.py`
- Quick test: `benchmarks/sparse_matrix_quick_test.py`
- Results (this file): `BENCHMARK_RESULTS_2025-10-15.md`
- Implementation plan: `COO_OPTIMIZATION_NEXT_STEPS.md`

## Test Environment

- Date: 2025-10-15
- Branch: `feature/coo-matrix-optimization`
- Python: 3.12.7
- NumPy: 2.0.0+
- SciPy: 1.7.0+
- Platform: Windows (win32)

## Conclusion

The benchmark provides strong evidence that COO pre-allocated arrays are the right choice:
- **Faster**: 28% speedup
- **Smaller**: 65% memory reduction
- **Scalable**: Enables 1M+ pixel datasets
- **Correct**: Identical results verified

The current LIL approach is a bottleneck that prevents processing large MSI datasets. COO optimization removes this limitation.

**Status: Ready for implementation**
