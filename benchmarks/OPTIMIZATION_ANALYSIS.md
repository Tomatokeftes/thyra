# MSI Resampling Optimization Analysis

## Current Performance (Baseline)

**Full Conversion Performance:**
- **Total time**: 35.4 seconds
- **Processing speed**: 951 spectra/s
- **Dataset**: 33,690 spectra, 300,000 m/z bins

**Time Breakdown:**
- Resampling (`_nearest_neighbor_resample`): 19.1s (54%)
- SDK reading (Bruker): 2.3s (6.5%)
- m/z conversion: 1.6s (4.5%)
- Sparse matrix operations: 0.6s (1.7%)
- I/O and other: ~12s (33%)

## Why JIT Optimization Was Rejected

### Initial Attempt: Numba JIT
We attempted to JIT-compile the `_nearest_neighbor_resample()` function using Numba, expecting 2-5x speedup.

**Results:**
- Isolated function speedup: **1.38x** (0.842ms → 0.611ms)
- Full pipeline speedup: **1.09x** (35.4s → 32.6s, with JIT compilation overhead)
- Time saved: Only 2.8 seconds

### Why So Little Improvement?

**NumPy already uses compiled C/Fortran code:**
- `searchsorted()` - binary search in C
- `bincount()` - accumulation in C
- `where()` - boolean indexing in C
- `clip()` - vectorized in C

**The bottleneck is algorithmic, not implementation:**
- 33,690 spectra × 800 peaks = 27 million operations
- Each operation: binary search on 300k bins
- This is inherently O(n × m × log(k)) where:
  - n = 33,690 spectra
  - m = ~800 peaks per spectrum
  - k = 300,000 bins

**Verdict:** JIT doesn't help when you're already calling highly-optimized C code.

## Why Vectorization Across Spectra Won't Work

### The Problem: Per-Spectrum Operation

**Each spectrum has different characteristics:**
- Different number of peaks (~800 average, but varies)
- Different m/z values (mass shift between pixels)
- Different peak patterns

**Vectorization would require:**
- Padding all spectra to max length → huge memory waste
- Batch processing → still need per-spectrum resampling due to unique m/z values
- Cache invalidation from huge arrays

**Why caching won't help:**
- Mass shift means m/z values differ slightly between pixels
- At 300k bin resolution (0.003 Da spacing), even tiny shifts map to different bins
- Example: One pixel has peak at 500.123 Da, another at 500.145 Da → different bins
- No reuse opportunity

**Verdict:** Resampling is inherently a per-spectrum operation with this data structure.

## Real Optimization Opportunities

### 1. Reduce Bin Count (Instant 3x speedup)
**Current**: 300,000 bins (0.003 Da spacing)
**Proposed**: 100,000 bins (0.01 Da spacing)

**Impact:**
- `searchsorted` speed: O(log k) → log(100k) vs log(300k) = 1.47x faster
- `bincount` speed: O(k) → 3x faster
- Memory: 3x less
- **Total estimated speedup**: 2-3x
- **New performance**: ~2,500-3,000 spec/s

**Trade-off:** Slightly lower mass resolution (still excellent for most applications)

### 2. Parallel Processing (Linear scaling)
**Current**: Single-threaded
**Proposed**: Multi-core processing

**Implementation:**
- Process spectra in parallel batches
- Each worker has independent COO arrays
- Merge at end

**Impact:**
- 4 cores → 4x speedup
- 8 cores → 6-7x speedup (diminishing returns from I/O)
- **New performance**: ~4,000-6,000 spec/s

**Trade-off:** More complex implementation, memory overhead per worker

### 3. Combined Approach (6-9x speedup)
**Reduce bins + Parallel processing**:
- 100k bins: 3x
- 4 cores: 4x
- **Combined**: ~10-12x theoretical, ~6-9x practical
- **New performance**: ~6,000-9,000 spec/s
- **Time**: 35s → 4-6s

## Recommendation

### Priority 1: Reduce Bins (Easy Win)
- Immediate 3x improvement
- Trivial to implement (change config parameter)
- No architectural changes needed
- Still excellent resolution for MS imaging

### Priority 2: Parallel Processing (Big Win)
- 4-6x additional improvement
- Moderate complexity
- Well-defined problem (embarrassingly parallel)
- CPU-bound bottleneck benefits most from parallelization

### Not Recommended:
- ~~JIT compilation~~ - Only 9% improvement, not worth complexity
- ~~Caching~~ - Mass shift prevents reuse
- ~~Cross-spectrum vectorization~~ - Incompatible data structure

## Current Code Status

The codebase uses pure NumPy with already-optimized operations:
- Vectorized searchsorted for nearest neighbor
- Bincount for intensity accumulation
- Sparse COO matrix construction
- All major operations use compiled C code under the hood

**This is already near-optimal for single-threaded, 300k bin resolution.**

Further gains require:
1. Reducing problem size (fewer bins)
2. Parallelization (more cores)
3. Algorithmic changes (different data structure - unlikely to help)

## Benchmark Scripts

Maintained scripts for performance analysis:
- `benchmarks/profile_full_conversion.py` - Full pipeline profiling
- `benchmarks/profile_resample_details.py` - Detailed operation breakdown
- `benchmarks/verify_data_integrity.py` - Data validation after changes

## Conclusion

**NumPy is already fast** because it uses compiled C/Fortran for all heavy lifting. The 54% of time spent in resampling is not due to Python overhead - it's the algorithmic cost of mapping 27 million values to 300k bins.

**To go faster, change the problem:**
- Reduce bins (easy, 3x)
- Add parallelism (moderate, 4x)
- Both (hard, 9x)
