# COO Matrix Optimization - Next Steps

## Benchmark Results Summary (100k pixels)

Test completed successfully on 2025-10-15 with the following results:

| Method            | Time (s) | Memory (GB) | Rate (pixels/s) | Speedup |
|-------------------|----------|-------------|-----------------|---------|
| LIL Matrix        | 631.9    | 16.98       | 158             | 1.00x   |
| COO Pre-allocated | 492.4    | 5.92        | 203             | 1.28x   |
| Batched LIL       | 914.6    | 4.80        | 109             | 0.69x   |

### Key Findings

1. **COO is 1.28x faster than LIL** (28% speedup)
   - Time savings: 139 seconds (2.3 minutes) on 100k pixels
   - Extrapolated to 1M pixels: ~23 minutes faster

2. **COO uses 65% less memory** (5.92 GB vs 16.98 GB)
   - Memory reduction: 11 GB saved on 100k pixels
   - This is why the full 1M pixel benchmark crashed at 54%
   - Critical for scaling to large datasets

3. **All methods verified identical** (200,004,247 NNZ each)

### Why the Full Benchmark Failed

The full 1M pixel benchmark ran out of memory at 54% (536k pixels) using LIL matrix:
- LIL needed ~16-17 GB for 100k pixels
- Extrapolated: ~160+ GB for 1M pixels (not feasible)
- COO would need ~60 GB for 1M pixels (much more reasonable)

## Decision: PROCEED with COO Optimization

Even though speedup is 1.28x (not 2-3x as hoped), the memory savings alone justify the change:
- 65% memory reduction enables processing larger datasets
- Prevents out-of-memory crashes on large MSI data
- 28% speedup is a bonus
- No downsides (identical results, cleaner code)

---

## Next Steps for Tomorrow

### Step 1: Verify Benchmark Data Integrity
**Time estimate: 5 minutes**

Check that the saved JSON results file was created correctly:

```bash
# Look for the results file
ls benchmarks/results/

# If it exists, check the contents
cat benchmarks/results/sparse_construction_quick_test.json
```

If the file wasn't saved, re-run the quick test to capture the data.

---

### Step 2: Locate Current LIL Matrix Usage in Production Code
**Time estimate: 15 minutes**

Find where the current LIL matrix construction happens in the actual codebase:

```bash
# Search for LIL matrix usage
poetry run python -c "import grep; grep.search('lil_matrix', 'thyra/')"

# Or use grep directly
grep -r "lil_matrix" thyra/ --include="*.py"

# Also look for the conversion process
grep -r "sparse.*matrix" thyra/converters/ --include="*.py"
```

Expected location: `thyra/converters/spatialdata_converter.py`

---

### Step 3: Review Current Implementation
**Time estimate: 10 minutes**

Read and understand the current sparse matrix construction code:

1. Open the converter file (likely `thyra/converters/spatialdata_converter.py`)
2. Find the method that builds the intensity matrix
3. Document the current approach (for comparison later)
4. Note any special cases or edge conditions

---

### Step 4: Implement COO Matrix Optimization
**Time estimate: 30-45 minutes**

Replace the LIL matrix approach with COO pre-allocated arrays.

**Before (current LIL approach):**
```python
# Current approach (slow, memory-intensive)
lil = sparse.lil_matrix((n_pixels, n_mz), dtype=np.float64)
for pixel_idx, mz_indices, intensities in data:
    lil[pixel_idx, mz_indices] = intensities
csr = lil.tocsr()
```

**After (new COO approach):**
```python
# Pre-allocate arrays
estimated_nnz = estimate_total_peaks(data)  # Calculate expected size
rows = np.empty(estimated_nnz, dtype=np.int32)
cols = np.empty(estimated_nnz, dtype=np.int32)
data = np.empty(estimated_nnz, dtype=np.float64)

# Fill arrays directly
current_idx = 0
for pixel_idx, mz_indices, intensities in data:
    n = len(mz_indices)
    rows[current_idx:current_idx + n] = pixel_idx
    cols[current_idx:current_idx + n] = mz_indices
    data[current_idx:current_idx + n] = intensities
    current_idx += n

# Trim and convert
rows = rows[:current_idx]
cols = cols[:current_idx]
data = data[:current_idx]
coo = sparse.coo_matrix((data, (rows, cols)), shape=(n_pixels, n_mz))
csr = coo.tocsr()
```

**Key implementation details:**
1. Add 10% buffer to estimated_nnz for variance
2. Use int32 for indices (saves memory vs int64)
3. Use float64 for data (required for scientific accuracy)
4. Trim arrays before creating COO matrix
5. Add progress logging for large datasets

---

### Step 5: Update Tests
**Time estimate: 20 minutes**

Ensure existing tests still pass with the new implementation:

```bash
# Run converter tests
poetry run pytest tests/unit/converters/ -v

# Run integration tests if available
poetry run pytest tests/integration/ -v

# Run full test suite
poetry run pytest
```

If tests fail, investigate whether:
- Results are identical (may just be floating-point precision)
- Test expects specific matrix type (update test)
- Actual bug in implementation (fix it)

---

### Step 6: Test on Real MSI Data
**Time estimate: 30 minutes**

Test the new implementation on actual MSI datasets:

```bash
# Convert a small dataset
poetry run thyra convert <path-to-small-dataset> --output test_output

# Convert a medium dataset (if available)
poetry run thyra convert <path-to-medium-dataset> --output test_output2

# Check the outputs are valid
poetry run python -c "
import spatialdata as sd
sdata = sd.read_zarr('test_output')
print(f'Loaded successfully: {sdata}')
"
```

Compare:
- Conversion time (should be faster)
- Memory usage (should be lower)
- Output correctness (should be identical)

---

### Step 7: Document Changes
**Time estimate: 15 minutes**

Update documentation to reflect the optimization:

1. Add a note to `CHANGELOG.md` or release notes
2. Update any performance documentation
3. Add comments in the code explaining the COO approach
4. Document the memory savings in README if relevant

Example changelog entry:
```markdown
## [Unreleased]

### Changed
- Optimized sparse matrix construction using COO format instead of LIL
  - 28% faster conversion (1.28x speedup)
  - 65% less memory usage (enables larger datasets)
  - Prevents out-of-memory crashes on datasets >500k pixels
```

---

### Step 8: Commit Changes
**Time estimate: 5 minutes**

Create a clean commit with the optimization:

```bash
# Stage changes
git add thyra/converters/

# Commit with conventional commit format
git commit -m "perf: optimize sparse matrix construction with COO format

Replace LIL matrix with pre-allocated COO arrays for building intensity matrices.

Performance improvements (tested on 100k pixels):
- 28% faster conversion time (632s -> 492s)
- 65% less memory usage (17 GB -> 6 GB)
- Enables processing of larger datasets without OOM errors

Benchmark results show identical output with improved efficiency.
This change prevents memory crashes observed at ~500k+ pixels.
"
```

---

## Optional: Full-Scale Benchmark on Powerful Machine

If you have access to a machine with 64+ GB RAM:

```bash
# Run the full 1M pixel benchmark
poetry run python benchmarks/sparse_matrix_construction_benchmark.py
```

This would give more authoritative results, but the 100k pixel test is sufficient for decision-making.

---

## Files to Keep

- `benchmarks/sparse_matrix_construction_benchmark.py` - Full benchmark
- `benchmarks/sparse_matrix_quick_test.py` - Quick test (useful for future validation)
- `BENCHMARK_QUICKSTART.md` - Instructions for running benchmarks
- `COO_OPTIMIZATION_NEXT_STEPS.md` - This file (tomorrow's plan)

## Files to Delete

- None - all files are useful for reference and future benchmarking

---

## Success Criteria

The optimization is successful if:
1. All existing tests pass
2. Real MSI data converts correctly
3. Conversion is faster (any speedup is good)
4. Memory usage is lower (prevents crashes)
5. Output data is identical to before

---

## Estimated Total Time

- Verification and setup: 30 minutes
- Implementation: 45 minutes
- Testing: 50 minutes
- Documentation: 20 minutes

**Total: ~2.5 hours**

---

## Questions to Consider

1. **Should we add memory profiling to the converter?**
   - Could track and log memory usage during conversion
   - Helps users understand resource requirements

2. **Should we add a progress bar to the actual converter?**
   - The benchmark has tqdm progress bars
   - Real conversions could benefit from the same

3. **Should we make batch size configurable?**
   - Currently hardcoded in benchmark
   - Could be useful for memory-constrained systems

---

## Contact/Notes

- Benchmark completed: 2025-10-15
- Branch: `feature/coo-matrix-optimization`
- Quick test results saved to: `benchmarks/results/sparse_construction_quick_test.json`
- Full benchmark crashed at 54% due to memory (expected behavior for LIL)

**Bottom line: The benchmark proves COO is better. Time to implement it in production!**
