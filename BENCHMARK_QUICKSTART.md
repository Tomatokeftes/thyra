# Quick Start: Running Sparse Matrix Benchmark on Powerful Machine

## What This Is

A comprehensive benchmark to test if COO (Coordinate) matrix construction is 2-3x faster than the current LIL (List of Lists) approach for MSI data conversion.

## Quick Commands

### 1. Clone/Pull the Repository

```bash
git clone https://github.com/Tomatokeftes/thyra.git
cd thyra
git checkout feature/coo-matrix-optimization
git pull
```

Or if already cloned:
```bash
cd thyra
git fetch
git checkout feature/coo-matrix-optimization
git pull
```

### 2. Run the Benchmark

```bash
# Option 1: With Poetry (recommended)
poetry install
poetry run python benchmarks/sparse_matrix_construction_benchmark.py

# Option 2: With system Python (auto-installs numpy & scipy)
python benchmarks/install_deps_and_run.py

# Option 3: Manual install
pip install numpy>=2.0.0 scipy>=1.7.0
python benchmarks/sparse_matrix_construction_benchmark.py
```

### 3. Expected Runtime

- **Full benchmark (1M pixels):** 40-60 minutes total
- **Quick test (100k pixels):** 4-6 minutes (edit script to reduce n_pixels)

### 4. What to Expect

The benchmark will:
1. Test LIL matrix (current, ~25-35 min, shows degradation)
2. Test COO pre-allocated (proposed, ~8-15 min, consistent performance)
3. Test batched LIL (alternative, ~15-25 min)
4. Verify all produce identical results
5. Print comparison table with recommendation

### 5. Success Criteria

**Good result (proceed with optimization):**
- COO is ≥2x faster than LIL
- Memory usage similar or lower
- All methods produce identical matrices

**Example output:**
```
Method                    Time (s)   Memory (GB)   Rate (spec/s)
LIL Matrix                1,850.3        9.45           540
COO Pre-allocated           548.7        6.23         1,823
  → 3.37x speedup vs baseline ✅

RECOMMENDATION: [PROCEED] Use COO Pre-allocated optimization
```

## Results Location

- Console output: Full details printed during run
- JSON file: `benchmarks/results/sparse_construction_benchmark.json`

## What to Report Back

After the benchmark completes, report:
1. **Speedup factor:** e.g., "COO is 3.2x faster than LIL"
2. **Memory comparison:** e.g., "COO used 6.2 GB vs LIL's 9.5 GB"
3. **Verification status:** "All methods produced identical results: YES/NO"
4. **Recommendation:** "PROCEED / CONSIDER / NOT RECOMMENDED"

## Optional: Quick Test First

To verify everything works before the long run, edit `sparse_matrix_construction_benchmark.py`:

```python
# Line ~437, change:
benchmark = SparseMatrixBenchmark(
    n_pixels=100_000,    # Changed from 1_000_000
    n_mz=400_000,
    avg_peaks=2000,
)
```

This runs in 4-6 minutes instead of 40-60 minutes.

## Files Added

- `benchmarks/sparse_matrix_construction_benchmark.py` - Main benchmark
- `benchmarks/install_deps_and_run.py` - Helper for auto-install
- `benchmarks/README_SPARSE_BENCHMARK.md` - Full documentation

## Current Branch Info

- Branch: `feature/coo-matrix-optimization`
- Commits ahead of main: 2
- Status: Ready to run benchmark

## Next Steps After Benchmark

1. ✅ Run benchmark on powerful machine
2. 📊 Share results (speedup, memory, verification)
3. 🚀 If ≥2x speedup → Implement COO in production code
4. 🧪 If ≥2x speedup → Run integration tests
5. 📝 If ≥2x speedup → Create PR with before/after metrics

---

**TL;DR:**
```bash
git checkout feature/coo-matrix-optimization
poetry run python benchmarks/sparse_matrix_construction_benchmark.py
# Wait 40-60 minutes
# Check if COO is ≥2x faster → Report back
```
