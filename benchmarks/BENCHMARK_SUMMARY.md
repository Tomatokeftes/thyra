# Thyra Benchmark Suite - Summary

## What We've Built

A comprehensive, publication-ready benchmark suite for comparing MSI data formats with focus on demonstrating SpatialData/Zarr advantages over traditional formats (ImzML, Bruker .d).

## Architecture

### Core Modules

1. **[config.py](config.py)** - Central configuration
   - Dataset registry
   - Benchmark parameters
   - Plot styling

2. **[utils.py](utils.py)** - Utility functions
   - File size calculations
   - Timing and profiling
   - Result management

3. **[storage_benchmark.py](storage_benchmark.py)** - Storage efficiency
   - File size comparison
   - Compression ratios
   - Conversion performance

4. **[spatial_access_benchmark.py](spatial_access_benchmark.py)** - Spatial patterns
   - Sequential access
   - ROI extraction
   - Random pixel access

5. **[spectral_access_benchmark.py](spectral_access_benchmark.py)** - Spectral patterns
   - M/z range queries
   - Ion image extraction
   - Multi-m/z queries

6. **[parallel_benchmark.py](parallel_benchmark.py)** - Parallel processing
   - Dask scalability
   - Worker count comparison
   - Parallel operations

7. **[bruker_interpolation_benchmark.py](bruker_interpolation_benchmark.py)** - Interpolation comparison
   - Raw vs resampled (300k bins)
   - Resolution vs performance trade-offs

8. **[visualize.py](visualize.py)** - Visualization
   - Publication-quality plots
   - Matplotlib-based figures
   - 300 DPI output

9. **[run_all.py](run_all.py)** - Main runner
   - Orchestrates all benchmarks
   - Command-line interface
   - Batch execution

## Key Features

### Uses Real Data
- Works with actual test data from `tests/data/`
- No synthetic data generation
- Realistic performance measurements

### Comprehensive Metrics

**Storage:**
- File sizes (bytes, MB, GB)
- Compression ratios
- Conversion time and throughput
- Memory usage

**Access Patterns:**
- Sequential, random, ROI access
- M/z queries and ion images
- Timing for each operation
- Speedup calculations

**Parallel Processing:**
- Scalability metrics
- Worker count comparison
- Efficiency measurements

### Publication Ready

**Outputs:**
- JSON files with detailed metrics
- High-resolution plots (300 DPI)
- Console summaries with tables
- Structured for paper inclusion

**Plots:**
- Bar charts for comparisons
- Line plots for scalability
- Multi-panel figures
- Color-coded by format

## Usage Workflows

### Quick Start (Minimal Data)
```bash
# Using minimal test data
cd benchmarks
poetry run python run_all.py
```

### Full Paper Benchmarks
```bash
# 1. Add your datasets to config.py
# 2. Run all benchmarks
poetry run python run_all.py

# Results in:
# - benchmarks/results/*.json
# - benchmarks/plots/*.png
# - benchmarks/converted/*.zarr
```

### Individual Benchmarks
```bash
# Just storage
poetry run python storage_benchmark.py

# Just spectral access (m/z queries)
poetry run python spectral_access_benchmark.py

# Just visualizations
poetry run python visualize.py
```

### Selective Execution
```bash
# Only specific benchmarks
poetry run python run_all.py --benchmarks storage spatial

# Skip visualization
poetry run python run_all.py --skip-viz
```

## What to Benchmark for Paper

### Must Have (Core Results)

1. **Storage Efficiency**
   - Show compression ratios (2-4x typical)
   - Conversion performance
   - One-time cost vs long-term benefits

2. **Ion Image Extraction** (Most Common Operation)
   - SpatialData: Direct m/z slicing
   - ImzML/Bruker: Must iterate all pixels
   - Expect 10-100x speedup

3. **ROI Extraction**
   - Common analysis workflow
   - Spatial subsetting performance
   - Shows Zarr chunking benefits

4. **Parallel Scaling**
   - Demonstrate Dask integration
   - Near-linear scaling for embarrassingly parallel ops
   - Important for large datasets

### Should Have (Strong Support)

5. **M/z Range Queries**
   - Various range sizes
   - Shows array-based access benefits

6. **Random vs Sequential Access**
   - Zarr chunk optimization

7. **Dataset Size Scaling**
   - Small (MB) to large (GB) datasets
   - Where traditional formats break down

### Nice to Have (If Time/Space)

8. **Bruker Interpolation Comparison**
   - Raw vs 300k bins trade-off
   - Industry standard comparison

9. **Multiple m/z Queries**
   - Batch processing scenarios

10. **Memory Usage**
    - Streaming vs loading all data

## Expected Results

### Storage
- **Compression**: 2-4x smaller
- **Conversion**: One-time cost, ~10-50 MB/s

### Access Patterns
- **Ion images**: 10-100x faster (SpatialData vs original)
- **ROI**: 2-10x faster
- **Sequential**: Comparable or slightly faster

### Parallel Processing
- **2 workers**: ~1.8x speedup
- **4 workers**: ~3.2x speedup
- **8 workers**: ~5-6x speedup (diminishing returns)

### Bruker Interpolation
- **Storage**: 3-10x smaller (300k vs raw)
- **Access**: Similar or faster (fewer m/z bins to process)
- **Resolution**: Sufficient for most workflows

## Cloud Benchmarking

For cloud storage demonstrations:

1. **Setup S3/GCS/Azure**
   - Free tier sufficient for initial tests
   - ~$10-20 for comprehensive benchmarks

2. **Key Differences**
   - Parallel chunk reads shine on cloud
   - Network latency vs throughput
   - Show scaling across machines

3. **Metrics to Report**
   - Same as local benchmarks
   - Add: Network bandwidth utilization
   - Compare: Local vs network vs cloud

## Customization Points

### config.py
```python
# Add your datasets
DATASETS = {...}

# Adjust parameters
N_RANDOM_PIXELS = 100
MZ_RANGES = [(100, 110), ...]
WORKER_COUNTS = [1, 2, 4, 8]
```

### Plotting
```python
# In config.py
PLOT_STYLE = {
    'imzml': '#A23B72',
    'bruker': '#F18F01',
    'spatialdata': '#2E86AB',
    'speedup': '#06A77D'
}
PLOT_DPI = 300
```

### Benchmark Operations
- Edit individual benchmark files
- Add/remove access patterns
- Adjust iteration counts
- Change m/z ranges

## Files to Include in Paper

### Main Text Figures
1. `storage_comparison.png` - Compression ratios
2. `spectral_access_comparison.png` - Ion image extraction speedup
3. `parallel_scalability.png` - Dask scaling

### Supplementary Material
1. `spatial_access_comparison.png` - Detailed access patterns
2. `bruker_interpolation_comparison.png` - Resolution trade-offs

### Tables (from JSON)
- Detailed timing metrics
- Storage statistics
- Conversion performance

## Next Steps

1. **Add Real Datasets**
   - Replace minimal test data
   - Add Bruker .d files
   - Include various sizes

2. **Run Full Benchmarks**
   ```bash
   poetry run python run_all.py
   ```

3. **Cloud Benchmarks** (Optional)
   - Setup S3 bucket
   - Install s3fs
   - Run with cloud paths

4. **Generate Paper Figures**
   - All plots in `plots/` directory
   - 300 DPI, publication ready
   - Adjust colors/layout as needed

5. **Extract Metrics**
   - JSON files in `results/`
   - Import to paper tables
   - Calculate additional statistics

## Maintenance

### Adding New Benchmarks
1. Create new module in `benchmarks/`
2. Follow existing patterns
3. Add to `run_all.py`
4. Add visualization in `visualize.py`

### Updating Datasets
1. Edit `config.py` DATASETS
2. Delete `converted/` to reconvert
3. Rerun benchmarks

### Debugging
- Each module runs standalone
- Check console output
- Inspect JSON results
- Verify converted Zarr files

## Contact

For questions or contributions, please open an issue on the Thyra repository.
