# Thyra Benchmarks

Comprehensive benchmark suite for comparing MSI data formats and demonstrating SpatialData/Zarr advantages.

## Overview

This benchmark suite compares:
- **ImzML** vs **SpatialData/Zarr** (from ImzML)
- **Bruker .d (original)** vs **Zarr (raw)** vs **Zarr (300k bins)**
  - Shows Zarr is better even at full 78M m/z resolution
  - Demonstrates massive gains from resampling

## Benchmark Categories

### 1. Storage Efficiency (`storage_benchmark.py`)
- File sizes (original vs Zarr)
- Compression ratios
- Conversion time and throughput
- Memory usage during conversion

### 2. Spatial Access Patterns (`spatial_access_benchmark.py`)
- Sequential access (full dataset iteration)
- Region of Interest (ROI) extraction
- Random pixel access
- Spatial slicing

### 3. Spectral Access Patterns (`spectral_access_benchmark.py`)
- M/z range queries
- Ion image extraction
- Multiple m/z value queries
- Spectral slicing

**Key insight**: SpatialData/Zarr enables direct m/z axis slicing, avoiding the need to iterate all pixels for m/z queries.

### 4. Parallel Processing (`parallel_benchmark.py`)
- Dask-based parallel computation
- Scalability with different worker counts (1, 2, 4, 8)
- Concurrent ion image extraction
- Parallel normalization operations

### 5. Bruker Format Comparison (`bruker_interpolation_benchmark.py`)
**Compares three Bruker storage formats:**
1. Original Bruker .d (SQLite with ragged arrays, ~3.2GB)
2. SpatialData/Zarr raw (dense array, ~78M m/z bins, ~880MB)
3. SpatialData/Zarr resampled (300k bins, ~50-100MB)

**Key demonstration:** Even with full 78M m/z resolution (no data loss), dense Zarr is 3.6x smaller than ragged SQLite, plus enables direct m/z slicing. Resampling provides massive additional gains.

## Quick Start

### Setup

1. Add your datasets to the configuration:

```python
# Edit config.py
DATASETS = {
    'your_imzml_data': DatasetConfig(
        name='your_imzml_data',
        path=Path('path/to/your.imzML'),
        format_type='imzml',
        description='Your ImzML dataset'
    ),
    'your_bruker_data': DatasetConfig(
        name='your_bruker_data',
        path=Path('path/to/your.d'),
        format_type='bruker',
        description='Your Bruker dataset',
        resampling_config={
            'method': 'bin_width_at_mz',
            'params': {'target_bins': 300000}
        }
    ),
}
```

2. Install dependencies (if needed):
```bash
poetry install
```

### Run All Benchmarks

```bash
cd benchmarks
poetry run python run_all.py
```

### Run Specific Benchmarks

```bash
# Storage only
poetry run python run_all.py --benchmarks storage

# Multiple specific benchmarks
poetry run python run_all.py --benchmarks storage spatial spectral

# Skip visualization generation
poetry run python run_all.py --skip-viz
```

### Run Individual Benchmark Modules

```bash
poetry run python storage_benchmark.py
poetry run python spatial_access_benchmark.py
poetry run python spectral_access_benchmark.py
poetry run python parallel_benchmark.py
poetry run python bruker_interpolation_benchmark.py
```

### Generate Visualizations

```bash
poetry run python visualize.py
```

## Output Structure

```
benchmarks/
├── converted/           # Converted Zarr datasets
│   └── *.zarr/
├── results/            # JSON benchmark results
│   ├── storage_benchmark.json
│   ├── spatial_access_benchmark.json
│   ├── spectral_access_benchmark.json
│   ├── parallel_benchmark.json
│   └── bruker_interpolation_benchmark.json
└── plots/              # Publication-quality figures
    ├── storage_comparison.png
    ├── spatial_access_comparison.png
    ├── spectral_access_comparison.png
    ├── parallel_scalability.png
    └── bruker_interpolation_comparison.png
```

## Configuration

Edit `config.py` to customize:
- Dataset paths and configurations
- Benchmark parameters (ROI size, m/z ranges, etc.)
- Parallel worker counts
- Plot styling and output settings

## Key Metrics Reported

### Storage
- Original file size (MB/GB)
- Zarr file size (MB/GB)
- Compression ratio
- Conversion throughput (MB/s)

### Access Patterns
- Time (seconds) for various operations
- Speedup factors (SpatialData vs original)
- Memory usage

### Parallel Scaling
- Speedup relative to single worker
- Efficiency at different worker counts

## For Paper/Publication

The benchmark suite generates:
1. **Quantitative metrics** (JSON files) for tables
2. **Publication-quality plots** (PNG, 300 DPI) for figures
3. **Comprehensive summaries** printed to console

## Notes

- Benchmarks use **real test data** from `tests/data/` directory
- First run will convert datasets (one-time cost)
- Subsequent runs reuse converted Zarr files
- Parallel benchmarks require converted datasets
- Results are deterministic (random seeds set)

## Cloud Benchmarking

For cloud storage benchmarks (S3, GCS, Azure Blob):

1. Install cloud storage libraries:
```bash
poetry add s3fs gcsfs adlfs
```

2. Configure credentials according to provider documentation

3. Update dataset paths in `config.py` to use cloud URIs:
```python
path=Path('s3://bucket-name/dataset.zarr')
```

## Troubleshooting

- **Missing dependencies**: Run `poetry install`
- **No datasets found**: Check paths in `config.py`
- **Bruker interpolation benchmark skipped**: Add both raw and resampled Bruker configs
- **Parallel benchmarks failing**: Ensure sufficient memory (2GB per worker)

## Contact

For questions or issues related to benchmarks, please open an issue on the Thyra repository.
