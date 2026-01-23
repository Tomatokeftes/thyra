# Streaming Converter Design

## Goal

Create a memory-efficient streaming converter for MSI data that:
- Keeps memory usage constant regardless of dataset size
- Processes data in chunks, writing to disk as we go
- Produces identical output to the standard converter

## Problem Statement

Current converters accumulate all COO triplets (row, col, data) in memory before
building the sparse matrix. For a 1M pixel x 100K mass bin dataset with ~500
peaks per spectrum:
- ~500M non-zero values
- Memory usage: 8+ GB just for COO triplets
- Dense matrix would be 800 GB

## Chosen Approach: Incremental COO with Chunked Zarr

**NOT using Dask** - After testing, we found a simpler pure Python/NumPy/Zarr
approach that achieves bounded memory during processing.

### How It Works

The streaming converter uses COO (Coordinate) format with chunked writes:

1. Process spectra in batches (default: 5000 spectra per chunk)
2. For each spectrum, calculate pixel_idx from (x, y, z) coordinates
3. Accumulate COO triplets (row, col, data) in memory
4. When chunk is full, concatenate and write to Zarr arrays on disk
5. Clear memory and continue with next chunk
6. At end, read all chunks back, create scipy COO matrix, convert to CSC/CSR

This approach handles sparse datasets correctly where not all pixels have spectra.

### Why Not Dask?

Dask was considered but not needed because:
- Sequential row-order processing is natural for MSI data
- Zarr handles chunked I/O efficiently
- No need for parallel coordination overhead
- Simpler implementation, easier to debug

### Why COO instead of CSR?

Initial testing with incremental CSR showed issues when:
- Not all pixels have spectra (sparse pixel coverage)
- Spectra don't come in strict pixel order

COO format naturally handles these cases since it just stores (row, col, data)
triplets without requiring contiguous row structure.

## Test Results

### Proof of Concept (250K pixels, 50K masses, 500 peaks/spectrum, 125M nnz)

| Approach | Time | During Processing | After Assembly |
|----------|------|-------------------|----------------|
| In-memory accumulation | ~60s* | 8.4 GB | N/A |
| COO chunks to Zarr | 20s | 167 MB | 2.0 GB |
| Incremental CSR | 12s | 77 MB | 1.5 GB |

*Extrapolated from 1M pixel test

### Real Data Test (pea.imzML: 17,423 pixels, 10,000 m/z bins, 37M nnz)

| Converter | Time | Memory (start -> peak) | Success |
|-----------|------|------------------------|---------|
| Standard (SpatialData2D) | 7.32s | 243 -> 733 MB | Yes |
| Streaming | 9.79s | 285 -> 750 MB | Yes |

**Validation Results:**
- Matrix shapes: IDENTICAL (17423, 10000)
- Non-zeros: IDENTICAL (37,212,927)
- Max data difference: **0.0** (bit-perfect match)
- Output: **IDENTICAL**

### Key Findings

1. **Processing phase is bounded** - Memory stays constant during chunk processing
2. **Final assembly spike is unavoidable** - Need to load sparse matrix components
3. **Streaming is slightly slower** - ~30% overhead from disk I/O
4. **Output is identical** - Bit-perfect match with standard converter

## Implementation

### Phase 1: Proof of Concept [DONE]
- [x] Create fake data generator
- [x] Test chunked writes to Zarr
- [x] Test incremental CSR construction
- [x] Verify memory stays bounded (77 MB)

### Phase 2: Full Implementation [DONE]
- [x] Implement StreamingSpatialDataConverter class
- [x] Handle real reader (ImzML, Bruker) instead of fake data
- [x] Handle all SpatialData components (tables, shapes, images)
- [x] Add progress reporting and timing

### Phase 3: Validation [DONE]
- [x] Compare outputs with standard converter (must be identical)
- [x] Memory profiling with real datasets
- [x] Performance benchmarking

## Usage

```python
from thyra.readers.imzml import ImzMLReader
from thyra.converters.spatialdata import StreamingSpatialDataConverter

reader = ImzMLReader("data.imzML")
converter = StreamingSpatialDataConverter(
    reader=reader,
    output_path="output.zarr",
    dataset_id="my_dataset",
    chunk_size=5000,  # Spectra per chunk (default: 5000)
    # temp_dir=Path("/tmp/streaming"),  # Optional: custom temp directory
)
success = converter.convert()
```

## Key Design Decisions

1. **Sparse format: COO -> CSC**
   - COO for accumulation (handles sparse pixel coverage)
   - Convert to CSC at end (SpatialData default)

2. **Chunk size: 5000 spectra (default)**
   - Good balance of I/O efficiency vs memory
   - Configurable via constructor

3. **No Dask dependency**
   - Simpler, fewer dependencies
   - Pure Python/NumPy/Zarr/SciPy

4. **Temporary Zarr storage**
   - Automatic cleanup after conversion
   - Optional custom temp directory for debugging

## Code Structure

```
thyra/converters/spatialdata/
    streaming_converter.py      # StreamingSpatialDataConverter class
```

Test files (not for production):
```
test_incremental_csr.py         # POC for incremental CSR approach
test_dask_streaming.py          # Earlier Dask exploration (not used)
test_streaming_concept.py       # Earlier COO chunking test
run_streaming_test2.py          # Final validation test
```

## Progress Log

### 2025-01-08
- Deleted old streaming_converter.py (was accumulating in memory, not streaming)
- Tested in-memory approach: 8.4 GB for 1M pixels (baseline)
- Tested COO chunking: 167 MB during processing, 2 GB at assembly
- Tested incremental CSR: 77 MB during processing, 1.5 GB at assembly
- Decision: Use incremental COO approach (handles sparse pixel coverage)
- Implemented full StreamingSpatialDataConverter
- Validated with pea.imzML: **IDENTICAL output** to standard converter
- Streaming converter: 9.79s, 285->750 MB (vs standard: 7.32s, 243->733 MB)

## References

- [Zarr Python](https://zarr.readthedocs.io/)
- [SciPy Sparse Matrices](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- [SpatialData](https://spatialdata.scverse.org/)
- [AnnData on-disk format](https://anndata.readthedocs.io/en/latest/fileformat-prose.html)
