# Complete Resampling Flow: From Input to SpatialData

This document traces the EXACT order of operations when running:
```bash
thyra input.d output.zarr --resample --resample-width-at-mz 0.005
```

---

## PHASE 1: Entry Point & Setup

### Step 1.1: CLI Argument Parsing (`thyra/__main__.py`)
```python
# Parse command line arguments
args.resample = True
args.resample_width_at_mz = 0.005  # User specified
args.resample_bins = None  # Not specified
```

### Step 1.2: Build Resampling Config (`thyra/__main__.py:179-194`)
```python
resampling_config = {
    "target_bins": None,  # Not specified by user
    "min_mz": None,       # Will use dataset min
    "max_mz": None,       # Will use dataset max
    "width_at_mz": 0.005, # User specified: 5 mDa
    "reference_mz": 1000.0  # Default
}
```

### Step 1.3: Call convert_msi (`thyra/convert.py:198`)
```python
convert_msi(
    input_path="input.d",
    output_path="output.zarr",
    resampling_config=resampling_config,
    ...
)
```

---

## PHASE 2: Reader Initialization

### Step 2.1: Detect Format (`thyra/convert.py:100`)
```python
input_format = detect_format(input_path)
# Returns: "bruker"
```

### Step 2.2: Create Reader (`thyra/convert.py:102-104`)
```python
reader_class = get_reader_class("bruker")
# Returns: BrukerReader class

reader = BrukerReader(input_path)
# Initializes:
# - Opens SQLite database (read-only mode)
# - Loads Bruker SDK DLL
# - Caches coordinate info
```

### Step 2.3: Get Essential Metadata (`thyra/convert.py:123`)
```python
essential_metadata = reader.get_essential_metadata()
# Returns:
# {
#   dimensions: (width, height, depth),
#   mass_range: (250.0, 1200.0),  # From GlobalMetadata table
#   pixel_size: (beam_x, beam_y),
#   n_spectra: 918855,
#   ...
# }
```

---

## PHASE 3: Converter Initialization

### Step 3.1: Create Converter (`thyra/convert.py:241`)
```python
converter_class = get_converter_class("spatialdata")
# Returns: SpatialData2DConverter or SpatialData3DConverter

converter = SpatialData2DConverter(
    reader=reader,
    output_path=output_path,
    resampling_config=resampling_config,  # Passed here!
    ...
)
```

### Step 3.2: Converter Stores Resampling Config (`thyra/converters/spatialdata/base_spatialdata_converter.py:155`)
```python
# In __init__:
self._resampling_config = resampling_config
if self._resampling_config:
    self._target_bins = resampling_config.get("target_bins", None)  # None
    self._min_mz = resampling_config.get("min_mz")  # None
    self._max_mz = resampling_config.get("max_mz")  # None
    self._width_at_mz = resampling_config.get("width_at_mz")  # 0.005
    self._reference_mz = resampling_config.get("reference_mz", 1000.0)
```

---

## PHASE 4: Conversion Starts

### Step 4.1: convert() Method Called (`thyra/core/base_converter.py:76`)
```python
converter.convert()
# Template method that calls:
# 1. _initialize_conversion()
# 2. _create_data_structures()
# 3. _process_spectra()
# 4. _finalize_conversion()
```

### Step 4.2: Initialize Conversion (`thyra/converters/spatialdata/base_spatialdata_converter.py:395`)
```python
def _initialize_conversion(self):
    # Load essential metadata
    self._dimensions = reader.get_essential_metadata().dimensions
    # (1469, 1007, 1) for your dataset

    # CRITICAL: Set up mass axis based on resampling config
    if self._resampling_config:
        # Build resampled mass axis BEFORE processing spectra
        self._build_resampled_mass_axis()
    else:
        # Build raw mass axis (scan all spectra)
        self._common_mass_axis = reader.get_common_mass_axis()
```

---

## PHASE 5: Build Resampled Mass Axis (THE KEY STEP!)

### Step 5.1: _build_resampled_mass_axis() (`base_spatialdata_converter.py:331`)
```python
def _build_resampled_mass_axis(self):
    # Get mass range from metadata
    mass_range = self.reader.get_essential_metadata().mass_range
    min_mz = 250.0  # From Bruker GlobalMetadata
    max_mz = 1200.0 # From Bruker GlobalMetadata

    # Override with user values if provided
    if self._min_mz is not None:
        min_mz = self._min_mz
    if self._max_mz is not None:
        max_mz = self._max_mz
```

### Step 5.2: Calculate Bin Count (`base_spatialdata_converter.py:345`)
```python
# User didn't specify target_bins, so calculate from width
if self._target_bins is None:
    target_bins = self._calculate_bins_from_width(min_mz, max_mz, axis_type)
else:
    target_bins = self._target_bins
```

### Step 5.3: _calculate_bins_from_width() (`base_spatialdata_converter.py:263`)
```python
def _calculate_bins_from_width(self, min_mz, max_mz, axis_type):
    # Use default: 5 mDa at m/z 1000
    width_at_mz = 0.005  # From user
    reference_mz = 1000.0

    # Get metadata for axis type selection
    metadata = reader.get_comprehensive_metadata()
    tree = ResamplingDecisionTree()
    axis_type = tree.select_axis_type(metadata)
    # Returns: AxisType.REFLECTOR_TOF for Bruker timsTOF

    # Calculate bins for REFLECTOR_TOF
    if axis_name == "reflector_tof":
        relative_resolution = reference_mz / width_at_mz
        # = 1000.0 / 0.005 = 200,000

        bins = int(np.log(max_mz / min_mz) * relative_resolution)
        # = int(ln(1200/250) * 200000)
        # = int(1.56862 * 200000)
        # = 313,723 bins
```

### Step 5.4: Generate Resampled Axis (`base_spatialdata_converter.py:360`)
```python
# Select appropriate generator
from thyra.resampling.mass_axis import ReflectorTOFAxisGenerator

generator = ReflectorTOFAxisGenerator()

mass_axis_result = generator.generate_axis(
    min_mz=250.0,
    max_mz=1200.0,
    target_bins=313723,
    reference_mz=1000.0,
    reference_width=0.005
)

# Generator creates m/z values with physics-based spacing
# For TOF: bin_width ∝ m/z (constant relative resolution)
self._common_mass_axis = mass_axis_result.mz_values
# Array of 313,723 m/z values from 250.0 to 1200.0
```

---

## PHASE 6: Create Data Structures

### Step 6.1: _create_data_structures() (`spatialdata_2d_converter.py:29` or `spatialdata_3d_converter.py:28`)
```python
# For 2D data (most common):
def _create_data_structures(self):
    n_x, n_y, n_z = self._dimensions  # (1007, 1469, 1)

    # Create sparse matrix for intensities
    sparse_matrix = sparse.lil_matrix(
        (n_x * n_y * n_z, len(self._common_mass_axis)),
        dtype=np.float64
    )
    # Shape: (1,479,283 pixels, 313,723 m/z bins)

    return {
        "sparse_matrix": sparse_matrix,  # LIL format for construction
        "coords_df": pd.DataFrame(),     # Will store coordinates
        "var_df": pd.DataFrame(),        # Will store m/z values
        "total_intensity": np.zeros(313723),  # For average spectrum
        "pixel_count": 0
    }
```

---

## PHASE 7: Process Spectra (THE SLOW PART!)

### Step 7.1: _process_spectra() (`base_spatialdata_converter.py:481`)
```python
def _process_spectra(self, data_structures):
    # Decide resampling strategy
    tree = ResamplingDecisionTree()
    strategy_type = tree.select_strategy(metadata)
    # Returns: ResamplingMethod.NEAREST_NEIGHBOR for timsTOF

    # Create strategy
    from thyra.resampling.strategies import NearestNeighborStrategy
    resampling_strategy = NearestNeighborStrategy(
        target_axis=self._common_mass_axis  # The 313,723 bin axis
    )

    # Iterate through all spectra
    with tqdm(total=n_spectra, desc="Converting spectra") as pbar:
        for coords, mzs, intensities in reader.iter_spectra():
            # coords: (x, y, z) - pixel position
            # mzs: Original m/z values from spectrum
            # intensities: Original intensity values

            # RESAMPLE THE SPECTRUM
            resampled_intensities = resampling_strategy.resample(
                original_mz=mzs,
                original_intensity=intensities,
                target_mz=self._common_mass_axis
            )
            # Returns: Array of 313,723 values (resampled to common axis)

            # Process the resampled spectrum
            self._process_spectrum(data_structures, coords, resampled_intensities)
            pbar.update(1)
```

### Step 7.2: Resampling Strategy: Nearest Neighbor
```python
# Inside NearestNeighborStrategy.resample():
def resample(self, original_mz, original_intensity, target_mz):
    # Find nearest target bin for each original peak
    indices = np.searchsorted(target_mz, original_mz)

    # Create output array (all zeros initially)
    resampled = np.zeros(len(target_mz), dtype=np.float64)

    # Assign each peak to nearest bin
    for i, (mz_val, intensity) in enumerate(zip(original_mz, original_intensity)):
        target_idx = indices[i]
        # Accumulate intensity if multiple peaks map to same bin
        resampled[target_idx] += intensity

    return resampled
    # Returns: Dense array of 313,723 values (mostly zeros for sparse data)
```

### Step 7.3: Add to Sparse Matrix (`base_spatialdata_converter.py:766`)
```python
def _add_to_sparse_matrix(self, sparse_matrix, pixel_idx, mz_indices, intensities):
    # Filter out zeros
    nonzero_mask = intensities != 0.0
    valid_mz_indices = mz_indices[nonzero_mask]  # ~1,254 indices
    valid_intensities = intensities[nonzero_mask]  # ~1,254 values

    # THIS IS THE SLOW LINE (Python list operations in LIL matrix)
    sparse_matrix[pixel_idx, valid_mz_indices] = valid_intensities
    # For pixel 0: Set values at ~1,254 positions
    # For pixel 1,000,000: SLOWER due to memory fragmentation
```

---

## PHASE 8: Finalize Conversion

### Step 8.1: Convert LIL to CSR (`spatialdata_2d_converter.py:150`)
```python
def _finalize_conversion(self, data_structures):
    # Convert sparse matrix to CSR format
    logging.info("Converting to CSR format...")
    sparse_matrix_csr = data_structures["sparse_matrix"].tocsr()
    # This takes 1-2 seconds for 1.86 billion non-zero values

    # Create AnnData object
    adata = AnnData(
        X=sparse_matrix_csr,  # (1,479,283 × 313,723) CSR matrix
        obs=coords_df,        # Pixel coordinates
        var=var_df            # M/Z values
    )
```

### Step 8.2: Create SpatialData Object
```python
from spatialdata import SpatialData

sdata = SpatialData(
    tables={dataset_id + "_z0": adata},
    shapes={...},  # Pixel geometries
    images={...}   # TIC image
)
```

### Step 8.3: Write to Zarr (`base_spatialdata_converter.py:210`)
```python
# Save to Zarr
sdata.write(output_path)
# Internally:
# - CSR matrix saved as:
#   X/data: array of non-zero values (float64)
#   X/indices: column indices for each value (int32)
#   X/indptr: row pointers (int32)
# - Each array chunked (10k-100k elements)
# - Compressed with blosc/zstd
# - Final size: ~5-8 GB on disk
```

---

## SUMMARY: Complete Data Flow

```
User Command
    ↓
CLI Arguments Parsed
    ↓
Resampling Config Created {width_at_mz: 0.005, reference_mz: 1000}
    ↓
Reader Created (BrukerReader)
    ├─ SQLite DB opened (read-only)
    ├─ Mass range loaded: 250-1200 Da
    └─ SDK initialized
    ↓
Converter Created (SpatialData2DConverter)
    └─ Resampling config stored
    ↓
converter.convert() called
    ↓
_initialize_conversion()
    └─ _build_resampled_mass_axis()
        ├─ Calculate bins: 313,723 (from 5mDa @ 1000)
        ├─ Select axis type: REFLECTOR_TOF
        └─ Generate 313,723 m/z values (250.0 to 1200.0)
    ↓
_create_data_structures()
    └─ Create LIL matrix (1,479,283 × 313,723)
    ↓
_process_spectra()  ← THIS IS WHERE SLOWDOWN HAPPENS
    └─ For each of 1,479,283 spectra:
        ├─ Read spectrum from Bruker SDK
        ├─ Resample to 313,723 bins (Nearest Neighbor)
        └─ Add to LIL matrix ← GETS SLOWER AT 90%
            (Python list reallocation overhead)
    ↓
_finalize_conversion()
    ├─ Convert LIL → CSR (fast, 1-2 seconds)
    ├─ Create AnnData(X=CSR matrix)
    ├─ Create SpatialData object
    └─ Write to Zarr
        ├─ X/data (non-zero values)
        ├─ X/indices (column indices)
        └─ X/indptr (row pointers)
```

---

## Key Insights

1. **Resampled axis is built BEFORE reading any spectra** - dimensions known upfront
2. **Each spectrum is resampled individually** as it's read
3. **LIL matrix accumulates resampled spectra** - this is where slowdown occurs
4. **Final CSR conversion is fast** - happens once at end
5. **Zarr storage is identical** regardless of how LIL matrix was built

The bottleneck is Step 7.3: Adding to LIL matrix degrades from 1500→400 spec/sec at 90%.
