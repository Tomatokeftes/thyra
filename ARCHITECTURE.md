# MSIConverter Architecture & Conversion Workflow

## Quick Overview

MSIConverter transforms Mass Spectrometry Imaging data through a **4-phase pipeline**:
1. **Detection** → Identify input format (ImzML/Bruker)
2. **Reading** → Extract spectra and metadata
3. **Processing** → Map to common mass axis and create sparse matrices
4. **Output** → Generate SpatialData/Zarr format

## Architecture Components

### 🏗️ Core Framework
```
msiconvert/core/
├── registry.py          # Plugin registration system
├── base_reader.py       # Abstract reader interface
├── base_converter.py    # Abstract converter interface
└── base_extractor.py    # Abstract metadata extractor
```

### 📖 Format Readers
```
msiconvert/readers/
├── imzml_reader.py      # ImzML format support
└── bruker/
    ├── bruker_reader.py # Bruker format support
    ├── sdk/             # Bruker SDK integration
    └── utils/           # Batch processing, caching
```

### 📊 Metadata System
```
msiconvert/metadata/
├── extractors/
│   ├── imzml_extractor.py   # ImzML metadata extraction
│   └── bruker_extractor.py  # Bruker metadata extraction
├── types.py             # Data type definitions
└── ontology/            # Ontology validation
```

### 🔄 Converters
```
msiconvert/converters/
└── spatialdata_converter.py # SpatialData/Zarr output
```

## Detailed Conversion Flow

### 🚀 **Phase 1: CLI Entry Point** (`__main__.py`)

```python
def main():
    # 1. Parse CLI arguments
    args = parser.parse_args()  # input, output, pixel-size, etc.

    # 2. Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        parser.error(f"Input does not exist: {input_path}")

    # 3. Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)

    # 4. Handle pixel size detection
    if args.pixel_size is None:
        # Auto-detect or interactive prompt
        final_pixel_size, detection_info = detect_pixel_size_interactive(reader, input_format)

    # 5. Launch conversion
    success = convert_msi(
        args.input, args.output,
        pixel_size_um=final_pixel_size,
        handle_3d=args.handle_3d
    )
```

**Key Functions:**
- `setup_logging()` → Configure output verbosity
- `detect_pixel_size_interactive()` → Auto-detect or prompt user
- `convert_msi()` → Main conversion orchestrator

---

### 🔍 **Phase 2: Format Detection & Reader Creation** (`convert.py`)

```python
def convert_msi(input_path, output_path, **kwargs):
    # 1. Detect input format
    input_format = detect_format(input_path)  # "imzml" or "bruker"

    # 2. Get appropriate reader class
    reader_class = get_reader_class(input_format)  # ImzMLReader or BrukerReader

    # 3. Initialize reader
    reader = reader_class(input_path)

    # 4. Get converter class
    converter_class = get_converter_class("spatialdata")  # SpatialDataConverter

    # 5. Create and run converter
    converter = converter_class(reader, output_path, **kwargs)
    return converter.convert()
```

**Registry System Flow:**
```python
# Registry populated at import time via decorators
@register_reader("imzml")
class ImzMLReader(BaseMSIReader): ...

@register_format_detector("imzml")
def detect_imzml_format(path): ...

# Runtime lookup
def detect_format(path):
    for format_name, detector_func in format_detectors.items():
        if detector_func(path):
            return format_name
    raise ValueError(f"Unknown format: {path}")
```

---

### 📖 **Phase 3: Reader Initialization & Metadata Extraction**

#### ImzML Reader Flow (`readers/imzml_reader.py`)
```python
class ImzMLReader(BaseMSIReader):
    def __init__(self, imzml_path):
        self.imzml_path = Path(imzml_path)
        self.ibd_path = self.imzml_path.with_suffix('.ibd')

        # Initialize pyimzML parser
        self.parser = ImzMLParser(str(self.imzml_path))

        # Create metadata extractor
        self.metadata_extractor = self._create_metadata_extractor()

    def _create_metadata_extractor(self):
        return ImzMLMetadataExtractor(self.parser, self.imzml_path)

    def get_common_mass_axis(self):
        # Continuous mode: use first spectrum
        # Processed mode: collect all unique m/z values
        if hasattr(self.parser, 'continuous') and self.parser.continuous:
            mzs, _ = self.parser.getspectrum(0)
            return np.array(mzs)
        else:
            # Collect unique m/z from all spectra
            all_mzs = set()
            for i in range(len(self.parser.coordinates)):
                mzs, _ = self.parser.getspectrum(i)
                all_mzs.update(mzs)
            return np.array(sorted(all_mzs))
```

#### Bruker Reader Flow (`readers/bruker/bruker_reader.py`)
```python
class BrukerReader(BaseMSIReader):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.tsf_path = self.data_path / "analysis.tsf"

        # Initialize Bruker SDK connection
        self.sdk_manager = DLLManager(self.data_path)
        self.conn = sqlite3.connect(str(self.tsf_path))

        # Create metadata extractor
        self.metadata_extractor = self._create_metadata_extractor()

        # Initialize batch processor for memory efficiency
        self.batch_processor = BatchProcessor(self.conn, batch_size=1000)

    def _create_metadata_extractor(self):
        return BrukerMetadataExtractor(self.conn, self.data_path)
```

#### Metadata Extraction (`metadata/extractors/`)
```python
class MetadataExtractor(ABC):
    def get_essential(self) -> EssentialMetadata:
        """Fast metadata for conversion setup."""
        if self._essential_cache is None:
            self._essential_cache = self._extract_essential_impl()
        return self._essential_cache

    def get_comprehensive(self) -> ComprehensiveMetadata:
        """Complete metadata for output."""
        if self._comprehensive_cache is None:
            essential = self.get_essential()
            self._comprehensive_cache = self._extract_comprehensive_impl()
        return self._comprehensive_cache

# Two-Phase Extraction Strategy:
# Phase 1 (Essential): dimensions, pixel_size, mass_range - needed for conversion setup
# Phase 2 (Comprehensive): all metadata - needed for final output
```

---

### 🔄 **Phase 4: Conversion Processing** (`converters/spatialdata_converter.py`)

```python
class SpatialDataConverter(BaseMSIConverter):
    def convert(self) -> bool:
        """Template method defining conversion workflow."""
        try:
            # 1. Initialize conversion
            self._initialize_conversion()

            # 2. Create data structures
            data_structures = self._create_data_structures()

            # 3. Process all spectra
            self._process_spectra(data_structures)

            # 4. Finalize data
            final_data = self._finalize_data(data_structures)

            # 5. Save output
            return self._save_output(final_data)

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False
```

#### Step-by-Step Processing:

**1. Initialize Conversion**
```python
def _initialize_conversion(self):
    # Load essential metadata for setup
    self.essential_metadata = self.reader.get_essential_metadata()
    self.dimensions = self.essential_metadata.dimensions
    self.pixel_size = self.essential_metadata.pixel_size

    # Build common mass axis
    self.common_mass_axis = self.reader.get_common_mass_axis()

    logger.info(f"Dataset: {self.dimensions} pixels, {len(self.common_mass_axis)} masses")
```

**2. Create Data Structures**
```python
def _create_data_structures(self):
    n_pixels = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    n_masses = len(self.common_mass_axis)

    return {
        # Sparse matrix for spectral data (pixels × masses)
        'spectral_matrix': sparse.lil_matrix((n_pixels, n_masses), dtype=np.float32),

        # Coordinate mapping (pixel_index → spatial_coords)
        'coordinates': [],

        # Total ion current per pixel
        'tic_values': np.zeros(n_pixels, dtype=np.float32),

        # Valid pixel mask
        'valid_pixels': np.zeros(n_pixels, dtype=bool)
    }
```

**3. Process Spectra (Core Loop)**
```python
def _process_spectra(self, data_structures):
    batch_size = self.batch_size
    processed_count = 0

    # Iterate through spectra in batches
    for batch in self.reader.iter_spectra(batch_size=batch_size):
        for spectrum_data in batch:
            # Extract data
            pixel_coords = spectrum_data['coordinates']  # (x, y, z)
            mzs = spectrum_data['mzs']                   # m/z values
            intensities = spectrum_data['intensities']   # intensity values

            # Process single spectrum
            self._process_single_spectrum(
                pixel_coords, mzs, intensities, data_structures
            )

            processed_count += 1

        # Progress reporting
        if processed_count % 1000 == 0:
            logger.info(f"Processed {processed_count} spectra")
```

**4. Single Spectrum Processing**
```python
def _process_single_spectrum(self, pixel_coords, mzs, intensities, data_structures):
    # Calculate pixel index from coordinates
    x, y, z = pixel_coords
    pixel_idx = self._coords_to_pixel_index(x, y, z)

    # Map spectrum m/z to common mass axis
    mass_indices = np.searchsorted(self.common_mass_axis, mzs)

    # Store intensities in sparse matrix
    spectral_matrix = data_structures['spectral_matrix']
    spectral_matrix[pixel_idx, mass_indices] = intensities

    # Calculate and store TIC
    tic = np.sum(intensities)
    data_structures['tic_values'][pixel_idx] = tic

    # Mark pixel as valid
    data_structures['valid_pixels'][pixel_idx] = True

    # Store coordinates
    data_structures['coordinates'].append((x, y, z))
```

**5. Finalize Data Structures**
```python
def _finalize_data(self, data_structures):
    # Convert sparse matrix to efficient format
    spectral_matrix = data_structures['spectral_matrix'].tocsr()

    # Create AnnData object (scanpy/anndata format)
    adata = AnnData(
        X=spectral_matrix,                           # Observations × Variables
        obs=pd.DataFrame({                           # Pixel metadata
            'x': [c[0] for c in coordinates],
            'y': [c[1] for c in coordinates],
            'z': [c[2] for c in coordinates],
            'tic': data_structures['tic_values']
        }),
        var=pd.DataFrame({                           # Mass axis metadata
            'mz': self.common_mass_axis
        })
    )

    # Create coordinate shapes (pixel boundaries)
    shapes = self._create_pixel_shapes()

    # Create TIC image
    tic_image = self._create_tic_image(data_structures['tic_values'])

    return {
        'tables': {self.dataset_id: adata},
        'shapes': {f"{self.dataset_id}_shapes": shapes},
        'images': {f"{self.dataset_id}_tic": tic_image}
    }
```

**6. Save Output**
```python
def _save_output(self, final_data):
    # Create SpatialData object
    sdata = SpatialData(
        tables=final_data['tables'],
        shapes=final_data['shapes'],
        images=final_data['images']
    )

    # Add comprehensive metadata
    comprehensive_metadata = self.reader.get_comprehensive_metadata()
    self.add_metadata(sdata, comprehensive_metadata)

    # Save to Zarr format
    sdata.write(str(self.output_path))

    logger.info(f"Conversion completed: {self.output_path}")
    return True
```

## Key Technical Details

### 🧠 **Memory Management Strategy**
- **Sparse Matrices**: Only store non-zero intensities (~1-5% of total matrix)
- **Batch Processing**: Process spectra in configurable batches (default: 1000)
- **Lazy Loading**: Load data only when needed
- **Progressive Cleanup**: Release memory after each phase

### ⚡ **Performance Optimizations**
- **Mass Axis Mapping**: Use `np.searchsorted()` for O(log n) m/z mapping
- **Efficient Data Types**: Float32 for intensities, appropriate integer types for indices
- **Chunked Zarr Storage**: Optimized for both random and sequential access
- **Coordinate Caching**: Cache frequently accessed coordinate calculations

### 🔍 **Pixel Size Detection**
```python
# ImzML: Extract from XML metadata
def detect_imzml_pixel_size(parser):
    # Check imzmldict first (parsed values)
    if hasattr(parser, 'imzmldict'):
        x_size = parser.imzmldict.get('pixel size x')
        y_size = parser.imzmldict.get('pixel size y')
        if x_size and y_size:
            return (float(x_size), float(y_size))

    # Fallback to XML parsing
    for cvparam in parser.metadata.root.findall('.//cvParam'):
        if cvparam.get('accession') == 'IMS:1000046':  # pixel size x
            x_size = float(cvparam.get('value'))
        elif cvparam.get('accession') == 'IMS:1000047':  # pixel size y
            y_size = float(cvparam.get('value'))

    return (x_size, y_size) if x_size and y_size else None

# Bruker: Extract from database
def detect_bruker_pixel_size(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT BeamScanSizeX, BeamScanSizeY FROM MaldiFrameLaserInfo LIMIT 1")
    result = cursor.fetchone()
    return (float(result[0]), float(result[1])) if result else None
```

### 📊 **Data Type Specifications**

**EssentialMetadata** (Fast extraction for conversion setup):
```python
@dataclass(frozen=True)
class EssentialMetadata:
    dimensions: Tuple[int, int, int]                     # (x_pixels, y_pixels, z_slices)
    coordinate_bounds: Tuple[float, float, float, float] # (min_x, max_x, min_y, max_y)
    mass_range: Tuple[float, float]                      # (min_mz, max_mz)
    pixel_size: Optional[Tuple[float, float]]            # (x_size_μm, y_size_μm)
    n_spectra: int                                       # Total spectrum count
    estimated_memory_gb: float                           # Memory usage estimate
    source_path: str                                     # Original file path
```

**ComprehensiveMetadata** (Complete extraction for final output):
```python
@dataclass
class ComprehensiveMetadata:
    essential: EssentialMetadata                         # Essential metadata
    format_specific: Dict[str, Any]                      # Format-specific parameters
    acquisition_params: Dict[str, Any]                   # Acquisition settings
    instrument_info: Dict[str, Any]                      # Instrument details
    raw_metadata: Dict[str, Any]                         # Original metadata
```

This architecture enables MSIConverter to:
- ✅ Handle datasets from MBs to 100+ GBs efficiently
- ✅ Support multiple input formats with consistent interface
- ✅ Provide rich metadata preservation and validation
- ✅ Generate cloud-ready, standardized output formats
- ✅ Maintain extensibility for new formats and features

The plugin-based registry system and template method pattern ensure that adding new formats requires minimal code changes while maintaining consistency across the conversion pipeline.
