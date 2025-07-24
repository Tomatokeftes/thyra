# MSIConverter Function Call Flow

## Complete Function-Level Call Tree

This document shows the exact function call sequence when running:
```bash
msiconvert data.imzML output.zarr --pixel-size 25
```

## 🚀 Entry Point: CLI Parsing

```
__main__.py:main()
├── argparse.ArgumentParser()
├── parser.parse_args()
├── Path(args.input).resolve()
├── Path(args.output).resolve()
└── setup_logging(log_level=args.log_level, log_file=args.log_file)
    └── logging_config.py:setup_logging()
        ├── logging.basicConfig()
        ├── logging.getLogger().setLevel()
        └── logging.StreamHandler() / logging.FileHandler()
```

## 🔍 Format Detection & Reader Setup

```
__main__.py:main()
└── convert.py:convert_msi()
    ├── registry.py:detect_format(input_path)
    │   ├── Path(input_path).suffix → ".imzML"
    │   ├── for format_name, detector_func in format_detectors.items():
    │   └── imzml_reader.py:detect_imzml_format()
    │       ├── Path(path).suffix.lower() == '.imzml'
    │       ├── Path(path).with_suffix('.ibd').exists()
    │       └── return True
    │
    ├── registry.py:get_reader_class("imzml")
    │   └── return readers["imzml"]  # ImzMLReader class
    │
    └── ImzMLReader(input_path)
        ├── __init__(self, imzml_path)
        │   ├── self.imzml_path = Path(imzml_path)
        │   ├── self.ibd_path = self.imzml_path.with_suffix('.ibd')
        │   ├── pyimzml.ImzMLParser(str(self.imzml_path))
        │   └── self._create_metadata_extractor()
        │       └── ImzMLMetadataExtractor(self.parser, self.imzml_path)
        │
        └── get_essential_metadata()
            └── metadata_extractor.get_essential()
                ├── if self._essential_cache is None:
                └── _extract_essential_impl()
                    ├── np.array(self.parser.coordinates)
                    ├── _calculate_dimensions(coords)
                    │   ├── coords_0based = coords - 1
                    │   ├── max_coords = np.max(coords_0based, axis=0)
                    │   └── return (max_coords[0]+1, max_coords[1]+1, max_coords[2]+1)
                    ├── _calculate_bounds(coords)
                    │   ├── x_coords = coords[:, 0].astype(float)
                    │   ├── y_coords = coords[:, 1].astype(float)
                    │   └── return (min(x), max(x), min(y), max(y))
                    ├── _get_mass_range_fast()
                    │   ├── first_mzs, _ = self.parser.getspectrum(0)
                    │   ├── min_mass = float(np.min(first_mzs))
                    │   ├── max_mass = float(np.max(first_mzs))
                    │   ├── for idx in [n//4, n//2, 3*n//4, n-1]:
                    │   │   ├── mzs, _ = self.parser.getspectrum(idx)
                    │   │   ├── min_mass = min(min_mass, np.min(mzs))
                    │   │   └── max_mass = max(max_mass, np.max(mzs))
                    │   └── return (min_mass, max_mass)
                    ├── _extract_pixel_size_fast()
                    │   ├── if hasattr(self.parser, 'imzmldict'):
                    │   ├── x_size = self.parser.imzmldict.get('pixel size x')
                    │   ├── y_size = self.parser.imzmldict.get('pixel size y')
                    │   └── return (float(x_size), float(y_size)) or None
                    ├── _estimate_memory(n_spectra)
                    │   ├── avg_peaks_per_spectrum = 1000
                    │   ├── bytes_per_value = 8
                    │   ├── estimated_bytes = n_spectra * avg_peaks * 2 * bytes_per_value
                    │   └── return estimated_bytes / (1024**3)
                    └── return EssentialMetadata(...)
```

## 📏 Pixel Size Detection (if not provided)

```
__main__.py:main()
└── if args.pixel_size is None:
    └── detect_pixel_size_interactive(reader, input_format)
        ├── reader.get_essential_metadata()  # Unified metadata extraction
        ├── essential_metadata.pixel_size  # Check if auto-detected
        ├── if pixel_size is None:
        │   ├── print("Could not automatically detect pixel size")
        │   ├── print("Please enter pixel size manually:")
        │   ├── input("Pixel size (micrometers): ")
        │   ├── float(user_input)
        │   └── if pixel_size <= 0: raise ValueError
        ├── print(f"Using pixel size: {pixel_size} μm")
        └── return pixel_size, detection_info, essential_metadata  # Returns metadata for reuse
```

## 🔄 Converter Setup & Initialization

```
convert.py:convert_msi()
├── registry.py:get_converter_class("spatialdata")
│   └── return converters["spatialdata"]  # SpatialDataConverter class
│
├── # Metadata reuse optimization - skips re-extraction if provided
├── if essential_metadata is None:
│   └── essential_metadata = reader.get_essential_metadata()
│
└── SpatialDataConverter(reader, output_path, **kwargs)
    ├── __init__(self, reader, output_path, dataset_id, pixel_size_um, handle_3d)
    │   ├── self.reader = reader
    │   ├── self.output_path = Path(output_path)
    │   ├── self.dataset_id = dataset_id
    │   ├── self.pixel_size_um = pixel_size_um
    │   ├── self.handle_3d = handle_3d
    │   └── self.batch_size = 1000  # Default batch size
    │
    └── convert()  # Template method starts here
        ├── _initialize_conversion()
        ├── _create_data_structures()
        ├── _process_spectra(data_structures)
        ├── _finalize_data(data_structures)
        └── _save_output(final_data)
```

## 🏗️ Data Structure Creation

```
SpatialDataConverter.convert()
└── _initialize_conversion()
    ├── self.reader.get_essential_metadata()  # Already cached
    ├── self.essential_metadata = essential_metadata
    ├── self.dimensions = essential_metadata.dimensions
    ├── self.pixel_size = essential_metadata.pixel_size or self.pixel_size_um
    ├── logger.info(f"Initializing conversion for {self.dimensions} dataset")
    │
    ├── self.reader.get_common_mass_axis()
    │   └── ImzMLReader.get_common_mass_axis()
    │       ├── if hasattr(self.parser, 'continuous') and self.parser.continuous:
    │       │   ├── mzs, _ = self.parser.getspectrum(0)  # First spectrum
    │       │   └── return np.array(mzs, dtype=np.float64)
    │       └── else:  # Processed mode
    │           ├── all_mzs = set()
    │           ├── for i in range(len(self.parser.coordinates)):
    │           │   ├── mzs, _ = self.parser.getspectrum(i)
    │           │   └── all_mzs.update(mzs)
    │           └── return np.array(sorted(all_mzs), dtype=np.float64)
    │
    ├── self.common_mass_axis = common_mass_axis
    ├── logger.info(f"Common mass axis: {len(self.common_mass_axis)} points")
    ├── logger.info(f"Mass range: {self.common_mass_axis[0]:.2f} - {self.common_mass_axis[-1]:.2f}")
    └── logger.info(f"Pixel size: {self.pixel_size} μm")

└── _create_data_structures()
    ├── n_pixels = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    ├── n_masses = len(self.common_mass_axis)
    ├── logger.info(f"Creating sparse matrix: {n_pixels} × {n_masses}")
    │
    └── return {
        'spectral_matrix': scipy.sparse.lil_matrix((n_pixels, n_masses), dtype=np.float32),
        'coordinates': [],
        'tic_values': np.zeros(n_pixels, dtype=np.float32),
        'valid_pixels': np.zeros(n_pixels, dtype=bool),
        'pixel_count': 0
    }
```

## 🔄 Spectrum Processing Loop

```
SpatialDataConverter.convert()
└── _process_spectra(data_structures)
    ├── batch_size = self.batch_size  # 1000
    ├── total_spectra = self.essential_metadata.n_spectra
    ├── logger.info(f"Processing {total_spectra} spectra in batches of {batch_size}")
    │
    └── for batch in self.reader.iter_spectra(batch_size=batch_size):
        ├── ImzMLReader.iter_spectra(batch_size)
        │   ├── coordinates = self.parser.coordinates
        │   ├── batch_count = 0
        │   ├── current_batch = []
        │   │
        │   └── for i, coord in enumerate(coordinates):
        │       ├── mzs, intensities = self.parser.getspectrum(i)
        │       ├── spectrum_data = {
        │       │   'coordinates': coord,      # (x, y, z)
        │       │   'mzs': np.array(mzs),
        │       │   'intensities': np.array(intensities)
        │       │   }
        │       ├── current_batch.append(spectrum_data)
        │       ├── if len(current_batch) >= batch_size:
        │       │   ├── yield current_batch
        │       │   └── current_batch = []
        │       └── if i % 1000 == 0:
        │           └── logger.debug(f"Loaded {i} spectra")
        │
        └── for spectrum_data in batch:
            └── _process_single_spectrum(
                   spectrum_data['coordinates'],
                   spectrum_data['mzs'],
                   spectrum_data['intensities'],
                   data_structures
               )
```

## 🔬 Single Spectrum Processing

```
_process_single_spectrum(pixel_coords, mzs, intensities, data_structures)
├── x, y, z = pixel_coords  # e.g., (1, 1, 1)
├── pixel_idx = self._coords_to_pixel_index(x, y, z)
│   ├── # Convert 3D coordinates to 1D pixel index
│   ├── x_idx, y_idx, z_idx = x-1, y-1, z-1  # Convert to 0-based
│   ├── dims_x, dims_y, dims_z = self.dimensions
│   └── return z_idx * (dims_x * dims_y) + y_idx * dims_x + x_idx
│
├── # Map spectrum m/z values to common mass axis indices
├── mass_indices = np.searchsorted(self.common_mass_axis, mzs)
│   # Uses binary search: O(log n) complexity
│
├── # Filter out m/z values outside the common range
├── valid_mask = (mass_indices < len(self.common_mass_axis))
├── mass_indices = mass_indices[valid_mask]
├── intensities = intensities[valid_mask]
│
├── # Store intensities in sparse matrix
├── spectral_matrix = data_structures['spectral_matrix']
├── spectral_matrix[pixel_idx, mass_indices] = intensities
│
├── # Calculate and store Total Ion Current (TIC)
├── tic = np.sum(intensities)
├── data_structures['tic_values'][pixel_idx] = tic
│
├── # Mark pixel as containing data
├── data_structures['valid_pixels'][pixel_idx] = True
│
├── # Store coordinate mapping
├── data_structures['coordinates'].append((x, y, z))
├── data_structures['pixel_count'] += 1
│
└── # Progress logging
    └── if data_structures['pixel_count'] % 1000 == 0:
        └── logger.info(f"Processed {data_structures['pixel_count']} spectra")
```

## 🎯 Data Finalization

```
SpatialDataConverter.convert()
└── _finalize_data(data_structures)
    ├── logger.info("Finalizing data structures...")
    │
    ├── # Convert sparse matrix to efficient CSR format
    ├── spectral_matrix = data_structures['spectral_matrix'].tocsr()
    ├── logger.info(f"Sparse matrix density: {spectral_matrix.nnz / spectral_matrix.size * 100:.2f}%")
    │
    ├── # Create observation (pixel) metadata DataFrame
    ├── coordinates = data_structures['coordinates']
    ├── obs_df = pd.DataFrame({
    │   'x': [c[0] for c in coordinates],
    │   'y': [c[1] for c in coordinates],
    │   'z': [c[2] for c in coordinates],
    │   'tic': data_structures['tic_values'][data_structures['valid_pixels']]
    │   })
    │
    ├── # Create variable (mass) metadata DataFrame
    ├── var_df = pd.DataFrame({
    │   'mz': self.common_mass_axis,
    │   'mass_index': np.arange(len(self.common_mass_axis))
    │   })
    │
    ├── # Create AnnData object (scanpy/single-cell format)
    ├── import anndata
    ├── adata = anndata.AnnData(
    │   X=spectral_matrix,           # Observations × Variables matrix
    │   obs=obs_df,                  # Pixel metadata
    │   var=var_df,                  # Mass axis metadata
    │   dtype=np.float32
    │   )
    │
    ├── # Create pixel boundary shapes
    ├── shapes_gdf = self._create_pixel_shapes()
    │   ├── import geopandas as gpd
    │   ├── from shapely.geometry import Polygon
    │   ├── shapes = []
    │   ├── for x, y, z in coordinates:
    │   │   ├── # Calculate pixel boundaries
    │   │   ├── x_min = (x - 1) * self.pixel_size[0]
    │   │   ├── x_max = x * self.pixel_size[0]
    │   │   ├── y_min = (y - 1) * self.pixel_size[1]
    │   │   ├── y_max = y * self.pixel_size[1]
    │   │   ├── polygon = Polygon([
    │   │   │   (x_min, y_min), (x_max, y_min),
    │   │   │   (x_max, y_max), (x_min, y_max)
    │   │   │   ])
    │   │   └── shapes.append(polygon)
    │   └── return gpd.GeoDataFrame({'geometry': shapes})
    │
    ├── # Create TIC image
    ├── tic_image = self._create_tic_image(data_structures['tic_values'])
    │   ├── import xarray as xr
    │   ├── # Reshape TIC values to spatial grid
    │   ├── tic_grid = np.zeros(self.dimensions[:2])  # 2D grid
    │   ├── for i, (x, y, z) in enumerate(coordinates):
    │   │   └── tic_grid[y-1, x-1] = data_structures['tic_values'][i]
    │   ├── # Create coordinate arrays
    │   ├── x_coords = np.arange(self.dimensions[0]) * self.pixel_size[0]
    │   ├── y_coords = np.arange(self.dimensions[1]) * self.pixel_size[1]
    │   └── return xr.DataArray(
    │       tic_grid,
    │       dims=['y', 'x'],
    │       coords={'x': x_coords, 'y': y_coords},
    │       name='total_ion_current'
    │       )
    │
    └── return {
        'tables': {self.dataset_id: adata},
        'shapes': {f"{self.dataset_id}_shapes": shapes_gdf},
        'images': {f"{self.dataset_id}_tic": tic_image}
    }
```

## 💾 Output Generation & Metadata Addition

```
SpatialDataConverter.convert()
└── _save_output(final_data)
    ├── logger.info("Creating SpatialData object...")
    ├── import spatialdata as sd
    ├── sdata = sd.SpatialData(
    │   tables=final_data['tables'],       # AnnData objects
    │   shapes=final_data['shapes'],       # GeoDataFrames
    │   images=final_data['images']        # xarray DataArrays
    │   )
    │
    ├── # Add comprehensive metadata
    ├── comprehensive_metadata = self.reader.get_comprehensive_metadata()
    │   └── ImzMLMetadataExtractor.get_comprehensive()
    │       ├── if self._comprehensive_cache is None:
    │       └── _extract_comprehensive_impl()
    │           ├── essential = self.get_essential()  # Already cached
    │           ├── format_specific = self._extract_imzml_specific()
    │           │   ├── imzml_version = "1.1.0"
    │           │   ├── file_mode = "continuous" or "processed"
    │           │   ├── ibd_file = str(self.imzml_path.with_suffix('.ibd'))
    │           │   ├── spectrum_count = len(self.parser.coordinates)
    │           │   └── scan_settings = {}
    │           ├── acquisition_params = self._extract_acquisition_params()
    │           │   ├── if not self.get_essential().has_pixel_size:
    │           │   │   └── pixel_size = self._extract_pixel_size_from_xml()
    │           │   └── # Extract scan direction, pattern, etc. from imzmldict
    │           ├── instrument_info = self._extract_instrument_info()
    │           │   └── # Extract instrument model, serial, software, etc.
    │           ├── raw_metadata = self._extract_raw_metadata()
    │           │   └── return dict(self.parser.imzmldict)
    │           └── return ComprehensiveMetadata(...)
    │
    ├── self.add_metadata(sdata, comprehensive_metadata)
    │   ├── conversion_info = {
    │   │   'dataset_id': self.dataset_id,
    │   │   'pixel_size_um': self.pixel_size,
    │   │   'conversion_timestamp': datetime.now().isoformat(),
    │   │   'msiconvert_version': msiconvert.__version__,
    │   │   'input_format': 'imzml',
    │   │   'output_format': 'spatialdata'
    │   │   }
    │   ├── sdata.metadata['conversion_info'] = conversion_info
    │   ├── sdata.metadata['essential_metadata'] = asdict(comprehensive_metadata.essential)
    │   ├── sdata.metadata['format_specific'] = comprehensive_metadata.format_specific
    │   ├── sdata.metadata['acquisition_params'] = comprehensive_metadata.acquisition_params
    │   ├── sdata.metadata['instrument_info'] = comprehensive_metadata.instrument_info
    │   └── sdata.metadata['raw_metadata'] = comprehensive_metadata.raw_metadata
    │
    ├── # Save to Zarr format
    ├── logger.info(f"Writing output to {self.output_path}")
    ├── sdata.write(str(self.output_path))
    │   ├── # SpatialData handles Zarr serialization
    │   ├── # Creates .zarr directory with:
    │   │   ├── tables/
    │   │   │   └── {dataset_id}/  # AnnData as Zarr
    │   │   ├── shapes/
    │   │   │   └── {dataset_id}_shapes/  # GeoDataFrame as Parquet
    │   │   ├── images/
    │   │   │   └── {dataset_id}_tic/  # xarray as Zarr
    │   │   └── .zattrs  # Metadata as JSON
    │   └── # Zarr uses chunked, compressed storage
    │
    ├── logger.info("Conversion completed successfully!")
    ├── logger.info(f"Output written to: {self.output_path}")
    ├── logger.info(f"Dataset dimensions: {self.dimensions}")
    ├── logger.info(f"Number of spectra: {self.essential_metadata.n_spectra}")
    ├── logger.info(f"Mass axis points: {len(self.common_mass_axis)}")
    └── return True
```

## 📊 Performance Metrics & Cleanup

```
convert.py:convert_msi()
├── conversion_success = converter.convert()
├── reader.close()
│   └── ImzMLReader.close()
│       ├── if hasattr(self.parser, 'close'):
│       │   └── self.parser.close()
│       ├── self.metadata_extractor.clear_cache()
│       └── logger.debug("ImzML reader closed")
│
├── if conversion_success:
│   ├── logger.info("✅ Conversion completed successfully")
│   ├── output_size = sum(f.stat().st_size for f in Path(output_path).rglob('*') if f.is_file())
│   ├── logger.info(f"Output size: {output_size / (1024**3):.2f} GB")
│   └── return True
├── else:
│   ├── logger.error("❌ Conversion failed")
│   └── return False
```

## Summary: Function Call Statistics

**Total Functions Called**: ~50-100 depending on dataset size and format
**Most Critical Path**:
1. `detect_format()` → `ImzMLReader()` → `get_essential_metadata()`
2. `get_common_mass_axis()` → `iter_spectra()` → `_process_single_spectrum()`
3. `_create_data_structures()` → `_finalize_data()` → `_save_output()`

**Performance Bottlenecks**:
- `iter_spectra()`: I/O bound for large files
- `np.searchsorted()`: CPU bound for mass axis mapping
- `sparse.lil_matrix.tocsr()`: Memory bound for matrix conversion
- `sdata.write()`: I/O bound for Zarr serialization

**Memory Management**:
- Sparse matrices reduce memory by ~95% vs dense arrays
- Batch processing prevents memory overflow on large datasets
- Caching prevents redundant metadata extraction
- Progressive cleanup releases memory after each phase
