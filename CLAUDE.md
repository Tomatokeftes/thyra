# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses Poetry for dependency management. Use the following commands:

### Installation and Setup
```bash
poetry install  # Install all dependencies including dev/test groups
```

### Testing
```bash
poetry run pytest                    # Run unit tests only (default)
poetry run pytest -m "not integration"  # Explicitly run unit tests
poetry run pytest -m integration     # Run integration tests only
poetry run pytest -m "unit or integration"  # Run all tests
poetry run pytest --cov=msiconvert   # Run with coverage report
```

### Code Quality
```bash
poetry run black .        # Format code
poetry run isort .         # Sort imports
poetry run flake8         # Lint code
```

### Running the CLI
```bash
poetry run msiconvert <input_path> <output_path>  # Basic conversion
poetry run msiconvert-check-ontology              # Ontology validation tool
```

### Building
```bash
poetry build  # Build distribution packages
```

## Project Architecture

MSIConverter is a library for converting Mass Spectrometry Imaging (MSI) data into SpatialData format. The architecture follows a plugin-based design with clear separation between readers and converters.

### Core Components

- **Registry System** (`msiconvert/core/registry.py`): Central registration system for readers and converters with automatic format detection
- **Base Classes**:
  - `BaseMSIReader` (`msiconvert/core/base_reader.py`): Abstract interface for reading MSI formats
  - `BaseMSIConverter` (`msiconvert/core/base_converter.py`): Template method pattern for conversion workflow
- **Main Conversion** (`msiconvert/convert.py`): High-level API that orchestrates the entire conversion process

### Supported Formats

#### Readers (Input)
- **ImzML**: `msiconvert/readers/imzml_reader.py` - Handles .imzML files
- **Bruker**: `msiconvert/readers/bruker/` - Handles Bruker .d directories with native SDK integration

#### Converters (Output)
- **SpatialData**: `msiconvert/converters/spatialdata_converter.py` - Converts to SpatialData/Zarr format

### Key Design Patterns

1. **Plugin Architecture**: Readers and converters register themselves via decorators (`@register_reader`, `@register_converter`)
2. **Template Method**: `BaseMSIConverter.convert()` defines conversion workflow with customizable steps
3. **Iterator Pattern**: Spectra processing uses generators for memory efficiency
4. **Factory Pattern**: Registry creates appropriate reader/converter instances

### Data Flow

1. Format detection via registered detectors
2. Reader instantiation and metadata extraction
3. Common mass axis generation across all spectra
4. Streaming spectrum processing with sparse matrix construction
5. Output format-specific finalization and persistence

### Testing Structure

- `tests/unit/`: Fast tests for individual components
- `tests/integration/`: End-to-end conversion tests
- Test markers: `@pytest.mark.unit` and `@pytest.mark.integration`
- Mock data fixtures in `conftest.py`

### Special Considerations

- **Memory Management**: Large datasets processed in chunks via streaming
- **Cross-platform**: Windows/Linux support with platform-specific Bruker SDK handling
- **Sparse Data**: Uses scipy.sparse matrices for efficient storage of spectral data
- **Metadata Preservation**: Extracts and validates ontology terms from source formats
