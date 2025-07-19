# MSIConverter

[![Tests](https://github.com/tvisvikis/MSIConverter/actions/workflows/tests.yml/badge.svg)](https://github.com/tvisvikis/MSIConverter/actions/workflows/tests.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modern Python library for converting Mass Spectrometry Imaging (MSI) data into the standardized **SpatialData/Zarr format**, enabling seamless integration with spatial omics analysis workflows.

## 🚀 Features

- **Multiple Input Formats**: ImzML, Bruker (.d directories)
- **SpatialData Output**: Modern, cloud-ready format with Zarr backend
- **Memory Efficient**: Handles large datasets (100+ GB) through streaming processing
- **Metadata Preservation**: Extracts and maintains all acquisition parameters
- **3D Support**: Process volume data or treat as 2D slices
- **Dry-Run Mode**: Preview conversions before processing
- **Cross-Platform**: Windows, macOS, and Linux support

## 📦 Installation

### Via pip (Recommended)
```bash
pip install msiconvert
```

### Via conda
```bash
conda install -c conda-forge msiconvert
```

### From source
```bash
git clone https://github.com/tvisvikis/MSIConverter.git
cd MSIConverter
poetry install
```

## 🏃‍♂️ Quick Start

### Command Line Interface

```bash
# Basic conversion
msiconvert input.imzML output.zarr

# With custom parameters
msiconvert data.d output.zarr --pixel-size 50 --dataset-id "experiment_001"

# Dry run to preview conversion
msiconvert input.imzML output.zarr --dry-run

# 3D volume processing
msiconvert volume.imzML output.zarr --handle-3d
```

### Python API

```python
from msiconvert import convert_msi

# Simple conversion
success = convert_msi(
    input_path="data/sample.imzML",
    output_path="output/sample.zarr",
    pixel_size_um=25.0
)

# Advanced usage with custom parameters
success = convert_msi(
    input_path="data/experiment.d",
    output_path="output/experiment.zarr",
    dataset_id="exp_001",
    pixel_size_um=10.0,
    handle_3d=True
)
```

## 📊 Supported Formats

### Input Formats
| Format | Extension | Description | Status |
|--------|-----------|-------------|--------|
| ImzML | `.imzML` | Open standard for MS imaging | ✅ Full support |
| Bruker | `.d` | Bruker proprietary format | ✅ Full support |

### Output Formats
| Format | Description | Benefits |
|--------|-------------|----------|
| SpatialData/Zarr | Modern spatial omics standard | Cloud-ready, efficient, standardized |

## 🛠️ Advanced Usage

### Configuration Options

```bash
# All available options
msiconvert input.imzML output.zarr \
    --pixel-size 25 \
    --dataset-id "my_experiment" \
    --handle-3d \
    --optimize-chunks \
    --log-level DEBUG \
    --log-file conversion.log
```

### Batch Processing

```python
import glob
from msiconvert import convert_msi

# Process multiple files
for input_file in glob.glob("data/*.imzML"):
    output_file = input_file.replace(".imzML", ".zarr")
    convert_msi(input_file, output_file)
```

### Working with SpatialData

```python
import spatialdata as sd

# Load converted data
sdata = sd.read_zarr("output/sample.zarr")

# Access the MSI data
msi_data = sdata.tables["msi_dataset"]
print(f"Shape: {msi_data.shape}")
print(f"Mass channels: {msi_data.var.index}")
```

## 🔧 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/tvisvikis/MSIConverter.git
cd MSIConverter

# Install with development dependencies
poetry install

# Install pre-commit hooks
poetry run pre-commit install
```

### Running Tests

```bash
# Unit tests only
poetry run pytest -m "not integration"

# All tests
poetry run pytest

# With coverage
poetry run pytest --cov=msiconvert
```

### Code Quality

```bash
# Format code
poetry run black .
poetry run isort .

# Run linting
poetry run flake8

# Run all checks
poetry run pre-commit run --all-files
```

## 📚 Documentation

- **API Documentation**: [Auto-generated docs](https://github.com/tvisvikis/MSIConverter#readme)
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Architecture Overview**: [docs/architecture.md](docs/architecture.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`poetry run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/tvisvikis/MSIConverter/issues)
- **Discussions**: [GitHub Discussions](https://github.com/tvisvikis/MSIConverter/discussions)
- **Email**: t.visvikis@maastrichtuniversity.nl

## 🏆 Citation

If you use MSIConverter in your research, please cite:

```bibtex
@software{msiconvert2024,
  title = {MSIConverter: Modern Mass Spectrometry Imaging Data Conversion},
  author = {Visvikis, Theodoros},
  year = {2024},
  url = {https://github.com/tvisvikis/MSIConverter}
}
```

## 🙏 Acknowledgments

- Built with [SpatialData](https://spatialdata.scverse.org/) ecosystem
- Powered by [Zarr](https://zarr.readthedocs.io/) for efficient storage
- Uses [pyimzML](https://github.com/alexandrovteam/pyimzML) for ImzML parsing

---

**MSIConverter** - Bridging traditional MSI formats with modern spatial omics workflows 🔬✨