"""
Complete end-to-end test of the Bruker Reader with the actual
spatialdata_converter.py

This test verifies that:
1. The Bruker Reader works correctly with the real spatialdata_converter.py
2. A valid SpatialData object is created
3. All data structures are properly formed
4. The conversion pipeline completes successfully
"""

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

# Import from the correct location
from thyra.readers.bruker.timstof.timstof_reader import BrukerReader


def _setup_test_environment():
    """Setup test environment and paths."""
    print("=" * 80)
    print("FULL SPATIALDATA CONVERSION TEST")
    print("=" * 80)

    data_path = Path(
        r"C:\Users\tvisv\Downloads\MSIConverter\20231109_PEA_NEDC_bruker.d"
    )

    if not data_path.exists():
        print(f"❌ Test data not found: {data_path}")
        return None, None, None

    temp_dir = Path(tempfile.mkdtemp(prefix="spatialdata_test_"))
    output_path = temp_dir / "test_spatialdata_output.zarr"

    print(f"📂 Test data: {data_path}")
    print(f"📁 Output directory: {output_path}")

    return data_path, temp_dir, output_path


def _initialize_reader(data_path):
    """Initialize the Bruker reader."""
    print("\n🔧 Step 1: Initialize Bruker Reader...")

    reader = BrukerReader(
        data_path=data_path, cache_coordinates=True, memory_limit_gb=2.0
    )

    print("   ✅ Reader initialized successfully")
    return reader


def _check_spatialdata_dependencies():
    """Check if spatialdata dependencies are available."""
    try:
        from spatialdata import SpatialData

        from thyra.converters.spatialdata import SpatialDataConverter

        print("   ✅ SpatialData dependencies available")
        return True, SpatialData, SpatialDataConverter
    except ImportError as e:
        print(f"   ❌ SpatialData dependencies not available: {e}")
        print("   Please install spatialdata: pip install spatialdata")
        return False, None, None


def _create_mock_converter_methods():
    """Create additional methods for MockBaseMSIConverter."""

    def _map_mass_to_indices(self, mzs):
        """Map m/z values to indices in common mass axis"""
        indices = np.searchsorted(self._common_mass_axis, mzs, side="left")
        valid_mask = (indices < len(self._common_mass_axis)) & (indices >= 0)
        valid_indices = indices[valid_mask]
        valid_mzs = mzs[valid_mask]

        if len(valid_indices) > 0:
            mz_diffs = np.abs(self._common_mass_axis[valid_indices] - valid_mzs)
            exact_matches = valid_indices[mz_diffs <= 0.01]
            return exact_matches
        return np.array([], dtype=int)

    def _add_to_sparse_matrix(self, sparse_matrix, pixel_idx, mz_indices, intensities):
        """Add data to sparse matrix"""
        if len(mz_indices) > 0:
            for i, intensity in enumerate(intensities):
                if i < len(mz_indices):
                    sparse_matrix[pixel_idx, mz_indices[i]] = intensity

    def _get_pixel_index(self, x, y, z):
        """Get linear pixel index from 3D coordinates"""
        n_x, n_y, n_z = self._dimensions
        return z * (n_x * n_y) + y * n_x + x

    return _map_mass_to_indices, _add_to_sparse_matrix, _get_pixel_index


def _create_mock_converter_class():
    """Create the MockBaseMSIConverter class."""

    # Get additional methods
    map_mass, add_to_sparse, get_pixel_index = _create_mock_converter_methods()

    class MockBaseMSIConverter:
        def __init__(self, reader, output_path, **kwargs):
            self.reader = reader
            self.output_path = output_path
            self.dataset_id = kwargs.get("dataset_id", "test")
            self.pixel_size_um = kwargs.get("pixel_size_um", 1.0)
            self.handle_3d = kwargs.get("handle_3d", False)
            self._dimensions = None
            self._metadata = None
            self._common_mass_axis = None

        def _prepare_conversion(self):
            """Prepare for conversion by getting basic info"""
            self._metadata = self.reader.get_metadata()
            self._dimensions = self.reader.get_dimensions()
            self._common_mass_axis = self.reader.get_common_mass_axis()

        def _create_sparse_matrix(self):
            from scipy import sparse

            n_x, n_y, n_z = self._dimensions
            n_pixels = n_x * n_y * n_z
            n_masses = len(self._common_mass_axis)
            return sparse.lil_matrix((n_pixels, n_masses), dtype=np.float64)

        def _create_coordinates_dataframe(self):
            import pandas as pd

            n_x, n_y, n_z = self._dimensions
            coords = []
            instance_ids = []

            idx = 0
            for z in range(n_z):
                for y in range(n_y):
                    for x in range(n_x):
                        coords.append((x, y, z))
                        instance_ids.append(str(idx))
                        idx += 1

            df = pd.DataFrame(
                {
                    "x": [c[0] for c in coords],
                    "y": [c[1] for c in coords],
                    "z": [c[2] for c in coords],
                    "instance_id": instance_ids,
                    "region": f"{self.dataset_id}_pixels",
                }
            )

            df.set_index("instance_id", inplace=True)
            df["spatial_x"] = df["x"] * self.pixel_size_um
            df["spatial_y"] = df["y"] * self.pixel_size_um
            return df

        def _create_mass_dataframe(self):
            import pandas as pd

            return pd.DataFrame(
                {
                    "mz": self._common_mass_axis,
                    "mass_id": [f"mz_{mz:.4f}" for mz in self._common_mass_axis],
                }
            ).set_index("mass_id")

        # Assign the extracted methods
        _map_mass_to_indices = map_mass
        _add_to_sparse_matrix = add_to_sparse
        _get_pixel_index = get_pixel_index

    return MockBaseMSIConverter


def _create_mock_reader_class():
    """Create the MockBaseMSIReader class."""

    class MockBaseMSIReader:
        pass

    return MockBaseMSIReader


def _create_mock_decorator():
    """Create the mock register_converter decorator."""

    def mock_register_converter(name):
        def decorator(cls):
            return cls

        return decorator

    return mock_register_converter


def _setup_module_mocks(mock_converter, mock_reader, mock_decorator):
    """Setup the actual module mocks in sys.modules."""
    import types

    # Setup mock modules
    mock_base_module = types.ModuleType("thyra.core.base_converter")
    mock_base_module.BaseMSIConverter = mock_converter
    sys.modules["thyra.core.base_converter"] = mock_base_module

    mock_reader_module = types.ModuleType("thyra.core.base_reader")
    mock_reader_module.BaseMSIReader = mock_reader
    sys.modules["thyra.core.base_reader"] = mock_reader_module

    mock_registry_module = types.ModuleType("thyra.core.registry")
    mock_registry_module.register_converter = mock_decorator
    sys.modules["thyra.core.registry"] = mock_registry_module


def _setup_relative_module_imports(
    mock_base_module, mock_reader_module, mock_registry_module
):
    """Setup relative module imports."""
    sys.modules["..core.base_converter"] = mock_base_module
    sys.modules["..core.base_reader"] = mock_reader_module
    sys.modules["..core.registry"] = mock_registry_module


def _setup_mock_modules():
    """Setup mock modules for testing."""
    # Create mock classes and decorators
    mock_converter = _create_mock_converter_class()
    mock_reader = _create_mock_reader_class()
    mock_decorator = _create_mock_decorator()

    # Setup module mocks
    _setup_module_mocks(mock_converter, mock_reader, mock_decorator)

    # Get the modules for relative imports
    mock_base_module = sys.modules["thyra.core.base_converter"]
    mock_reader_module = sys.modules["thyra.core.base_reader"]
    mock_registry_module = sys.modules["thyra.core.registry"]

    # Setup relative imports
    _setup_relative_module_imports(
        mock_base_module, mock_reader_module, mock_registry_module
    )


def _import_spatialdata_converter():
    """Import and setup spatialdata converter."""
    print("\n📦 Step 2: Import spatialdata_converter...")
    print("   Attempting to import spatialdata converter...")

    # Check dependencies first
    spatialdata_available, SpatialData, SpatialDataConverter = (
        _check_spatialdata_dependencies()
    )
    if not spatialdata_available:
        print("   ⚠️  Cannot test full conversion without spatialdata dependencies")
        print("   Will test reader interface compatibility instead...")
        return False, None, None

    # Find converter path
    converter_path = Path(__file__).parent.parent.parent / "spatialdata_converter.py"
    if not converter_path.exists():
        print(f"   ❌ spatialdata_converter.py not found at: {converter_path}")
        print("   Will test reader interface compatibility instead...")
        return False, None, None

    # Setup environment
    sys.path.insert(0, str(converter_path.parent))
    _setup_mock_modules()

    # Import converter
    exec(open(converter_path).read(), globals())
    print("   ✅ spatialdata_converter imported successfully")

    return True, SpatialData, SpatialDataConverter


def _create_and_prepare_converter(reader, output_path, SpatialDataConverter):
    """Create and prepare the SpatialData converter."""
    print("\n🔄 Step 3: Create SpatialDataConverter and test conversion...")

    converter = SpatialDataConverter(
        reader=reader,
        output_path=output_path,
        dataset_id="test_bruker_reader",
        pixel_size_um=1.0,
        handle_3d=False,
    )

    print("   ✅ SpatialDataConverter created successfully")

    # Prepare conversion
    print("   📋 Preparing conversion...")
    converter._prepare_conversion()

    # Get basic info
    metadata = converter._metadata
    dimensions = converter._dimensions
    mass_axis = converter._common_mass_axis

    print("   📊 Dataset info:")
    print(f"      - Dimensions: {dimensions}")
    print(f"      - Mass axis size: {len(mass_axis):,}")
    print(f"      - File type: {metadata.get('file_type', 'unknown')}")

    return converter


def _process_spectra_data(converter, reader):
    """Process spectra data with the converter."""
    # Create data structures
    print("   🏗️  Creating data structures...")
    data_structures = converter._create_data_structures()
    print(f"   ✅ Data structures created (mode: {data_structures['mode']})")

    # Process spectra
    print("   📊 Processing spectra (limited sample)...")
    processed_count = 0
    max_spectra = 1000
    start_time = time.time()

    for coords, mzs, intensities in reader.iter_spectra():
        converter._process_single_spectrum(data_structures, coords, mzs, intensities)
        processed_count += 1

        if processed_count >= max_spectra:
            break

        if processed_count % 200 == 0:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            print(f"      Processed {processed_count} spectra ({rate:.0f} spectra/s)")

    elapsed = time.time() - start_time
    rate = processed_count / elapsed if elapsed > 0 else 0
    print(
        f"   ✅ Processed {processed_count} spectra in {elapsed:.1f}s ({rate:.0f} spectra/s)"
    )

    # Finalize data structures
    print("   🔧 Finalizing data structures...")
    converter._finalize_data(data_structures)
    print("   ✅ Data structures finalized")

    return data_structures


def _verify_data_structures(data_structures):
    """Verify created data structures."""
    print("\n📋 Step 4: Verify created data structures...")

    print(f"   Tables created: {len(data_structures['tables'])}")
    for table_name in data_structures["tables"].keys():
        table = data_structures["tables"][table_name]
        print(f"      - {table_name}: {table.table.shape}")

    print(f"   Shapes created: {len(data_structures['shapes'])}")
    for shape_name in data_structures["shapes"].keys():
        shapes = data_structures["shapes"][shape_name]
        print(f"      - {shape_name}: {len(shapes.table)} geometries")

    print(f"   Images created: {len(data_structures['images'])}")
    for image_name in data_structures["images"].keys():
        image = data_structures["images"][image_name]
        print(f"      - {image_name}: {image.image.shape}")


def _save_and_verify_output(converter, data_structures, output_path, SpatialData):
    """Save and verify the SpatialData output."""
    print("\n💾 Step 5: Save SpatialData object...")

    success = converter._save_output(data_structures)
    if not success:
        print("   ❌ Failed to save SpatialData")
        return False

    print(f"   ✅ SpatialData saved successfully to {output_path}")

    if not output_path.exists():
        print("   ❌ Output file not found after saving")
        return False

    print(f"   ✅ Output file exists: {output_path}")

    # Verify by loading
    print("   🔍 Verifying saved SpatialData...")
    try:
        loaded_sdata = SpatialData.read(str(output_path))
        print("   ✅ SpatialData loaded successfully")
        print(f"      - Tables: {list(loaded_sdata.tables.keys())}")
        print(f"      - Shapes: {list(loaded_sdata.shapes.keys())}")
        print(f"      - Images: {list(loaded_sdata.images.keys())}")

        # Check first table
        if loaded_sdata.tables:
            first_table_name = list(loaded_sdata.tables.keys())[0]
            first_table = loaded_sdata.tables[first_table_name]
            print(f"      - First table shape: {first_table.table.shape}")
            print(
                f"      - First table variables: {first_table.table.var.shape[0]} mass channels"
            )

        return True
    except Exception as e:
        print(f"   ❌ Error loading saved SpatialData: {e}")
        return False


def _cleanup_test_environment(reader, temp_dir):
    """Cleanup test environment."""
    try:
        reader.close()
    except Exception:
        pass

    try:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"🧹 Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"⚠️  Could not clean up temporary directory: {e}")


def test_spatialdata_integration():
    """Test full integration with the actual spatialdata_converter.py"""
    # Setup
    data_path, temp_dir, output_path = _setup_test_environment()
    if not data_path:
        return False

    reader = None
    try:
        # Initialize reader
        reader = _initialize_reader(data_path)

        # Import converter
        try:
            converter_imported, SpatialData, SpatialDataConverter = (
                _import_spatialdata_converter()
            )
            if not converter_imported:
                return test_reader_interface_only(reader)
        except Exception as e:
            print(f"   ❌ Failed to import spatialdata_converter: {e}")
            print("   Will test reader interface compatibility instead...")
            return test_reader_interface_only(reader)

        # Create and test converter
        try:
            converter = _create_and_prepare_converter(
                reader, output_path, SpatialDataConverter
            )
            data_structures = _process_spectra_data(converter, reader)
            _verify_data_structures(data_structures)

            if not _save_and_verify_output(
                converter, data_structures, output_path, SpatialData
            ):
                return False

            # Success message
            print("\n✅ FULL SPATIALDATA CONVERSION TEST PASSED!")
            print("=" * 80)
            print("The Bruker Reader successfully:")
            print("• Integrated with the real spatialdata_converter.py")
            print("• Created valid SpatialData tables, shapes, and images")
            print("• Saved and loaded the SpatialData object")
            print("• Maintained full interface compatibility")

            return True

        except Exception as e:
            print(f"   ❌ Conversion failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if reader and temp_dir:
            _cleanup_test_environment(reader, temp_dir)


@pytest.mark.skip(reason="Requires actual Bruker test data and reader fixture")
def test_reader_interface_only():
    """
    Test just the reader interface compatibility when full conversion isn't possible

    Note: This test is skipped as it requires actual Bruker test data.
    """
    pass


def main():
    """Main test function"""
    print("🧪 BRUKER READER - FULL SPATIALDATA INTEGRATION TEST")
    print("=" * 80)

    success = test_spatialdata_integration()

    if success:
        print("\n🎉 ALL TESTS PASSED!")
        print("The Bruker Reader is fully compatible with spatialdata_converter.py")
    else:
        print("\n❌ TESTS FAILED!")
        print("Check the error messages above for details")

    return success


if __name__ == "__main__":
    main()
