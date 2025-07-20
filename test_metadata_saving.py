#!/usr/bin/env python3

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "msiconvert"))

from msiconvert.convert import convert_msi


def test_metadata_saving():
    """Test that pixel size detection metadata is saved in SpatialData."""
    # Test with a small conversion
    input_path = Path(r"C:\Users\tvisv\OneDrive\Data\MSIConvert Data\pea.imzML")

    if not input_path.exists():
        print(f"Test file not found: {input_path}")
        return

    # Create a temporary output directory
    temp_dir = tempfile.mkdtemp()
    output_path = Path(temp_dir) / "test_output.zarr"

    try:
        print(f"Testing conversion with automatic pixel size detection...")
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")

        # Run conversion with automatic detection (no pixel_size_um specified)
        success = convert_msi(
            input_path=str(input_path),
            output_path=str(output_path),
            dataset_id="test_metadata",
        )

        if success:
            print("✓ Conversion successful!")

            # Now let's check if the metadata was saved
            try:
                import spatialdata

                sdata = spatialdata.read_zarr(str(output_path))

                if hasattr(sdata, "metadata") and sdata.metadata:
                    print(f"✓ Metadata found: {list(sdata.metadata.keys())}")

                    if "pixel_size_detection" in sdata.metadata:
                        detection_info = sdata.metadata["pixel_size_detection"]
                        print(f"✓ Pixel size detection metadata found:")
                        for key, value in detection_info.items():
                            print(f"  {key}: {value}")
                    else:
                        print("✗ No pixel_size_detection metadata found")
                        print(f"Available metadata keys: {list(sdata.metadata.keys())}")
                else:
                    print("✗ No metadata found in SpatialData object")

            except ImportError:
                print("SpatialData not available for metadata verification")

        else:
            print("✗ Conversion failed")

    finally:
        # Clean up
        try:
            import shutil

            shutil.rmtree(temp_dir)
            print(f"Cleaned up: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {temp_dir}: {e}")


if __name__ == "__main__":
    test_metadata_saving()
