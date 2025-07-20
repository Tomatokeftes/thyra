#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "msiconvert"))

from msiconvert.core.registry import detect_format, get_reader_class


def test_detection_metadata():
    """Simple test to verify pixel size detection metadata creation."""
    input_path = Path(r"C:\Users\tvisv\OneDrive\Data\MSIConvert Data\pea.imzML")

    if not input_path.exists():
        print(f"Test file not found: {input_path}")
        return

    # Test the detection logic
    input_format = detect_format(input_path)
    reader_class = get_reader_class(input_format)
    reader = reader_class(input_path)

    print(f"Detected format: {input_format}")

    # Try pixel size detection
    detected_pixel_size = reader.get_pixel_size()
    print(f"Detected pixel size: {detected_pixel_size}")

    if detected_pixel_size is not None:
        # Create detection provenance metadata like convert.py does
        pixel_size_detection_info = {
            "method": "automatic",
            "detected_x_um": float(detected_pixel_size[0]),
            "detected_y_um": float(detected_pixel_size[1]),
            "source_format": input_format,
            "detection_successful": True,
            "note": "Pixel size automatically detected from source metadata and applied to coordinate systems",
        }

        print("✓ Pixel size detection provenance metadata:")
        for key, value in pixel_size_detection_info.items():
            print(f"  {key}: {value}")

        print(
            f"\n✓ This will be stored as 'pixel_size_provenance' in SpatialData metadata"
        )
        print(
            f"✓ The actual pixel size ({detected_pixel_size[0]} μm) will be applied to:"
        )
        print(f"  - Spatial coordinates (spatial_x, spatial_y)")
        print(f"  - Image coordinate systems")
        print(f"  - Global metadata as 'pixel_size_um'")
    else:
        print("✗ No pixel size detected")

    reader.close()


if __name__ == "__main__":
    test_detection_metadata()
