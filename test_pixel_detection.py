#!/usr/bin/env python3

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "msiconvert"))

# Set up logging to see all messages
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")

from msiconvert.core.registry import detect_format, get_reader_class


def test_pixel_detection(file_path):
    print(f"Testing pixel size detection for: {file_path}")

    # Detect format
    input_format = detect_format(file_path)
    print(f"Detected format: {input_format}")

    # Create reader
    reader_class = get_reader_class(input_format)
    print(f"Using reader: {reader_class.__name__}")

    reader = reader_class(file_path)

    # Test pixel size detection
    print("Attempting pixel size detection...")
    print(f"Reader type: {type(reader)}")
    print(f"Reader MRO: {type(reader).__mro__}")
    print(f"Reader has get_pixel_size method: {hasattr(reader, 'get_pixel_size')}")
    if hasattr(reader, "get_pixel_size"):
        print(f"get_pixel_size method: {reader.get_pixel_size}")
        print(f"get_pixel_size method class: {reader.get_pixel_size.__qualname__}")

    # Check if the method is implemented in the reader class
    if "get_pixel_size" in type(reader).__dict__:
        print("✓ get_pixel_size is implemented directly in the reader class")
    else:
        print("✗ get_pixel_size is inherited from a parent class")

    pixel_size = reader.get_pixel_size()
    print(f"get_pixel_size returned: {pixel_size}")

    if pixel_size is not None:
        print(
            f"✓ SUCCESS: Detected pixel size: {pixel_size[0]:.1f} x {pixel_size[1]:.1f} μm"
        )
    else:
        print("✗ FAILED: Could not detect pixel size")

        # If it's a Bruker reader, let's investigate the database
        if hasattr(reader, "conn") and reader.conn is not None:
            print("  Investigating Bruker database tables...")
            cursor = reader.conn.cursor()
            try:
                # List all tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                print(f"  Available tables: {[table[0] for table in tables]}")

                # Check for GlobalMetadata
                if any("GlobalMetadata" in table[0] for table in tables):
                    cursor.execute(
                        "SELECT Key, Value FROM GlobalMetadata WHERE Key LIKE '%pixel%' OR Key LIKE '%size%' OR Key LIKE '%step%'"
                    )
                    pixel_metadata = cursor.fetchall()
                    if pixel_metadata:
                        print(f"  Pixel-related metadata: {pixel_metadata}")
                    else:
                        print("  No pixel-related metadata found in GlobalMetadata")

                # Check MaldiFrameLaserInfo table specifically
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='MaldiFrameLaserInfo'"
                )
                if cursor.fetchone():
                    cursor.execute("PRAGMA table_info(MaldiFrameLaserInfo)")
                    columns = cursor.fetchall()
                    print(
                        f"  MaldiFrameLaserInfo columns: {[col[1] for col in columns]}"
                    )

                    # Check actual data in the table
                    cursor.execute("SELECT COUNT(*) FROM MaldiFrameLaserInfo")
                    row_count = cursor.fetchone()[0]
                    print(f"  MaldiFrameLaserInfo row count: {row_count}")

                    if row_count > 0:
                        cursor.execute(
                            "SELECT BeamScanSizeX, BeamScanSizeY, SpotSize FROM MaldiFrameLaserInfo LIMIT 5"
                        )
                        sample_data = cursor.fetchall()
                        print(f"  Sample data: {sample_data}")

                        # Test the exact same query that our get_pixel_size method uses
                        print("  Testing the exact detection logic...")
                        cursor.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name='MaldiFrameLaserInfo'"
                        )
                        table_exists = cursor.fetchone()
                        print(f"  Table check result: {table_exists}")

                        if table_exists:
                            cursor.execute(
                                "SELECT BeamScanSizeX, BeamScanSizeY, SpotSize FROM MaldiFrameLaserInfo LIMIT 1"
                            )
                            detection_result = cursor.fetchone()
                            print(f"  Detection query result: {detection_result}")
                            if detection_result:
                                beam_x, beam_y, spot = detection_result
                                print(
                                    f"  Parsed values: beam_x={beam_x}, beam_y={beam_y}, spot={spot}"
                                )
                                if beam_x is not None and beam_y is not None:
                                    print(
                                        f"  ✓ Should have detected: ({float(beam_x)}, {float(beam_y)})"
                                    )
                                else:
                                    print(f"  ✗ NULL values detected")
                            else:
                                print(f"  ✗ No result from detection query")
                else:
                    print("  MaldiFrameLaserInfo table not found")
            except Exception as e:
                print(f"  Database investigation failed: {e}")

    # Test other properties used by dry-run
    print("Testing other properties...")
    print(f"Shape: {reader.shape}")
    print(f"Number of spectra: {reader.n_spectra}")
    print("Testing mass range calculation...")
    try:
        mass_range = reader.mass_range
        print(f"Mass range: {mass_range}")
    except Exception as e:
        print(f"Error with mass_range: {e}")

    reader.close()
    return pixel_size


if __name__ == "__main__":
    # Test with pea.imzML
    pea_path = Path(r"C:\Users\tvisv\OneDrive\Data\MSIConvert Data\pea.imzML")
    if pea_path.exists():
        print("=== Testing ImzML Format ===")
        test_pixel_detection(pea_path)
    else:
        print(f"File not found: {pea_path}")

    print("\n" + "=" * 50 + "\n")

    # Test with Bruker data
    bruker_path = Path(
        r"C:\Users\tvisv\OneDrive\Data\MSIConvert Data\20231109_PEA_NEDC_bruker.d"
    )
    if bruker_path.exists():
        print("=== Testing Bruker Format ===")
        test_pixel_detection(bruker_path)
    else:
        print(f"File not found: {bruker_path}")
