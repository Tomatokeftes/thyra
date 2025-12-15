# Bruker SDK DLL Location

This folder is the designated location for the Bruker SDK library files.

## Required Files

Place the appropriate Bruker SDK library file in this directory:

- Windows: `timsdata.dll`
- Linux: `libtimsdata.so`
- macOS: `libtimsdata.dylib` (limited support)

## Where to Get the SDK

The Bruker SDK libraries may also be obtained from:

1. Bruker Daltonics official channels
2. The Bruker timsTOF software installation
3. Contact your Bruker representative
4. Some GitHub repositories

## Installation

1. Obtain the SDK library file for your platform
2. Copy it to this directory
3. The library will be automatically detected by Thyra

## Alternative Locations

If you cannot place the DLL in this folder, you can also:

1. Set the `BRUKER_SDK_PATH` environment variable to point to the DLL location
2. Place the DLL in the same directory as your data (.d folder)
3. Place the DLL in your current working directory
4. Add the SDK directory to your system PATH

However, placing the DLL in this repository folder is the MOST RELIABLE method.
