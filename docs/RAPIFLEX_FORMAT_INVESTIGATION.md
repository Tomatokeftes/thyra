# Bruker rapifleX/FlexImaging Data Format Investigation

This document summarizes the reverse-engineering investigation of Bruker rapifleX MALDI-TOF imaging data format.

## Overview

**Key finding: No SDK required - the format can be read directly with pure Python.**

| File | Purpose | Format |
|------|---------|--------|
| `.dat` | Spectral intensities | Binary (header + offset table + float32 arrays) |
| `_poslog.txt` | X/Y coordinates | Text (raster positions R00X{x}Y{y}) |
| `_info.txt` | Metadata | Text (mass range, raster, spots count) |
| `.mis` | Imaging sequence | XML (method, teaching points) |
| `.d/` folder | MCF containers | Not needed for basic reading |

## File Structure

Unlike timsTOF which uses a single `.d` folder, rapifleX data is organized as:

```
20230303_Brain GirT/              <- Parent folder (NOT .d!)
  +-- 20230303_Brain GirT.dat     <- Main spectral data (4.2 GB)
  +-- 20230303_Brain GirT.mis     <- Method/alignment XML
  +-- 20230303_Brain GirT_info.txt
  +-- 20230303_Brain GirT_poslog.txt
  +-- *.tif                       <- Optical images
  +-- 20230303_Brain GirT.d/      <- MCF containers (not needed)
        +-- *.mcf                 <- Binary containers
        +-- *.mcf_idx             <- SQLite index files
```

## .dat File Structure

The main spectral data is stored in a binary `.dat` file with the following structure:

```
Offset    Size              Content
------    ----              -------
0         48 bytes          Fixed header
48        n_spots * 4       Offset table (uint32 per spectrum)
varies    n_datapoints * 4  Float32 intensity arrays (one per valid spectrum)
```

### Header Fields (48 bytes)

| Offset | Type | Value (example) | Description |
|--------|------|-----------------|-------------|
| 0 | uint32 | 48 | Header size |
| 4 | uint32 | 256 | Unknown |
| 8 | uint32 | 1391 | First raster X coordinate |
| 12 | uint32 | 312 | First raster Y coordinate |
| 16 | uint32 | 424 | Raster width |
| 20 | uint32 | 494 | Raster height |
| 24 | uint32 | 29100 | Data points per spectrum |

### Offset Table

- Located at byte 48
- Contains one uint32 offset per spot position
- Offset of 0 indicates no data for that position (off-tissue)
- Valid offsets point to the start of float32 intensity arrays

### Spectral Data

- Each spectrum is stored as `n_datapoints` float32 values
- Intensities are stored in order corresponding to the m/z axis
- m/z axis is computed linearly from mass_start to mass_end

## _info.txt Metadata

Plain text file with key acquisition parameters:

```
FlexImaging Info File
Name of Sample: 20230303_Brain GirT
Number of Spots: 38744
Number of Shots: 100
Spectrum Size: 30271
Detector Gain: 2.08
Mass Start: 99.9979
Mass End: 600.009
Acquisition Mode: REFLECTOR
Instrument Serial Number: 1834948.10106
Laser Power: 53
Sample Rate: 1.25
DataPoints: 29100
Method: D:\Methods\flexControlMethods\Bryn\RP_100-1000_Da.par
flexImaging Version: 5.1.46.0_1455_51
flexControl Version: 4.0.46.0_867_879
Raster: 20,20
Start Time: Fri, 03.03.2023 15:49:59
End Time: Fri, 03.03.2023 16:23:26
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| Number of Spots | Total raster positions |
| DataPoints | Number of intensity values per spectrum |
| Mass Start/End | m/z range in Daltons |
| Raster | Step size in micrometers (X,Y) |
| Acquisition Mode | REFLECTOR or LINEAR |

## _poslog.txt Coordinates

Tab-separated text file with position information:

```
#Timestamp Pos X Y Z
2023-03-03 15:50:10.355891 R00X1391Y312 47965.0 -22720.0 0.00
2023-03-03 15:50:10.399959 R00X1392Y312 47985.0 -22720.0 0.00
...
```

### Fields

| Column | Description |
|--------|-------------|
| Timestamp | Acquisition time |
| Pos | Raster position ID (R{region}X{x}Y{y}) |
| X | Physical X coordinate (micrometers) |
| Y | Physical Y coordinate (micrometers) |
| Z | Physical Z coordinate (micrometers) |

### Coordinate Extraction

Raster coordinates are encoded in the position ID:
- Format: `R{region}X{raster_x}Y{raster_y}`
- Example: `R00X1391Y312` = region 0, raster X=1391, raster Y=312

## .mis File (Imaging Sequence)

XML file containing method information and teaching points for image alignment:

```xml
<ImagingSequence flexImagingVersion="5.1.46.0_1453_60">
  <Method>D:\Methods\flexControlMethods\Bryn\RP_100-1000_Da.par</Method>
  <ImageFile>20230303_Brain deriv GirT_0000.tif</ImageFile>
  <Raster>20,20</Raster>
  <TeachPoint>1020,2344;-31922,23383</TeachPoint>
  <TeachPoint>20604,2348;6957,23375</TeachPoint>
  <TeachPoint>760,10940;-32477,6220</TeachPoint>
  ...
</ImagingSequence>
```

### Teaching Points

Teaching points define the affine transformation between image pixels and stage coordinates:

| Image (pixels) | Stage (micrometers) |
|----------------|---------------------|
| (1020, 2344) | (-31922, 23383) |
| (20604, 2348) | (6957, 23375) |
| (760, 10940) | (-32477, 6220) |

The transformation matrix:
- scale_x: ~1.985 (stage units per pixel)
- scale_y: ~-1.997 (negative due to Y-axis flip)
- Near-zero shear (image is axis-aligned)

## TIFF Optical Images

| File | Size | Purpose |
|------|------|---------|
| `*deriv GirT.tif` | 5417 x 2833 | Low-resolution optical overview |
| `*_0000.tif` | 21744 x 11464 | High-resolution reference image (used for teaching) |
| `*_0001.tif` | 5436 x 2866 | Derived/processed image |

## Data Coverage

Not all raster positions contain spectral data. In the test dataset:

- **Total positions**: 38,744
- **Valid spectra**: 15,664 (40.4%)
- **Empty positions**: 23,080 (off-tissue or acquisition skipped)

The offset table in the .dat file indicates which positions have data (offset > 0).

## Coordinate System Alignment Issue

**Important**: The teaching point stage coordinates and poslog physical coordinates use different reference frames:

| Source | X Range | Y Range |
|--------|---------|---------|
| Teaching points | -32,477 to 6,957 | 6,220 to 23,383 |
| Poslog | 47,965 to 56,425 | -32,580 to -22,720 |

These ranges do not overlap, indicating:
1. Different coordinate origins
2. Possible stage recalibration between teaching and acquisition
3. Manual alignment verification may be required

## Comparison with timsTOF Format

| Feature | timsTOF (.tdf) | rapifleX (.dat) |
|---------|---------------|-----------------|
| SDK required | Yes (timsdata.dll) | No |
| Data format | SQLite + binary | Pure binary |
| m/z storage | Per-spectrum (profile) | Uniform axis (profile) |
| Coordinates | In SQLite database | Separate _poslog.txt file |
| Metadata | SQLite tables | Text files (_info.txt) |
| Ion mobility | Yes | No |
| Container | .d folder | Parent folder with .dat |

## Reading Algorithm

```python
def read_rapiflex(folder_path):
    # 1. Parse _info.txt for metadata
    info = parse_info_file(folder_path / "*_info.txt")
    n_spots = info['Number of Spots']
    n_datapoints = info['DataPoints']
    mass_start = info['Mass Start']
    mass_end = info['Mass End']

    # 2. Compute m/z axis (linear)
    mz_axis = np.linspace(mass_start, mass_end, n_datapoints)

    # 3. Parse _poslog.txt for coordinates
    positions = parse_poslog(folder_path / "*_poslog.txt")

    # 4. Read .dat file
    with open(folder_path / "*.dat", 'rb') as f:
        # Skip 48-byte header
        f.seek(48)

        # Read offset table
        offsets = np.fromfile(f, dtype=np.uint32, count=n_spots)

        # Read each valid spectrum
        for i, offset in enumerate(offsets):
            if offset > 0:
                f.seek(offset)
                intensities = np.fromfile(f, dtype=np.float32, count=n_datapoints)
                yield positions[i], mz_axis, intensities
```

## Implementation Recommendations

1. **Create `RapifleXReader`** class that:
   - Detects parent folder (contains .dat, .mis, _poslog.txt)
   - Reads spectra from .dat file directly (no SDK needed)
   - Parses coordinates from _poslog.txt
   - Extracts metadata from _info.txt and .mis

2. **Format detection**: Look for .dat file + _poslog.txt in same folder

3. **TIFF integration**: Store as additional image layers in SpatialData, but flag that alignment may need manual verification

4. **Teaching points**: Store as metadata for potential future alignment tools

## References

- Bruker FlexImaging User Manual
- Test data: `20230303_Brain GirT` (Brain tissue MALDI imaging)
- Acquisition: rapifleX MALDI-TOF, REFLECTOR mode, 100-600 Da
