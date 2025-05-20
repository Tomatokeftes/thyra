#!/usr/bin/env python
"""
Simple script to calculate and compare the total intensity sum across different spectrum sources.
"""

import numpy as np
from pathlib import Path
import scipy.sparse as sparse
from pyimzml.ImzMLParser import ImzMLParser
import zarr
from spatialdata import SpatialData

def load_spatialdata_spectrum(zarr_path):
    """Load the stored average spectrum from SpatialData."""
    print(f"Loading SpatialData from {zarr_path}")
    try:
        # Try to load with regular SpatialData
        sdata = SpatialData.read(zarr_path)
    except Exception as e:
        print(f"Version mismatch: {e}")
        try:
            # Alternative approach: open zarr directly
            z = zarr.open(zarr_path, mode='r')
            
            # Find average spectrum table
            if 'tables' in z and any('average' in k for k in z['tables'].keys()):
                avg_key = [k for k in z['tables'].keys() if 'average' in k][0]
                # Get X matrix data
                if 'X' in z['tables'][avg_key]:
                    intensities = z['tables'][avg_key]['X'][...]
                    if len(intensities.shape) > 1:
                        intensities = intensities.flatten()
                    return intensities
            else:
                print("Could not find average spectrum in zarr store")
                return None
        except Exception as e2:
            print(f"Error opening zarr directly: {e2}")
            return None
    
    # Find the average spectrum table
    avg_keys = [key for key in sdata.tables.keys() if 'average' in key.lower()]
    if not avg_keys:
        print("No average spectrum found in SpatialData")
        return None
        
    avg_key = avg_keys[0]
    avg_table = sdata.tables[avg_key]
    
    # Extract intensities
    try:
        intensities = avg_table.X.toarray().flatten() if sparse.issparse(avg_table.X) else avg_table.X.flatten()
        return intensities
    except Exception as e:
        print(f"Error extracting stored intensities: {e}")
        return None

def recalculate_spectrum_from_spatialdata(zarr_path):
    """Recalculate average spectrum from SpatialData raw data."""
    print(f"Recalculating average spectrum from SpatialData")
    try:
        # Try to load with regular SpatialData
        sdata = SpatialData.read(zarr_path)
    except Exception as e:
        print(f"Version mismatch: {e}")
        try:
            # Alternative approach: open zarr directly
            z = zarr.open(zarr_path, mode='r')
            
            # Find main data table
            main_keys = [k for k in z['tables'].keys() if 'average' not in k]
            if not main_keys:
                print("Could not find main data table in zarr store")
                return None
                
            main_key = main_keys[0]
            
            # Get dimensions for counting pixels
            total_pixels = 0
            # Try to get from tic image
            if 'images' in z and any('tic' in k for k in z['images'].keys()):
                tic_key = [k for k in z['images'].keys() if 'tic' in k][0]
                if 'raster' in z['images'][tic_key]:
                    total_pixels = z['images'][tic_key]['raster'].size
            
            # Process X matrix data - this is complex as it might be sparse
            # For simplicity, let's sum all intensities and divide by pixels
            if 'X' in z['tables'][main_key]:
                # This is simplified and might not work with all sparse formats
                total_intensity = np.sum(z['tables'][main_key]['X'])
                if total_pixels > 0:
                    n_mass_vals = z['tables'][main_key]['X'].shape[1]
                    avg_intensity = total_intensity / total_pixels / n_mass_vals
                    return np.ones(n_mass_vals) * avg_intensity
            
            print("Could not recalculate from zarr directly")
            return None
        except Exception as e2:
            print(f"Error processing zarr directly: {e2}")
            return None
            
    # Find the main data table
    main_keys = [key for key in sdata.tables.keys() if 'average' not in key.lower()]
    if not main_keys:
        print("No main data table found in SpatialData")
        return None
        
    main_key = main_keys[0]
    main_table = sdata.tables[main_key]
    
    # Get intensity matrix
    X = main_table.X
    
    # Calculate average
    try:
        # Simple approach - get the mean along axis 0
        if sparse.issparse(X):
            avg_intensities = np.array(X.mean(axis=0)).flatten()
        else:
            avg_intensities = X.mean(axis=0)
            
        return avg_intensities
    except Exception as e:
        print(f"Error recalculating intensities: {e}")
        return None

def get_imzml_spectrum(imzml_path):
    """Extract average spectrum from imzML file using two-pass approach with unified m/z axis."""
    print(f"Loading imzML file: {imzml_path}")
    try:
        parser = ImzMLParser(str(imzml_path))
        
        # First pass: Collect all unique m/z values
        print("First pass: collecting all unique m/z values")
        all_mzs = set()
        for idx, (x, y, z) in enumerate(parser.coordinates):
            mzs, intensities = parser.getspectrum(idx)
            
            # Skip empty spectra
            if len(mzs) == 0 or len(intensities) == 0:
                continue
                
            # Add all m/z values to our set
            all_mzs.update(mzs)
            
            # Progress indication
            if idx % 1000 == 0 and idx > 0:
                print(f"Processed {idx} spectra for unique m/z values")
        
        # Convert to sorted numpy array for searchsorted
        unified_mzs = np.array(sorted(all_mzs))
        print(f"Created unified m/z axis with {len(unified_mzs)} points")
        
        # Second pass: Map intensities to unified m/z axis
        print("Second pass: mapping intensities to unified m/z axis")
        total_intensities = np.zeros(len(unified_mzs))
        total_spectra = 0
        
        for idx, (x, y, z) in enumerate(parser.coordinates):
            mzs, intensities = parser.getspectrum(idx)
            
            # Skip empty spectra
            if len(mzs) == 0 or len(intensities) == 0:
                continue
                
            # Find indices in the unified m/z axis using searchsorted
            indices = np.searchsorted(unified_mzs, mzs)
            
            # Skip if any index is out of bounds (shouldn't happen with our approach)
            if np.any(indices >= len(unified_mzs)):
                print(f"Warning: Index out of bounds for spectrum {idx}")
                continue
                
            # Add intensities to the appropriate bins
            for i, intensity in zip(indices, intensities):
                total_intensities[i] += intensity
                
            total_spectra += 1
            
            # Progress indication
            if idx % 1000 == 0 and idx > 0:
                print(f"Processed {idx} spectra for intensities")
        
        if total_spectra > 0:
            avg_intensities = total_intensities / total_spectra
            return unified_mzs, avg_intensities
        else:
            print("No valid spectra found")
            return None, None
    except Exception as e:
        print(f"Error processing imzML: {e}")
        return None, None

def count_pixels(zarr_path):
    """Count total pixels and pixels with data."""
    try:
        # Open zarr store
        z = zarr.open(zarr_path, mode='r')
        
        # Get dimensions from TIC image
        if 'images' in z and any('tic' in k for k in z['images'].keys()):
            tic_key = [k for k in z['images'].keys() if 'tic' in k][0]
            if 'raster' in z['images'][tic_key]:
                tic_array = z['images'][tic_key]['raster'][...]
                total_pixels = tic_array.size
                nonzero_pixels = np.count_nonzero(tic_array)
                return total_pixels, nonzero_pixels
                
        # Try to get from main table
        if 'tables' in z:
            main_keys = [k for k in z['tables'].keys() if 'average' not in k]
            if main_keys:
                main_key = main_keys[0]
                if 'X' in z['tables'][main_key]:
                    data_pixels = z['tables'][main_key]['X'].shape[0]
                    return None, data_pixels
        
        return None, None
    except Exception as e:
        print(f"Error counting pixels: {e}")
        return None, None

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare total intensity sums across spectrum sources")
    parser.add_argument("zarr_path", help="Path to SpatialData zarr store")
    parser.add_argument("--imzml", help="Path to original imzML file (optional)")
    args = parser.parse_args()
    
    zarr_path = args.zarr_path
    imzml_path = args.imzml
    
    # Count pixels
    total_pixels, nonzero_pixels = count_pixels(zarr_path)
    if total_pixels is not None and nonzero_pixels is not None:
        print(f"Total pixels in grid: {total_pixels}")
        print(f"Non-zero pixels in TIC image: {nonzero_pixels}")
        print(f"Ratio of total/nonzero: {total_pixels/nonzero_pixels:.6f}")
    
    # Load stored spectrum
    stored_intensities = load_spatialdata_spectrum(zarr_path)
    if stored_intensities is not None:
        print(f"\nStored spectrum info:")
        print(f"  Length: {len(stored_intensities)}")
        print(f"  Total sum: {np.sum(stored_intensities):.6f}")
        print(f"  Mean value: {np.mean(stored_intensities):.6f}")
        print(f"  Max value: {np.max(stored_intensities):.6f}")
    
    # Recalculate spectrum
    recalc_intensities = recalculate_spectrum_from_spatialdata(zarr_path)
    if recalc_intensities is not None:
        print(f"\nRecalculated spectrum info:")
        print(f"  Length: {len(recalc_intensities)}")
        print(f"  Total sum: {np.sum(recalc_intensities):.6f}")
        print(f"  Mean value: {np.mean(recalc_intensities):.6f}")
        print(f"  Max value: {np.max(recalc_intensities):.6f}")
    
    # Compare ratios
    if stored_intensities is not None and recalc_intensities is not None:
        if len(stored_intensities) == len(recalc_intensities):
            ratio = np.sum(stored_intensities) / np.sum(recalc_intensities)
            print(f"\nRatio of stored/recalculated total sum: {ratio:.6f}")
        else:
            print("\nCannot compute ratio - different lengths")
    
    # Load imzML spectrum if provided
    if imzml_path:
        imzml_mzs, imzml_intensities = get_imzml_spectrum(imzml_path)
        if imzml_intensities is not None:
            print(f"\nimzML spectrum info:")
            print(f"  Length: {len(imzml_intensities)}")
            print(f"  Mass range: {imzml_mzs[0]:.4f} - {imzml_mzs[-1]:.4f}")
            print(f"  Total sum: {np.sum(imzml_intensities):.6f}")
            print(f"  Mean value: {np.mean(imzml_intensities):.6f}")
            print(f"  Max value: {np.max(imzml_intensities):.6f}")
            
            # Compare with stored and recalculated
            if stored_intensities is not None:
                ratio = np.sum(stored_intensities) / np.sum(imzml_intensities)
                print(f"\nRatio of stored/imzML total sum: {ratio:.6f}")
                print(f"Length comparison: stored {len(stored_intensities)} vs imzML {len(imzml_intensities)}")

if __name__ == "__main__":
    main()