# import matplotlib.pyplot as plt
# import numpy as np
# from pyimzml.ImzMLParser import ImzMLParser

# def extract_imzml_data(imzml_path):
#     """Extract dimensions, m/z values, and intensity information from an imzML file."""
#     with ImzMLParser(str(imzml_path)) as parser:
#         # Get pixel dimensions from the metadata
#         x_dim = parser.imzmldict['max count of pixels x']
#         y_dim = parser.imzmldict['max count of pixels y']
#         mz_length = parser.mzLengths[0]  # Assumes uniform m/z length in continuous mode
        
#         # Retrieve m/z values and initialize intensity sum
#         mz_values = parser.getspectrum(0)[0]  # Extract m/z values from the first pixel
#         total_intensity_sum = np.zeros((y_dim, x_dim))
#         pixel_count = 0
        
#         # Initialize an array to accumulate intensities for the average spectrum
#         intensity_sums = np.zeros(mz_length)

#         # Sum all intensity values across all pixels
#         for x, y, _ in parser.coordinates:
#             mzs, intensities = parser.getspectrum(pixel_count)
            
#             # Check that m/z values are consistent
#             if not np.allclose(mz_values, mzs):
#                 raise ValueError("m/z values are inconsistent across pixels.")
            
#             # Accumulate intensities for each m/z channel
#             intensity_sums += intensities
            
#             # Accumulate intensities for the total ion image (TIC)
#             total_intensity_sum[y-1, x-1] = np.sum(intensities)
#             pixel_count += 1

#         # Calculate the average mass spectrum
#         average_spectrum = intensity_sums / pixel_count

#     return x_dim, y_dim, mz_values, average_spectrum, total_intensity_sum, pixel_count, mz_values

# def plot_average_mass_spectrum(mz_values, average_spectrum):
#     """Plot the average mass spectrum as a stem plot."""
#     plt.figure(figsize=(10, 5))
#     plt.stem(mz_values, average_spectrum, basefmt=" ")
#     plt.xlabel("m/z")
#     plt.ylabel("Average Intensity")
#     plt.title("Average Mass Spectrum")
#     plt.grid(True)
#     plt.show()

# def plot_total_ion_image(total_intensity_sum):
#     """Plot the total ion image (TIC) as a heatmap."""
#     plt.figure(figsize=(6, 6))
#     plt.imshow(total_intensity_sum, cmap="viridis", origin="lower")
#     plt.colorbar(label="Total Ion Count")
#     plt.xlabel("X")
#     plt.ylabel("Y")
#     plt.title("Total Ion Image (TIC)")
#     plt.show()

# # Load and plot data
# imzml_path = "tests/data/test_continuous.imzML"  # Update this path as needed
# x_dim, y_dim, mz_values, average_spectrum, total_intensity_sum, pixel_count, mz_values = extract_imzml_data(imzml_path)
# print(f"Dimensions: {x_dim} x {y_dim}")
# print(f"Number of m/z values: {len(mz_values)}")
# print(f"Average mass spectrum shape: {average_spectrum.shape}")
# print(f"Total ion image shape: {total_intensity_sum.shape}")
# print(f"Number of pixels: {pixel_count}")
# print(f"m/z values: {mz_values}")


# # Plot the average mass spectrum
# plot_average_mass_spectrum(mz_values, average_spectrum)

# # Plot the total ion image
# plot_total_ion_image(total_intensity_sum)


# import numpy as np
# from pyimzml.ImzMLParser import ImzMLParser
# import matplotlib.pyplot as plt

# def generate_tic_image(imzml_path):
#     """Generate a Total Ion Count (TIC) image from a processed imzML file."""
#     parser = ImzMLParser(str(imzml_path))

#     # Get dimensions from the imzML file metadata
#     x_dim = parser.imzmldict['max count of pixels x']
#     y_dim = parser.imzmldict['max count of pixels y']


#     # Initialize TIC image array
#     tic_image = np.zeros((y_dim + 1, x_dim + 1))

#     # Sum intensities for each pixel to create the TIC
#     for idx, (x, y, _) in enumerate(parser.coordinates):
#         mzs, intensities = parser.getspectrum(idx)
#         tic_image[y -1, x - 1] = np.sum(intensities)  # One-based to zero-based index adjustment

#     return tic_image

# def plot_tic_image(tic_image):
#     """Plot the TIC image using matplotlib."""
#     plt.imshow(tic_image, cmap='viridis', origin="upper")
#     plt.colorbar(label="Total Ion Count")
#     plt.xlabel("X Coordinate")
#     plt.ylabel("Y Coordinate")
#     plt.title("Total Ion Image (TIC)")
#     plt.show()

# # Usage example
# imzml_path = r"C:\Users\tvisv\Downloads\MSIConverter\test_processed.imzML"  # Replace with the path to your processed imzML file
# tic_image = generate_tic_image(imzml_path)
# plot_tic_image(tic_image)


##################################################################

import logging
from pathlib import Path
import cProfile
from msiconvert.imzml.convertor import ImzMLToZarrConvertor

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Paths to your actual imzML and ibd files
    imzml_file = Path(r"C:\Users\tvisv\OneDrive\Desktop\Taste of MSI\rsc Taste of MSI\Ingredient Classification MALDI\Downbinned\20240605_pea_pos.imzML")
    ibd_file = Path(r"C:\Users\tvisv\OneDrive\Desktop\Taste of MSI\rsc Taste of MSI\Ingredient Classification MALDI\Downbinned\20240605_pea_pos.ibd")
    output_dir = Path("pea_downbinned_rechunker.zarr")

    # Initialize converter
    converter = ImzMLToZarrConvertor(imzml_file, ibd_file)

    # Run conversion and log the outcome
    success = converter.convert(output_dir)
    if success:
        logging.info(f"Conversion completed successfully. Zarr output stored at {output_dir}")
    else:
        logging.error("Conversion failed.")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()

    # Save profile results to a file
    profile_path = "profile_rechunker.prof"
    profiler.dump_stats(profile_path)
    logging.info(f"Profiling results saved to {profile_path}")


##############################################

# import zarr
# import numpy as np
# import matplotlib.pyplot as plt  # or plotly for interactivity
# import xarray as xr

# # Load the Zarr store
# zarr_path = r"C:\Users\tvisv\Downloads\MSIConverter\pea_original_dask.zarr"  # Replace with your actual Zarr path
# # Load the Zarr data into an Xarray dataset
# ds = xr.open_zarr(zarr_path)

# # Display the structure of the dataset
# print(ds)

# total_ion_image = ds['0'].sum(dim='c')  # Summing across the m/z (channel) dimension for a TII

# # Plot the total ion image
# total_ion_image.plot()

# # Rotate 180 degrees
# plt.gca().invert_yaxis()

# plt.show()


# import xarray as xr

# # Load the Zarr file (assuming it's in a folder named 'output.zarr')
# zarr_path = r"C:\Users\tvisv\Downloads\MSIConverter\pea_downbinned_dask.zarr"
# ds = xr.open_zarr(zarr_path)

# # Print metadata from the Dataset's attributes
# print("=== Global Metadata ===")
# for key, value in ds.attrs.items():
#     print(f"{key}: {value}")

# # Print metadata for each data variable (e.g., intensity values)
# print("\n=== Data Variables ===")
# for var_name, var_data in ds.data_vars.items():
#     print(f"\nData Variable: {var_name}")
#     print(f"Dimensions: {var_data.dims}")
#     print(f"Shape: {var_data.shape}")
#     print(f"Dtype: {var_data.dtype}")
#     print("Attributes:")
#     for attr_key, attr_value in var_data.attrs.items():
#         print(f"  {attr_key}: {attr_value}")

# # Print information about dimensions
# print("\n=== Dimensions ===")
# for dim_name, dim_size in ds.dims.items():
#     print(f"{dim_name}: {dim_size}")
