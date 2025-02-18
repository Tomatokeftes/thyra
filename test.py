# from rechunk_zarr import rechunk_zarr

# # Rechunk your array
# rechunk_zarr(
#     source_path=r"C:\Users\P70078823\Desktop\Random\burger.zarr",
#     target_path=r"C:\Users\P70078823\Desktop\Random\burger3.zarr",
#     chunks=(448284, 1, 1, 1)
# )
##################################################################
# import zarr
# import dask.array as da

# # Define the path to your Zarr store
# zarr_store_path = r"C:\Users\P70078823\Desktop\MSIConverter\pea.zarr"

# # Open the Zarr store
# root = zarr.open(zarr_store_path, mode='r')

# # Access the main intensity dataset
# zarr_array = root["0"]  # Adjust if needed

# # Convert to Dask array for lazy evaluation
# dask_array = da.from_zarr(zarr_array)

# def count_low_intensity_values(dask_array, threshold: float = 0.01):
#     """
#     Counts how many nonzero values in the dataset are below a certain intensity threshold.

#     Parameters:
#     -----------
#     dask_array : dask.array.Array
#         The Dask array wrapping the Zarr dataset.
#     threshold : float, optional
#         The intensity threshold (default is 0.01).

#     Returns:
#     --------
#     tuple(int, int, float)
#         - Count of nonzero values below the threshold.
#         - Total number of nonzero values.
#         - Percentage of low-intensity values out of all nonzero values.
#     """
#     # Count all nonzero values
#     total_nonzero_count = da.sum(dask_array != 0).compute()

#     # Count how many of these are below the threshold
#     low_intensity_count = da.sum((dask_array != 0) & (dask_array < threshold)).compute()

#     # Compute percentage
#     percentage_low_intensity = (low_intensity_count / total_nonzero_count) * 100 if total_nonzero_count > 0 else 0

#     return int(low_intensity_count), int(total_nonzero_count), percentage_low_intensity

# # Set the threshold for low-intensity values
# threshold_value = 100  # Adjust as needed

# # Compute counts
# low_count, total_nonzero, low_percentage = count_low_intensity_values(dask_array, threshold=threshold_value)

# # Print results
# print(f"Total nonzero entries: {total_nonzero}")
# print(f"Nonzero entries below {threshold_value}: {low_count}")
# print(f"Percentage of low-intensity nonzero entries: {low_percentage:.6f}%")



##################################################################
# import zarr
# import numpy as np
# import dask.array as da

# # Define the path to your Zarr store
# zarr_store_path = r"C:\Users\P70078823\Desktop\MSIConverter\pea.zarr"

# # Open the Zarr store
# root = zarr.open(zarr_store_path, mode='r')

# # Access the main intensity dataset
# zarr_array = root["0"]  # Adjust if needed

# # Convert to Dask array for lazy evaluation
# dask_array = da.from_zarr(zarr_array)

# def count_approximate_value(dask_array, target_value=1.0, atol=1e-6):
#     """
#     Counts the number of occurrences of a value (handling floating-point precision) in the Zarr array.

#     Parameters:
#     -----------
#     dask_array : dask.array.Array
#         The Dask array wrapping the Zarr dataset.
#     target_value : float, optional
#         The value to count occurrences of (default is 1.0).
#     atol : float, optional
#         Absolute tolerance for floating-point comparisons (default is 1e-6).

#     Returns:
#     --------
#     tuple(int, float)
#         - The total count of entries approximately equal to `target_value`.
#         - The percentage of these entries relative to the total dataset size.
#     """
#     total_elements = dask_array.size  # Get the total number of elements
#     approx_count = da.sum(da.map_blocks(np.isclose, dask_array, target_value, atol=atol)).compute()  # Use np.isclose with Dask
#     percentage = (approx_count / total_elements) * 100  # Compute percentage

#     return int(approx_count), percentage

# # Compute the count and percentage of values approximately equal to 1.0
# count_ones, percentage_ones = count_approximate_value(dask_array, target_value=0.0, atol=1e-6)

# # Print results
# print(f"Total number of entries approximately equal to 1: {count_ones}")
# print(f"Percentage of dataset approximately equal to 1: {percentage_ones:.6f}%")

##################################################################
import logging
from pathlib import Path
import cProfile

from msiconvert.io.msi_convert import MSIToZarrConverter

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Path to your specific imzML file or Bruker .d directory
    input_path = Path(r"C:\Users\P70078823\OneDrive\Desktop\Taste of MSI\rsc\Ingredient Classification MALDI\Original\20240605_pea_pos.imzML")  # or "C:\path\to\your\dataset.d"
    output_dir = Path("pealz4BYTEshuffle.zarr")

    # Initialize converter with the input path and output path
    converter = MSIToZarrConverter(input_path, output_dir)

    # Run conversion and log the outcome
    success = converter.convert()
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


##################################################################
# import zarr
# import pandas as pd

# def export_first_chunk_mzs_to_csv(zarr_store_path: str, output_csv_path: str):
#     """
#     Export the m/z values from the first chunk of the mzs array in a Zarr file store to a CSV file.
    
#     Parameters:
#     -----------
#     zarr_store_path : str
#         Path to the Zarr file store.
#     output_csv_path : str
#         Path to save the output CSV file.
#     """
#     # Open the Zarr file store
#     zarr_store = zarr.open(zarr_store_path, mode='r')
    
#     # Navigate to the mzs array
#     try:
#         mzs_array = zarr_store['labels']['mzs']['0']  # Adjusted path based on your structure
#     except KeyError:
#         raise KeyError("The 'mzs' array was not found in the expected location within the Zarr store.")
    
#     # Extract the first chunk size along the m/z axis
#     mz_chunk_size = mzs_array.chunks[0]  # Size of the first chunk along the m/z axis

#     # Extract the first chunk and flatten spatial dimensions
#     first_chunk = mzs_array[:mz_chunk_size, 0, :17, :17].reshape(mz_chunk_size, -1)

#     # Create a DataFrame with each pixel as a row and m/z values as columns
#     df = pd.DataFrame(first_chunk.T)  # Transpose to have pixels as rows

#     # Save the DataFrame to a CSV file
#     df.to_csv(output_csv_path, index=False)

# # Define the paths
# zarr_store_path = "pea_continuous.zarr"
# output_csv_path = "first_chunk_mzs.csv"

# # Export the first chunk to CSV
# export_first_chunk_mzs_to_csv(zarr_store_path, output_csv_path)

# # Provide the path to the saved CSV file
# output_csv_path


##############################################

# import zarr
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm

# def plot_total_mass_spectrum_and_total_ion_image(zarr_store_path: str):
#     """
#     Plot the total mass spectrum and total ion image from the processed MSI data stored in Zarr.
    
#     Parameters:
#     -----------
#     zarr_store_path : str
#         Path to the Zarr file store containing the processed MSI data.
#     """
#     # Open the Zarr file store
#     zarr_store = zarr.open(zarr_store_path, mode='r')

#     # Access the shape of the intensity array and initialize variables
#     intensities = zarr_store['0']  # 4D array: (c, z, y, x)
#     y_size, x_size = intensities.shape[2], intensities.shape[3]
    
#     # Check the shape of the m/z array
#     mzs_array = zarr_store['labels']['mzs']['0']
#     print("Shape of mzs array:", mzs_array.shape)

#     # Initialize variables for constructing a common mass axis
#     all_mzs = []

#     # Collect all unique m/z values from each pixel, ensuring bounds are respected
#     for y in range(y_size):
#         for x in range(x_size):
#             if y < mzs_array.shape[2] and x < mzs_array.shape[3]:  # Ensure indices are within bounds
#                 mzs = mzs_array[:, 0, y, x]  # Retrieve m/z values for the pixel
#                 all_mzs.extend(mzs)

#     # Create a sorted, unique, and common mass axis
#     common_mz_axis = np.unique(np.sort(all_mzs))

#     # Initialize an array for the total mass spectrum
#     total_mass_spectrum = np.zeros_like(common_mz_axis, dtype=np.float64)

#     # Iterate through each pixel to align and sum intensities using searchsorted
#     for y in tqdm(range(y_size), desc="Processing pixels"):
#         for x in range(x_size):
#             if y < mzs_array.shape[2] and x < mzs_array.shape[3]:  # Ensure indices are within bounds
#                 mzs = mzs_array[:, 0, y, x]  # m/z values for this pixel
#                 intensity_array = intensities[:, 0, y, x]  # Intensities for this pixel

#                 # Align intensities to the common mass axis using searchsorted
#                 indices = np.searchsorted(common_mz_axis, mzs)
#                 np.add.at(total_mass_spectrum, indices, intensity_array)

#     # Plot the total mass spectrum
#     plt.figure(figsize=(10, 5))
#     plt.plot(common_mz_axis, total_mass_spectrum, color='blue')
#     plt.xlabel('m/z')
#     plt.ylabel('Total Intensity')
#     plt.title('Total Mass Spectrum')
#     plt.grid(True)
#     plt.show()

#     # Calculate the total ion image by summing over the m/z axis
#     total_ion_image = np.sum(intensities, axis=0).reshape(y_size, x_size)  # Sum over m/z axis
#     # Plot the total ion image
#     plt.figure(figsize=(8, 8))
#     plt.imshow(total_ion_image, cmap='viridis', origin='upper')
#     plt.colorbar(label='Total Intensity')
#     plt.title('Total Ion Image')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.show()

# # Example usage
# zarr_store_path = 'output.zarr'
# plot_total_mass_spectrum_and_total_ion_image(zarr_store_path)



########################################

# import zarr
# import numpy as np
# import matplotlib.pyplot as plt
# import xarray as xr

# # Load the Zarr store
# zarr_path = r"pea.zarr"  # Replace with your actual Zarr path
# ds = xr.open_zarr(zarr_path)

# # Compute the total ion current (TIC) image by summing across the m/z (c) dimension
# tic_image = ds['0'].sum(dim='c')

# # Replace zero or null values with NaN for visualization purposes
# tic_image = tic_image.where(tic_image > 0, np.nan)

# # Plot the TIC image with adjusted color limits
# plt.figure(figsize=(8, 6))
# tic_image.plot(cmap='viridis', vmin=np.nanpercentile(tic_image, 1), vmax=np.nanpercentile(tic_image, 99))  # Set color limits to 1st and 99th percentile

# # Rotate 180 degrees if needed
# plt.gca().invert_yaxis()

# plt.title("Median-Normalized Total Ion Image (TIC) with Improved Color Scaling")
# plt.show()


################################################

# import zarr
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the Zarr store
# zarr_path = r"C:\Users\tvisv\Downloads\MSIConverter\20240826_xenium_0041899.zarr"
# zarr_store = zarr.open_group(zarr_path, mode='r')

# # Access the intensity, m/z, and lengths data
# intensities = zarr_store['0']  # (c, z, y, x)
# mzs = zarr_store['labels/mzs/0']  # (c, z, y, x)
# lengths = zarr_store['labels/lengths/0']  # (1, 1, y, x)

# # Initialize a dictionary to accumulate intensities for each m/z value
# mass_spectrum_accumulator = {}

# # Iterate over the y and x coordinates to access each pixel's m/z and intensity data
# for y in range(intensities.shape[2]):  # Loop over the y dimension
#     for x in range(intensities.shape[3]):  # Loop over the x dimension
#         length = lengths[0, 0, y, x]  # Get the length for the current pixel
#         mz_values = mzs[:length, 0, y, x]  # Get m/z values for the current pixel up to `length`
#         intensity_values = intensities[:length, 0, y, x]  # Get intensity values for the current pixel up to `length`

#         # Accumulate intensities in the dictionary
#         for mz, intensity in zip(mz_values, intensity_values):
#             if mz not in mass_spectrum_accumulator:
#                 mass_spectrum_accumulator[mz] = 0
#             mass_spectrum_accumulator[mz] += intensity

# # Convert the accumulator dictionary to sorted arrays for plotting
# sorted_mz = np.array(sorted(mass_spectrum_accumulator.keys()))
# sorted_intensity = np.array([mass_spectrum_accumulator[mz] for mz in sorted_mz])

# # Plot the total mass spectrum using a stem plot
# plt.figure(figsize=(14, 6))
# plt.stem(sorted_mz, sorted_intensity, linefmt='b-', markerfmt='bo', basefmt='r-')
# plt.xlabel('m/z')
# plt.ylabel('Total Intensity')
# plt.title('Total Mass Spectrum (Processed Data)')
# plt.grid(True)

# plt.show()



###############################################

# from dask.distributed import Client
# import dask.array as da
# import zarr
# from numcodecs import Blosc

# if __name__ == "__main__":
#     # Start the Dask client to create a dashboard for monitoring
#     client = Client()
#     print("Dask dashboard is running. Visit the following URL: ", client.dashboard_link)

#     # Open the root of the Zarr store in append mode
#     zarr_store = zarr.open_group(r'C:\Users\tvisv\Downloads\MSIConverter\20240826_xenium_0041899.zarr', mode='a')

#     # Define a compressor (e.g., Blosc with specific settings)
#     compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)

#     # Rechunk and write the intensity data in the '0' folder with compression
#     intensity_array = da.from_array(zarr_store['0'], chunks=zarr_store['0'].chunks)
#     rechunked_intensity = intensity_array.rechunk((1000, 1, 256, 256))  # Adjust chunk size as needed
#     da.to_zarr(rechunked_intensity, r'C:\Users\tvisv\Downloads\MSIConverter\20240826_xenium_0041899.zarr/2', compressor=compressor)

#     # Rechunk and write the m/z data in the 'labels/mzs' folder with compression
#     mzs_array = da.from_array(zarr_store['labels/mzs/0'], chunks=zarr_store['labels/mzs/0'].chunks)
#     rechunked_mzs = mzs_array.rechunk((1000, 1, 256, 256))  # Adjust chunk size as needed
#     da.to_zarr(rechunked_mzs, r'C:\Users\tvisv\Downloads\MSIConverter\20240826_xenium_0041899.zarr/labels/mzs3', compressor=compressor)

#     # Rechunk and write the m/z data in the 'labels/mzs' folder with compression
#     lengths_array = da.from_array(zarr_store['labels/lengths/0'], chunks=zarr_store['labels/lengths/0'].chunks)
#     rechunked_mzs = lengths_array.rechunk((1, 1, 256, 256))  # Adjust chunk size as needed
#     da.to_zarr(rechunked_mzs, r'C:\Users\tvisv\Downloads\MSIConverter\20240826_xenium_0041899.zarr/labels/lengths2', compressor=compressor)



####################

# import os
# import csv

# def save_folder_file_sizes(folder_path, output_csv_path):
#     # Collect file sizes
#     file_sizes = []
#     for root, _, files in os.walk(folder_path):
#         for file in files:
#             file_path = os.path.join(root, file)
#             file_size = os.path.getsize(file_path)
#             file_sizes.append((file_path, file_size))
    
#     # Save to CSV
#     with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(['File Path', 'File Size (Bytes)'])
#         for file_path, file_size in file_sizes:
#             writer.writerow([file_path, file_size])
    
#     print(f"File sizes saved to {output_csv_path}")

# # Specify the folder path and output CSV path
# folder_path = r"20240826_xenium_0041899.zarr/1"  # Replace with the folder you want to scan
# output_csv_path = 'filesizes_intensities2.csv'  # Replace with your desired CSV output path

# # Run the function
# save_folder_file_sizes(folder_path, output_csv_path)

##############################################

# import zarr
# import dask.array as da

# # Load the Zarr store
# zarr_path = "20240826_xenium_0041899.zarr"  # Replace with your Zarr store path
# zarr_group = zarr.open_group(zarr_path, mode='r')

# import dask.array as da
# import numpy as np

# # Load the m/z values and intensity data lazily
# mzs = da.from_zarr(zarr_group['labels/mzs/0'])
# intensities = da.from_zarr(zarr_group['0'])

# # Define the m/z range
# mz_min, mz_max = 200, 250

# # Function to process each chunk
# def process_chunk(mzs_chunk, intensities_chunk):
#     # Create a mask for the m/z range within this chunk
#     mask = (mzs_chunk >= mz_min) & (mzs_chunk <= mz_max)
    
#     # Apply the mask to filter m/z values and corresponding intensities
#     mzs_filtered = mzs_chunk[mask]
#     intensities_filtered = intensities_chunk[mask].sum(axis=(1, 2, 3))
    
#     # Ensure the output arrays have consistent shapes
#     if mzs_filtered.size == 0:
#         mzs_filtered = np.array([], dtype=mzs_chunk.dtype)
#         intensities_filtered = np.array([], dtype=intensities_chunk.dtype)
    
#     return mzs_filtered, intensities_filtered

# # Use Dask's map_blocks to process data in chunks
# mzs_filtered, intensities_filtered = da.map_blocks(
#     process_chunk, mzs, intensities, 
#     dtype=(mzs.dtype, intensities.dtype),
#     chunks=(mzs.chunks[0],)  # Specify chunks for the output
# )

# # Compute the results
# mzs_filtered_computed = mzs_filtered.compute()
# intensities_filtered_computed = intensities_filtered.compute()

# import plotly.graph_objects as go

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=mzs_filtered_computed, y=intensities_filtered_computed, mode='lines'))
# fig.update_layout(title='Total Mass Spectrum (200-250 m/z)', xaxis_title='m/z', yaxis_title='Intensity')
# fig.show()

################################################################

# import zarr
# import numpy as np
# import matplotlib.pyplot as plt

# def plot_mass_spectrum_of_pixel(zarr_store_path, y, x):
#     """
#     Plot the mass spectrum of the pixel at the specified (y, x) coordinates from the processed Zarr store.

#     Parameters:
#     -----------
#     zarr_store_path : str
#         Path to the processed Zarr store.
#     y : int
#         The y-coordinate of the pixel.
#     x : int
#         The x-coordinate of the pixel.
#     """
#     # Open the Zarr store
#     zarr_store = zarr.open(zarr_store_path, mode='r')
    
#     # Access the Zarr arrays
#     mzs_array = zarr_store['labels']['mzs']['0']
#     intensities_array = zarr_store['0']
#     lengths_array = zarr_store['labels/lengths']['0']
    
#     # Get the length of the m/z values for the selected pixel
#     length = lengths_array[0, 0, y, x]
    
#     # Extract the m/z values and intensities for the selected pixel
#     mz_values = mzs_array[:length, 0, y, x]
#     intensity_values = intensities_array[:length, 0, y, x]
    
#     # Plot the mass spectrum as a stemplot
#     plt.figure(figsize=(10, 5))
#     plt.stem(mz_values, intensity_values, linefmt='b-', markerfmt='bo', basefmt='r-')
#     plt.xlabel('m/z')
#     plt.ylabel('Intensity')
#     plt.title(f'Mass Spectrum of Pixel ({y}, {x})')
#     plt.grid(True)
#     plt.show()

# # Example usage
# zarr_store_path = '20240826_xenium_0041899.zarr'
# y, x = 100, 50  # The coordinates of the pixel to plot
# plot_mass_spectrum_of_pixel(zarr_store_path, y, x)
