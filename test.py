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

# import logging
# from pathlib import Path
# import cProfile
# from msiconvert.imzml.convertor import ImzMLToZarrConvertor

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# def main():
#     # Paths to your actual imzML and ibd files
#     imzml_file = Path(r"C:\Users\tvisv\OneDrive\Desktop\Taste of MSI\rsc Taste of MSI\Ingredient Classification MALDI\Original\20240605_pea_pos.imzML")
#     ibd_file = Path(r"C:\Users\tvisv\OneDrive\Desktop\Taste of MSI\rsc Taste of MSI\Ingredient Classification MALDI\Original\20240605_pea_pos.ibd")
#     output_dir = Path("pea_processed_new.zarr")

#     # Initialize converter
#     converter = ImzMLToZarrConvertor(imzml_file, ibd_file)

#     # Run conversion and log the outcome
#     success = converter.convert(output_dir)
#     if success:
#         logging.info(f"Conversion completed successfully. Zarr output stored at {output_dir}")
#     else:
#         logging.error("Conversion failed.")

# if __name__ == "__main__":
#     profiler = cProfile.Profile()
#     profiler.enable()
#     main()
#     profiler.disable()

#     # Save profile results to a file
#     profile_path = "profile_rechunker.prof"
#     profiler.dump_stats(profile_path)
#     logging.info(f"Profiling results saved to {profile_path}")

##################################################################

# import zarr
# import numpy as np

# def check_mzs_chunks_for_duplicates(zarr_store_path: str):
#     """
#     Check the chunks of the mzs array in a Zarr file store for duplicates.
    
#     Parameters:
#     -----------
#     zarr_store_path : str
#         Path to the Zarr file store.
        
#     Returns:
#     --------
#     bool
#         True if duplicates are found across chunks, False otherwise.
#     """
#     # Open the Zarr file store
#     zarr_store = zarr.open(zarr_store_path, mode='r')
    
#     # Navigate to the mzs array
#     try:
#         mzs_array = zarr_store['labels']['mzs']['0']  # Adjusted path based on your structure
#     except KeyError:
#         raise KeyError("The 'mzs' array was not found in the expected location within the Zarr store.")
    
#     # Retrieve shape and chunk sizes
#     shape = mzs_array.shape
#     chunk_sizes = mzs_array.chunks

#     print(f"mzs array shape: {shape}")
#     print(f"Chunk sizes: {chunk_sizes}")

#     # Generate slices for each chunk along each dimension
#     def get_chunk_slices(shape, chunk_sizes):
#         """Generate slices for each chunk along each dimension."""
#         slices = []
#         for dim_size, chunk_size in zip(shape, chunk_sizes):
#             dim_slices = [slice(i, min(i + chunk_size, dim_size)) for i in range(0, dim_size, chunk_size)]
#             slices.append(dim_slices)
#         return slices

#     # Generate all combinations of chunk slices
#     chunk_slices = get_chunk_slices(shape, chunk_sizes)
#     all_chunk_combinations = np.array(np.meshgrid(*chunk_slices, indexing='ij')).T.reshape(-1, len(shape))

#     # Collect all m/z values from each chunk
#     chunk_contents = []
#     for chunk_slice in all_chunk_combinations:
#         chunk = mzs_array.oindex[tuple(chunk_slice)]
#         chunk_contents.append(chunk.flatten())

#     # Check for duplicates
#     all_mzs = np.concatenate(chunk_contents)
#     unique_mzs = np.unique(all_mzs)
    
#     if len(unique_mzs) < len(all_mzs):
#         print("Duplicates found in the mzs chunks.")
#         return True
#     else:
#         print("No duplicates found in the mzs chunks.")
#         return False

# # Example usage
# zarr_store_path = "pea_continuous.zarr"
# duplicates_found = check_mzs_chunks_for_duplicates(zarr_store_path)
# print(f"Duplicates found: {duplicates_found}")

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

# def plot_total_mass_spectrum_and_total_ion_image(zarr_store_path: str):
#     """
#     Plot the total mass spectrum and total ion image from the processed MSI data.
    
#     Parameters:
#     -----------
#     zarr_store_path : str
#         Path to the Zarr file store containing the processed MSI data.
#     """
#     # Open the Zarr file store
#     zarr_store = zarr.open(zarr_store_path, mode='r')
    
#     # Access the m/z values and intensity data
#     mzs = zarr_store['labels']['mzs']['0'][:, 0, 0, 0]  # 1D array of m/z values
#     intensities = zarr_store['0']  # 4D array: (c, 1, y, x)
    
#     # Calculate the total mass spectrum
#     total_mass_spectrum = np.sum(intensities, axis=(1, 2, 3))  # Sum over all pixels
#     # Plot the total mass spectrum
#     plt.figure(figsize=(10, 5))
#     plt.plot(mzs, total_mass_spectrum, color='blue')
#     plt.xlabel('m/z')
#     plt.ylabel('Total Intensity')
#     plt.title('Total Mass Spectrum')
#     plt.grid(True)
#     plt.show()
    
#     # Calculate the total ion image
#     total_ion_image = np.sum(intensities, axis=0).reshape(intensities.shape[2], intensities.shape[3])  # Sum over m/z axis
#     # Plot the total ion image
#     plt.figure(figsize=(8, 8))
#     plt.imshow(total_ion_image, cmap='viridis', origin='lower')
#     plt.colorbar(label='Total Intensity')
#     plt.title('Total Ion Image')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.show()

# # Example usage
# zarr_store_path = 'pea_processed_new.zarr'
# plot_total_mass_spectrum_and_total_ion_image(zarr_store_path)



########################################

# import zarr
# import numpy as np
# import matplotlib.pyplot as plt
# import xarray as xr

# # Load the Zarr store
# zarr_path = r"C:\Users\tvisv\Downloads\MSIConverter\20240826_xenium_0041899.zarr"  # Replace with your actual Zarr path
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

import zarr
import numpy as np
import matplotlib.pyplot as plt

def plot_mass_spectrum_of_pixel(zarr_store_path, y, x):
    """
    Plot the mass spectrum of the pixel at the specified (y, x) coordinates from the processed Zarr store.

    Parameters:
    -----------
    zarr_store_path : str
        Path to the processed Zarr store.
    y : int
        The y-coordinate of the pixel.
    x : int
        The x-coordinate of the pixel.
    """
    # Open the Zarr store
    zarr_store = zarr.open(zarr_store_path, mode='r')
    
    # Access the Zarr arrays
    mzs_array = zarr_store['labels']['mzs']['0']
    intensities_array = zarr_store['0']
    lengths_array = zarr_store['labels/lengths']['0']
    
    # Get the length of the m/z values for the selected pixel
    length = lengths_array[0, 0, y, x]
    
    # Extract the m/z values and intensities for the selected pixel
    mz_values = mzs_array[:length, 0, y, x]
    intensity_values = intensities_array[:length, 0, y, x]
    
    # Plot the mass spectrum as a stemplot
    plt.figure(figsize=(10, 5))
    plt.stem(mz_values, intensity_values, linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.xlabel('m/z')
    plt.ylabel('Intensity')
    plt.title(f'Mass Spectrum of Pixel ({y}, {x})')
    plt.grid(True)
    plt.show()

# Example usage
zarr_store_path = '20240826_xenium_0041899.zarr'
y, x = 100, 50  # The coordinates of the pixel to plot
plot_mass_spectrum_of_pixel(zarr_store_path, y, x)

