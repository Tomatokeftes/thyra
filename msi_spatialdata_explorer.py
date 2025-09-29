#!/usr/bin/env python3
"""
MSI SpatialData Explorer

Simple exploration of Mass Spectrometry Imaging (MSI) datasets stored in SpatialData/Zarr format.
Updated for compatibility with Zarr 3 and new SpatialData format.

Features:
- Load and visualize Total Ion Current (TIC) images
- Plot precomputed average mass spectra
- Export plots
- Interactive ion image calculation

Usage:
    python msi_spatialdata_explorer.py [zarr_path]
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import zarr

warnings.filterwarnings("ignore")


def load_msi_dataset(zarr_path):
    """Load MSI dataset from zarr file with Zarr 3 compatibility."""
    if not Path(zarr_path).exists():
        print(f"Error: Dataset not found at {zarr_path}")
        return None, None, None, None

    print(f"Loading dataset: {zarr_path}")

    try:
        store = zarr.open_group(zarr_path, mode="r")
        print("Successfully opened zarr store")
    except Exception as e:
        print(f"Error opening zarr store: {e}")
        return None, None, None, None

    print("Store structure:")
    print(f"  Root keys: {list(store.keys())}")

    # Load components
    tic_data = _load_tic_image(store)
    avg_spectrum, mz_values, table_info = _load_table_data(store)

    return tic_data, avg_spectrum, mz_values, table_info


def _load_tic_image(store):
    """Load TIC image from zarr store."""
    tic_data = None
    if "images" not in store:
        return None

    print(f"  Images: {list(store['images'].keys())}")
    for img_name in store["images"].keys():
        img_group = store["images"][img_name]
        print(f"    {img_name}: {list(img_group.keys())}")

        try:
            img_data = None
            if "0" in img_group:
                img_data = img_group["0"]
            elif hasattr(img_group, "shape"):
                img_data = img_group

            if img_data is not None and hasattr(img_data, "shape"):
                if len(img_data.shape) == 3:
                    tic_data = img_data[0]  # Remove channel dimension
                else:
                    tic_data = img_data
                print(f"    TIC image loaded from {img_name}: {tic_data.shape}")
                break
        except Exception as e:
            print(f"    Error loading image {img_name}: {e}")
            continue

    return tic_data


def _load_table_data(store):
    """Load mass spectrum data and table info from zarr store."""
    avg_spectrum = None
    mz_values = None
    table_info = None

    if "tables" not in store:
        return avg_spectrum, mz_values, table_info

    print(f"  Tables: {list(store['tables'].keys())}")
    for table_name in store["tables"].keys():
        table = store["tables"][table_name]
        print(f"    {table_name}: {list(table.keys())}")

        # Load m/z values and spectrum
        mz_values = _load_mz_values(table)
        avg_spectrum = _load_average_spectrum(table)
        table_info = _load_table_info(table_name, table)

        # If we found data, break (use first table with data)
        if mz_values is not None or avg_spectrum is not None:
            break

    return avg_spectrum, mz_values, table_info


def _load_mz_values(table):
    """Load m/z values from table var section."""
    if "var" not in table:
        return None

    var_keys = list(table["var"].keys())
    print(f"      var: {var_keys}")

    mz_candidates = ["mz", "m/z", "mass", "mass_over_charge"]
    for mz_col in mz_candidates:
        if mz_col in table["var"]:
            mz_values = table["var"][mz_col][:]
            print(f"      m/z values loaded from '{mz_col}': {len(mz_values)} values")
            print(f"      m/z range: {mz_values.min():.2f} - {mz_values.max():.2f}")
            return mz_values
    return None


def _load_average_spectrum(table):
    """Load average spectrum from table uns section."""
    if "uns" not in table:
        return None

    uns_keys = list(table["uns"].keys())
    print(f"      uns: {uns_keys}")

    spectrum_candidates = ["average_spectrum", "avg_spectrum", "mean_spectrum"]
    for spec_col in spectrum_candidates:
        if spec_col in table["uns"]:
            avg_spectrum = table["uns"][spec_col][:]
            print(f"      Average spectrum loaded from '{spec_col}': {avg_spectrum.shape}")
            return avg_spectrum
    return None


def _load_table_info(table_name, table):
    """Load table info for interactive features."""
    if "X" not in table:
        return None

    table_info = {"name": table_name, "table": table, "has_X": True}
    x_group = table["X"]
    if hasattr(x_group, "keys"):
        x_keys = list(x_group.keys())
        print(f"      X matrix keys: {x_keys}")
        if all(key in x_group for key in ["data", "indices", "indptr"]):
            n_pixels = x_group["indptr"].shape[0] - 1
            print(f"      Sparse matrix: {n_pixels} pixels")
    else:
        print(f"      X matrix shape: {x_group.shape}")

    return table_info


def plot_tic(tic_data, save_path=None):
    """Plot Total Ion Current image."""
    if tic_data is None:
        print("No TIC data available")
        return None

    plt.figure(figsize=(10, 8))

    im = plt.imshow(tic_data, cmap="viridis", aspect="auto", origin="lower")
    plt.title("Total Ion Current (TIC)", fontsize=14, fontweight="bold")
    plt.xlabel("X coordinate (pixels)")
    plt.ylabel("Y coordinate (pixels)")

    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label("TIC Intensity", rotation=270, labelpad=20)

    # Add basic statistics
    tic_flat = tic_data.flatten()
    tic_nonzero = tic_flat[tic_flat > 0]

    stats_text = f"""Image size: {tic_data.shape[1]} x {tic_data.shape[0]} pixels
Non-zero pixels: {len(tic_nonzero):,} ({len(tic_nonzero)/len(tic_flat)*100:.1f}%)
TIC range: {tic_flat.min():.2e} - {tic_flat.max():.2e}"""

    if len(tic_nonzero) > 0:
        stats_text += f"\nMean TIC: {tic_nonzero.mean():.2e}"

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"TIC plot saved: {save_path}")

    plt.show()
    return plt.gcf()


def plot_mass_spectrum(avg_spectrum, mz_values, save_path=None):
    """Plot average mass spectrum."""
    if avg_spectrum is None or mz_values is None:
        print("No mass spectrum data available")
        return None

    plt.figure(figsize=(12, 6))

    plt.plot(mz_values, avg_spectrum, linewidth=0.7, color="red")
    plt.title("Average Mass Spectrum", fontsize=14, fontweight="bold")
    plt.xlabel("m/z")
    plt.ylabel("Average Intensity")
    plt.grid(True, alpha=0.3)

    # Add basic statistics
    nonzero_spectrum = avg_spectrum[avg_spectrum > 0]

    stats_text = f"""m/z range: {mz_values.min():.2f} - {mz_values.max():.2f}
Total points: {len(avg_spectrum):,}
Non-zero points: {len(nonzero_spectrum):,}
Max intensity: {avg_spectrum.max():.2e}"""

    if len(nonzero_spectrum) > 0:
        stats_text += f"\nMean intensity (non-zero): {nonzero_spectrum.mean():.2e}"

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Spectrum plot saved: {save_path}")

    plt.show()
    return plt.gcf()


def plot_combined_view(tic_data, avg_spectrum, mz_values, save_path=None):
    """Plot TIC and spectrum side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # TIC plot
    if tic_data is not None:
        im = ax1.imshow(tic_data, cmap="viridis", aspect="auto", origin="lower")
        ax1.set_title("Total Ion Current (TIC)", fontsize=14)
        ax1.set_xlabel("X coordinate (pixels)")
        ax1.set_ylabel("Y coordinate (pixels)")
        plt.colorbar(im, ax=ax1, shrink=0.8)
    else:
        ax1.text(
            0.5,
            0.5,
            "No TIC data available",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title("TIC (Not Available)")

    # Spectrum plot
    if avg_spectrum is not None and mz_values is not None:
        ax2.plot(mz_values, avg_spectrum, linewidth=0.7, color="red")
        ax2.set_title("Average Mass Spectrum", fontsize=14)
        ax2.set_xlabel("m/z")
        ax2.set_ylabel("Average Intensity")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "No spectrum data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Spectrum (Not Available)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Combined plot saved: {save_path}")

    plt.show()
    return fig


def export_plots(tic_data, avg_spectrum, mz_values, zarr_path, output_dir="./plots"):
    """Export plots to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    dataset_name = Path(zarr_path).stem

    print(f"Exporting plots to: {output_path}")

    # Export individual plots
    if tic_data is not None:
        tic_output = output_path / f"{dataset_name}_TIC.png"
        plot_tic(tic_data, save_path=tic_output)
        plt.close()

    if avg_spectrum is not None and mz_values is not None:
        spectrum_output = output_path / f"{dataset_name}_spectrum.png"
        plot_mass_spectrum(avg_spectrum, mz_values, save_path=spectrum_output)
        plt.close()

    # Export combined
    combined_output = output_path / f"{dataset_name}_combined.png"
    plot_combined_view(tic_data, avg_spectrum, mz_values, save_path=combined_output)
    plt.close()

    print("All plots exported successfully")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Explore MSI SpatialData datasets")
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default="output/pea_spatialdata.zarr",
        help="Path to SpatialData zarr file",
    )
    parser.add_argument("--export", action="store_true", help="Export plots to files")
    parser.add_argument(
        "--output-dir", default="./plots", help="Output directory for exported plots"
    )

    args = parser.parse_args()

    print("MSI SpatialData Explorer")
    print("=" * 40)

    # Load dataset
    tic_data, avg_spectrum, mz_values, table_info = load_msi_dataset(args.zarr_path)

    if tic_data is None and avg_spectrum is None:
        print("No data could be loaded from the dataset.")
        print("Please check the zarr file format and structure.")
        return 1

    print("\nDataset Summary:")
    if tic_data is not None:
        print(f"[+] TIC image: {tic_data.shape}")
    else:
        print("[-] No TIC image found")

    if avg_spectrum is not None and mz_values is not None:
        print(f"[+] Average spectrum: {len(avg_spectrum)} points")
        print(f"[+] m/z range: {mz_values.min():.2f} - {mz_values.max():.2f}")
    else:
        print("[-] No spectrum data found")

    if table_info and table_info["has_X"]:
        print("[+] Intensity matrix available for interactive features")
    else:
        print("[-] No intensity matrix found")

    print("\nDisplaying plots...")

    # Display plots
    if tic_data is not None:
        plot_tic(tic_data)

    if avg_spectrum is not None and mz_values is not None:
        plot_mass_spectrum(avg_spectrum, mz_values)

    plot_combined_view(tic_data, avg_spectrum, mz_values)

    # Export if requested
    if args.export:
        export_plots(tic_data, avg_spectrum, mz_values, args.zarr_path, args.output_dir)

    print("\nExploration complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
