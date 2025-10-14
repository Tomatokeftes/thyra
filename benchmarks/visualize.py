"""
Visualization module for benchmark results.

Creates publication-quality plots for all benchmark metrics.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from config import BenchmarkConfig
from utils import load_metrics


class BenchmarkVisualizer:
    """Create publication-quality visualizations of benchmark results."""

    def __init__(self):
        BenchmarkConfig.setup_directories()
        self.colors = BenchmarkConfig.PLOT_STYLE
        self.dpi = BenchmarkConfig.PLOT_DPI

    def plot_storage_comparison(self, results: List[Dict[str, Any]], output_path: Path):
        """Plot storage efficiency comparison."""
        if not results:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: File sizes
        datasets = [r["dataset"] for r in results]
        input_sizes = [r["input_size_mb"] for r in results]
        output_sizes = [r["output_size_mb"] for r in results]

        x = np.arange(len(datasets))
        width = 0.35

        ax1.bar(
            x - width / 2,
            input_sizes,
            width,
            label="Original Format",
            color=self.colors["bruker"],
            alpha=0.8,
        )
        ax1.bar(
            x + width / 2,
            output_sizes,
            width,
            label="SpatialData/Zarr",
            color=self.colors["spatialdata"],
            alpha=0.8,
        )

        ax1.set_xlabel("Dataset", fontsize=12, weight="bold")
        ax1.set_ylabel("Size (MB)", fontsize=12, weight="bold")
        ax1.set_title("Storage Comparison", fontsize=14, weight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, axis="y", alpha=0.3)

        # Plot 2: Compression ratios
        compression_ratios = [r["compression_ratio"] for r in results]
        colors = [self.colors["speedup"]] * len(datasets)

        bars = ax2.bar(datasets, compression_ratios, color=colors, alpha=0.8)
        ax2.set_xlabel("Dataset", fontsize=12, weight="bold")
        ax2.set_ylabel("Compression Ratio", fontsize=12, weight="bold")
        ax2.set_title("Compression Efficiency", fontsize=14, weight="bold")
        ax2.set_xticklabels(datasets, rotation=45, ha="right")
        ax2.grid(True, axis="y", alpha=0.3)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}x",
                ha="center",
                va="bottom",
                fontsize=10,
                weight="bold",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()

    def plot_spatial_access_comparison(
        self, results: List[Dict[str, Any]], output_path: Path
    ):
        """Plot spatial access pattern comparison."""
        if not results:
            return

        n_datasets = len(results)
        fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6))

        if n_datasets == 1:
            axes = [axes]

        for idx, result in enumerate(results):
            ax = axes[idx]

            # Get format types
            orig_format = "imzml" if "imzml" in result else "bruker"
            orig_data = result.get(orig_format, {})
            sd_data = result.get("spatialdata", {})

            if not orig_data or not sd_data:
                continue

            # Extract times
            operations = ["Sequential", "ROI", "Random"]
            orig_times = [
                orig_data.get("sequential_access_sec", 0),
                orig_data.get("roi_access_sec", 0),
                orig_data.get("random_access_sec", 0),
            ]
            sd_times = [
                sd_data.get("sequential_access_sec", 0),
                sd_data.get("roi_access_sec", 0),
                sd_data.get("random_access_sec", 0),
            ]

            x = np.arange(len(operations))
            width = 0.35

            ax.bar(
                x - width / 2,
                orig_times,
                width,
                label=f"{orig_format.upper()}",
                color=self.colors[orig_format],
                alpha=0.8,
            )
            ax.bar(
                x + width / 2,
                sd_times,
                width,
                label="SpatialData",
                color=self.colors["spatialdata"],
                alpha=0.8,
            )

            ax.set_xlabel("Access Pattern", fontsize=12, weight="bold")
            ax.set_ylabel("Time (seconds)", fontsize=12, weight="bold")
            ax.set_title(
                f'{result["dataset"]}\nSpatial Access Performance',
                fontsize=13,
                weight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(operations)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)

            # Add speedup labels
            if "speedups" in result:
                speedups = result["speedups"]
                keys = ["sequential_access_sec", "roi_access_sec", "random_access_sec"]
                for i, key in enumerate(keys):
                    if key in speedups:
                        speedup = speedups[key]
                        y_pos = max(orig_times[i], sd_times[i]) * 1.05
                        ax.text(
                            i,
                            y_pos,
                            f"{speedup:.1f}x",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            weight="bold",
                            color=self.colors["speedup"],
                        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()

    def plot_spectral_access_comparison(
        self, results: List[Dict[str, Any]], output_path: Path
    ):
        """Plot spectral (m/z) access pattern comparison."""
        if not results:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for result in results:
            dataset_name = result["dataset"]
            orig_format = "imzml" if "imzml" in result else "bruker"

            # Plot 1: M/z range query speedups
            if "mz_range_speedups" in result:
                speedups = result["mz_range_speedups"]
                mz_ranges = list(speedups.keys())
                speedup_values = list(speedups.values())

                axes[0].bar(
                    mz_ranges,
                    speedup_values,
                    color=self.colors["speedup"],
                    alpha=0.8,
                    label=dataset_name,
                )

            # Plot 2: Ion image extraction speedup
            if "ion_image_speedup" in result:
                speedup = result["ion_image_speedup"]
                axes[1].bar(
                    dataset_name, speedup, color=self.colors["speedup"], alpha=0.8
                )

        axes[0].set_xlabel("M/z Range (Da)", fontsize=12, weight="bold")
        axes[0].set_ylabel("Speedup Factor", fontsize=12, weight="bold")
        axes[0].set_title("M/z Range Query Performance", fontsize=14, weight="bold")
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].grid(True, axis="y", alpha=0.3)
        axes[0].legend()

        axes[1].set_ylabel("Speedup Factor", fontsize=12, weight="bold")
        axes[1].set_title(
            "Ion Image Extraction Performance", fontsize=14, weight="bold"
        )
        axes[1].grid(True, axis="y", alpha=0.3)

        # Add value labels
        for ax in axes:
            for container in ax.containers:
                ax.bar_label(container, fmt="%.1fx", weight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()

    def plot_parallel_scalability(
        self, results: List[Dict[str, Any]], output_path: Path
    ):
        """Plot parallel processing scalability."""
        if not results:
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for result in results:
            dataset_name = result["dataset"]

            # Ion images
            if "ion_images_speedups" in result:
                data = result["ion_images_speedups"]
                workers = [d["n_workers"] for d in data]
                speedups = [d["speedup"] for d in data]
                axes[0].plot(
                    workers,
                    speedups,
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    label=dataset_name,
                )

            # Normalization
            if "normalization_speedups" in result:
                data = result["normalization_speedups"]
                workers = [d["n_workers"] for d in data]
                speedups = [d["speedup"] for d in data]
                axes[1].plot(
                    workers,
                    speedups,
                    marker="s",
                    linewidth=2,
                    markersize=8,
                    label=dataset_name,
                )

            # M/z slicing
            if "mz_slicing_speedups" in result:
                data = result["mz_slicing_speedups"]
                workers = [d["n_workers"] for d in data]
                speedups = [d["speedup"] for d in data]
                axes[2].plot(
                    workers,
                    speedups,
                    marker="^",
                    linewidth=2,
                    markersize=8,
                    label=dataset_name,
                )

        # Format all subplots
        titles = ["Ion Image Extraction", "TIC Normalization", "M/z Range Extraction"]
        for ax, title in zip(axes, titles):
            ax.set_xlabel("Number of Workers", fontsize=12, weight="bold")
            ax.set_ylabel("Speedup Factor", fontsize=12, weight="bold")
            ax.set_title(title, fontsize=13, weight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Add ideal scaling line
            max_workers = max(BenchmarkConfig.WORKER_COUNTS)
            ax.plot(
                [1, max_workers],
                [1, max_workers],
                "--",
                color="gray",
                alpha=0.5,
                label="Ideal",
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()

    def plot_bruker_interpolation(self, result: Dict[str, Any], output_path: Path):
        """Plot Bruker format comparison (3-way: original .d, raw zarr, resampled zarr)."""
        if not result:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        storage = result.get("storage", {})
        access = result.get("access_patterns", {})

        # Plot 1: Storage comparison (3 formats)
        if storage:
            labels = []
            sizes = []
            colors_list = []

            # Bruker original
            if "bruker_original_mb" in storage:
                labels.append("Bruker .d\n(SQLite)")
                sizes.append(storage["bruker_original_mb"])
                colors_list.append(self.colors["bruker"])

            # Zarr raw
            if "zarr_raw_mb" in storage:
                raw_bins = storage.get("raw_mz_stats", {}).get("n_mz_bins", 0)
                labels.append(f"Zarr Raw\n({raw_bins:,} bins)")
                sizes.append(storage["zarr_raw_mb"])
                colors_list.append("#8B4789")

            # Zarr resampled
            if "zarr_resampled_mb" in storage:
                res_bins = storage.get("resampled_mz_stats", {}).get("n_mz_bins", 0)
                labels.append(f"Zarr\n({res_bins:,} bins)")
                sizes.append(storage["zarr_resampled_mb"])
                colors_list.append(self.colors["spatialdata"])

            bars = ax1.bar(range(len(labels)), sizes, color=colors_list, alpha=0.8)

            ax1.set_ylabel("Size (MB)", fontsize=12, weight="bold")
            ax1.set_title(
                "Storage Comparison:\nBruker Formats", fontsize=14, weight="bold"
            )
            ax1.set_xticks(range(len(labels)))
            ax1.set_xticklabels(labels, fontsize=10)
            ax1.grid(True, axis="y", alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.0f} MB",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    weight="bold",
                )

            # Add compression annotations
            if len(sizes) >= 2:
                # Bruker to Zarr raw
                if "compression_bruker_to_zarr_raw" in storage:
                    ratio = storage["compression_bruker_to_zarr_raw"]
                    ax1.annotate(
                        "",
                        xy=(1, sizes[1]),
                        xytext=(0, sizes[0]),
                        arrowprops=dict(arrowstyle="<->", color="green", lw=2),
                    )
                    ax1.text(
                        0.5,
                        (sizes[0] + sizes[1]) / 2,
                        f"{ratio:.1f}x",
                        ha="center",
                        fontsize=11,
                        weight="bold",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
                    )

        # Plot 2: Access pattern comparison (3 formats)
        if access:
            operations = ["ROI", "M/z Slice", "Ion Image"]

            bruker_times = []
            raw_times = []
            res_times = []

            # Collect data
            if "bruker_original" in access:
                bruker_data = access["bruker_original"]
                bruker_times = [
                    bruker_data.get("roi_access_sec", 0),
                    bruker_data.get("mz_slice_sec", 0),
                    bruker_data.get("ion_image_sec", 0),
                ]

            if "zarr_raw" in access:
                raw_data = access["zarr_raw"]
                raw_times = [
                    raw_data.get("roi_access_sec", 0),
                    raw_data.get("mz_slice_sec", 0),
                    raw_data.get("ion_image_sec", 0),
                ]

            if "zarr_resampled" in access:
                res_data = access["zarr_resampled"]
                res_times = [
                    res_data.get("roi_access_sec", 0),
                    res_data.get("mz_slice_sec", 0),
                    res_data.get("ion_image_sec", 0),
                ]

            x = np.arange(len(operations))
            width = 0.25

            if bruker_times:
                ax2.bar(
                    x - width,
                    bruker_times,
                    width,
                    label="Bruker .d",
                    color=self.colors["bruker"],
                    alpha=0.8,
                )
            if raw_times:
                ax2.bar(
                    x, raw_times, width, label="Zarr Raw", color="#8B4789", alpha=0.8
                )
            if res_times:
                ax2.bar(
                    x + width,
                    res_times,
                    width,
                    label="Zarr 300k",
                    color=self.colors["spatialdata"],
                    alpha=0.8,
                )

            ax2.set_xlabel("Access Pattern", fontsize=12, weight="bold")
            ax2.set_ylabel("Time (seconds)", fontsize=12, weight="bold")
            ax2.set_title(
                "Access Performance:\nBruker Formats", fontsize=14, weight="bold"
            )
            ax2.set_xticks(x)
            ax2.set_xticklabels(operations)
            ax2.legend(fontsize=10)
            ax2.grid(True, axis="y", alpha=0.3)

            # Add speedup annotations for m/z slice (most important)
            if bruker_times and res_times and bruker_times[1] > 0 and res_times[1] > 0:
                speedup = bruker_times[1] / res_times[1]
                y_pos = max(bruker_times[1], res_times[1]) * 1.1
                ax2.text(
                    1,
                    y_pos,
                    f"{speedup:.1f}x faster",
                    ha="center",
                    fontsize=10,
                    weight="bold",
                    bbox=dict(
                        boxstyle="round", facecolor=self.colors["speedup"], alpha=0.5
                    ),
                )

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        plt.close()

    def create_all_plots(self):
        """Create all available plots from benchmark results."""
        print(f"\n{'='*70}")
        print("CREATING BENCHMARK VISUALIZATIONS")
        print(f"{'='*70}\n")

        results_dir = BenchmarkConfig.RESULTS_DIR
        plots_dir = BenchmarkConfig.PLOTS_DIR

        # Storage benchmark
        storage_file = results_dir / "storage_benchmark.json"
        if storage_file.exists():
            print("Creating storage comparison plot...")
            data = load_metrics(storage_file)
            self.plot_storage_comparison(data, plots_dir / "storage_comparison.png")

        # Spatial access benchmark
        spatial_file = results_dir / "spatial_access_benchmark.json"
        if spatial_file.exists():
            print("Creating spatial access comparison plot...")
            data = load_metrics(spatial_file)
            self.plot_spatial_access_comparison(
                data, plots_dir / "spatial_access_comparison.png"
            )

        # Spectral access benchmark
        spectral_file = results_dir / "spectral_access_benchmark.json"
        if spectral_file.exists():
            print("Creating spectral access comparison plot...")
            data = load_metrics(spectral_file)
            self.plot_spectral_access_comparison(
                data, plots_dir / "spectral_access_comparison.png"
            )

        # Parallel benchmark
        parallel_file = results_dir / "parallel_benchmark.json"
        if parallel_file.exists():
            print("Creating parallel scalability plot...")
            data = load_metrics(parallel_file)
            self.plot_parallel_scalability(data, plots_dir / "parallel_scalability.png")

        # Bruker interpolation benchmark
        bruker_file = results_dir / "bruker_interpolation_benchmark.json"
        if bruker_file.exists():
            print("Creating Bruker interpolation comparison plot...")
            data = load_metrics(bruker_file)
            self.plot_bruker_interpolation(
                data, plots_dir / "bruker_interpolation_comparison.png"
            )

        print(f"\n{'='*70}")
        print(f"All plots saved to: {plots_dir}")
        print(f"{'='*70}")


def main():
    """Create all benchmark visualizations."""
    visualizer = BenchmarkVisualizer()
    visualizer.create_all_plots()


if __name__ == "__main__":
    main()
