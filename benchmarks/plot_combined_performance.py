#!/usr/bin/env python
"""Create combined performance figure: initialization times + access latencies.

This creates a publication-quality figure showing:
(a) Top panel: Bar chart of initialization times (log scale to avoid label overlap)
(b) Bottom panel: Raincloud plots of access pattern latencies
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import ptitprince as pt
import numpy as np


# EDIT THESE VALUES based on your benchmark run:
INITIALIZATION_TIMES = {
    'ImzML (Processed)': 58.0,  # seconds - UPDATE THIS
    'Bruker .d': 0.5,            # seconds - UPDATE THIS
    'SpatialData': 8.0,          # seconds - UPDATE THIS
}


def plot_combined_figure(
    csv_path: str,
    init_times: dict,
    output_path: str,
    font_size: int = 12,
    width: int = 12,
    height: int = 10,
    use_log_init: bool = True,
):
    """Create combined figure with initialization and latency plots."""

    # Load latency results
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} measurements")
    if 'dataset' in df.columns:
        print(f"Datasets: {df['dataset'].unique()}")
    print(f"Formats: {df['format'].unique()}")
    print(f"Access patterns: {df['access_pattern'].unique()}")

    # Rename columns for better labels
    df = df.rename(
        columns={
            "access_pattern": "Access Pattern",
            "format": "Format",
            "latency_seconds": "Latency (seconds)",
        }
    )

    # Get unique values
    access_patterns = sorted(df["Access Pattern"].unique())
    formats = sorted(df["Format"].unique())

    print(f"\nAccess patterns: {access_patterns}")
    print(f"Formats: {formats}")

    # Create figure with 2 subplots
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.5], hspace=0.35)

    # =================================================================
    # TOP PANEL (a): Initialization Times
    # =================================================================
    ax_init = fig.add_subplot(gs[0])

    init_formats = list(init_times.keys())
    init_values = list(init_times.values())

    # Colors matching Set2 palette
    init_colors = ['#fc8d62', '#8da0cb', '#66c2a5']  # Orange, Blue, Teal

    # Create bars
    bars = ax_init.bar(init_formats, init_values, color=init_colors,
                       alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on top of bars
    for bar, value in zip(bars, init_values):
        height = bar.get_height()
        # Position label above bar with some padding
        y_pos = height * 1.05 if use_log_init else height + 1
        ax_init.text(bar.get_x() + bar.get_width()/2., y_pos,
                     f'{value:.1f}s',
                     ha='center', va='bottom', fontsize=font_size, fontweight='bold')

    # Labels
    ylabel = 'Time (seconds, log scale)' if use_log_init else 'Time (seconds)'
    ax_init.set_ylabel(ylabel, fontsize=font_size, fontweight='bold')
    ax_init.set_title('(a) Format Initialization Overhead',
                      fontsize=font_size+2, fontweight='bold', loc='left', pad=10)

    # Use log scale for y-axis if requested
    if use_log_init:
        ax_init.set_yscale('log')
        # Set y-limits to give space for labels
        ax_init.set_ylim(bottom=min(init_values)*0.5, top=max(init_values)*2)
    else:
        # Extend y-axis to 65 to avoid label overlap
        ax_init.set_ylim(top=65)

    # Grid
    ax_init.grid(True, alpha=0.3, axis='y')
    ax_init.set_axisbelow(True)
    ax_init.tick_params(axis='both', which='major', labelsize=font_size)

    # =================================================================
    # BOTTOM PANEL (b): Raincloud Plots
    # =================================================================
    ax_rain = fig.add_subplot(gs[1])

    # Create raincloud plot
    ax_rain = pt.RainCloud(
        x="Access Pattern",
        y="Latency (seconds)",
        hue="Format",
        data=df,
        palette="Set2",
        order=access_patterns,
        hue_order=formats,
        width_viol=0.6,
        ax=ax_rain,
        orient="h",
        alpha=0.65,
        jitter=0.03,
        move=0.2,
    )

    # Set log scale for x-axis (latency)
    ax_rain.set_xscale("log")
    ax_rain.set_xlabel("Latency per access (seconds, log scale)",
                       fontsize=font_size, fontweight='bold')
    ax_rain.set_ylabel("", fontsize=font_size)
    ax_rain.set_title('(b) Access Pattern Latencies',
                      fontsize=font_size+2, fontweight='bold', loc='left', pad=10)

    # Font sizes
    ax_rain.tick_params(axis='both', which='major', labelsize=font_size)

    # Format legend
    handles, labels = ax_rain.get_legend_handles_labels()
    n_formats = len(df["Format"].unique())
    ax_rain.legend(
        handles[0:n_formats],
        labels[0:n_formats],
        loc="lower left",
        prop={"size": font_size},
        title="Format",
        title_fontsize=font_size,
    )

    # Grid
    ax_rain.grid(True, alpha=0.3, axis="x")

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nCombined figure saved to: {output_path}")

    # Also save as PDF
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF version saved to: {pdf_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("INITIALIZATION TIMES")
    print("=" * 70)
    for fmt, time in init_times.items():
        print(f"{fmt:25s}: {time:8.3f} seconds")

    # Speedups
    if 'SpatialData' in init_times:
        print("\nInitialization vs SpatialData:")
        sd_time = init_times['SpatialData']
        for fmt, time in init_times.items():
            if fmt != 'SpatialData':
                ratio = time / sd_time
                if ratio > 1:
                    print(f"  {fmt:25s}: {ratio:.2f}x slower")
                else:
                    print(f"  {fmt:25s}: {1/ratio:.2f}x faster")

    print("\n" + "=" * 70)
    print("ACCESS LATENCY SUMMARY (MEDIAN)")
    print("=" * 70)

    for pattern in access_patterns:
        print(f"\n{pattern}:")
        pattern_df = df[df["Access Pattern"] == pattern]

        for fmt in sorted(formats):
            fmt_df = pattern_df[pattern_df["Format"] == fmt]
            if len(fmt_df) > 0:
                median = fmt_df["Latency (seconds)"].median()
                if median < 0.001:
                    print(f"  {fmt:25s}: {median*1000:.3f} ms ({median*1e6:.1f} μs)")
                elif median < 1:
                    print(f"  {fmt:25s}: {median*1000:.3f} ms")
                else:
                    print(f"  {fmt:25s}: {median:.3f} s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create combined performance figure (initialization + latencies)"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="benchmarks/results/xenium_comparison.csv",
        help="Input CSV file with latency results",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="benchmarks/results/combined_performance.png",
        help="Output filename",
    )
    parser.add_argument("--font-size", type=int, default=12, help="Font size")
    parser.add_argument("--width", type=int, default=12, help="Figure width")
    parser.add_argument("--height", type=int, default=10, help="Figure height")
    parser.add_argument("--no-log-init", action="store_true",
                       help="Don't use log scale for initialization times (extends to y=65)")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_combined_figure(
        str(csv_path),
        INITIALIZATION_TIMES,
        str(output_path),
        font_size=args.font_size,
        width=args.width,
        height=args.height,
        use_log_init=not args.no_log_init,
    )

    return 0


if __name__ == "__main__":
    exit(main())
