#!/usr/bin/env python
"""Plot raincloud plots for MSI latency benchmark results."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import ptitprince as pt


def plot_latency_results(
    csv_path: str,
    output_path: str,
    font_size: int = 14,
    width: int = 12,
    height: int = 6,
):
    """Create raincloud plots for latency benchmark results."""
    # Load results
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} measurements")

    # Handle both CSV formats (with or without 'dataset' column)
    if "dataset" in df.columns:
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

    # Get unique access patterns and formats
    access_patterns = sorted(df["Access Pattern"].unique())
    formats = sorted(df["Format"].unique())

    print(f"\nAccess patterns found: {access_patterns}")
    print(f"Formats found: {formats}")

    # Create figure
    fig, ax = plt.subplots(figsize=(width, height))

    # Create raincloud plot
    ax = pt.RainCloud(
        x="Access Pattern",
        y="Latency (seconds)",
        hue="Format",
        data=df,
        palette="Set2",
        order=access_patterns,
        hue_order=formats,
        width_viol=0.6,
        ax=ax,
        orient="h",
        alpha=0.65,
        jitter=0.03,
        move=0.2,
    )

    # Set log scale for x-axis (latency)
    ax.set_xscale("log")
    ax.set_xlabel("Latency per access (seconds)", fontsize=font_size)
    ax.set_ylabel("")

    # Set font sizes
    for item in (
        [ax.title, ax.xaxis.label, ax.yaxis.label]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(font_size)

    # Format legend
    handles, labels = ax.get_legend_handles_labels()
    n_formats = len(df["Format"].unique())
    plt.legend(
        handles[0:n_formats],
        labels[0:n_formats],
        loc="lower left",
        prop={"size": font_size},
        title="Format",
        title_fontsize=font_size,
    )

    # Add grid
    ax.grid(True, alpha=0.3, axis="x")

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nRaincloud plot saved to: {output_path}")

    # Also save as PDF
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF version saved to: {pdf_path}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for pattern in df["Access Pattern"].unique():
        print(f"\n{pattern}:")
        pattern_df = df[df["Access Pattern"] == pattern]

        for fmt in sorted(df["Format"].unique()):
            fmt_df = pattern_df[pattern_df["Format"] == fmt]
            if len(fmt_df) > 0:
                latencies = fmt_df["Latency (seconds)"]
                print(f"  {fmt}:")
                print(f"    Mean: {latencies.mean():.6f} s")
                print(f"    Median: {latencies.median():.6f} s")
                print(f"    Std: {latencies.std():.6f} s")
                print(f"    Min: {latencies.min():.6f} s")
                print(f"    Max: {latencies.max():.6f} s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Plot raincloud plots for MSI latency benchmark"
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="benchmarks/results/latency_three_formats.csv",
        help="Input CSV file",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="benchmarks/results/latency_three_formats.png",
        help="Output filename",
    )
    parser.add_argument("--font-size", type=int, default=14, help="Font size")
    parser.add_argument("--width", type=int, default=12, help="Figure width")
    parser.add_argument("--height", type=int, default=6, help="Figure height")

    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_latency_results(
        str(csv_path),
        str(output_path),
        font_size=args.font_size,
        width=args.width,
        height=args.height,
    )

    return 0


if __name__ == "__main__":
    exit(main())
