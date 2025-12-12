"""
Thyra Architecture and Workflow Diagrams Generator

This script generates publication-ready diagrams for the Thyra paper.
Run with: python generate_diagrams.py

Requirements:
    pip install matplotlib graphviz

For graphviz, also install the system package:
    - Windows: choco install graphviz OR download from graphviz.org
    - macOS: brew install graphviz
    - Linux: apt install graphviz
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

# =============================================================================
# Color Scheme (Publication-friendly)
# =============================================================================
COLORS = {
    "core": "#E3F2FD",  # Light blue
    "core_border": "#1565C0",  # Dark blue
    "reader": "#E8F5E9",  # Light green
    "reader_border": "#2E7D32",  # Dark green
    "converter": "#FFF3E0",  # Light orange
    "converter_border": "#EF6C00",  # Dark orange
    "output": "#F3E5F5",  # Light purple
    "output_border": "#7B1FA2",  # Dark purple
    "resampling": "#FFFDE7",  # Light yellow
    "resampling_border": "#F9A825",  # Dark yellow
    "future": "#FAFAFA",  # Light gray
    "future_border": "#9E9E9E",  # Gray
    "registry": "#FFEBEE",  # Light red
    "registry_border": "#C62828",  # Dark red
    "downstream": "#E0F2F1",  # Light teal
    "downstream_border": "#00695C",  # Dark teal
    "arrow": "#424242",  # Dark gray
    "text": "#212121",  # Almost black
}


def create_architecture_diagram():
    """Create the plugin architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        8,
        11.5,
        "Thyra Plugin Architecture",
        fontsize=20,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["text"],
    )

    # ==========================================================================
    # CORE SECTION (Top center)
    # ==========================================================================
    # Core box
    core_box = FancyBboxPatch(
        (4, 8),
        8,
        3,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["core"],
        edgecolor=COLORS["core_border"],
        linewidth=2,
    )
    ax.add_patch(core_box)
    ax.text(
        8,
        10.7,
        "THYRA CORE",
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["core_border"],
    )

    # Abstract Base Classes
    abc_boxes = [
        (4.3, 9.2, "BaseMSIReader\n(ABC)", 2.2),
        (6.9, 9.2, "BaseMSIConverter\n(ABC)", 2.2),
        (9.5, 9.2, "MetadataExtractor\n(ABC)", 2.2),
    ]
    for x, y, label, width in abc_boxes:
        box = FancyBboxPatch(
            (x, y),
            width,
            1.2,
            boxstyle="round,pad=0.05",
            facecolor="white",
            edgecolor=COLORS["core_border"],
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + 0.6,
            label,
            fontsize=9,
            ha="center",
            va="center",
            color=COLORS["text"],
        )

    # Registry
    reg_box = FancyBboxPatch(
        (5.5, 8.2),
        5,
        0.8,
        boxstyle="round,pad=0.05",
        facecolor=COLORS["registry"],
        edgecolor=COLORS["registry_border"],
        linewidth=1.5,
    )
    ax.add_patch(reg_box)
    ax.text(
        8,
        8.6,
        "Plugin Registry (@register_reader / @register_converter)",
        fontsize=9,
        ha="center",
        va="center",
        color=COLORS["registry_border"],
    )

    # ==========================================================================
    # READERS SECTION (Left)
    # ==========================================================================
    reader_box = FancyBboxPatch(
        (0.5, 4),
        4,
        3.5,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["reader"],
        edgecolor=COLORS["reader_border"],
        linewidth=2,
    )
    ax.add_patch(reader_box)
    ax.text(
        2.5,
        7.2,
        "INPUT READERS",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["reader_border"],
    )

    readers = [
        (0.8, 5.8, "ImzMLReader\n.imzml + .ibd", False),
        (0.8, 4.8, "BrukerReader\n.d directories", False),
        (2.6, 5.3, "Future Readers\n(Community)", True),
    ]
    for x, y, label, is_future in readers:
        color = COLORS["future"] if is_future else "white"
        border = COLORS["future_border"] if is_future else COLORS["reader_border"]
        style = "--" if is_future else "-"
        box = FancyBboxPatch(
            (x, y),
            1.7,
            0.9,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor=border,
            linewidth=1,
            linestyle=style,
        )
        ax.add_patch(box)
        ax.text(
            x + 0.85,
            y + 0.45,
            label,
            fontsize=7,
            ha="center",
            va="center",
            color=COLORS["text"],
        )

    # ==========================================================================
    # CONVERTERS SECTION (Right)
    # ==========================================================================
    conv_box = FancyBboxPatch(
        (11.5, 4),
        4,
        3.5,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["converter"],
        edgecolor=COLORS["converter_border"],
        linewidth=2,
    )
    ax.add_patch(conv_box)
    ax.text(
        13.5,
        7.2,
        "OUTPUT CONVERTERS",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["converter_border"],
    )

    converters = [
        (11.8, 5.8, "SpatialData2D\nConverter", False),
        (11.8, 4.8, "SpatialData3D\nConverter", False),
        (13.6, 5.3, "Future Converters\n(Community)", True),
    ]
    for x, y, label, is_future in converters:
        color = COLORS["future"] if is_future else "white"
        border = COLORS["future_border"] if is_future else COLORS["converter_border"]
        style = "--" if is_future else "-"
        box = FancyBboxPatch(
            (x, y),
            1.7,
            0.9,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor=border,
            linewidth=1,
            linestyle=style,
        )
        ax.add_patch(box)
        ax.text(
            x + 0.85,
            y + 0.45,
            label,
            fontsize=7,
            ha="center",
            va="center",
            color=COLORS["text"],
        )

    # ==========================================================================
    # RESAMPLING SECTION (Center)
    # ==========================================================================
    resamp_box = FancyBboxPatch(
        (5.5, 4),
        5,
        3.5,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["resampling"],
        edgecolor=COLORS["resampling_border"],
        linewidth=2,
    )
    ax.add_patch(resamp_box)
    ax.text(
        8,
        7.2,
        "RESAMPLING SYSTEM",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["resampling_border"],
    )

    ax.text(
        8,
        6.5,
        "Decision Tree",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["text"],
    )

    # Mass axis generators
    ax.text(6.8, 5.8, "Mass Axis Generators:", fontsize=8, ha="center", va="center")
    ax.text(
        6.8,
        5.4,
        "Linear | TOF | FTICR | Orbitrap",
        fontsize=7,
        ha="center",
        va="center",
        style="italic",
    )

    # Resampling strategies
    ax.text(9.2, 5.8, "Strategies:", fontsize=8, ha="center", va="center")
    ax.text(
        9.2,
        5.4,
        "Nearest Neighbor\nTIC-Preserving",
        fontsize=7,
        ha="center",
        va="center",
        style="italic",
    )

    # ==========================================================================
    # OUTPUT SECTION (Bottom)
    # ==========================================================================
    out_box = FancyBboxPatch(
        (4, 0.5),
        8,
        3,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["output"],
        edgecolor=COLORS["output_border"],
        linewidth=2,
    )
    ax.add_patch(out_box)
    ax.text(
        8,
        3.2,
        "SPATIALDATA / ZARR OUTPUT",
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["output_border"],
    )

    outputs = [
        (4.3, 1.5, "Tables\n(AnnData)"),
        (6.1, 1.5, "Shapes\n(GeoDataFrame)"),
        (7.9, 1.5, "Images\n(xarray)"),
        (9.7, 1.5, "Metadata\n(Dict)"),
    ]
    for x, y, label in outputs:
        box = FancyBboxPatch(
            (x, y),
            1.6,
            1.2,
            boxstyle="round,pad=0.03",
            facecolor="white",
            edgecolor=COLORS["output_border"],
            linewidth=1,
        )
        ax.add_patch(box)
        ax.text(
            x + 0.8,
            y + 0.6,
            label,
            fontsize=8,
            ha="center",
            va="center",
            color=COLORS["text"],
        )

    # ==========================================================================
    # ARROWS
    # ==========================================================================
    arrow_style = dict(
        arrowstyle="->", color=COLORS["arrow"], lw=1.5, connectionstyle="arc3,rad=0.1"
    )

    # Core to Readers
    ax.annotate("", xy=(2.5, 7.5), xytext=(5.5, 8.5), arrowprops=arrow_style)

    # Core to Converters
    ax.annotate("", xy=(13.5, 7.5), xytext=(10.5, 8.5), arrowprops=arrow_style)

    # Readers to Resampling
    ax.annotate(
        "",
        xy=(5.5, 5.5),
        xytext=(4.5, 5.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.5),
    )

    # Resampling to Converters
    ax.annotate(
        "",
        xy=(11.5, 5.5),
        xytext=(10.5, 5.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.5),
    )

    # Converters to Output
    ax.annotate(
        "",
        xy=(8, 3.5),
        xytext=(8, 4),
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=1.5),
    )

    # Legend
    legend_items = [
        ("Current Implementation", "white", COLORS["reader_border"], "-"),
        ("Community Extension Point", COLORS["future"], COLORS["future_border"], "--"),
    ]
    for i, (label, fc, ec, ls) in enumerate(legend_items):
        box = FancyBboxPatch(
            (0.5, 0.5 + i * 0.6),
            0.4,
            0.4,
            boxstyle="round,pad=0.02",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1,
            linestyle=ls,
        )
        ax.add_patch(box)
        ax.text(1.1, 0.7 + i * 0.6, label, fontsize=8, ha="left", va="center")

    plt.tight_layout()
    return fig


def create_workflow_diagram():
    """Create the data flow workflow diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        9,
        7.6,
        "Thyra Data Processing Workflow",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["text"],
    )

    # ==========================================================================
    # Stage boxes
    # ==========================================================================
    stages = [
        (0.2, 2, 2.2, 4.5, "INPUT\nFORMATS", COLORS["reader"], COLORS["reader_border"]),
        (
            2.8,
            2,
            2.2,
            4.5,
            "FORMAT\nDETECTION",
            COLORS["registry"],
            COLORS["registry_border"],
        ),
        (5.4, 2, 2.4, 4.5, "DATA\nREADING", COLORS["core"], COLORS["core_border"]),
        (
            8.2,
            2,
            2.4,
            4.5,
            "RESAMPLING\n(Optional)",
            COLORS["resampling"],
            COLORS["resampling_border"],
        ),
        (
            11,
            2,
            2.4,
            4.5,
            "CONVERSION",
            COLORS["converter"],
            COLORS["converter_border"],
        ),
        (13.8, 2, 2.2, 4.5, "OUTPUT", COLORS["output"], COLORS["output_border"]),
        (
            16.4,
            2,
            1.4,
            4.5,
            "ANALYSIS",
            COLORS["downstream"],
            COLORS["downstream_border"],
        ),
    ]

    for x, y, w, h, label, fc, ec in stages:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor=fc,
            edgecolor=ec,
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(
            x + w / 2,
            y + h - 0.4,
            label,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="center",
            color=ec,
        )

    # ==========================================================================
    # Stage contents
    # ==========================================================================
    # Input formats
    inputs = ["ImzML\n(.imzml)", "Bruker\n(.d)", "Future\nFormats"]
    for i, label in enumerate(inputs):
        is_future = i == 2
        box = FancyBboxPatch(
            (0.4, 5.2 - i * 1.1),
            1.8,
            0.9,
            boxstyle="round,pad=0.02",
            facecolor="white" if not is_future else COLORS["future"],
            edgecolor=(
                COLORS["reader_border"] if not is_future else COLORS["future_border"]
            ),
            linewidth=1,
            linestyle="-" if not is_future else "--",
        )
        ax.add_patch(box)
        ax.text(1.3, 5.65 - i * 1.1, label, fontsize=7, ha="center", va="center")

    # Format detection
    ax.text(3.9, 4.5, "Registry\nLookup", fontsize=8, ha="center", va="center")
    ax.text(3.9, 3.5, "Extension\nMapping", fontsize=8, ha="center", va="center")

    # Data reading
    reading_items = [
        "iter_spectra()",
        "get_mass_axis()",
        "Essential\nMetadata",
        "Comprehensive\nMetadata",
    ]
    for i, label in enumerate(reading_items):
        ax.text(6.6, 5.5 - i * 0.9, label, fontsize=7, ha="center", va="center")

    # Resampling
    ax.text(
        9.4,
        5.5,
        "Decision\nTree",
        fontsize=7,
        ha="center",
        va="center",
        fontweight="bold",
    )
    ax.text(9.4, 4.5, "Mass Axis\nGenerator", fontsize=7, ha="center", va="center")
    ax.text(9.4, 3.5, "Resampling\nStrategy", fontsize=7, ha="center", va="center")

    # Conversion steps
    steps = [
        "1. Initialize",
        "2. Create\n    Structures",
        "3. Process\n    Spectra",
        "4. Finalize",
        "5. Save",
    ]
    for i, label in enumerate(steps):
        ax.text(12.2, 5.7 - i * 0.7, label, fontsize=7, ha="center", va="center")

    # Output components
    outputs = [
        "Tables\n(AnnData)",
        "Shapes\n(GeoPandas)",
        "Images\n(xarray)",
        "Metadata",
    ]
    for i, label in enumerate(outputs):
        ax.text(14.9, 5.5 - i * 0.9, label, fontsize=7, ha="center", va="center")

    # Analysis tools
    tools = ["Scanpy", "Squidpy", "Napari", "Custom"]
    for i, label in enumerate(tools):
        ax.text(17.1, 5.5 - i * 0.9, label, fontsize=7, ha="center", va="center")

    # ==========================================================================
    # Arrows between stages
    # ==========================================================================
    arrow_positions = [
        (2.4, 4.25, 2.8, 4.25),
        (5.0, 4.25, 5.4, 4.25),
        (7.8, 4.25, 8.2, 4.25),
        (10.6, 4.25, 11.0, 4.25),
        (13.4, 4.25, 13.8, 4.25),
        (16.0, 4.25, 16.4, 4.25),
    ]

    for x1, y1, x2, y2 in arrow_positions:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2),
        )

    # Stage numbers
    for i, x in enumerate([1.3, 3.9, 6.6, 9.4, 12.2, 14.9, 17.1]):
        ax.text(
            x,
            1.6,
            f"Stage {i+1}",
            fontsize=8,
            ha="center",
            va="center",
            color=COLORS["arrow"],
            style="italic",
        )

    plt.tight_layout()
    return fig


def create_simple_architecture():
    """Create a simplified architecture diagram focused on extensibility."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(
        7,
        9.5,
        "Thyra Extensible Plugin Architecture",
        fontsize=18,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["text"],
    )

    # Central core
    core = plt.Circle(
        (7, 5),
        1.5,
        facecolor=COLORS["core"],
        edgecolor=COLORS["core_border"],
        linewidth=3,
    )
    ax.add_patch(core)
    ax.text(
        7,
        5.3,
        "THYRA",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["core_border"],
    )
    ax.text(
        7,
        4.7,
        "CORE",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        color=COLORS["core_border"],
    )

    # Abstract base classes around core
    abc_angles = [60, 180, 300]
    abc_labels = ["BaseMSI\nReader", "BaseMSI\nConverter", "Metadata\nExtractor"]
    abc_colors = [
        COLORS["reader_border"],
        COLORS["converter_border"],
        COLORS["resampling_border"],
    ]

    for angle, label, color in zip(abc_angles, abc_labels, abc_colors):
        rad = np.radians(angle)
        x = 7 + 2.3 * np.cos(rad)
        y = 5 + 2.3 * np.sin(rad)
        circle = plt.Circle(
            (x, y), 0.7, facecolor="white", edgecolor=color, linewidth=2
        )
        ax.add_patch(circle)
        ax.text(x, y, label, fontsize=7, ha="center", va="center", fontweight="bold")

    # Plugin implementations
    # Readers (left side)
    reader_box = FancyBboxPatch(
        (0.5, 3.5),
        2.5,
        3,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["reader"],
        edgecolor=COLORS["reader_border"],
        linewidth=2,
    )
    ax.add_patch(reader_box)
    ax.text(
        1.75,
        6.2,
        "READERS",
        fontsize=10,
        fontweight="bold",
        ha="center",
        color=COLORS["reader_border"],
    )

    reader_items = [
        ("ImzMLReader", False),
        ("BrukerReader", False),
        ("ThermoReader", True),
        ("WatersReader", True),
    ]
    for i, (label, is_future) in enumerate(reader_items):
        y = 5.5 - i * 0.6
        style = "--" if is_future else "-"
        color = COLORS["future_border"] if is_future else COLORS["reader_border"]
        fc = COLORS["future"] if is_future else "white"
        box = FancyBboxPatch(
            (0.7, y),
            2.1,
            0.5,
            boxstyle="round,pad=0.02",
            facecolor=fc,
            edgecolor=color,
            linewidth=1,
            linestyle=style,
        )
        ax.add_patch(box)
        ax.text(1.75, y + 0.25, label, fontsize=7, ha="center", va="center")

    # Converters (right side)
    conv_box = FancyBboxPatch(
        (11, 3.5),
        2.5,
        3,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["converter"],
        edgecolor=COLORS["converter_border"],
        linewidth=2,
    )
    ax.add_patch(conv_box)
    ax.text(
        12.25,
        6.2,
        "CONVERTERS",
        fontsize=10,
        fontweight="bold",
        ha="center",
        color=COLORS["converter_border"],
    )

    conv_items = [
        ("SpatialData2D", False),
        ("SpatialData3D", False),
        ("AnnData-only", True),
        ("OME-ZARR", True),
    ]
    for i, (label, is_future) in enumerate(conv_items):
        y = 5.5 - i * 0.6
        style = "--" if is_future else "-"
        color = COLORS["future_border"] if is_future else COLORS["converter_border"]
        fc = COLORS["future"] if is_future else "white"
        box = FancyBboxPatch(
            (11.2, y),
            2.1,
            0.5,
            boxstyle="round,pad=0.02",
            facecolor=fc,
            edgecolor=color,
            linewidth=1,
            linestyle=style,
        )
        ax.add_patch(box)
        ax.text(12.25, y + 0.25, label, fontsize=7, ha="center", va="center")

    # Output (bottom)
    out_box = FancyBboxPatch(
        (4.5, 0.5),
        5,
        1.5,
        boxstyle="round,pad=0.1",
        facecolor=COLORS["output"],
        edgecolor=COLORS["output_border"],
        linewidth=2,
    )
    ax.add_patch(out_box)
    ax.text(
        7,
        1.7,
        "SPATIALDATA / ZARR",
        fontsize=11,
        fontweight="bold",
        ha="center",
        color=COLORS["output_border"],
    )
    ax.text(
        7,
        1.1,
        "Tables | Shapes | Images | Metadata",
        fontsize=9,
        ha="center",
        color=COLORS["text"],
    )

    # Arrows
    # Readers to core
    ax.annotate(
        "",
        xy=(5.5, 5.8),
        xytext=(3, 5.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2),
    )

    # Core to converters
    ax.annotate(
        "",
        xy=(11, 5.5),
        xytext=(8.5, 5.8),
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2),
    )

    # Converters to output
    ax.annotate(
        "",
        xy=(7, 2),
        xytext=(7, 3.5),
        arrowprops=dict(arrowstyle="->", color=COLORS["arrow"], lw=2),
    )

    # Legend
    ax.text(0.5, 1.5, "Legend:", fontsize=9, fontweight="bold")
    box1 = FancyBboxPatch(
        (0.5, 0.9),
        0.4,
        0.4,
        boxstyle="round,pad=0.02",
        facecolor="white",
        edgecolor=COLORS["reader_border"],
        linewidth=1,
    )
    ax.add_patch(box1)
    ax.text(1.1, 1.1, "Current Implementation", fontsize=8, ha="left", va="center")

    box2 = FancyBboxPatch(
        (0.5, 0.4),
        0.4,
        0.4,
        boxstyle="round,pad=0.02",
        facecolor=COLORS["future"],
        edgecolor=COLORS["future_border"],
        linewidth=1,
        linestyle="--",
    )
    ax.add_patch(box2)
    ax.text(1.1, 0.6, "Community Extension Point", fontsize=8, ha="left", va="center")

    plt.tight_layout()
    return fig


def main():
    """Generate all diagrams and save them."""
    print("Generating Thyra diagrams...")

    # Architecture diagram
    print("  Creating architecture diagram...")
    fig1 = create_architecture_diagram()
    fig1.savefig(
        "architecture_diagram.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig1.savefig(
        "architecture_diagram.svg",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig1.savefig(
        "architecture_diagram.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("    Saved: architecture_diagram.png/svg/pdf")

    # Workflow diagram
    print("  Creating workflow diagram...")
    fig2 = create_workflow_diagram()
    fig2.savefig(
        "workflow_diagram.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig2.savefig(
        "workflow_diagram.svg", bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    fig2.savefig(
        "workflow_diagram.pdf", bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    print("    Saved: workflow_diagram.png/svg/pdf")

    # Simple architecture (for graphical abstract)
    print("  Creating simplified architecture diagram...")
    fig3 = create_simple_architecture()
    fig3.savefig(
        "architecture_simple.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig3.savefig(
        "architecture_simple.svg",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig3.savefig(
        "architecture_simple.pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print("    Saved: architecture_simple.png/svg/pdf")

    print("\nAll diagrams generated successfully!")
    print("\nFiles created:")
    print("  - architecture_diagram.png/svg/pdf  (detailed architecture)")
    print("  - workflow_diagram.png/svg/pdf      (data flow pipeline)")
    print("  - architecture_simple.png/svg/pdf   (simplified for abstract)")

    plt.show()


if __name__ == "__main__":
    main()
