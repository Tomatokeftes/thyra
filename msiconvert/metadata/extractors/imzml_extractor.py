# msiconvert/metadata/extractors/imzml_extractor.py
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pyimzml.ImzMLParser import ImzMLParser

from ...core.base_extractor import MetadataExtractor
from ..types import ComprehensiveMetadata, EssentialMetadata

logger = logging.getLogger(__name__)


class ImzMLMetadataExtractor(MetadataExtractor):
    """ImzML-specific metadata extractor with optimized two-phase extraction."""

    def __init__(self, parser: ImzMLParser, imzml_path: Path):
        """
        Initialize ImzML metadata extractor.

        Args:
            parser: Initialized ImzML parser
            imzml_path: Path to the ImzML file
        """
        super().__init__(parser)
        self.parser = parser
        self.imzml_path = imzml_path

    def _extract_essential_impl(self) -> EssentialMetadata:
        """Extract essential metadata optimized for speed."""
        # Single coordinate scan for efficiency
        coords = np.array(self.parser.coordinates)

        if len(coords) == 0:
            raise ValueError("No coordinates found in ImzML file")

        dimensions = self._calculate_dimensions(coords)
        coordinate_bounds = self._calculate_bounds(coords)
        mass_range = self._get_mass_range_fast()
        pixel_size = self._extract_pixel_size_fast()
        n_spectra = len(coords)
        estimated_memory = self._estimate_memory(n_spectra)

        return EssentialMetadata(
            dimensions=dimensions,
            coordinate_bounds=coordinate_bounds,
            mass_range=mass_range,
            pixel_size=pixel_size,
            n_spectra=n_spectra,
            estimated_memory_gb=estimated_memory,
            source_path=str(self.imzml_path),
        )

    def _extract_comprehensive_impl(self) -> ComprehensiveMetadata:
        """Extract comprehensive metadata with full XML parsing."""
        essential = self.get_essential()

        return ComprehensiveMetadata(
            essential=essential,
            format_specific=self._extract_imzml_specific(),
            acquisition_params=self._extract_acquisition_params(),
            instrument_info=self._extract_instrument_info(),
            raw_metadata=self._extract_raw_metadata(),
        )

    def _calculate_dimensions(self, coords: NDArray[np.int_]) -> Tuple[int, int, int]:
        """Calculate dataset dimensions from coordinates."""
        if len(coords) == 0:
            return (0, 0, 0)

        # Coordinates are 1-based in ImzML, convert to 0-based for calculation
        coords_0based = coords - 1

        max_coords = np.max(coords_0based, axis=0)
        return (int(max_coords[0]) + 1, int(max_coords[1]) + 1, int(max_coords[2]) + 1)

    def _calculate_bounds(
        self, coords: NDArray[np.int_]
    ) -> Tuple[float, float, float, float]:
        """Calculate coordinate bounds (min_x, max_x, min_y, max_y)."""
        if len(coords) == 0:
            return (0.0, 0.0, 0.0, 0.0)

        # Convert to spatial coordinates (assuming 1-based indexing)
        x_coords = coords[:, 0].astype(float)
        y_coords = coords[:, 1].astype(float)

        return (
            float(np.min(x_coords)),
            float(np.max(x_coords)),
            float(np.min(y_coords)),
            float(np.max(y_coords)),
        )

    def _get_mass_range_fast(self) -> Tuple[float, float]:
        """Fast mass range extraction using first and last spectrum."""
        try:
            # Get mass range from first spectrum
            first_mzs, _ = self.parser.getspectrum(0)
            if len(first_mzs) == 0:
                return (0.0, 1000.0)  # Default range

            min_mass = float(np.min(first_mzs))
            max_mass = float(np.max(first_mzs))

            # Check a few more spectra to get better range estimate
            n_spectra = len(self.parser.coordinates)
            check_indices = [
                n_spectra // 4,
                n_spectra // 2,
                3 * n_spectra // 4,
                n_spectra - 1,
            ]

            for idx in check_indices:
                if idx < n_spectra and idx != 0:
                    try:
                        mzs, _ = self.parser.getspectrum(idx)
                        if len(mzs) > 0:
                            min_mass = min(min_mass, float(np.min(mzs)))
                            max_mass = max(max_mass, float(np.max(mzs)))
                    except Exception:
                        continue  # Skip problematic spectra

            return (min_mass, max_mass)

        except Exception as e:
            logger.warning(f"Failed to extract mass range: {e}")
            return (0.0, 1000.0)  # Default range

    def _extract_pixel_size_fast(self) -> Optional[Tuple[float, float]]:
        """Fast pixel size extraction from imzmldict first."""
        if hasattr(self.parser, "imzmldict") and self.parser.imzmldict:
            # Check for pixel size parameters in the parsed dictionary
            x_size = self.parser.imzmldict.get("pixel size x")
            y_size = self.parser.imzmldict.get("pixel size y")

            if x_size is not None and y_size is not None:
                try:
                    return (float(x_size), float(y_size))
                except (ValueError, TypeError):
                    pass

        return None  # Defer to comprehensive extraction

    def _estimate_memory(self, n_spectra: int) -> float:
        """Estimate memory usage in GB."""
        # Rough estimate: assume average 1000 peaks per spectrum, 8 bytes per float
        avg_peaks_per_spectrum = 1000
        bytes_per_value = 8  # float64
        estimated_bytes = (
            n_spectra * avg_peaks_per_spectrum * 2 * bytes_per_value
        )  # mz + intensity
        return estimated_bytes / (1024**3)  # Convert to GB

    def _extract_imzml_specific(self) -> Dict[str, Any]:
        """Extract ImzML format-specific metadata."""
        format_specific = {
            "imzml_version": "1.1.0",  # Default version
            "file_mode": "continuous"
            if getattr(self.parser, "continuous", False)
            else "processed",
            "ibd_file": str(self.imzml_path.with_suffix(".ibd")),
            "uuid": None,
            "spectrum_count": len(self.parser.coordinates),
            "scan_settings": {},
        }

        # Extract UUID if available
        try:
            if hasattr(self.parser, "metadata") and hasattr(
                self.parser.metadata, "file_description"
            ):
                cv_params = getattr(
                    self.parser.metadata.file_description, "cv_params", []
                )
                if cv_params and len(cv_params) > 0:
                    format_specific["uuid"] = cv_params[0][2]
        except Exception as e:
            logger.debug(f"Could not extract UUID: {e}")

        return format_specific

    def _extract_acquisition_params(self) -> Dict[str, Any]:
        """Extract acquisition parameters from XML metadata."""
        params = {}

        # Extract pixel size with full XML parsing if not found in fast extraction
        if not self.get_essential().has_pixel_size:
            pixel_size = self._extract_pixel_size_from_xml()
            if pixel_size:
                params["pixel_size_x_um"] = pixel_size[0]
                params["pixel_size_y_um"] = pixel_size[1]

        # Add other acquisition parameters from imzmldict
        if hasattr(self.parser, "imzmldict") and self.parser.imzmldict:
            acquisition_keys = [
                "scan direction",
                "scan pattern",
                "scan type",
                "laser power",
                "laser frequency",
                "laser spot size",
            ]
            for key in acquisition_keys:
                if key in self.parser.imzmldict:
                    params[key.replace(" ", "_")] = self.parser.imzmldict[key]

        return params

    def _extract_instrument_info(self) -> Dict[str, Any]:
        """Extract instrument information."""
        instrument = {}

        if hasattr(self.parser, "imzmldict") and self.parser.imzmldict:
            instrument_keys = [
                "instrument model",
                "instrument serial number",
                "software",
                "software version",
            ]
            for key in instrument_keys:
                if key in self.parser.imzmldict:
                    instrument[key.replace(" ", "_")] = self.parser.imzmldict[key]

        return instrument

    def _extract_raw_metadata(self) -> Dict[str, Any]:
        """Extract raw metadata from imzmldict."""
        raw_metadata = {}

        if hasattr(self.parser, "imzmldict") and self.parser.imzmldict:
            raw_metadata = dict(self.parser.imzmldict)

        return raw_metadata

    def _extract_pixel_size_from_xml(self) -> Optional[Tuple[float, float]]:
        """Extract pixel size using full XML parsing as fallback."""
        try:
            if not hasattr(self.parser, "metadata") or not hasattr(
                self.parser.metadata, "root"
            ):
                return None

            root = self.parser.metadata.root

            # Define namespaces for XML parsing
            namespaces = {
                "mzml": "http://psi.hupo.org/ms/mzml",
                "ims": "http://www.maldi-msi.org/download/imzml/imagingMS.obo",
            }

            x_size = None
            y_size = None

            # Search for cvParam elements with the pixel size accessions
            for cvparam in root.findall(".//mzml:cvParam", namespaces):
                accession = cvparam.get("accession")
                if accession == "IMS:1000046":  # pixel size x
                    x_size = float(cvparam.get("value", 0))
                elif accession == "IMS:1000047":  # pixel size y
                    y_size = float(cvparam.get("value", 0))

            if x_size is not None and y_size is not None:
                logger.info(
                    f"Detected pixel size from XML: x={x_size} μm, y={y_size} μm"
                )
                return (x_size, y_size)

        except Exception as e:
            logger.warning(f"Failed to parse XML metadata for pixel size: {e}")

        return None
