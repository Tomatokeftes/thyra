# thyra/metadata/extractors/imzml_extractor.py
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pyimzml.ImzMLParser import ImzMLParser

from ...core.base_extractor import MetadataExtractor
from ..types import ComprehensiveMetadata, EssentialMetadata

logger = logging.getLogger(__name__)


class ImzMLMetadataExtractor(MetadataExtractor):
    """ImzML-specific metadata extractor with optimized two-phase extraction."""

    def __init__(self, parser: ImzMLParser, imzml_path: Path):
        """Initialize ImzML metadata extractor.

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
        mass_range, total_peaks = self._get_mass_range_complete()
        pixel_size = self._extract_pixel_size_fast()
        n_spectra = len(coords)
        estimated_memory = self._estimate_memory(n_spectra)

        # Check for centroid spectrum
        spectrum_type = self._detect_centroid_spectrum()

        return EssentialMetadata(
            dimensions=dimensions,
            coordinate_bounds=coordinate_bounds,
            mass_range=mass_range,
            pixel_size=pixel_size,
            n_spectra=n_spectra,
            total_peaks=total_peaks,
            estimated_memory_gb=estimated_memory,
            source_path=str(self.imzml_path),
            spectrum_type=spectrum_type,  # Add spectrum type to essential
            # metadata
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
        return (
            int(max_coords[0]) + 1,
            int(max_coords[1]) + 1,
            int(max_coords[2]) + 1,
        )

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

    def _get_mass_range_complete(self) -> Tuple[Tuple[float, float], int]:
        """Complete mass range extraction by scanning ALL spectra.

        Required for resampling to ensure no m/z values are missed.
        Also counts total peaks for COO matrix pre-allocation.

        Returns:
            Tuple of ((min_mass, max_mass), total_peaks)
        """
        try:
            logger.info(
                "Scanning ALL spectra for complete mass range and peak count..."
            )

            n_spectra = len(self.parser.coordinates)
            min_mass = float("inf")
            max_mass = float("-inf")
            total_peaks = 0

            from tqdm import tqdm

            with tqdm(
                total=n_spectra,
                desc="Scanning mass range and counting peaks",
                unit="spectrum",
            ) as pbar:
                for idx in range(n_spectra):
                    try:
                        mzs, _ = self.parser.getspectrum(idx)
                        if len(mzs) > 0:
                            min_mass = min(min_mass, float(np.min(mzs)))
                            max_mass = max(max_mass, float(np.max(mzs)))
                            total_peaks += len(mzs)
                    except Exception as e:
                        logger.debug(f"Failed to read spectrum {idx}: {e}")
                        continue  # Skip problematic spectra
                    pbar.update(1)

            if min_mass == float("inf"):
                logger.warning("No valid spectra found")
                return ((0.0, 1000.0), 0)

            logger.info(f"Complete mass range: {min_mass:.2f} - {max_mass:.2f} m/z")
            logger.info(f"Total peaks: {total_peaks:,}")
            return ((min_mass, max_mass), total_peaks)

        except Exception as e:
            logger.error(f"Complete mass range scan failed: {e}")
            return ((0.0, 1000.0), 0)

    def get_mass_range_for_resampling(self) -> Tuple[float, float]:
        """Get accurate mass range required for resampling.

        This performs a complete scan of all spectra to ensure no m/z
        values are missed when building the resampled axis.
        """
        mass_range, _ = self._get_mass_range_complete()
        return mass_range

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
        # Rough estimate: assume average 1000 peaks per spectrum,
        # 8 bytes per float
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
            "file_mode": (
                "continuous"
                if getattr(self.parser, "continuous", False)
                else "processed"
            ),
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

        # Extract pixel size with full XML parsing if not found in fast
        # extraction
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
        """Extract raw metadata from imzmldict and spectrum cvParams."""
        raw_metadata = {}

        if hasattr(self.parser, "imzmldict") and self.parser.imzmldict:
            raw_metadata = dict(self.parser.imzmldict)

        # Extract spectrum-level cvParams for centroid detection
        cv_params = self._extract_spectrum_cvparams()
        if cv_params:
            raw_metadata["cvParams"] = cv_params

        return raw_metadata

    def _extract_spectrum_cvparams(self) -> Optional[List[Dict[str, Any]]]:
        """Extract cvParams from first spectrum for centroid detection."""
        try:
            if not hasattr(self.parser, "metadata") or not self.parser.metadata:
                return None

            # Look for spectrum-level cvParams in the metadata
            if hasattr(self.parser.metadata, "file_description"):
                file_desc = self.parser.metadata.file_description
                if hasattr(file_desc, "param_by_name"):
                    params = file_desc.param_by_name
                    cv_params = []

                    # Check for centroid spectrum in file description
                    for name, value in params.items():
                        cv_params.append({"name": name, "value": value})

                    return cv_params

            return None
        except Exception as e:
            logger.debug(f"Could not extract spectrum cvParams: {e}")
            return None

    def _detect_centroid_spectrum(self) -> Optional[str]:
        """Detect if this is a centroid spectrum by looking for MS:1000127."""
        try:
            # Method 1: Parse XML directly from file for MS:1000127
            result = self._check_xml_for_centroid()
            if result:
                return result

            # Method 2: Check if parser metadata has processed flag
            result = self._check_parser_metadata_for_centroid()
            if result:
                return result

            return None
        except Exception as e:
            logger.debug(f"Could not detect centroid spectrum: {e}")
            return None

    def _check_xml_for_centroid(self) -> Optional[str]:
        """Check XML for MS:1000127 centroid spectrum marker."""
        try:
            ET = self._get_xml_parser()
            tree = ET.parse(self.imzml_path)  # nosec B314
            root = tree.getroot()

            # Look for cvParam with accession MS:1000127
            for elem in root.iter():
                if elem.tag.endswith("cvParam"):
                    accession = elem.get("accession", "")
                    name = elem.get("name", "")
                    if accession == "MS:1000127" and name == "centroid spectrum":
                        logger.info("Detected centroid spectrum from MS:1000127")
                        return "centroid spectrum"
        except Exception as e:
            logger.debug(f"XML parsing method failed: {e}")
        return None

    def _get_xml_parser(self):
        """Get XML parser, preferring defusedxml for security."""
        try:
            # Use defusedxml for secure parsing
            import defusedxml.ElementTree as ET

            return ET
        except ImportError:
            # Fallback to standard library with warning
            import xml.etree.ElementTree as ET  # nosec B405

            logger.warning("defusedxml not available, using xml.etree.ElementTree")
            return ET

    def _check_parser_metadata_for_centroid(self) -> Optional[str]:
        """Check parser metadata for processed flag indicating centroid data."""
        if not (hasattr(self.parser, "metadata") and self.parser.metadata):
            return None

        if not hasattr(self.parser.metadata, "file_description"):
            return None

        file_desc = self.parser.metadata.file_description
        if not hasattr(file_desc, "param_by_name"):
            return None

        params = file_desc.param_by_name
        # If it's processed data, it's likely centroided
        if params.get("processed", False):
            logger.info("Assuming centroid spectrum for processed ImzML data")
            return "centroid spectrum"

        return None

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
                logger.info(f"Detected pixel size from XML: x={x_size}μm, y={y_size}μm")
                return (x_size, y_size)

        except Exception as e:
            logger.warning(f"Failed to parse XML metadata for pixel size: {e}")

        return None
