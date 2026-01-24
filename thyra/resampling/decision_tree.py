"""Decision tree for automatic resampling strategy selection.

This module provides the main interface for automatic resampling decisions.
It uses the Strategy pattern via InstrumentDetectorChain for extensible
instrument detection.
"""

import logging
from typing import Any, Dict, Optional

from .constants import SpectrumType
from .data_characteristics import DataCharacteristics
from .instrument_detectors import InstrumentDetectorChain
from .types import AxisType, ResamplingMethod

logger = logging.getLogger(__name__)


class ResamplingDecisionTree:
    """Decision tree for resampling strategy selection based on instrument metadata.

    This class provides the main interface for automatic resampling decisions.
    It uses DataCharacteristics to consolidate metadata and InstrumentDetectorChain
    to select the appropriate resampling strategy using the Strategy pattern.

    Example:
        >>> tree = ResamplingDecisionTree()
        >>> method = tree.select_strategy(metadata)
        >>> axis_type = tree.select_axis_type(metadata)
    """

    def __init__(self):
        """Initialize the decision tree with default detector chain."""
        self._detector_chain = InstrumentDetectorChain()

    def select_strategy(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> ResamplingMethod:
        """Automatically select appropriate resampling method.

        Parameters
        ----------
        metadata : Optional[Dict[str, Any]]
            Metadata dictionary containing instrument information

        Returns
        -------
        ResamplingMethod
            Selected resampling strategy

        Raises
        ------
        NotImplementedError
            When metadata is None (cannot auto-detect without data)
        """
        if metadata is None:
            raise NotImplementedError(
                "Automatic strategy selection requires metadata. "
                "Please provide metadata or specify the resampling method manually."
            )

        # Convert metadata to DataCharacteristics
        characteristics = DataCharacteristics.from_metadata(metadata)

        # Also check legacy metadata format for backwards compatibility
        self._enhance_characteristics_from_legacy(characteristics, metadata)

        # Use detector chain to find matching instrument
        return self._detector_chain.get_resampling_method(characteristics)

    def select_axis_type(self, metadata: Optional[Dict[str, Any]] = None) -> AxisType:
        """Automatically select appropriate mass axis type.

        Parameters
        ----------
        metadata : Optional[Dict[str, Any]]
            Metadata dictionary containing instrument information

        Returns
        -------
        AxisType
            Recommended axis type for the instrument
        """
        if metadata is None:
            logger.info("No metadata provided, using default CONSTANT axis type")
            return AxisType.CONSTANT

        # Convert metadata to DataCharacteristics
        characteristics = DataCharacteristics.from_metadata(metadata)

        # Also check legacy metadata format for backwards compatibility
        self._enhance_characteristics_from_legacy(characteristics, metadata)

        # Use detector chain to find matching instrument
        return self._detector_chain.get_axis_type(characteristics)

    def _enhance_characteristics_from_legacy(
        self, characteristics: DataCharacteristics, metadata: Dict[str, Any]
    ) -> None:
        """Enhance DataCharacteristics from legacy metadata formats.

        This provides backwards compatibility with existing metadata structures
        that may not match the new DataCharacteristics format.
        """
        # Check for legacy centroid spectrum detection
        if characteristics.spectrum_type is None:
            if self._is_imzml_centroid_spectrum_legacy(metadata):
                characteristics.spectrum_type = SpectrumType.CENTROID
            elif self._is_profile_spectrum_legacy(metadata):
                characteristics.spectrum_type = SpectrumType.PROFILE

        # Check for legacy timsTOF detection
        if not characteristics.is_timstof:
            if self._detect_timstof_from_bruker_metadata(metadata):
                characteristics.is_timstof = True
                characteristics.instrument_name = "timsTOF Maldi 2"

        # Check for legacy Rapiflex detection
        if not characteristics.is_rapiflex_format:
            if self._check_rapiflex_format_legacy(metadata):
                characteristics.is_rapiflex_format = True

        # Extract peak statistics if not present
        if characteristics.total_peaks is None or characteristics.n_spectra is None:
            essential = metadata.get("essential_metadata", {})
            if isinstance(essential, dict):
                if characteristics.total_peaks is None:
                    characteristics.total_peaks = essential.get("total_peaks")
                if characteristics.n_spectra is None:
                    characteristics.n_spectra = essential.get("n_spectra")

    # =========================================================================
    # Legacy detection methods (for backwards compatibility)
    # =========================================================================

    def _is_imzml_centroid_spectrum_legacy(self, metadata: Dict[str, Any]) -> bool:
        """Check for ImzML centroid spectrum from legacy metadata formats."""
        # Check essential metadata for spectrum_type
        if "essential_metadata" in metadata:
            essential = metadata["essential_metadata"]
            if isinstance(essential, dict) and "spectrum_type" in essential:
                return bool(essential["spectrum_type"] == SpectrumType.CENTROID)

        # Fallback: Look for cvParam with exact name
        if "cvParams" in metadata:
            cv_params = metadata["cvParams"]
            if isinstance(cv_params, list):
                for param in cv_params:
                    if (
                        isinstance(param, dict)
                        and param.get("name") == SpectrumType.CENTROID
                    ):
                        return True

        # Also check for spectrum_type at root level
        if "spectrum_type" in metadata:
            return bool(metadata["spectrum_type"] == SpectrumType.CENTROID)

        return False

    def _is_profile_spectrum_legacy(self, metadata: Dict[str, Any]) -> bool:
        """Check for profile spectrum from legacy metadata formats."""
        if "essential_metadata" in metadata:
            essential = metadata["essential_metadata"]
            if isinstance(essential, dict) and "spectrum_type" in essential:
                return bool(essential["spectrum_type"] == SpectrumType.PROFILE)

        if "spectrum_type" in metadata:
            return bool(metadata["spectrum_type"] == SpectrumType.PROFILE)

        return False

    def _detect_timstof_from_bruker_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Detect timsTOF from Bruker-specific metadata structure."""
        if "GlobalMetadata" in metadata:
            global_meta = metadata["GlobalMetadata"]
            if "InstrumentName" in global_meta:
                instrument_name = str(global_meta["InstrumentName"]).strip()
                if instrument_name == "timsTOF Maldi 2":
                    return True
        return False

    def _check_rapiflex_format_legacy(self, metadata: Dict[str, Any]) -> bool:
        """Check if format_specific indicates Rapiflex."""
        format_specific = metadata.get("format_specific", {})
        if isinstance(format_specific, dict):
            return bool(format_specific.get("format") == "Rapiflex")
        return False
