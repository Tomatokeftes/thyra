"""Decision tree for automatic resampling strategy selection."""

import logging
from typing import Any, Dict, Optional

from .types import AxisType, ResamplingMethod

logger = logging.getLogger(__name__)


class ResamplingDecisionTree:
    """Implements decision tree for resampling strategy selection based on
    instrument metadata."""

    def select_strategy(
        self, metadata: Optional[Dict[str, Any]] = None
    ) -> ResamplingMethod:
        """Automatically select appropriate resampling method based on
        instrument metadata.

        Currently implemented:
        - Bruker timsTOF detection -> NEAREST_NEIGHBOR (optimal for
          centroid data)
        - All other instruments -> NotImplementedError (to be implemented
          in future phases)

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
            For non-timsTOF instruments (to be implemented)
        """
        if metadata is None:
            raise NotImplementedError(
                "Automatic strategy selection without metadata not yet "
                "implemented. Currently only Bruker timsTOF detection is "
                "supported."
            )

        # Check for ImzML centroid spectrum detection (exact cvParam match)
        if self._is_imzml_centroid_spectrum(metadata):
            logger.info(
                "ImzML centroid spectrum detected, using NEAREST_NEIGHBOR " "strategy"
            )
            return ResamplingMethod.NEAREST_NEIGHBOR

        # Check Bruker GlobalMetadata for timsTOF detection (for .d files)
        if self._is_bruker_metadata(metadata):
            if self._detect_timstof_from_bruker_metadata(metadata):
                logger.info(
                    "timsTOF detected from Bruker metadata, using "
                    "NEAREST_NEIGHBOR strategy"
                )
                return ResamplingMethod.NEAREST_NEIGHBOR

        # For now, everything else is not implemented
        instrument_name = self._extract_instrument_name(metadata)
        if instrument_name:
            raise NotImplementedError(
                f"Automatic strategy selection for instrument "
                f"'{instrument_name}' not yet implemented. Currently only "
                f"Bruker timsTOF detection is supported. Please specify the "
                f"resampling method manually."
            )
        else:
            raise NotImplementedError(
                "Automatic strategy selection for non-timsTOF instruments not "
                "yet implemented. Currently only Bruker timsTOF detection is "
                "supported. Please specify the resampling method manually."
            )

    def select_axis_type(self, metadata: Optional[Dict[str, Any]] = None) -> AxisType:
        """Automatically select appropriate mass axis type based on
        instrument metadata.

        Parameters
        ----------
        metadata : Optional[Dict[str, Any]]
            Metadata dictionary containing instrument information

        Returns
        -------
        AxisType
            Recommended axis type for the instrument

        Raises
        ------
        NotImplementedError
            For non-timsTOF instruments (to be implemented)
        """
        if metadata is None:
            return AxisType.CONSTANT  # Default to uniform spacing

        # Check for ImzML centroid spectrum detection
        if self._is_imzml_centroid_spectrum(metadata):
            logger.info(
                "ImzML centroid spectrum detected, using REFLECTOR_TOF axis " "type"
            )
            return (
                AxisType.REFLECTOR_TOF
            )  # Most ImzML centroid data benefits from constant relative
            # resolution

        # Check Bruker GlobalMetadata for timsTOF detection
        if self._is_bruker_metadata(metadata):
            if self._detect_timstof_from_bruker_metadata(metadata):
                logger.info(
                    "timsTOF detected from Bruker metadata, using "
                    "REFLECTOR_TOF axis type"
                )
                return AxisType.REFLECTOR_TOF

        # Default to uniform spacing for unknown instruments
        logger.info("Unknown instrument, using uniform axis spacing")
        return AxisType.CONSTANT

    def _extract_instrument_name(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract instrument name from metadata."""
        # Common metadata keys for instrument information
        instrument_keys = [
            "instrument_name",
            "instrument",
            "instrument_model",
            "InstrumentName",
            "Instrument",
            "InstrumentModel",
            "ms_instrument_name",
            "ms_instrument",
            "device_name",
        ]

        for key in instrument_keys:
            if key in metadata and metadata[key]:
                return str(metadata[key]).strip()

        return None

    def _extract_spectrum_type(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract spectrum type from metadata (for ImzML files)."""
        # Common keys for spectrum type information
        spectrum_keys = [
            "spectrum_type",
            "data_type",
            "acquisition_mode",
            "SpectrumType",
            "DataType",
            "AcquisitionMode",
            "ms_spectrum_type",
            "spectrum_representation",
        ]

        for key in spectrum_keys:
            if key in metadata and metadata[key]:
                return str(metadata[key]).strip().lower()

        return None

    def _is_imzml_centroid_spectrum(self, metadata: Dict[str, Any]) -> bool:
        """Check for ImzML centroid spectrum from essential metadata."""
        # Check essential metadata for spectrum_type
        if "essential_metadata" in metadata:
            essential = metadata["essential_metadata"]
            if isinstance(essential, dict) and "spectrum_type" in essential:
                return essential["spectrum_type"] == "centroid spectrum"

        # Fallback: Look for cvParam with exact name="centroid spectrum"
        if "cvParams" in metadata:
            cv_params = metadata["cvParams"]
            if isinstance(cv_params, list):
                for param in cv_params:
                    if (
                        isinstance(param, dict)
                        and param.get("name") == "centroid spectrum"
                    ):
                        return True

        # Also check for spectrum_type containing exact match
        if "spectrum_type" in metadata:
            return metadata["spectrum_type"] == "centroid spectrum"

        return False

    def _is_bruker_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Check if metadata appears to be from Bruker instruments."""
        bruker_keys = [
            "GlobalMetadata",
            "AcquisitionKeys",
            "Method",
            "InstrumentFamily",
            "InstrumentName",
        ]

        return bool(any(key in metadata for key in bruker_keys))

    def _detect_timstof_from_bruker_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Detect timsTOF from Bruker-specific metadata structure."""
        # Check GlobalMetadata for specific timsTOF instrument name
        if "GlobalMetadata" in metadata:
            global_meta = metadata["GlobalMetadata"]

            # Check for specific instrument name "timsTOF Maldi 2"
            if "InstrumentName" in global_meta:
                instrument_name = str(global_meta["InstrumentName"]).strip()
                if instrument_name == "timsTOF Maldi 2":
                    return True

        return False
