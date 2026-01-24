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

        Only handles truly legacy formats not covered by DataCharacteristics.from_metadata():
        - cvParams list format for spectrum type
        - Root level spectrum_type (without essential_metadata wrapper)
        """
        if characteristics.spectrum_type is None:
            characteristics.spectrum_type = self._detect_spectrum_type_legacy(metadata)

    def _detect_spectrum_type_legacy(
        self, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Detect spectrum type from legacy metadata formats.

        Checks cvParams list and root-level spectrum_type key.
        """
        # Check cvParams list (legacy ImzML format)
        cv_params = metadata.get("cvParams")
        if isinstance(cv_params, list):
            for param in cv_params:
                if isinstance(param, dict):
                    name = param.get("name")
                    if name == SpectrumType.CENTROID:
                        return SpectrumType.CENTROID
                    if name == SpectrumType.PROFILE:
                        return SpectrumType.PROFILE

        # Check root level spectrum_type
        spectrum_type = metadata.get("spectrum_type")
        if spectrum_type in (SpectrumType.CENTROID, SpectrumType.PROFILE):
            return spectrum_type

        return None
