"""Decision tree for automatic resampling strategy selection.

This module provides the main interface for automatic resampling decisions.
It uses the Strategy pattern via InstrumentDetectorChain for extensible
instrument detection.
"""

import logging
from typing import Any, Dict, Optional

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

        # Use detector chain to find matching instrument
        return self._detector_chain.get_axis_type(characteristics)
