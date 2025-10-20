"""Nearest neighbor resampling strategy for centroid data.

This strategy is optimal for centroid data (e.g., from timsTOF
instruments) where each peak represents a discrete mass value.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from .base import ResamplingStrategy, Spectrum


class NearestNeighborStrategy(ResamplingStrategy):
    """Nearest neighbor resampling strategy for centroid data."""

    def resample(
        self, spectrum: Spectrum, target_axis: npt.NDArray[np.floating[Any]]
    ) -> Spectrum:
        """Resample spectrum using nearest neighbor interpolation.

        For each target m/z value, finds the nearest original m/z value
        and assigns its intensity. This preserves the discrete nature
        of centroid data.

        Parameters
        ----------
        spectrum : Spectrum
            Input spectrum to resample
        target_axis : npt.NDArray[np.floating[Any]]
            Target mass axis values

        Returns
        -------
        Spectrum
            Resampled spectrum with target_axis as mz values
        """
        if len(spectrum.mz) == 0:
            # Handle empty spectrum
            return Spectrum(
                mz=target_axis.copy(),
                intensity=np.zeros_like(target_axis),
                coordinates=spectrum.coordinates,
                metadata=spectrum.metadata,
            )

        # Find nearest neighbors for each target m/z
        # np.searchsorted finds insertion points, we need to check both sides
        indices = np.searchsorted(spectrum.mz, target_axis, side="left")

        # Clip indices to valid range
        indices = np.clip(indices, 0, len(spectrum.mz) - 1)

        # For each target point, check if the left or right neighbor is closer
        resampled_intensity = np.zeros_like(target_axis)

        for i, (target_mz, idx) in enumerate(zip(target_axis, indices)):
            # Check boundaries
            if idx == 0:
                # First point, use it
                nearest_idx = 0
            elif idx >= len(spectrum.mz):
                # Beyond last point, use last point
                nearest_idx = len(spectrum.mz) - 1
            else:
                # Check which neighbor is closer
                left_dist = abs(target_mz - spectrum.mz[idx - 1])
                right_dist = abs(target_mz - spectrum.mz[idx])

                if left_dist <= right_dist:
                    nearest_idx = idx - 1
                else:
                    nearest_idx = idx

            resampled_intensity[i] = spectrum.intensity[nearest_idx]

        return Spectrum(
            mz=target_axis.copy(),
            intensity=resampled_intensity,
            coordinates=spectrum.coordinates,
            metadata=spectrum.metadata,
        )
