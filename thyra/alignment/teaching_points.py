# thyra/alignment/teaching_points.py
"""Teaching point alignment for FlexImaging optical-MSI registration.

This module handles the alignment between optical images and MSI data
using teaching point calibration data from FlexImaging .mis files.

Coordinate Systems:
- Image pixels: (x, y) in the optical reference image (origin at top-left)
- Stage coordinates (teaching): From teaching point calibration
- Stage coordinates (poslog): From acquisition position log
- MSI raster: (x, y) grid positions in the MSI dataset (0-based)

Key Challenge:
FlexImaging uses different coordinate frames for teaching (image calibration)
and acquisition (stage movement). These frames may have large offsets that
cannot be reliably computed without additional reference data.

The module provides:
1. Reliable image <-> teaching stage transformation from teaching points
2. Estimated MSI <-> image transformation (may need manual verification)
3. Methods to manually specify alignment offsets if automatic alignment fails
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .affine import AffineTransform

logger = logging.getLogger(__name__)


@dataclass
class TeachingPoint:
    """A single teaching point correspondence.

    Attributes:
        image_x: X coordinate in image pixels
        image_y: Y coordinate in image pixels
        stage_x: X coordinate in stage units (micrometers)
        stage_y: Y coordinate in stage units (micrometers)
    """

    image_x: int
    image_y: int
    stage_x: int
    stage_y: int

    @classmethod
    def from_dict(cls, data: Dict[str, Tuple[int, int]]) -> "TeachingPoint":
        """Create from parsed dictionary format.

        Args:
            data: Dictionary with 'image' and 'stage' tuples

        Returns:
            TeachingPoint instance
        """
        img = data["image"]
        stage = data["stage"]
        return cls(
            image_x=img[0],
            image_y=img[1],
            stage_x=stage[0],
            stage_y=stage[1],
        )


@dataclass
class RasterPosition:
    """A position from the poslog with raster and physical coordinates.

    Attributes:
        raster_x: Raster grid X coordinate
        raster_y: Raster grid Y coordinate
        phys_x: Physical stage X coordinate (micrometers)
        phys_y: Physical stage Y coordinate (micrometers)
    """

    raster_x: int
    raster_y: int
    phys_x: float
    phys_y: float


@dataclass
class AlignmentResult:
    """Result of teaching point alignment computation.

    Attributes:
        image_to_stage: Affine transform from image pixels to stage coords
        stage_to_image: Inverse transform (stage to image pixels)
        msi_to_image: Transform from MSI raster to image pixels
        image_to_msi: Transform from image pixels to MSI raster
        stage_offset: Estimated offset between teaching and poslog stages
        rmse: Root mean square error of teaching point fit
        warnings: List of alignment warnings
    """

    image_to_stage: AffineTransform
    stage_to_image: AffineTransform
    msi_to_image: Optional[AffineTransform] = None
    image_to_msi: Optional[AffineTransform] = None
    stage_offset: Optional[Tuple[float, float]] = None
    rmse: float = 0.0
    warnings: List[str] = field(default_factory=list)


class TeachingPointAlignment:
    """Computes alignment between optical images and MSI data.

    This class handles the coordinate system transformations needed to
    align optical images with MSI raster data using FlexImaging teaching
    points.

    The workflow:
    1. Parse teaching points from .mis metadata
    2. Compute image -> stage affine transformation
    3. Determine offset between teaching stage and poslog stage coords
    4. Compute final image -> MSI raster transformation

    Example:
        >>> aligner = TeachingPointAlignment()
        >>> result = aligner.compute_alignment(
        ...     teaching_points=reader.mis_metadata['teaching_points'],
        ...     poslog_positions=reader._positions,
        ...     raster_step=(20.0, 20.0),
        ... )
        >>> # Transform optical image coordinates to MSI raster
        >>> msi_coords = result.image_to_msi.transform_point(img_x, img_y)
    """

    def __init__(self):
        """Initialize the alignment calculator."""
        pass

    def compute_alignment(
        self,
        teaching_points: List[Dict[str, Tuple[int, int]]],
        poslog_positions: Optional[List[Dict[str, Any]]] = None,
        raster_step: Tuple[float, float] = (20.0, 20.0),
        raster_offset: Tuple[int, int] = (0, 0),
    ) -> AlignmentResult:
        """Compute alignment transformations from teaching points.

        Args:
            teaching_points: List of teaching point dictionaries with
                'image' and 'stage' keys containing (x, y) tuples
            poslog_positions: Optional list of position dictionaries from
                poslog parsing, used to estimate stage coordinate offset
            raster_step: (step_x, step_y) raster step size in micrometers
            raster_offset: (offset_x, offset_y) offset of first raster position

        Returns:
            AlignmentResult with computed transformations
        """
        warnings: List[str] = []

        # Parse teaching points
        if len(teaching_points) < 3:
            n_pts = len(teaching_points)
            raise ValueError(f"At least 3 teaching points required, got {n_pts}")

        points = [TeachingPoint.from_dict(tp) for tp in teaching_points]
        logger.info(f"Processing {len(points)} teaching points")

        # Compute image -> stage transformation
        image_to_stage = self._compute_image_to_stage(points)

        # Compute RMSE for quality assessment
        src_pts = [(float(p.image_x), float(p.image_y)) for p in points]
        dst_pts = [(float(p.stage_x), float(p.stage_y)) for p in points]
        rmse, _ = image_to_stage.compute_residuals(src_pts, dst_pts)

        if rmse > 10.0:  # More than 10 um error
            warnings.append(
                f"Teaching point fit has high error (RMSE={rmse:.2f} um). "
                "Alignment may be inaccurate."
            )

        logger.info(f"Image->Stage RMSE: {rmse:.4f} um")

        # Compute inverse
        stage_to_image = image_to_stage.inverse()

        # Try to compute stage offset and MSI transformations
        stage_offset = None
        msi_to_image = None
        image_to_msi = None

        if poslog_positions and len(poslog_positions) > 0:
            stage_offset = self._estimate_stage_offset(
                points, poslog_positions, raster_step
            )

            if stage_offset is not None:
                logger.info(
                    f"Estimated stage offset: ({stage_offset[0]:.1f}, "
                    f"{stage_offset[1]:.1f}) um"
                )

                # Compute MSI <-> image transformations
                msi_to_image, image_to_msi = self._compute_msi_transforms(
                    image_to_stage,
                    stage_to_image,
                    stage_offset,
                    raster_step,
                    raster_offset,
                )
            else:
                warnings.append(
                    "Could not determine stage coordinate offset. "
                    "MSI-to-image transform may require manual calibration."
                )

        return AlignmentResult(
            image_to_stage=image_to_stage,
            stage_to_image=stage_to_image,
            msi_to_image=msi_to_image,
            image_to_msi=image_to_msi,
            stage_offset=stage_offset,
            rmse=rmse,
            warnings=warnings,
        )

    def _compute_image_to_stage(self, points: List[TeachingPoint]) -> AffineTransform:
        """Compute affine transformation from image pixels to stage coords.

        Args:
            points: List of teaching points

        Returns:
            AffineTransform from image to stage coordinates
        """
        src_points = [(float(p.image_x), float(p.image_y)) for p in points]
        dst_points = [(float(p.stage_x), float(p.stage_y)) for p in points]

        return AffineTransform.from_points(src_points, dst_points)

    def _estimate_stage_offset(
        self,
        teaching_points: List[TeachingPoint],
        poslog_positions: List[Dict[str, Any]],
        raster_step: Tuple[float, float],
    ) -> Optional[Tuple[float, float]]:
        """Estimate offset between teaching stage and poslog stage coords.

        This attempts to find the translation offset between the two
        coordinate systems by analyzing the relationship between
        raster positions and their physical coordinates.

        The poslog records: raster (x, y) -> physical stage (px, py)
        The relationship should be: px = raster_x * step_x + offset_x

        Args:
            teaching_points: Teaching point data
            poslog_positions: Position log entries
            raster_step: Raster step size (step_x, step_y) in um

        Returns:
            Estimated (offset_x, offset_y) or None if cannot determine
        """
        if not poslog_positions:
            return None

        # Extract poslog data
        raster_x_vals = []
        raster_y_vals = []
        phys_x_vals = []
        phys_y_vals = []

        for pos in poslog_positions:
            raster_x_vals.append(pos["raster_x"])
            raster_y_vals.append(pos["raster_y"])
            phys_x_vals.append(pos["phys_x"])
            phys_y_vals.append(pos["phys_y"])

        raster_x = np.array(raster_x_vals)
        raster_y = np.array(raster_y_vals)
        phys_x = np.array(phys_x_vals)
        phys_y = np.array(phys_y_vals)

        step_x, step_y = raster_step

        # Estimate the origin offset for poslog coordinates
        # phys = raster * step + poslog_origin
        # poslog_origin_x = phys_x - raster_x * step_x
        poslog_origin_x = float(np.median(phys_x - raster_x * step_x))
        poslog_origin_y = float(np.median(phys_y - raster_y * step_y))

        logger.debug(
            f"Poslog coordinate origin: "
            f"({poslog_origin_x:.1f}, {poslog_origin_y:.1f})"
        )

        # Get teaching point stage coordinate ranges
        teaching_x = [p.stage_x for p in teaching_points]
        teaching_y = [p.stage_y for p in teaching_points]

        teaching_center_x = np.mean(teaching_x)
        teaching_center_y = np.mean(teaching_y)

        # Get poslog physical coordinate ranges
        poslog_center_x = np.mean(phys_x)
        poslog_center_y = np.mean(phys_y)

        logger.debug(
            f"Teaching stage center: "
            f"({teaching_center_x:.1f}, {teaching_center_y:.1f})"
        )
        logger.debug(
            f"Poslog physical center: "
            f"({poslog_center_x:.1f}, {poslog_center_y:.1f})"
        )

        # The offset is the difference between poslog physical coords
        # and teaching stage coords for the same physical location
        # offset = poslog_physical - teaching_stage
        offset_x = poslog_center_x - teaching_center_x
        offset_y = poslog_center_y - teaching_center_y

        logger.info(
            f"Stage coordinate offset (poslog - teaching): "
            f"({offset_x:.1f}, {offset_y:.1f}) um"
        )

        return (offset_x, offset_y)

    def _compute_msi_transforms(
        self,
        image_to_stage: AffineTransform,
        stage_to_image: AffineTransform,
        stage_offset: Tuple[float, float],
        raster_step: Tuple[float, float],
        raster_offset: Tuple[int, int],
    ) -> Tuple[AffineTransform, AffineTransform]:
        """Compute transformations between MSI raster and image coordinates.

        The chain of transformations:
        MSI raster -> poslog stage -> teaching stage -> image

        Args:
            image_to_stage: Transform from image pixels to teaching stage
            stage_to_image: Inverse transform
            stage_offset: (offset_x, offset_y) between poslog and teaching
            raster_step: (step_x, step_y) raster step in um
            raster_offset: (offset_x, offset_y) of first raster position

        Returns:
            Tuple of (msi_to_image, image_to_msi) transforms
        """
        step_x, step_y = raster_step
        offset_x, offset_y = stage_offset

        # MSI raster (0-based) to poslog stage coordinates
        # poslog_x = (raster_x + raster_offset_x) * step_x + poslog_origin_x
        # But we work with normalized 0-based raster coords, so:
        # poslog_x = raster_x * step_x + origin_x (origin accounts for offset)

        # To go from poslog stage to teaching stage:
        # teaching_x = poslog_x - offset_x

        # Combined: MSI raster -> teaching stage
        # teaching_x = raster_x * step_x + origin_x - offset_x
        # teaching_y = raster_y * step_y + origin_y - offset_y

        # Build MSI -> teaching stage transform
        # (using 0-based raster coordinates)
        msi_to_teaching = AffineTransform.from_scale_translate(
            scale_x=step_x,
            scale_y=step_y,
            tx=-offset_x,  # Translate from poslog to teaching space
            ty=-offset_y,
        )

        # Chain: MSI -> teaching stage -> image
        msi_to_image = msi_to_teaching.compose(stage_to_image)

        # Inverse: image -> MSI
        image_to_msi = msi_to_image.inverse()

        return msi_to_image, image_to_msi

    def validate_alignment(
        self,
        result: AlignmentResult,
        image_shape: Tuple[int, int],
        msi_shape: Tuple[int, int],
    ) -> List[str]:
        """Validate alignment by checking if coordinates map sensibly.

        Args:
            result: AlignmentResult to validate
            image_shape: (height, width) of optical image
            msi_shape: (height, width) of MSI raster

        Returns:
            List of validation warnings (empty if OK)
        """
        warnings = []
        img_h, img_w = image_shape
        msi_h, msi_w = msi_shape

        if result.msi_to_image is None:
            warnings.append("MSI-to-image transform not available")
            return warnings

        # Check corners of MSI raster map to within image bounds
        corners = [
            (0, 0),
            (msi_w - 1, 0),
            (0, msi_h - 1),
            (msi_w - 1, msi_h - 1),
        ]

        for cx, cy in corners:
            ix, iy = result.msi_to_image.transform_point(cx, cy)

            # Allow some margin outside image bounds
            margin = 0.2  # 20% margin
            if ix < -img_w * margin or ix > img_w * (1 + margin):
                warnings.append(
                    f"MSI corner ({cx}, {cy}) maps to image X={ix:.0f}, "
                    f"outside valid range [0, {img_w}]"
                )
            if iy < -img_h * margin or iy > img_h * (1 + margin):
                warnings.append(
                    f"MSI corner ({cx}, {cy}) maps to image Y={iy:.0f}, "
                    f"outside valid range [0, {img_h}]"
                )

        return warnings
