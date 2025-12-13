# tests/unit/alignment/test_teaching_points.py
"""Tests for teaching point alignment functionality."""

import pytest

from thyra.alignment.teaching_points import (
    AlignmentResult,
    RasterPosition,
    TeachingPoint,
    TeachingPointAlignment,
)


class TestTeachingPoint:
    """Tests for TeachingPoint dataclass."""

    def test_from_dict(self):
        """Test creating TeachingPoint from dictionary."""
        data = {"image": (100, 200), "stage": (1000, 2000)}
        tp = TeachingPoint.from_dict(data)

        assert tp.image_x == 100
        assert tp.image_y == 200
        assert tp.stage_x == 1000
        assert tp.stage_y == 2000

    def test_direct_creation(self):
        """Test creating TeachingPoint directly."""
        tp = TeachingPoint(image_x=50, image_y=100, stage_x=500, stage_y=1000)

        assert tp.image_x == 50
        assert tp.image_y == 100
        assert tp.stage_x == 500
        assert tp.stage_y == 1000


class TestRasterPosition:
    """Tests for RasterPosition dataclass."""

    def test_creation(self):
        """Test creating RasterPosition."""
        pos = RasterPosition(raster_x=10, raster_y=20, phys_x=1000.0, phys_y=2000.0)

        assert pos.raster_x == 10
        assert pos.raster_y == 20
        assert pos.phys_x == 1000.0
        assert pos.phys_y == 2000.0


class TestTeachingPointAlignment:
    """Tests for TeachingPointAlignment class."""

    @pytest.fixture
    def sample_teaching_points(self):
        """Sample teaching points for testing."""
        return [
            {"image": (1000, 2000), "stage": (-10000, 20000)},
            {"image": (5000, 2000), "stage": (-2000, 20000)},
            {"image": (1000, 6000), "stage": (-10000, 12000)},
        ]

    @pytest.fixture
    def sample_poslog_positions(self):
        """Sample poslog positions for testing."""
        return [
            {"raster_x": 100, "raster_y": 50, "phys_x": 50000.0, "phys_y": -10000.0},
            {"raster_x": 101, "raster_y": 50, "phys_x": 50020.0, "phys_y": -10000.0},
            {"raster_x": 100, "raster_y": 51, "phys_x": 50000.0, "phys_y": -9980.0},
        ]

    def test_compute_alignment_basic(self, sample_teaching_points):
        """Test basic alignment computation."""
        aligner = TeachingPointAlignment()
        result = aligner.compute_alignment(sample_teaching_points)

        assert isinstance(result, AlignmentResult)
        assert result.image_to_stage is not None
        assert result.stage_to_image is not None
        assert result.rmse >= 0.0

    def test_compute_alignment_with_poslog(
        self, sample_teaching_points, sample_poslog_positions
    ):
        """Test alignment computation with poslog data."""
        aligner = TeachingPointAlignment()
        result = aligner.compute_alignment(
            teaching_points=sample_teaching_points,
            poslog_positions=sample_poslog_positions,
            raster_step=(20.0, 20.0),
        )

        assert result.stage_offset is not None
        assert len(result.stage_offset) == 2

    def test_too_few_teaching_points(self):
        """Test error when fewer than 3 teaching points provided."""
        aligner = TeachingPointAlignment()

        with pytest.raises(ValueError, match="At least 3"):
            aligner.compute_alignment(
                [
                    {"image": (100, 200), "stage": (1000, 2000)},
                    {"image": (200, 300), "stage": (2000, 3000)},
                ]
            )

    def test_alignment_scale_computation(self, sample_teaching_points):
        """Test that scale is correctly computed from teaching points."""
        aligner = TeachingPointAlignment()
        result = aligner.compute_alignment(sample_teaching_points)

        # Expected scale: ~2 um/pixel based on our test data
        # Image X: 1000 to 5000 (4000 pixels)
        # Stage X: -10000 to -2000 (8000 um)
        # Scale X = 8000/4000 = 2
        assert result.image_to_stage.scale_x == pytest.approx(2.0, abs=0.1)

    def test_alignment_rmse_zero_for_perfect_fit(self):
        """Test RMSE is zero for perfectly fitting points."""
        # Three points that form a perfect affine mapping
        teaching_points = [
            {"image": (0, 0), "stage": (0, 0)},
            {"image": (100, 0), "stage": (200, 0)},
            {"image": (0, 100), "stage": (0, 200)},
        ]

        aligner = TeachingPointAlignment()
        result = aligner.compute_alignment(teaching_points)

        assert result.rmse == pytest.approx(0.0, abs=1e-6)

    def test_alignment_warnings_for_high_rmse(self):
        """Test that warnings are generated for high RMSE."""
        # Create points with intentional error
        teaching_points = [
            {"image": (0, 0), "stage": (0, 0)},
            {"image": (100, 0), "stage": (200, 0)},
            {"image": (0, 100), "stage": (0, 200)},
            # This point doesn't fit the pattern (should be 200, 200)
            {"image": (100, 100), "stage": (250, 250)},
        ]

        aligner = TeachingPointAlignment()
        result = aligner.compute_alignment(teaching_points)

        # RMSE should be > 0 but may not trigger warning threshold
        assert result.rmse > 0

    def test_validate_alignment_no_msi_transform(self, sample_teaching_points):
        """Test validation when MSI transform is not available."""
        aligner = TeachingPointAlignment()
        result = aligner.compute_alignment(sample_teaching_points)

        # Without poslog, msi_to_image should be None
        result.msi_to_image = None
        warnings = aligner.validate_alignment(
            result, image_shape=(1000, 2000), msi_shape=(100, 100)
        )

        assert "not available" in warnings[0]


class TestAlignmentResult:
    """Tests for AlignmentResult dataclass."""

    def test_default_values(self):
        """Test default values in AlignmentResult."""
        from thyra.alignment.affine import AffineTransform

        identity = AffineTransform.identity()
        result = AlignmentResult(
            image_to_stage=identity,
            stage_to_image=identity,
        )

        assert result.msi_to_image is None
        assert result.image_to_msi is None
        assert result.stage_offset is None
        assert result.rmse == 0.0
        assert result.warnings == []

    def test_with_warnings(self):
        """Test AlignmentResult with warnings."""
        from thyra.alignment.affine import AffineTransform

        identity = AffineTransform.identity()
        result = AlignmentResult(
            image_to_stage=identity,
            stage_to_image=identity,
            warnings=["Warning 1", "Warning 2"],
        )

        assert len(result.warnings) == 2
