# tests/unit/alignment/test_affine.py
"""Tests for affine transformation utilities."""

import numpy as np
import pytest

from thyra.alignment.affine import AffineTransform


class TestAffineTransform:
    """Tests for the AffineTransform class."""

    def test_identity_transform(self):
        """Test identity transformation."""
        t = AffineTransform.identity()

        # Check matrix is identity
        np.testing.assert_array_almost_equal(t.matrix, np.eye(3))

        # Check properties
        assert t.scale_x == pytest.approx(1.0)
        assert t.scale_y == pytest.approx(1.0)
        assert t.rotation == pytest.approx(0.0)
        assert t.translation == (0.0, 0.0)

    def test_scale_translate_transform(self):
        """Test creating transform from scale and translation."""
        t = AffineTransform.from_scale_translate(
            scale_x=2.0, scale_y=3.0, tx=10.0, ty=20.0
        )

        assert t.scale_x == pytest.approx(2.0)
        assert t.scale_y == pytest.approx(3.0)
        assert t.translation == pytest.approx((10.0, 20.0))

    def test_transform_single_point(self):
        """Test transforming a single point."""
        t = AffineTransform.from_scale_translate(
            scale_x=2.0, scale_y=3.0, tx=10.0, ty=20.0
        )

        # Point (5, 5) -> (2*5 + 10, 3*5 + 20) = (20, 35)
        x, y = t.transform_point(5.0, 5.0)
        assert x == pytest.approx(20.0)
        assert y == pytest.approx(35.0)

    def test_transform_multiple_points(self):
        """Test transforming multiple points at once."""
        t = AffineTransform.from_scale_translate(
            scale_x=2.0, scale_y=1.0, tx=0.0, ty=0.0
        )

        points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        transformed = t.transform_points(points)

        expected = np.array([[2.0, 2.0], [6.0, 4.0], [10.0, 6.0]])
        np.testing.assert_array_almost_equal(transformed, expected)

    def test_inverse_transform(self):
        """Test inverse transformation."""
        t = AffineTransform.from_scale_translate(
            scale_x=2.0, scale_y=3.0, tx=10.0, ty=20.0
        )
        t_inv = t.inverse()

        # Transform point and then inverse should return original
        x, y = 5.0, 7.0
        x1, y1 = t.transform_point(x, y)
        x2, y2 = t_inv.transform_point(x1, y1)

        assert x2 == pytest.approx(x)
        assert y2 == pytest.approx(y)

    def test_compose_transforms(self):
        """Test composing two transformations."""
        # First: scale by 2
        t1 = AffineTransform.from_scale_translate(
            scale_x=2.0, scale_y=2.0, tx=0.0, ty=0.0
        )
        # Second: translate by (10, 10)
        t2 = AffineTransform.from_scale_translate(
            scale_x=1.0, scale_y=1.0, tx=10.0, ty=10.0
        )

        # Composed: first scale, then translate
        composed = t1.compose(t2)

        # Point (5, 5) -> scale -> (10, 10) -> translate -> (20, 20)
        x, y = composed.transform_point(5.0, 5.0)
        assert x == pytest.approx(20.0)
        assert y == pytest.approx(20.0)

    def test_from_points_minimum_points(self):
        """Test that at least 3 points are required."""
        src = [(0, 0), (1, 0)]
        dst = [(0, 0), (2, 0)]

        with pytest.raises(ValueError, match="At least 3 point pairs"):
            AffineTransform.from_points(src, dst)

    def test_from_points_mismatched_counts(self):
        """Test that source and destination counts must match."""
        # Both have >= 3 points but different counts
        src = [(0, 0), (1, 0), (0, 1), (1, 1)]
        dst = [(0, 0), (2, 0), (0, 2)]

        with pytest.raises(ValueError, match="must match"):
            AffineTransform.from_points(src, dst)

    def test_from_points_scale_only(self):
        """Test computing affine from points with pure scaling."""
        # Points that represent 2x scaling
        src = [(0, 0), (1, 0), (0, 1)]
        dst = [(0, 0), (2, 0), (0, 2)]

        t = AffineTransform.from_points(src, dst)

        assert t.scale_x == pytest.approx(2.0, abs=0.01)
        assert t.scale_y == pytest.approx(2.0, abs=0.01)
        assert t.rotation == pytest.approx(0.0, abs=0.1)

    def test_from_points_with_translation(self):
        """Test computing affine from points with translation."""
        # Points with translation of (10, 20)
        src = [(0, 0), (1, 0), (0, 1)]
        dst = [(10, 20), (11, 20), (10, 21)]

        t = AffineTransform.from_points(src, dst)

        x, y = t.transform_point(0, 0)
        assert x == pytest.approx(10.0, abs=0.01)
        assert y == pytest.approx(20.0, abs=0.01)

    def test_from_points_real_world_teaching_points(self):
        """Test with realistic teaching point data (scale ~2 um/pixel)."""
        # Simulated teaching points: image pixels -> stage um
        src = [(1000, 2000), (5000, 2000), (1000, 6000)]
        dst = [(-10000, 20000), (-2000, 20000), (-10000, 12000)]

        t = AffineTransform.from_points(src, dst)

        # Scale magnitude should be approximately 2 um/pixel
        assert t.scale_x == pytest.approx(2.0, abs=0.1)
        # Y scale magnitude is 2 (scale_y returns magnitude)
        assert t.scale_y == pytest.approx(2.0, abs=0.1)
        # The actual Y axis is flipped (negative in matrix)
        assert t.matrix[1, 1] == pytest.approx(-2.0, abs=0.1)

    def test_compute_residuals(self):
        """Test residual computation for validation."""
        t = AffineTransform.from_scale_translate(
            scale_x=2.0, scale_y=2.0, tx=0.0, ty=0.0
        )

        src = [(1, 1), (2, 2), (3, 3)]
        # Perfect fit destinations
        dst = [(2, 2), (4, 4), (6, 6)]

        rmse, residuals = t.compute_residuals(src, dst)

        assert rmse == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_array_almost_equal(residuals, [0, 0, 0])

    def test_compute_residuals_with_error(self):
        """Test residual computation with intentional error."""
        t = AffineTransform.from_scale_translate(
            scale_x=2.0, scale_y=2.0, tx=0.0, ty=0.0
        )

        src = [(1, 1), (2, 2)]
        # One point off by sqrt(2) (1 unit in x and y)
        dst = [(2, 2), (5, 5)]  # Second point is off by (1, 1)

        rmse, residuals = t.compute_residuals(src, dst)

        assert residuals[0] == pytest.approx(0.0, abs=1e-10)
        assert residuals[1] == pytest.approx(np.sqrt(2), abs=1e-10)

    def test_to_spatialdata_matrix(self):
        """Test conversion to SpatialData format."""
        t = AffineTransform.from_scale_translate(
            scale_x=2.0, scale_y=3.0, tx=10.0, ty=20.0
        )

        matrix = t.to_spatialdata_matrix()

        # Should be a copy, not the original
        assert matrix is not t.matrix
        np.testing.assert_array_equal(matrix, t.matrix)

    def test_repr(self):
        """Test string representation."""
        t = AffineTransform.identity()
        repr_str = repr(t)

        assert "AffineTransform" in repr_str
        assert "scale" in repr_str
        assert "rotation" in repr_str
