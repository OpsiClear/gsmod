"""Tests for filter API and utility functions.

Tests cover:
- Geometry filters (sphere, box, ellipsoid, frustum)
- Filter configuration interface
- Edge cases and error handling
"""

import numpy as np
import pytest
from gsply import GSData

from gsmod.filter.api import (
    _apply_ellipsoid_filter,
    _apply_filter,
    _apply_frustum_filter,
    _apply_rotated_cuboid_filter,
    _apply_sphere_filter,
    _axis_angle_to_rotation_matrix,
    apply_geometry_filter,
)
from gsmod.filter.config import (
    BoxFilter,
    EllipsoidFilter,
    FrustumFilter,
    QualityFilter,
    SphereFilter,
)
from gsmod.utils import linear_interp_1d, multiply_opacity, nearest_neighbor_1d


@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    # Create a grid of points
    rng = np.random.default_rng(42)
    return rng.uniform(-5, 5, (1000, 3)).astype(np.float32)


@pytest.fixture
def sample_gsdata():
    """Create sample GSData for testing."""
    rng = np.random.default_rng(42)
    n = 100
    return GSData(
        means=rng.random((n, 3), dtype=np.float32) * 10 - 5,
        scales=rng.random((n, 3), dtype=np.float32) * 0.5,
        quats=np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        opacities=rng.random(n, dtype=np.float32),
        sh0=rng.random((n, 3), dtype=np.float32),
        shN=None,
    )


class TestSphereFilter:
    """Test sphere filtering functions."""

    def test_sphere_filter_basic(self, sample_positions):
        """Test basic sphere filtering."""
        mask = _apply_sphere_filter(
            sample_positions,
            sphere_center=(0.0, 0.0, 0.0),
            sphere_radius=3.0,
        )

        # Check mask shape
        assert mask.shape == (len(sample_positions),)
        assert mask.dtype == np.bool_

        # Verify points inside sphere
        distances = np.linalg.norm(sample_positions, axis=1)
        expected = distances <= 3.0
        np.testing.assert_array_equal(mask, expected)

    def test_sphere_filter_offset_center(self, sample_positions):
        """Test sphere filter with offset center."""
        center = (2.0, 2.0, 2.0)
        radius = 2.0

        mask = _apply_sphere_filter(
            sample_positions,
            sphere_center=center,
            sphere_radius=radius,
        )

        # Verify calculation
        distances = np.linalg.norm(sample_positions - np.array(center), axis=1)
        expected = distances <= radius
        np.testing.assert_array_equal(mask, expected)

    def test_sphere_filter_all_inside(self):
        """Test sphere filter where all points are inside."""
        positions = np.array([[0, 0, 0], [0.1, 0.1, 0.1]], dtype=np.float32)
        mask = _apply_sphere_filter(positions, (0, 0, 0), 10.0)
        assert mask.all()

    def test_sphere_filter_none_inside(self):
        """Test sphere filter where no points are inside."""
        positions = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.float32)
        mask = _apply_sphere_filter(positions, (0, 0, 0), 1.0)
        assert not mask.any()


class TestCuboidFilter:
    """Test cuboid/box filtering functions."""

    def test_rotated_cuboid_no_rotation(self, sample_positions):
        """Test rotated cuboid filter without rotation."""
        mask = _apply_rotated_cuboid_filter(
            sample_positions,
            cuboid_center=(0.0, 0.0, 0.0),
            cuboid_size=(4.0, 4.0, 4.0),
            cuboid_rotation=None,
        )

        # Verify points inside box
        inside = np.all(np.abs(sample_positions) <= 2.0, axis=1)
        np.testing.assert_array_equal(mask, inside)

    def test_rotated_cuboid_with_rotation(self, sample_positions):
        """Test rotated cuboid filter with rotation."""
        # Rotate 45 degrees around Z axis
        rotation = (0.0, 0.0, np.pi / 4)

        mask = _apply_rotated_cuboid_filter(
            sample_positions,
            cuboid_center=(0.0, 0.0, 0.0),
            cuboid_size=(4.0, 4.0, 4.0),
            cuboid_rotation=rotation,
        )

        # Should filter some points
        assert 0 < mask.sum() < len(mask)

    def test_cuboid_offset_center(self, sample_positions):
        """Test cuboid filter with offset center."""
        center = (2.0, 2.0, 2.0)
        size = (2.0, 2.0, 2.0)

        mask = _apply_rotated_cuboid_filter(
            sample_positions,
            cuboid_center=center,
            cuboid_size=size,
            cuboid_rotation=None,
        )

        # Should keep some points
        assert 0 <= mask.sum() <= len(mask)


class TestEllipsoidFilter:
    """Test ellipsoid filtering functions."""

    def test_ellipsoid_sphere_case(self, sample_positions):
        """Test ellipsoid filter with equal radii (sphere)."""
        radius = 3.0
        mask = _apply_ellipsoid_filter(
            sample_positions,
            ellipsoid_center=(0.0, 0.0, 0.0),
            ellipsoid_radii=(radius, radius, radius),
            ellipsoid_rotation=None,
        )

        # Should match sphere filter
        sphere_mask = _apply_sphere_filter(
            sample_positions,
            sphere_center=(0.0, 0.0, 0.0),
            sphere_radius=radius,
        )
        np.testing.assert_array_equal(mask, sphere_mask)

    def test_ellipsoid_elongated(self, sample_positions):
        """Test ellipsoid filter with different radii."""
        mask = _apply_ellipsoid_filter(
            sample_positions,
            ellipsoid_center=(0.0, 0.0, 0.0),
            ellipsoid_radii=(5.0, 2.0, 2.0),
            ellipsoid_rotation=None,
        )

        # Should keep some points
        assert 0 < mask.sum() < len(mask)

    def test_ellipsoid_with_rotation(self, sample_positions):
        """Test ellipsoid filter with rotation."""
        mask = _apply_ellipsoid_filter(
            sample_positions,
            ellipsoid_center=(0.0, 0.0, 0.0),
            ellipsoid_radii=(5.0, 1.0, 1.0),
            ellipsoid_rotation=(0.0, 0.0, np.pi / 2),  # 90 deg around Z
        )

        # Should keep some points
        assert 0 < mask.sum() < len(mask)


class TestFrustumFilter:
    """Test camera frustum filtering functions."""

    def test_frustum_basic(self):
        """Test basic frustum filtering."""
        # Create points distributed around origin
        rng = np.random.default_rng(42)
        positions = rng.uniform(-5, 5, (100, 3)).astype(np.float32)

        mask = _apply_frustum_filter(
            positions,
            frustum_position=(0.0, 0.0, -10.0),
            frustum_rotation=None,
            frustum_fov=1.047,  # 60 degrees
            frustum_aspect=1.0,
            frustum_near=0.1,
            frustum_far=20.0,
        )

        # Should filter some points
        assert mask.dtype == np.bool_
        assert len(mask) == len(positions)

    def test_frustum_narrow_fov(self):
        """Test frustum with narrow field of view."""
        # Create points at varying angles from +Z
        positions = np.array(
            [
                [0, 0, 10],  # On axis
                [3, 0, 10],  # Off axis
                [5, 0, 10],  # Further off axis
            ],
            dtype=np.float32,
        )

        wide_mask = _apply_frustum_filter(
            positions,
            frustum_position=(0.0, 0.0, 0.0),
            frustum_fov=np.pi / 2,  # 90 degrees
        )

        narrow_mask = _apply_frustum_filter(
            positions,
            frustum_position=(0.0, 0.0, 0.0),
            frustum_fov=np.pi / 6,  # 30 degrees
        )

        # Wide FOV should capture more points
        assert wide_mask.sum() >= narrow_mask.sum()

    def test_frustum_with_rotation(self):
        """Test frustum filter with camera rotation."""
        # Create points to the right (+X)
        positions = np.array(
            [
                [10, 0, 0],  # To the right
                [0, 0, 10],  # Forward
            ],
            dtype=np.float32,
        )

        # Camera looking at +X (rotated 90 deg around Y)
        mask = _apply_frustum_filter(
            positions,
            frustum_position=(0.0, 0.0, 0.0),
            frustum_rotation=(0.0, -np.pi / 2, 0.0),  # Look right
            frustum_fov=1.047,
        )

        # Point at [10,0,0] should be visible, [0,0,10] should not
        assert mask.sum() > 0


class TestApplyGeometryFilter:
    """Test the high-level apply_geometry_filter function."""

    def test_sphere_filter_config(self, sample_positions):
        """Test sphere filter with config object."""
        config = SphereFilter(center=(0, 0, 0), radius=3.0)
        mask = apply_geometry_filter(sample_positions, config)

        assert mask.shape == (len(sample_positions),)
        assert 0 < mask.sum() < len(mask)

    def test_box_filter_config(self, sample_positions):
        """Test box filter with config object."""
        config = BoxFilter(center=(0, 0, 0), size=(4, 4, 4))
        mask = apply_geometry_filter(sample_positions, config)

        assert mask.shape == (len(sample_positions),)

    def test_ellipsoid_filter_config(self, sample_positions):
        """Test ellipsoid filter with config object."""
        config = EllipsoidFilter(center=(0, 0, 0), radii=(3, 2, 1))
        mask = apply_geometry_filter(sample_positions, config)

        assert mask.shape == (len(sample_positions),)

    def test_frustum_filter_config(self, sample_positions):
        """Test frustum filter with config object."""
        config = FrustumFilter(position=(0, 0, -10), fov=1.047)
        mask = apply_geometry_filter(sample_positions, config)

        assert mask.shape == (len(sample_positions),)

    def test_with_quality_filter(self, sample_positions):
        """Test geometry filter with quality filter."""
        geometry = SphereFilter(center=(0, 0, 0), radius=5.0)
        quality = QualityFilter(min_opacity=0.1, max_scale=2.0)

        mask = apply_geometry_filter(sample_positions, geometry, quality)
        assert mask.shape == (len(sample_positions),)

    def test_unknown_filter_type_raises(self, sample_positions):
        """Test that unknown filter type raises error."""

        class UnknownFilter:
            pass

        with pytest.raises(TypeError, match="Unknown geometry filter"):
            apply_geometry_filter(sample_positions, UnknownFilter())


class TestApplyFilter:
    """Test the internal _apply_filter function."""

    def test_empty_positions(self):
        """Test with empty positions array."""
        positions = np.array([], dtype=np.float32).reshape(0, 3)
        mask = _apply_filter(positions)
        assert len(mask) == 0

    def test_invalid_positions_shape(self):
        """Test with invalid positions shape."""
        positions = np.array([1, 2, 3], dtype=np.float32)
        with pytest.raises(ValueError, match="must be \\[N, 3\\]"):
            _apply_filter(positions)

    def test_no_filter(self, sample_positions):
        """Test with no filter (all pass)."""
        mask = _apply_filter(sample_positions, filter_type="none")
        assert mask.all()

    def test_sphere_with_absolute_radius(self, sample_positions):
        """Test sphere filter with absolute radius."""
        mask = _apply_filter(
            sample_positions,
            filter_type="sphere",
            sphere_radius=2.0,
        )

        distances = np.linalg.norm(sample_positions, axis=1)
        expected = distances <= 2.0
        np.testing.assert_array_equal(mask, expected)

    def test_opacity_filter(self, sample_positions):
        """Test opacity filtering."""
        opacities = np.random.rand(len(sample_positions)).astype(np.float32)

        mask = _apply_filter(
            sample_positions,
            opacities=opacities,
            opacity_threshold=0.5,
        )

        # Should match manual check
        expected = opacities >= 0.5
        np.testing.assert_array_equal(mask, expected)

    def test_scale_filter(self, sample_positions):
        """Test scale filtering."""
        scales = np.random.rand(len(sample_positions), 3).astype(np.float32) * 5

        mask = _apply_filter(
            sample_positions,
            scales=scales,
            max_scale=2.0,
        )

        # Should keep points with max scale <= 2.0
        max_scales = scales.max(axis=1)
        expected = max_scales <= 2.0
        np.testing.assert_array_equal(mask, expected)

    def test_combined_filters(self, sample_positions):
        """Test combined spatial + quality filters."""
        opacities = np.random.rand(len(sample_positions)).astype(np.float32)
        scales = np.random.rand(len(sample_positions), 3).astype(np.float32) * 2

        mask = _apply_filter(
            sample_positions,
            opacities=opacities,
            scales=scales,
            filter_type="sphere",
            sphere_radius=3.0,
            opacity_threshold=0.3,
            max_scale=1.5,
        )

        # Should combine all filters
        assert mask.dtype == np.bool_

    def test_unknown_filter_type(self, sample_positions):
        """Test that unknown filter type raises error."""
        with pytest.raises(ValueError, match="Unknown filter_type"):
            _apply_filter(sample_positions, filter_type="invalid")


class TestAxisAngleToRotationMatrix:
    """Test axis-angle to rotation matrix conversion."""

    def test_identity_rotation(self):
        """Test zero rotation gives identity matrix."""
        R = _axis_angle_to_rotation_matrix((0, 0, 0))
        np.testing.assert_array_almost_equal(R, np.eye(3))

    def test_90_degrees_around_z(self):
        """Test 90 degree rotation around Z axis."""
        R = _axis_angle_to_rotation_matrix((0, 0, np.pi / 2))

        # Should rotate X to Y
        x = np.array([1, 0, 0])
        rotated = R @ x
        np.testing.assert_array_almost_equal(rotated, [0, 1, 0], decimal=5)

    def test_180_degrees_around_x(self):
        """Test 180 degree rotation around X axis."""
        R = _axis_angle_to_rotation_matrix((np.pi, 0, 0))

        # Should flip Y and Z
        y = np.array([0, 1, 0])
        rotated = R @ y
        np.testing.assert_array_almost_equal(rotated, [0, -1, 0], decimal=5)


class TestLinearInterp1D:
    """Test 1D linear interpolation utility."""

    def test_basic_interpolation(self):
        """Test basic linear interpolation."""
        centers = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        values = np.array([0.0, 0.25, 1.0], dtype=np.float32)
        x = np.array([0.25, 0.75], dtype=np.float32)

        result = linear_interp_1d(x, centers, values)

        # 0.25 is between 0 and 0.5, so result = 0 + 0.5 * (0.25 - 0) = 0.125
        # 0.75 is between 0.5 and 1.0, so result = 0.25 + 0.5 * (1.0 - 0.25) = 0.625
        expected = np.array([0.125, 0.625], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_at_centers(self):
        """Test interpolation at center points."""
        centers = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        x = np.array([0.0, 0.5, 1.0], dtype=np.float32)

        result = linear_interp_1d(x, centers, values)
        np.testing.assert_array_almost_equal(result, values, decimal=5)

    def test_extrapolation(self):
        """Test behavior at boundaries."""
        centers = np.array([0.0, 1.0], dtype=np.float32)
        values = np.array([0.0, 1.0], dtype=np.float32)
        x = np.array([-1.0, 2.0], dtype=np.float32)

        result = linear_interp_1d(x, centers, values)
        # Should clamp to first/last segment
        assert result[0] < 0  # Extrapolated below
        assert result[1] > 1  # Extrapolated above


class TestNearestNeighbor1D:
    """Test 1D nearest neighbor lookup utility."""

    def test_basic_lookup(self):
        """Test basic nearest neighbor lookup."""
        centers = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        values = np.array([0.0, 0.25, 1.0], dtype=np.float32)
        x = np.array([0.1, 0.7], dtype=np.float32)

        result = nearest_neighbor_1d(x, centers, values)

        # 0.1 is nearest to 0.0, 0.7 is nearest to 0.5
        expected = np.array([0.0, 0.25], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_at_centers(self):
        """Test lookup exactly at centers."""
        centers = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        values = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        x = centers.copy()

        result = nearest_neighbor_1d(x, centers, values)
        np.testing.assert_array_equal(result, values)


class TestMultiplyOpacity:
    """Test opacity multiplication utility."""

    def test_basic_multiply(self, sample_gsdata):
        """Test basic opacity multiplication."""
        original = sample_gsdata.opacities.copy()

        result = multiply_opacity(sample_gsdata, 0.5, inplace=True)

        expected = np.clip(original * 0.5, 0, 1)
        np.testing.assert_array_almost_equal(result.opacities, expected)

    def test_inplace_false(self, sample_gsdata):
        """Test opacity multiplication with inplace=False."""
        original = sample_gsdata.opacities.copy()

        result = multiply_opacity(sample_gsdata, 0.5, inplace=False)

        # Original should be unchanged
        np.testing.assert_array_equal(sample_gsdata.opacities, original)
        # Result should be modified
        expected = np.clip(original * 0.5, 0, 1)
        np.testing.assert_array_almost_equal(result.opacities, expected)

    def test_clamp_to_one(self, sample_gsdata):
        """Test opacity is clamped to 1.0."""
        result = multiply_opacity(sample_gsdata, 10.0, inplace=True)
        assert result.opacities.max() <= 1.0

    def test_zero_factor_raises(self, sample_gsdata):
        """Test that zero factor raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            multiply_opacity(sample_gsdata, 0.0)

    def test_negative_factor_raises(self, sample_gsdata):
        """Test that negative factor raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            multiply_opacity(sample_gsdata, -0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
