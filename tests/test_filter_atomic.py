"""Tests for Filter atomic class with boolean operators."""

import numpy as np
import pytest

from gsmod import Filter, FilterValues, GSDataPro


def create_test_data(n: int = 1000, seed: int = 42) -> GSDataPro:
    """Create test GSDataPro with random data."""
    np.random.seed(seed)
    means = np.random.randn(n, 3).astype(np.float32)
    scales = np.abs(np.random.randn(n, 3).astype(np.float32)) * 0.1
    quats = np.random.randn(n, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.rand(n).astype(np.float32)
    sh0 = np.random.rand(n, 3).astype(np.float32)
    return GSDataPro(means, scales, quats, opacities, sh0, None)


@pytest.fixture
def sample_data():
    """Create sample GSDataPro with known positions and opacities."""
    return create_test_data(1000, seed=42)


@pytest.fixture
def simple_data():
    """Create simple GSDataPro with predictable values for testing."""
    data = create_test_data(100, seed=0)
    # Set up predictable opacities
    data.opacities[:50] = 0.2  # Low opacity
    data.opacities[50:] = 0.8  # High opacity
    return data


class TestFilterFactoryMethods:
    """Test Filter factory methods."""

    def test_min_opacity(self, simple_data):
        """Test min_opacity filter."""
        f = Filter.min_opacity(0.5)
        mask = f.get_mask(simple_data)
        # Only high opacity (>= 0.5) should pass
        assert mask.sum() == 50  # Last 50 have 0.8 opacity
        assert mask[50:].all()  # All high opacity pass
        assert not mask[:50].any()  # No low opacity pass

    def test_max_opacity(self, simple_data):
        """Test max_opacity filter."""
        f = Filter.max_opacity(0.5)
        mask = f.get_mask(simple_data)
        # Only low opacity (<= 0.5) should pass
        assert mask.sum() == 50
        assert mask[:50].all()

    def test_min_scale(self, sample_data):
        """Test min_scale filter."""
        threshold = np.median(sample_data.scales.max(axis=1))
        f = Filter.min_scale(threshold)
        mask = f.get_mask(sample_data)
        max_scales = sample_data.scales.max(axis=1)
        expected = max_scales >= threshold
        np.testing.assert_array_equal(mask, expected)

    def test_max_scale(self, sample_data):
        """Test max_scale filter."""
        threshold = np.median(sample_data.scales.max(axis=1))
        f = Filter.max_scale(threshold)
        mask = f.get_mask(sample_data)
        max_scales = sample_data.scales.max(axis=1)
        expected = max_scales <= threshold
        np.testing.assert_array_equal(mask, expected)

    def test_sphere(self, sample_data):
        """Test sphere filter."""
        f = Filter.sphere(radius=2.0)
        mask = f.get_mask(sample_data)
        distances = np.linalg.norm(sample_data.means, axis=1)
        expected = distances <= 2.0
        np.testing.assert_array_equal(mask, expected)

    def test_sphere_with_center(self, sample_data):
        """Test sphere filter with custom center."""
        center = (1.0, 1.0, 1.0)
        f = Filter.sphere(radius=2.0, center=center)
        mask = f.get_mask(sample_data)
        distances = np.linalg.norm(sample_data.means - np.array(center), axis=1)
        expected = distances <= 2.0
        np.testing.assert_array_equal(mask, expected)

    def test_box(self, sample_data):
        """Test box filter."""
        f = Filter.box(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))
        mask = f.get_mask(sample_data)
        means = sample_data.means
        expected = (
            (means[:, 0] >= -1)
            & (means[:, 0] <= 1)
            & (means[:, 1] >= -1)
            & (means[:, 1] <= 1)
            & (means[:, 2] >= -1)
            & (means[:, 2] <= 1)
        )
        np.testing.assert_array_equal(mask, expected)

    def test_ellipsoid(self, sample_data):
        """Test ellipsoid filter."""
        f = Filter.ellipsoid(center=(0, 0, 0), radii=(2, 1, 3))
        mask = f.get_mask(sample_data)
        means = sample_data.means
        # Manual check: (x/rx)^2 + (y/ry)^2 + (z/rz)^2 <= 1
        normalized = means / np.array([2, 1, 3])
        expected = np.sum(normalized**2, axis=1) <= 1
        np.testing.assert_array_equal(mask, expected)

    def test_frustum(self, sample_data):
        """Test frustum filter."""
        f = Filter.frustum(
            position=(0, 0, -5),
            rotation=None,
            fov=np.pi / 3,  # 60 degrees
            aspect=1.0,
            near=1.0,
            far=10.0,
        )
        mask = f.get_mask(sample_data)
        # Just verify it returns a valid mask
        assert mask.dtype == bool
        assert len(mask) == len(sample_data.means)


class TestFilterOutsideVariants:
    """Test outside_ factory methods."""

    def test_outside_sphere(self, sample_data):
        """Test outside_sphere is inverse of sphere."""
        inside = Filter.sphere(radius=2.0)
        outside = Filter.outside_sphere(radius=2.0)
        mask_inside = inside.get_mask(sample_data)
        mask_outside = outside.get_mask(sample_data)
        # Should be exact inverse
        np.testing.assert_array_equal(mask_outside, ~mask_inside)

    def test_outside_box(self, sample_data):
        """Test outside_box is inverse of box."""
        inside = Filter.box(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))
        outside = Filter.outside_box(min_corner=(-1, -1, -1), max_corner=(1, 1, 1))
        mask_inside = inside.get_mask(sample_data)
        mask_outside = outside.get_mask(sample_data)
        np.testing.assert_array_equal(mask_outside, ~mask_inside)

    def test_outside_ellipsoid(self, sample_data):
        """Test outside_ellipsoid is inverse of ellipsoid."""
        inside = Filter.ellipsoid(center=(0, 0, 0), radii=(2, 1, 3))
        outside = Filter.outside_ellipsoid(center=(0, 0, 0), radii=(2, 1, 3))
        mask_inside = inside.get_mask(sample_data)
        mask_outside = outside.get_mask(sample_data)
        np.testing.assert_array_equal(mask_outside, ~mask_inside)


class TestFilterBooleanOperators:
    """Test boolean operators &, |, ~."""

    def test_and_operator(self, simple_data):
        """Test AND operator combines filters correctly."""
        opacity = Filter.min_opacity(0.5)
        sphere = Filter.sphere(radius=10.0)  # Large enough to include most

        combined = opacity & sphere
        mask = combined.get_mask(simple_data)

        # Should be intersection
        mask_op = opacity.get_mask(simple_data)
        mask_sp = sphere.get_mask(simple_data)
        expected = mask_op & mask_sp
        np.testing.assert_array_equal(mask, expected)

    def test_or_operator(self, simple_data):
        """Test OR operator combines filters correctly."""
        low_opacity = Filter.max_opacity(0.3)
        high_opacity = Filter.min_opacity(0.7)

        combined = low_opacity | high_opacity
        mask = combined.get_mask(simple_data)

        # Should be union
        mask_low = low_opacity.get_mask(simple_data)
        mask_high = high_opacity.get_mask(simple_data)
        expected = mask_low | mask_high
        np.testing.assert_array_equal(mask, expected)

    def test_not_operator(self, simple_data):
        """Test NOT operator inverts filter correctly."""
        f = Filter.min_opacity(0.5)
        inverted = ~f

        mask = f.get_mask(simple_data)
        mask_inv = inverted.get_mask(simple_data)
        np.testing.assert_array_equal(mask_inv, ~mask)

    def test_complex_combination(self, simple_data):
        """Test complex boolean combination."""
        # (high opacity AND inside sphere) OR low opacity
        high_op = Filter.min_opacity(0.5)
        sphere = Filter.sphere(radius=10.0)
        low_op = Filter.max_opacity(0.3)

        combined = (high_op & sphere) | low_op
        mask = combined.get_mask(simple_data)

        # Manual computation
        m_high = high_op.get_mask(simple_data)
        m_sphere = sphere.get_mask(simple_data)
        m_low = low_op.get_mask(simple_data)
        expected = (m_high & m_sphere) | m_low
        np.testing.assert_array_equal(mask, expected)

    def test_double_negation(self, simple_data):
        """Test double negation returns to original."""
        f = Filter.min_opacity(0.5)
        double_neg = ~~f

        mask_orig = f.get_mask(simple_data)
        mask_double = double_neg.get_mask(simple_data)
        np.testing.assert_array_equal(mask_orig, mask_double)


class TestFilterApplication:
    """Test Filter application methods."""

    def test_get_mask(self, sample_data):
        """Test get_mask returns correct type and shape."""
        f = Filter.min_opacity(0.5)
        mask = f.get_mask(sample_data)
        assert mask.dtype == bool
        assert len(mask) == len(sample_data.means)

    def test_call_inplace_false(self, sample_data):
        """Test __call__ with inplace=False creates copy."""
        f = Filter.min_opacity(0.5)
        original_len = len(sample_data.means)

        result = f(sample_data, inplace=False)

        # Original unchanged
        assert len(sample_data.means) == original_len
        # Result is filtered
        assert len(result.means) <= original_len

    def test_call_inplace_true(self, sample_data):
        """Test __call__ with inplace=True modifies in place."""
        f = Filter.min_opacity(0.5)
        original_len = len(sample_data.means)

        result = f(sample_data, inplace=True)

        # Should be same object
        assert result is sample_data
        # Should be filtered
        assert len(sample_data.means) <= original_len


class TestFilterConversion:
    """Test Filter <-> FilterValues conversion."""

    def test_from_values_min_opacity(self):
        """Test from_values with min_opacity."""
        values = FilterValues(min_opacity=0.3)
        f = Filter.from_values(values)
        assert "min_opacity" in f._description

    def test_from_values_multiple(self):
        """Test from_values with multiple criteria."""
        values = FilterValues(min_opacity=0.3, max_scale=2.0, sphere_radius=5.0)
        f = Filter.from_values(values)
        # Should combine with AND
        assert "&" in f._description

    def test_from_values_neutral(self):
        """Test from_values with neutral values."""
        values = FilterValues()
        f = Filter.from_values(values)
        # Should pass everything
        assert f._description == "all"

    def test_from_values_invert(self):
        """Test from_values with invert=True."""
        values = FilterValues(sphere_radius=5.0, invert=True)
        f = Filter.from_values(values)
        assert f._description.startswith("~")

    def test_to_values_simple(self):
        """Test to_values for simple filter."""
        f = Filter.min_opacity(0.5)
        values = f.to_values()
        assert values.min_opacity == 0.5

    def test_to_values_sphere(self):
        """Test to_values for sphere filter."""
        f = Filter.sphere(radius=5.0)
        values = f.to_values()
        assert values.sphere_radius == 5.0

    def test_to_values_and_combination(self):
        """Test to_values works for AND combinations (uses fused kernel)."""
        f = Filter.min_opacity(0.5) & Filter.sphere(radius=5.0)
        values = f.to_values()
        assert values.min_opacity == 0.5
        assert values.sphere_radius == 5.0

    def test_to_values_or_raises(self):
        """Test to_values raises for OR combinations."""
        f = Filter.min_opacity(0.5) | Filter.sphere(radius=5.0)
        with pytest.raises(ValueError, match="Cannot convert complex"):
            f.to_values()

    def test_to_values_not_raises(self):
        """Test to_values raises for NOT combinations."""
        f = ~Filter.min_opacity(0.5)
        with pytest.raises(ValueError, match="Cannot convert complex"):
            f.to_values()


class TestFilterEquivalence:
    """Test Filter produces same results as FilterValues."""

    def test_min_opacity_equivalence(self, sample_data):
        """Test Filter.min_opacity matches FilterValues.min_opacity."""
        threshold = 0.5
        f = Filter.min_opacity(threshold)
        values = FilterValues(min_opacity=threshold)

        # Apply both
        data1 = sample_data.clone()
        data2 = sample_data.clone()

        f(data1, inplace=True)
        data2.filter(values, inplace=True)

        assert len(data1.means) == len(data2.means)
        np.testing.assert_array_equal(data1.means, data2.means)

    def test_sphere_equivalence(self, sample_data):
        """Test Filter.sphere matches FilterValues.sphere_radius."""
        f = Filter.sphere(radius=2.0)
        values = FilterValues(sphere_radius=2.0)

        data1 = sample_data.clone()
        data2 = sample_data.clone()

        f(data1, inplace=True)
        data2.filter(values, inplace=True)

        assert len(data1.means) == len(data2.means)
        np.testing.assert_array_equal(data1.means, data2.means)

    def test_combined_equivalence(self, sample_data):
        """Test combined Filter matches combined FilterValues."""
        f = Filter.min_opacity(0.3) & Filter.sphere(radius=3.0)
        values = FilterValues(min_opacity=0.3, sphere_radius=3.0)

        data1 = sample_data.clone()
        data2 = sample_data.clone()

        f(data1, inplace=True)
        data2.filter(values, inplace=True)

        assert len(data1.means) == len(data2.means)


class TestFilterRepr:
    """Test Filter string representation."""

    def test_repr_simple(self):
        """Test repr for simple filter."""
        f = Filter.min_opacity(0.5)
        assert "min_opacity" in repr(f)

    def test_repr_combined(self):
        """Test repr for combined filter."""
        f = Filter.min_opacity(0.5) & Filter.sphere(radius=5.0)
        assert "&" in repr(f)

    def test_repr_inverted(self):
        """Test repr for inverted filter."""
        f = ~Filter.sphere(radius=5.0)
        assert "~" in repr(f)


class TestFilterEdgeCases:
    """Test Filter edge cases."""

    def test_empty_data(self):
        """Test Filter handles empty data."""
        data = create_test_data(0, seed=0)
        f = Filter.min_opacity(0.5)
        mask = f.get_mask(data)
        assert len(mask) == 0

    def test_all_pass(self, sample_data):
        """Test filter that passes everything."""
        f = Filter.min_opacity(0.0)
        mask = f.get_mask(sample_data)
        assert mask.all()

    def test_none_pass(self, sample_data):
        """Test filter that passes nothing."""
        f = Filter.min_opacity(2.0)  # Impossible threshold
        mask = f.get_mask(sample_data)
        assert not mask.any()
