"""Tests for GSDataPro simplified API."""

import numpy as np
import pytest

from gsmod import (
    GSDataPro,
    ColorValues,
    FilterValues,
    TransformValues,
    CINEMATIC,
    WARM,
    STRICT_FILTER,
    DOUBLE_SIZE,
)
from gsmod.config.presets import (
    get_color_preset,
    color_from_dict,
    filter_from_dict,
    transform_from_dict,
)


def create_test_data(n: int = 1000) -> GSDataPro:
    """Create test GSDataPro with random data."""
    pro = GSDataPro.__new__(GSDataPro)
    pro.means = np.random.randn(n, 3).astype(np.float32)
    pro.scales = np.abs(np.random.randn(n, 3).astype(np.float32)) * 0.1
    pro.quats = np.random.randn(n, 4).astype(np.float32)
    pro.quats = pro.quats / np.linalg.norm(pro.quats, axis=1, keepdims=True)
    pro.opacities = np.random.rand(n).astype(np.float32)
    pro.sh0 = np.random.rand(n, 3).astype(np.float32)
    pro.shN = None
    return pro


class TestGSDataProColor:
    """Test GSDataPro.color() method."""

    def test_color_neutral(self):
        """Neutral color values should not change data."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        data.color(ColorValues())

        np.testing.assert_array_equal(data.sh0, original_sh0)

    def test_color_brightness(self):
        """Brightness adjustment should scale colors."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        data.color(ColorValues(brightness=1.5))

        # Colors should be brighter (scaled)
        assert not np.allclose(data.sh0, original_sh0)
        # Result should be clipped to [0, 1]
        assert np.all(data.sh0 >= 0)
        assert np.all(data.sh0 <= 1)

    def test_color_preset(self):
        """Color preset should apply correctly."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        data.color(CINEMATIC)

        assert not np.allclose(data.sh0, original_sh0)

    def test_color_composition(self):
        """Color values should compose correctly."""
        data = create_test_data()

        # Compose multiple color values
        values = WARM + ColorValues(brightness=1.2)
        data.color(values)

        # Should complete without error
        assert len(data.sh0) == 1000

    def test_color_inplace_false(self):
        """inplace=False should return a copy."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        result = data.color(ColorValues(brightness=1.5), inplace=False)

        # Original should be unchanged
        np.testing.assert_array_equal(data.sh0, original_sh0)
        # Result should be different
        assert not np.allclose(result.sh0, original_sh0)

    def test_color_chaining(self):
        """Methods should chain correctly."""
        data = create_test_data()

        result = data.color(ColorValues(brightness=1.1)).color(ColorValues(contrast=1.1))

        assert result is data  # Should return self


class TestGSDataProFilter:
    """Test GSDataPro.filter() method."""

    def test_filter_neutral(self):
        """Neutral filter should not change data."""
        data = create_test_data()
        original_count = len(data.means)

        data.filter(FilterValues())

        assert len(data.means) == original_count

    def test_filter_min_opacity(self):
        """Min opacity filter should reduce count."""
        data = create_test_data()

        data.filter(FilterValues(min_opacity=0.5))

        # Should have fewer points
        assert len(data.means) < 1000
        # Remaining should have opacity >= 0.5
        assert np.all(data.opacities >= 0.5)

    def test_filter_sphere(self):
        """Sphere filter should keep points within radius."""
        data = create_test_data()

        data.filter(FilterValues(sphere_radius=1.0, sphere_center=(0, 0, 0)))

        # Remaining points should be within radius
        distances = np.linalg.norm(data.means, axis=1)
        assert np.all(distances <= 1.0)

    def test_filter_box(self):
        """Box filter should keep points within bounds."""
        data = create_test_data()

        data.filter(FilterValues(box_min=(-0.5, -0.5, -0.5), box_max=(0.5, 0.5, 0.5)))

        # Remaining points should be within box
        assert np.all(data.means >= -0.5)
        assert np.all(data.means <= 0.5)

    def test_filter_preset(self):
        """Filter preset should apply correctly."""
        data = create_test_data()

        data.filter(STRICT_FILTER)

        assert len(data.means) <= 1000

    def test_filter_composition(self):
        """Filter values should compose (stricter wins)."""
        f1 = FilterValues(min_opacity=0.3)
        f2 = FilterValues(min_opacity=0.5)
        combined = f1 + f2

        assert combined.min_opacity == 0.5  # max

    def test_filter_inplace_false(self):
        """inplace=False should return a filtered copy."""
        data = create_test_data()
        original_count = len(data.means)

        result = data.filter(FilterValues(min_opacity=0.9), inplace=False)

        # Original should be unchanged
        assert len(data.means) == original_count
        # Result should be filtered
        assert len(result.means) < original_count


class TestGSDataProTransform:
    """Test GSDataPro.transform() method."""

    def test_transform_neutral(self):
        """Neutral transform should not change data."""
        data = create_test_data()
        original_means = data.means.copy()

        data.transform(TransformValues())

        np.testing.assert_array_almost_equal(data.means, original_means)

    def test_transform_scale(self):
        """Scale transform should scale positions."""
        data = create_test_data()
        original_means = data.means.copy()

        data.transform(TransformValues.from_scale(2.0))

        np.testing.assert_array_almost_equal(data.means, original_means * 2.0)

    def test_transform_translation(self):
        """Translation transform should move positions."""
        data = create_test_data()
        original_means = data.means.copy()

        data.transform(TransformValues.from_translation(1.0, 2.0, 3.0))

        expected = original_means + np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(data.means, expected)

    def test_transform_preset(self):
        """Transform preset should apply correctly."""
        data = create_test_data()
        original_means = data.means.copy()

        data.transform(DOUBLE_SIZE)

        np.testing.assert_array_almost_equal(data.means, original_means * 2.0)

    def test_transform_composition(self):
        """Transform values should compose (matrix multiplication)."""
        t1 = TransformValues.from_translation(1, 0, 0)
        t2 = TransformValues.from_scale(2.0)
        combined = t1 + t2  # translate then scale

        # The translation should also be scaled
        assert combined.translation[0] == 2.0

    def test_transform_inplace_false(self):
        """inplace=False should return a transformed copy."""
        data = create_test_data()
        original_means = data.means.copy()

        result = data.transform(TransformValues.from_scale(2.0), inplace=False)

        # Original should be unchanged
        np.testing.assert_array_equal(data.means, original_means)
        # Result should be transformed
        np.testing.assert_array_almost_equal(result.means, original_means * 2.0)


class TestChaining:
    """Test method chaining."""

    def test_full_chain(self):
        """Full chain of operations should work."""
        data = create_test_data()

        result = (
            data
            .color(ColorValues(brightness=1.1))
            .filter(FilterValues(min_opacity=0.3))
            .transform(TransformValues.from_scale(2.0))
        )

        assert result is data  # All inplace
        assert len(data.means) < 1000  # Filtered

    def test_mixed_inplace(self):
        """Mix of inplace and copy operations."""
        data = create_test_data()
        original_count = len(data.means)

        # First operation creates copy
        result = data.color(ColorValues(brightness=1.1), inplace=False)
        # Continue chaining
        result.filter(FilterValues(min_opacity=0.5))

        # Original unchanged
        assert len(data.means) == original_count
        # Result filtered
        assert len(result.means) < original_count


class TestPresetLoading:
    """Test preset loading functions."""

    def test_get_color_preset(self):
        """Get color preset by name."""
        preset = get_color_preset("cinematic")
        assert isinstance(preset, ColorValues)

    def test_color_from_dict(self):
        """Load color from dict."""
        d = {"brightness": 1.2, "temperature": 0.3}
        values = color_from_dict(d)
        assert values.brightness == 1.2
        assert values.temperature == 0.3

    def test_filter_from_dict(self):
        """Load filter from dict."""
        d = {"min_opacity": 0.5, "sphere_radius": 5.0}
        values = filter_from_dict(d)
        assert values.min_opacity == 0.5
        assert values.sphere_radius == 5.0

    def test_transform_from_dict(self):
        """Load transform from dict."""
        d = {"scale": 2.0, "translation": [1, 2, 3]}
        values = transform_from_dict(d)
        assert values.scale == 2.0
        assert values.translation == (1, 2, 3)

    def test_transform_from_dict_factory(self):
        """Load transform using factory method syntax."""
        d = {"from_rotation_euler": [0, 45, 0]}
        values = transform_from_dict(d)
        assert values.rotation != (1.0, 0.0, 0.0, 0.0)  # Not identity


class TestClone:
    """Test clone functionality."""

    def test_clone_deep_copy(self):
        """Clone should create deep copy."""
        data = create_test_data()
        clone = data.clone()

        # Modify original
        data.means[0] = [999, 999, 999]

        # Clone should be unaffected
        assert not np.array_equal(clone.means[0], [999, 999, 999])


class TestCombinedOperations:
    """Test combined filter + transform + color operations."""

    def test_filter_transform_color(self):
        """Full pipeline: filter -> transform -> color."""
        data = create_test_data()
        original_count = len(data.means)

        # Filter
        data.filter(FilterValues(min_opacity=0.3))
        filtered_count = len(data.means)
        assert filtered_count < original_count

        # Transform
        original_means = data.means.copy()
        data.transform(TransformValues.from_scale(2.0))
        np.testing.assert_array_almost_equal(data.means, original_means * 2.0)

        # Color
        original_sh0 = data.sh0.copy()
        data.color(ColorValues(brightness=1.5))
        assert not np.allclose(data.sh0, original_sh0)

    def test_transform_color_filter(self):
        """Different order: transform -> color -> filter."""
        data = create_test_data()

        # Transform first
        data.transform(TransformValues.from_translation(10, 0, 0))

        # Color second
        data.color(ColorValues(saturation=1.3))

        # Filter last - now filter around translated position
        data.filter(FilterValues(sphere_center=(10, 0, 0), sphere_radius=2.0))

        # Should have some points near (10, 0, 0)
        assert len(data.means) > 0
        distances = np.linalg.norm(data.means - np.array([10, 0, 0]), axis=1)
        assert np.all(distances <= 2.0)

    def test_multiple_transforms(self):
        """Multiple transform operations in sequence."""
        data = create_test_data()
        original_means = data.means.copy()

        data.transform(TransformValues.from_scale(2.0))
        data.transform(TransformValues.from_translation(1, 0, 0))

        # Should be scaled then translated
        expected = original_means * 2.0 + np.array([1, 0, 0])
        np.testing.assert_array_almost_equal(data.means, expected)

    def test_multiple_colors(self):
        """Multiple color operations in sequence."""
        data = create_test_data()

        data.color(ColorValues(brightness=1.1))
        data.color(ColorValues(contrast=1.1))
        data.color(ColorValues(saturation=1.2))

        # Should complete without error
        assert len(data.sh0) > 0

    def test_multiple_filters(self):
        """Multiple filter operations in sequence."""
        data = create_test_data()

        data.filter(FilterValues(min_opacity=0.2))
        count_after_first = len(data.means)

        data.filter(FilterValues(sphere_radius=2.0))
        count_after_second = len(data.means)

        # Each filter should reduce or maintain count
        assert count_after_second <= count_after_first

    def test_complex_workflow(self):
        """Complex realistic workflow."""
        data = create_test_data(n=5000)

        # Step 1: Initial cleanup
        data.filter(FilterValues(min_opacity=0.1, max_scale=0.5))

        # Step 2: Color grading
        data.color(CINEMATIC)

        # Step 3: Scale up
        data.transform(TransformValues.from_scale(1.5))

        # Step 4: Region of interest
        data.filter(FilterValues(sphere_radius=3.0))

        # Step 5: Final color adjustment
        data.color(ColorValues(brightness=1.1))

        # Should complete and have valid data
        assert len(data.means) > 0
        assert data.sh0.shape[0] == data.means.shape[0]
        assert data.opacities.shape[0] == data.means.shape[0]
