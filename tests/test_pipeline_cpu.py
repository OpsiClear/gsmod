"""Tests for unified CPU Pipeline class."""

import numpy as np

from gsmod import ColorValues, FilterValues, GSDataPro, Pipeline, TransformValues


def create_test_data(n: int = 1000, seed: int = 42) -> GSDataPro:
    """Create test GSDataPro with random data."""
    np.random.seed(seed)
    means = np.random.randn(n, 3).astype(np.float32) * 2
    scales = np.abs(np.random.randn(n, 3).astype(np.float32)) * 0.1
    quats = np.random.randn(n, 4).astype(np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = np.random.rand(n).astype(np.float32)
    sh0 = np.random.rand(n, 3).astype(np.float32)
    return GSDataPro(means, scales, quats, opacities, sh0, None)


class TestPipelineColorMethods:
    """Test Pipeline color methods."""

    def test_brightness(self):
        """Test brightness method."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        pipe = Pipeline().brightness(1.5)
        pipe(data, inplace=True)

        assert not np.allclose(data.sh0, original_sh0)

    def test_contrast(self):
        """Test contrast method."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        pipe = Pipeline().contrast(1.3)
        pipe(data, inplace=True)

        assert not np.allclose(data.sh0, original_sh0)

    def test_saturation(self):
        """Test saturation method."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        pipe = Pipeline().saturation(1.5)
        pipe(data, inplace=True)

        assert not np.allclose(data.sh0, original_sh0)

    def test_gamma(self):
        """Test gamma method."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        pipe = Pipeline().gamma(1.2)
        pipe(data, inplace=True)

        assert not np.allclose(data.sh0, original_sh0)

    def test_temperature(self):
        """Test temperature method."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        pipe = Pipeline().temperature(0.5)
        pipe(data, inplace=True)

        assert not np.allclose(data.sh0, original_sh0)

    def test_multiple_color_ops(self):
        """Test chaining multiple color operations."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        pipe = Pipeline().brightness(1.2).saturation(1.3).contrast(1.1)
        pipe(data, inplace=True)

        assert not np.allclose(data.sh0, original_sh0)
        assert len(pipe) == 3

    def test_color_values_direct(self):
        """Test color() method with ColorValues."""
        data = create_test_data()
        original_sh0 = data.sh0.copy()

        pipe = Pipeline().color(ColorValues(brightness=1.3, saturation=1.2))
        pipe(data, inplace=True)

        assert not np.allclose(data.sh0, original_sh0)


class TestPipelineTransformMethods:
    """Test Pipeline transform methods."""

    def test_translate(self):
        """Test translate method."""
        data = create_test_data()
        original_means = data.means.copy()

        pipe = Pipeline().translate([1.0, 2.0, 3.0])
        pipe(data, inplace=True)

        np.testing.assert_allclose(data.means, original_means + [1.0, 2.0, 3.0], rtol=1e-5)

    def test_scale_uniform(self):
        """Test uniform scale method."""
        data = create_test_data()
        original_means = data.means.copy()

        pipe = Pipeline().scale(2.0)
        pipe(data, inplace=True)

        np.testing.assert_allclose(data.means, original_means * 2.0, rtol=1e-5)

    def test_scale_nonuniform(self):
        """Test non-uniform scale method."""
        data = create_test_data()
        original_means = data.means.copy()

        pipe = Pipeline().scale([2.0, 1.0, 0.5])
        pipe(data, inplace=True)

        np.testing.assert_allclose(data.means, original_means * [2.0, 1.0, 0.5], rtol=1e-5)

    def test_rotate_euler(self):
        """Test Euler rotation method."""
        data = create_test_data()
        original_means = data.means.copy()

        pipe = Pipeline().rotate_euler([0, 90, 0])
        pipe(data, inplace=True)

        # Positions should change
        assert not np.allclose(data.means, original_means)

    def test_center_at_origin(self):
        """Test center_at_origin method."""
        data = create_test_data()
        # Shift data off-center
        data.means = data.means + np.array([10.0, 20.0, 30.0])

        pipe = Pipeline().center_at_origin()
        pipe(data, inplace=True)

        centroid = np.mean(data.means, axis=0)
        np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-5)

    def test_normalize_scale(self):
        """Test normalize_scale method."""
        data = create_test_data()

        pipe = Pipeline().normalize_scale(target_size=2.0)
        pipe(data, inplace=True)

        min_b, max_b = data.compute_bounds()
        largest_dim = np.max(max_b - min_b)
        np.testing.assert_allclose(largest_dim, 2.0, rtol=0.01)

    def test_transform_values_direct(self):
        """Test transform() method with TransformValues."""
        data = create_test_data()
        original_means = data.means.copy()

        pipe = Pipeline().transform(TransformValues.from_scale(2.0))
        pipe(data, inplace=True)

        np.testing.assert_allclose(data.means, original_means * 2.0, rtol=1e-5)


class TestPipelineFilterMethods:
    """Test Pipeline filter methods."""

    def test_min_opacity(self):
        """Test min_opacity filter."""
        data = create_test_data()
        original_len = len(data.means)

        pipe = Pipeline().min_opacity(0.5)
        pipe(data, inplace=True)

        assert len(data.means) < original_len
        assert np.all(data.opacities >= 0.5)

    def test_max_scale(self):
        """Test max_scale filter."""
        data = create_test_data()
        original_len = len(data.means)

        pipe = Pipeline().max_scale(0.1)
        pipe(data, inplace=True)

        assert len(data.means) <= original_len

    def test_within_sphere(self):
        """Test within_sphere filter."""
        data = create_test_data()
        original_len = len(data.means)

        pipe = Pipeline().within_sphere(radius=2.0)
        pipe(data, inplace=True)

        assert len(data.means) < original_len
        distances = np.linalg.norm(data.means, axis=1)
        assert np.all(distances <= 2.0)

    def test_outside_sphere(self):
        """Test outside_sphere filter."""
        data = create_test_data()
        original_len = len(data.means)

        pipe = Pipeline().outside_sphere(radius=1.0)
        pipe(data, inplace=True)

        assert len(data.means) < original_len
        distances = np.linalg.norm(data.means, axis=1)
        assert np.all(distances >= 1.0)

    def test_within_box(self):
        """Test within_box filter."""
        data = create_test_data()
        original_len = len(data.means)

        pipe = Pipeline().within_box((-1, -1, -1), (1, 1, 1))
        pipe(data, inplace=True)

        assert len(data.means) < original_len
        assert np.all(data.means >= -1) and np.all(data.means <= 1)

    def test_filter_values_direct(self):
        """Test filter() method with FilterValues."""
        data = create_test_data()
        original_len = len(data.means)

        pipe = Pipeline().filter(FilterValues(min_opacity=0.3, sphere_radius=3.0))
        pipe(data, inplace=True)

        assert len(data.means) < original_len


class TestPipelineExecution:
    """Test Pipeline execution."""

    def test_execution_order(self):
        """Test that operations execute in order."""
        data = create_test_data()

        # Scale first, then translate
        pipe1 = Pipeline().scale(2.0).translate([1, 0, 0])
        result1 = pipe1(data.clone(), inplace=True)

        # Reset and do translate first, then scale
        data2 = create_test_data()
        pipe2 = Pipeline().translate([1, 0, 0]).scale(2.0)
        result2 = pipe2(data2.clone(), inplace=True)

        # Results should differ due to order
        assert not np.allclose(result1.means, result2.means)

    def test_inplace_false(self):
        """Test inplace=False creates copy."""
        data = create_test_data()
        original_means = data.means.copy()

        pipe = Pipeline().translate([1, 0, 0])
        result = pipe(data, inplace=False)

        # Original unchanged
        np.testing.assert_array_equal(data.means, original_means)
        # Result is different
        assert not np.allclose(result.means, original_means)

    def test_inplace_true(self):
        """Test inplace=True modifies original."""
        data = create_test_data()
        original_means = data.means.copy()

        pipe = Pipeline().translate([1, 0, 0])
        result = pipe(data, inplace=True)

        # Should be same object
        assert result is data
        # Means should be modified
        assert not np.allclose(data.means, original_means)

    def test_empty_pipeline(self):
        """Test empty pipeline returns data unchanged."""
        data = create_test_data()
        original_means = data.means.copy()

        pipe = Pipeline()
        pipe(data, inplace=True)

        np.testing.assert_array_equal(data.means, original_means)
        assert len(pipe) == 0


class TestPipelineUtilities:
    """Test Pipeline utility methods."""

    def test_reset(self):
        """Test reset clears operations."""
        pipe = Pipeline().brightness(1.2).translate([1, 0, 0])
        assert len(pipe) == 2

        pipe.reset()
        assert len(pipe) == 0

    def test_clone(self):
        """Test clone creates independent copy."""
        pipe1 = Pipeline().brightness(1.2)
        pipe2 = pipe1.clone()

        pipe2.saturation(1.3)

        assert len(pipe1) == 1
        assert len(pipe2) == 2

    def test_repr(self):
        """Test string representation."""
        pipe = Pipeline().brightness(1.2).translate([1, 0, 0]).min_opacity(0.1)
        repr_str = repr(pipe)

        assert "Pipeline" in repr_str
        assert "color" in repr_str
        assert "transform" in repr_str
        assert "filter" in repr_str


class TestPipelineComplexChains:
    """Test complex pipeline chains."""

    def test_full_processing_chain(self):
        """Test full color + transform + filter chain."""
        data = create_test_data()
        original_len = len(data.means)

        pipe = (
            Pipeline()
            # Color adjustments
            .brightness(1.2)
            .saturation(1.3)
            .temperature(0.2)
            # Transform
            .center_at_origin()
            .normalize_scale(2.0)
            # Filter
            .min_opacity(0.3)
            .within_sphere(radius=1.5)
        )

        result = pipe(data, inplace=True)

        # Check filtering happened
        assert len(result.means) < original_len

        # Check centering worked
        centroid = np.mean(result.means, axis=0)
        np.testing.assert_allclose(centroid, [0, 0, 0], atol=0.1)

    def test_filter_then_transform(self):
        """Test filtering then transforming."""
        data = create_test_data()
        original_x_mean = np.mean(data.means[:, 0])

        pipe = Pipeline().min_opacity(0.3).translate([10, 0, 0])
        result = pipe(data, inplace=True)

        # Mean x should be shifted by ~10
        result_x_mean = np.mean(result.means[:, 0])
        np.testing.assert_allclose(result_x_mean - original_x_mean, 10.0, atol=1.0)

    def test_multiple_filters(self):
        """Test chaining multiple filters."""
        data = create_test_data()
        original_len = len(data.means)

        pipe = Pipeline().min_opacity(0.2).max_scale(0.5).within_sphere(radius=3.0)
        result = pipe(data, inplace=True)

        assert len(result.means) < original_len
