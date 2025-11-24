"""Tests for Transform (chained transforms with matrix composition)."""

import numpy as np
import pytest
from gsply import GSData

from gsmod.transform.pipeline import Transform


@pytest.fixture
def sample_gsdata():
    """Generate sample GSData for testing."""
    rng = np.random.default_rng(42)
    n = 1000

    means = rng.standard_normal((n, 3)).astype(np.float32)
    quaternions = rng.standard_normal((n, 4)).astype(np.float32)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=1, keepdims=True)
    scales = (rng.random((n, 3)).astype(np.float32) + 0.1) * 0.5
    opacities = rng.random(n, dtype=np.float32)
    sh0 = rng.random((n, 3), dtype=np.float32)

    return GSData(
        means=means,
        scales=scales,
        quats=quaternions,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


class TestTransform:
    """Test Transform functionality."""

    def test_initialization(self):
        """Test Transform initialization."""
        pipeline = Transform()

        assert pipeline._transforms == []
        assert pipeline._compiled_matrix is None
        assert pipeline._is_dirty

    def test_method_chaining(self):
        """Test that methods return self for chaining."""
        pipeline = Transform()

        # Test chaining
        result = (
            pipeline.scale(2.0)
            .rotate_quat(np.array([1.0, 0.0, 0.0, 0.0]))
            .translate([1.0, 2.0, 3.0])
        )

        assert result is pipeline
        assert len(pipeline._transforms) == 3

    def test_compilation(self):
        """Test that compilation creates transformation matrix."""
        pipeline = Transform()

        # Before compilation
        assert pipeline._compiled_matrix is None
        assert pipeline._is_dirty

        # Add operations and compile
        pipeline.translate([1, 0, 0]).scale(2.0).compile()

        # After compilation
        assert pipeline._compiled_matrix is not None
        assert not pipeline._is_dirty
        assert pipeline._compiled_matrix.shape == (4, 4)

    def test_auto_compilation_on_apply(self, sample_gsdata):
        """Test that apply() auto-compiles if needed."""
        pipeline = Transform()

        pipeline.translate([1, 0, 0]).scale(2.0)

        # Not compiled yet
        assert pipeline._compiled_matrix is None

        # Apply triggers compilation
        pipeline.apply(sample_gsdata, inplace=False)

        # Now compiled
        assert pipeline._compiled_matrix is not None
        assert not pipeline._is_dirty

    def test_translation_pipeline(self, sample_gsdata):
        """Test translation via pipeline."""
        pipeline = Transform()
        original_means = sample_gsdata.means.copy()
        original_quats = sample_gsdata.quats.copy()
        original_scales = sample_gsdata.scales.copy()

        translation = [1.0, 2.0, 3.0]
        pipeline.translate(translation)

        result = pipeline.apply(sample_gsdata, inplace=False)

        # Means should be translated
        expected_means = original_means + np.array(translation, dtype=np.float32)
        np.testing.assert_allclose(result.means, expected_means, atol=1e-5)

        # Quaternions and scales unchanged
        np.testing.assert_allclose(result.quats, original_quats, atol=1e-5)
        np.testing.assert_allclose(result.scales, original_scales, atol=1e-5)

    def test_scaling_pipeline(self, sample_gsdata):
        """Test scaling via pipeline."""
        pipeline = Transform()
        original_means = sample_gsdata.means.copy()
        original_quats = sample_gsdata.quats.copy()
        original_scales = sample_gsdata.scales.copy()

        scale_factor = 2.0
        pipeline.scale(scale_factor)

        result = pipeline.apply(sample_gsdata, inplace=False)

        # Means should be scaled
        np.testing.assert_allclose(result.means, original_means * scale_factor, atol=1e-5)

        # Scales should be scaled
        np.testing.assert_allclose(result.scales, original_scales * scale_factor, atol=1e-5)

        # Quaternions unchanged
        np.testing.assert_allclose(result.quats, original_quats, atol=1e-5)

    def test_rotation_pipeline(self, sample_gsdata):
        """Test rotation via pipeline."""
        pipeline = Transform()
        original_scales = sample_gsdata.scales.copy()

        # 90 degree rotation around Z axis
        rotation_quat = np.array([0.7071, 0.0, 0.0, 0.7071], dtype=np.float32)
        pipeline.rotate_quat(rotation_quat)

        result = pipeline.apply(sample_gsdata, inplace=False)

        # Test a specific point rotation
        rng = np.random.default_rng(42)
        test_gsdata = GSData(
            means=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            scales=rng.random((1, 3), dtype=np.float32),
            quats=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            opacities=np.array([1.0], dtype=np.float32),
            sh0=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            shN=None,
        )
        test_result = pipeline.apply(test_gsdata, inplace=False)
        expected_point = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(test_result.means, expected_point, atol=1e-4)

        # Scales unchanged
        np.testing.assert_allclose(result.scales, original_scales, atol=1e-5)

    def test_combined_operations(self, sample_gsdata):
        """Test combining multiple operations."""
        pipeline = Transform()
        original_means = sample_gsdata.means.copy()
        original_scales = sample_gsdata.scales.copy()

        # Apply scale, rotate, translate (in that order)
        scale_factor = 2.0
        rotation_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity
        translation = [1.0, 2.0, 3.0]

        pipeline.scale(scale_factor).rotate_quat(rotation_quat).translate(translation)

        result = pipeline.apply(sample_gsdata, inplace=False)

        # Expected: scale first, then translate
        expected_means = original_means * scale_factor + np.array(translation, dtype=np.float32)
        np.testing.assert_allclose(result.means, expected_means, atol=1e-5)

        # Scales should be scaled
        np.testing.assert_allclose(result.scales, original_scales * scale_factor, atol=1e-5)

    def test_rotation_formats(self, sample_gsdata):
        """Test different rotation formats."""
        rng = np.random.default_rng(42)
        test_gsdata = GSData(
            means=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            scales=rng.random((1, 3), dtype=np.float32),
            quats=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            opacities=np.array([1.0], dtype=np.float32),
            sh0=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            shN=None,
        )

        # Test Euler angles
        pipeline_euler = Transform()
        pipeline_euler.rotate_euler([0.0, 0.0, np.pi / 2])
        result_euler = pipeline_euler.apply(test_gsdata, inplace=False)

        # Test axis-angle
        pipeline_axis = Transform()
        pipeline_axis.rotate_axis_angle(axis=[0.0, 0.0, 1.0], angle=np.pi / 2)
        result_axis = pipeline_axis.apply(test_gsdata, inplace=False)

        # Both should give similar results (90 degree rotation around Z)
        expected = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(result_euler.means, expected, atol=1e-4)
        np.testing.assert_allclose(result_axis.means, expected, atol=1e-4)

    def test_center_for_rotation(self):
        """Test rotation around a center point."""
        pipeline = Transform()

        # Point at (2, 0, 0), rotate 90 degrees around Z at center (1, 0, 0)
        rng = np.random.default_rng(42)
        test_gsdata = GSData(
            means=np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
            scales=rng.random((1, 3), dtype=np.float32),
            quats=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            opacities=np.array([1.0], dtype=np.float32),
            sh0=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            shN=None,
        )
        rotation_quat = np.array([0.7071, 0.0, 0.0, 0.7071], dtype=np.float32)
        center = [1.0, 0.0, 0.0]

        pipeline.rotate_quat(rotation_quat, center=center)
        result = pipeline.apply(test_gsdata, inplace=False)

        # Should end up at (1, 1, 0)
        expected = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(result.means, expected, atol=1e-4)

    def test_center_for_scaling(self):
        """Test scaling around a center point."""
        pipeline = Transform()

        # Point at (2, 2, 2), scale by 2x around center (1, 1, 1)
        rng = np.random.default_rng(42)
        test_gsdata = GSData(
            means=np.array([[2.0, 2.0, 2.0]], dtype=np.float32),
            scales=rng.random((1, 3), dtype=np.float32),
            quats=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            opacities=np.array([1.0], dtype=np.float32),
            sh0=np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            shN=None,
        )
        scale_factor = 2.0
        center = [1.0, 1.0, 1.0]

        pipeline.scale(scale_factor, center=center)
        result = pipeline.apply(test_gsdata, inplace=False)

        # Should end up at (3, 3, 3)
        expected = np.array([[3.0, 3.0, 3.0]], dtype=np.float32)
        np.testing.assert_allclose(result.means, expected, atol=1e-4)

    def test_set_center(self):
        """Test setting a default center for operations."""
        pipeline = Transform()
        center = [1.0, 1.0, 1.0]
        pipeline.set_center(center)

        assert np.array_equal(pipeline._center, np.array(center, dtype=np.float32))

    def test_inplace_vs_copy(self, sample_gsdata):
        """Test inplace vs copy behavior."""
        pipeline = Transform().translate([1, 0, 0])
        original_means = sample_gsdata.means.copy()

        # Test copy
        gsdata_copy = GSData(
            means=sample_gsdata.means.copy(),
            scales=sample_gsdata.scales.copy(),
            quats=sample_gsdata.quats.copy(),
            opacities=sample_gsdata.opacities.copy(),
            sh0=sample_gsdata.sh0.copy(),
            shN=None,
        )
        result = pipeline.apply(gsdata_copy, inplace=False)
        assert np.allclose(gsdata_copy.means, original_means)  # Original unchanged
        assert result is not gsdata_copy  # Different GSData

        # Test inplace
        gsdata_inplace = GSData(
            means=sample_gsdata.means.copy(),
            scales=sample_gsdata.scales.copy(),
            quats=sample_gsdata.quats.copy(),
            opacities=sample_gsdata.opacities.copy(),
            sh0=sample_gsdata.sh0.copy(),
            shN=None,
        )
        result_inplace = pipeline.apply(gsdata_inplace, inplace=True)
        assert not np.allclose(gsdata_inplace.means, original_means)  # Original modified
        assert result_inplace is gsdata_inplace  # Same GSData

    def test_callable_interface(self, sample_gsdata):
        """Test that pipeline can be called as a function."""
        pipeline = Transform().translate([1, 0, 0]).scale(2.0)
        original_means = sample_gsdata.means.copy()

        # Should work as a callable
        result = pipeline(sample_gsdata, inplace=False)

        assert result.means.shape == original_means.shape
        assert not np.array_equal(result.means, original_means)

    def test_reset(self):
        """Test reset functionality."""
        pipeline = Transform()

        # Add operations and compile
        pipeline.translate([1, 0, 0]).scale(2.0).compile()
        assert pipeline._compiled_matrix is not None
        assert len(pipeline._transforms) == 2

        # Reset
        pipeline.reset()

        # Should be cleared
        assert pipeline._transforms == []
        assert pipeline._compiled_matrix is None
        assert pipeline._compiled_quat is None
        assert pipeline._compiled_scale is None
        assert pipeline._center is None
        assert pipeline._is_dirty

    def test_get_matrix(self):
        """Test getting the compiled transformation matrix."""
        pipeline = Transform()

        # Add a simple translation
        pipeline.translate([1.0, 2.0, 3.0])

        # Get matrix (should trigger compilation)
        matrix = pipeline.get_matrix()

        assert matrix is not None
        assert matrix.shape == (4, 4)
        # Check translation part
        np.testing.assert_allclose(matrix[:3, 3], [1.0, 2.0, 3.0], atol=1e-5)
        # Check rotation part (should be identity)
        np.testing.assert_allclose(matrix[:3, :3], np.eye(3), atol=1e-5)

    def test_repr(self):
        """Test string representation."""
        pipeline = Transform()
        pipeline.translate([1, 0, 0]).scale(2.0)

        repr_str = repr(pipeline)
        assert "Transform" in repr_str
        assert "2 operations" in repr_str
        assert "not compiled" in repr_str

        # After compilation
        pipeline.compile()
        repr_str = repr(pipeline)
        assert "compiled" in repr_str

    def test_empty_pipeline(self, sample_gsdata):
        """Test applying an empty pipeline."""
        pipeline = Transform()
        original_means = sample_gsdata.means.copy()
        original_quats = sample_gsdata.quats.copy()
        original_scales = sample_gsdata.scales.copy()

        # Empty pipeline should act as identity
        result = pipeline.apply(sample_gsdata, inplace=False)

        # Should return unchanged data
        np.testing.assert_allclose(result.means, original_means, atol=1e-5)
        np.testing.assert_allclose(result.quats, original_quats, atol=1e-5)
        np.testing.assert_allclose(result.scales, original_scales, atol=1e-5)
