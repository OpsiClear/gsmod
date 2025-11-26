"""Tests for GPU-accelerated operations in gsmod.torch."""

import pytest

# Skip all tests if PyTorch is not available
pytest.importorskip("torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from gsply import GSData  # noqa: E402

from gsmod.torch import ColorGPU, FilterGPU, GSTensorPro, PipelineGPU, TransformGPU  # noqa: E402


@pytest.fixture
def sample_gsdata():
    """Create sample GSData for testing."""
    n = 1000
    np.random.seed(42)

    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32),
        scales=np.random.rand(n, 3).astype(np.float32) * 0.5 + 0.1,  # Positive scales
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32),
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,  # Set to None for compatibility
    )

    # Normalize quaternions
    norms = np.linalg.norm(data.quats, axis=1, keepdims=True)
    data.quats /= norms

    return data


@pytest.fixture
def sample_gstensor(sample_gsdata):
    """Create sample GSTensorPro for testing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return GSTensorPro.from_gsdata(sample_gsdata, device=device)


class TestGSTensorPro:
    """Test GSTensorPro basic functionality."""

    def test_from_gsdata(self, sample_gsdata):
        """Test conversion from GSData."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gstensor = GSTensorPro.from_gsdata(sample_gsdata, device=device)

        assert isinstance(gstensor, GSTensorPro)
        assert len(gstensor) == len(sample_gsdata)
        assert gstensor.device.type == device
        assert gstensor.dtype == torch.float32

    def test_to_gsdata(self, sample_gstensor):
        """Test conversion back to GSData."""
        gsdata = sample_gstensor.to_gsdata()

        assert isinstance(gsdata, GSData)
        assert len(gsdata) == len(sample_gstensor)
        assert isinstance(gsdata.means, np.ndarray)

    def test_slicing(self, sample_gstensor):
        """Test slicing operations."""
        # Slice
        subset = sample_gstensor[10:20]
        assert len(subset) == 10

        # Boolean mask
        mask = sample_gstensor.opacities > 0.5
        filtered = sample_gstensor[mask]
        assert len(filtered) <= len(sample_gstensor)

    def test_clone(self, sample_gstensor):
        """Test cloning."""
        clone = sample_gstensor.clone()

        assert clone is not sample_gstensor
        assert len(clone) == len(sample_gstensor)
        assert torch.allclose(clone.means, sample_gstensor.means)


class TestColorOperations:
    """Test GPU color operations."""

    def test_brightness(self, sample_gstensor):
        """Test brightness adjustment."""
        original = sample_gstensor.clone()
        sample_gstensor.adjust_brightness(1.2, inplace=True)

        # Check that values changed
        assert not torch.allclose(sample_gstensor.sh0, original.sh0)

    def test_contrast(self, sample_gstensor):
        """Test contrast adjustment."""
        original = sample_gstensor.clone()
        sample_gstensor.adjust_contrast(1.1, inplace=True)

        assert not torch.allclose(sample_gstensor.sh0, original.sh0)

    def test_saturation(self, sample_gstensor):
        """Test saturation adjustment."""
        original = sample_gstensor.clone()
        sample_gstensor.adjust_saturation(1.3, inplace=True)

        assert not torch.allclose(sample_gstensor.sh0, original.sh0)

    def test_gamma(self, sample_gstensor):
        """Test gamma correction."""
        original = sample_gstensor.clone()
        sample_gstensor.adjust_gamma(0.8, inplace=True)

        assert not torch.allclose(sample_gstensor.sh0, original.sh0)

    def test_temperature(self, sample_gstensor):
        """Test temperature adjustment."""
        original = sample_gstensor.clone()
        sample_gstensor.adjust_temperature(0.2, inplace=True)

        assert not torch.allclose(sample_gstensor.sh0, original.sh0)

    def test_color_preset(self, sample_gstensor):
        """Test color preset application."""
        original = sample_gstensor.clone()
        sample_gstensor.apply_color_preset("cinematic", strength=0.8, inplace=True)

        assert not torch.allclose(sample_gstensor.sh0, original.sh0)


class TestTransformOperations:
    """Test GPU transform operations."""

    def test_translate(self, sample_gstensor):
        """Test translation."""
        original_means = sample_gstensor.means.clone()
        sample_gstensor.translate([1.0, 0.0, 0.5], inplace=True)

        # Check translation applied
        expected = original_means + torch.tensor([1.0, 0.0, 0.5], device=sample_gstensor.device)
        assert torch.allclose(sample_gstensor.means, expected)

    def test_scale_uniform(self, sample_gstensor):
        """Test uniform scaling."""
        original_means = sample_gstensor.means.clone()
        sample_gstensor.scale_uniform(2.0, inplace=True)

        # Check scaling applied to positions
        expected = original_means * 2.0
        assert torch.allclose(sample_gstensor.means, expected)

    def test_scale_nonuniform(self, sample_gstensor):
        """Test non-uniform scaling."""
        original_means = sample_gstensor.means.clone()
        sample_gstensor.scale_nonuniform([1.0, 2.0, 0.5], inplace=True)

        # Check scaling applied
        scale = torch.tensor([1.0, 2.0, 0.5], device=sample_gstensor.device)
        expected = original_means * scale
        assert torch.allclose(sample_gstensor.means, expected)

    def test_rotate_euler(self, sample_gstensor):
        """Test Euler angle rotation."""
        original_means = sample_gstensor.means.clone()
        sample_gstensor.rotate_euler([0, np.pi / 4, 0], order="XYZ", inplace=True)

        # Check that positions changed (except for points at origin)
        if torch.any(torch.norm(original_means, dim=1) > 1e-6):
            assert not torch.allclose(sample_gstensor.means, original_means)

    def test_center_at_origin(self, sample_gstensor):
        """Test centering at origin."""
        sample_gstensor.center_at_origin(inplace=True)

        # Check that mean position is near origin
        center = torch.mean(sample_gstensor.means, dim=0)
        assert torch.allclose(center, torch.zeros(3, device=sample_gstensor.device), atol=1e-5)

    def test_normalize_scale(self, sample_gstensor):
        """Test scale normalization."""
        sample_gstensor.normalize_scale(target_size=2.0, inplace=True)

        # Check that bounding box fits in target size
        min_bounds, max_bounds = sample_gstensor.compute_bounds()
        scene_size = torch.max(max_bounds - min_bounds)
        assert scene_size <= 2.0 + 1e-5


class TestFilterOperations:
    """Test GPU filter operations."""

    def test_filter_within_sphere(self, sample_gstensor):
        """Test spherical filtering."""
        mask = sample_gstensor.filter_within_sphere(radius=1.0)

        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert len(mask) == len(sample_gstensor)

    def test_filter_within_box(self, sample_gstensor):
        """Test box filtering."""
        mask = sample_gstensor.filter_within_box([-1, -1, -1], [1, 1, 1])

        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert len(mask) == len(sample_gstensor)

    def test_filter_min_opacity(self, sample_gstensor):
        """Test opacity filtering."""
        mask = sample_gstensor.filter_min_opacity(0.5)

        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert len(mask) == len(sample_gstensor)

    def test_filter_max_scale(self, sample_gstensor):
        """Test scale filtering."""
        mask = sample_gstensor.filter_max_scale(0.1)

        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert len(mask) == len(sample_gstensor)


class TestAdvancedFilterOperations:
    """Test advanced GPU filter operations (rotated box, ellipsoid, frustum)."""

    def test_filter_rotated_box(self, sample_gstensor):
        """Test rotated box filter via FilterGPU pipeline."""
        pipeline = FilterGPU().within_rotated_box(
            center=[0, 0, 0],
            size=[2, 2, 2],
            rotation=[0, np.pi / 4, 0],  # 45 deg around Y
        )
        mask = pipeline.compute_mask(sample_gstensor)

        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert len(mask) == len(sample_gstensor)
        # Should filter some points
        assert mask.sum() < len(sample_gstensor)

    def test_filter_rotated_box_no_rotation(self, sample_gstensor):
        """Test rotated box filter without rotation (should match AABB)."""
        pipeline = FilterGPU().within_rotated_box(
            center=[0, 0, 0],
            size=[2, 2, 2],
            rotation=None,
        )
        mask_rotated = pipeline.compute_mask(sample_gstensor)

        # Compare with AABB filter
        mask_aabb = sample_gstensor.filter_within_box([-1, -1, -1], [1, 1, 1])

        assert torch.equal(mask_rotated, mask_aabb)

    def test_filter_rotated_box_outside(self, sample_gstensor):
        """Test outside rotated box filter."""
        pipeline = FilterGPU().outside_rotated_box(
            center=[0, 0, 0],
            size=[2, 2, 2],
            rotation=[0, np.pi / 4, 0],
        )
        mask = pipeline.compute_mask(sample_gstensor)

        # Should be inverse of within_rotated_box
        pipeline_within = FilterGPU().within_rotated_box(
            center=[0, 0, 0],
            size=[2, 2, 2],
            rotation=[0, np.pi / 4, 0],
        )
        mask_within = pipeline_within.compute_mask(sample_gstensor)

        assert torch.equal(mask, ~mask_within)

    def test_filter_ellipsoid(self, sample_gstensor):
        """Test ellipsoid filter via FilterGPU pipeline."""
        pipeline = FilterGPU().within_ellipsoid(
            center=[0, 0, 0],
            radii=[2.0, 1.0, 1.5],
            rotation=None,
        )
        mask = pipeline.compute_mask(sample_gstensor)

        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert len(mask) == len(sample_gstensor)

    def test_filter_ellipsoid_with_rotation(self, sample_gstensor):
        """Test ellipsoid filter with rotation."""
        pipeline = FilterGPU().within_ellipsoid(
            center=[0, 0, 0],
            radii=[3.0, 1.0, 1.0],
            rotation=[0, 0, np.pi / 6],  # 30 deg around Z
        )
        mask = pipeline.compute_mask(sample_gstensor)

        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool

    def test_filter_ellipsoid_outside(self, sample_gstensor):
        """Test outside ellipsoid filter."""
        pipeline = FilterGPU().outside_ellipsoid(
            center=[0, 0, 0],
            radii=[2.0, 1.0, 1.5],
        )
        mask = pipeline.compute_mask(sample_gstensor)

        pipeline_within = FilterGPU().within_ellipsoid(
            center=[0, 0, 0],
            radii=[2.0, 1.0, 1.5],
        )
        mask_within = pipeline_within.compute_mask(sample_gstensor)

        assert torch.equal(mask, ~mask_within)

    def test_filter_frustum(self, sample_gstensor):
        """Test frustum filter via FilterGPU pipeline."""
        pipeline = FilterGPU().within_frustum(
            position=[0, 0, 5],
            rotation=None,
            fov=np.pi / 2,  # 90 deg
            aspect=1.0,
            near=0.1,
            far=20.0,
        )
        mask = pipeline.compute_mask(sample_gstensor)

        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
        assert len(mask) == len(sample_gstensor)

    def test_filter_frustum_with_rotation(self, sample_gstensor):
        """Test frustum filter with rotation."""
        # Camera looking down +X axis (rotated 90 deg around Y)
        pipeline = FilterGPU().within_frustum(
            position=[0, 0, 0],
            rotation=[0, np.pi / 2, 0],
            fov=1.047,  # 60 deg
            aspect=16 / 9,
            near=0.1,
            far=50.0,
        )
        mask = pipeline.compute_mask(sample_gstensor)

        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool

    def test_filter_frustum_outside(self, sample_gstensor):
        """Test outside frustum filter."""
        pipeline = FilterGPU().outside_frustum(
            position=[0, 0, 5],
            fov=np.pi / 2,
            near=0.1,
            far=20.0,
        )
        mask = pipeline.compute_mask(sample_gstensor)

        pipeline_within = FilterGPU().within_frustum(
            position=[0, 0, 5],
            fov=np.pi / 2,
            near=0.1,
            far=20.0,
        )
        mask_within = pipeline_within.compute_mask(sample_gstensor)

        assert torch.equal(mask, ~mask_within)

    def test_filter_pipeline_invert_new_filters(self, sample_gstensor):
        """Test that invert() works for new filter types."""
        pipeline = (
            FilterGPU()
            .within_rotated_box([0, 0, 0], [2, 2, 2], [0, 0.5, 0])
            .within_ellipsoid([0, 0, 0], [2, 1, 1])
            .within_frustum([0, 0, 5], None, 1.0, 1.0, 0.1, 20.0)
        )

        inverted = pipeline.invert()

        # Check that operations were inverted
        assert len(inverted._operations) == 3
        assert inverted._operations[0][0] == "outside_rotated_box"
        assert inverted._operations[1][0] == "outside_ellipsoid"
        assert inverted._operations[2][0] == "outside_frustum"

    def test_filter_combined_advanced(self, sample_gstensor):
        """Test combining multiple advanced filters."""
        pipeline = (
            FilterGPU()
            .within_rotated_box([0, 0, 0], [4, 4, 4], [0, np.pi / 4, 0])
            .within_ellipsoid([0, 0, 0], [3, 2, 2])
            .min_opacity(0.1)
        )

        mask = pipeline.compute_mask(sample_gstensor, mode="and")
        filtered = sample_gstensor[mask]

        assert len(filtered) <= len(sample_gstensor)


class TestGSTensorProFilterWithRotation:
    """Test GSTensorPro.filter() with rotation parameters."""

    def test_filter_with_box_rotation(self, sample_gstensor):
        """Test GSTensorPro.filter() with box_rot parameter."""
        from gsmod.config.values import FilterValues

        # Filter with rotated box
        values = FilterValues(
            box_min=(-1.0, -1.0, -1.0),
            box_max=(1.0, 1.0, 1.0),
            box_rot=(0.0, np.pi / 4, 0.0),  # 45 deg around Y
        )

        original_len = len(sample_gstensor)
        result = sample_gstensor.clone()
        result.filter(values, inplace=True)

        assert len(result) <= original_len

    def test_filter_with_ellipsoid_rotation(self, sample_gstensor):
        """Test GSTensorPro.filter() with ellipsoid_rot parameter."""
        from gsmod.config.values import FilterValues

        values = FilterValues(
            ellipsoid_center=(0.0, 0.0, 0.0),
            ellipsoid_radii=(2.0, 1.0, 1.5),
            ellipsoid_rot=(0.0, 0.0, np.pi / 6),  # 30 deg around Z
        )

        original_len = len(sample_gstensor)
        result = sample_gstensor.clone()
        result.filter(values, inplace=True)

        assert len(result) <= original_len

    def test_filter_with_frustum_rotation(self, sample_gstensor):
        """Test GSTensorPro.filter() with frustum_rot parameter."""
        from gsmod.config.values import FilterValues

        values = FilterValues(
            frustum_pos=(0.0, 0.0, 5.0),
            frustum_rot=(0.0, 0.0, 0.0),  # No rotation
            frustum_fov=1.047,
            frustum_aspect=1.0,
            frustum_near=0.1,
            frustum_far=20.0,
        )

        original_len = len(sample_gstensor)
        result = sample_gstensor.clone()
        result.filter(values, inplace=True)

        assert len(result) <= original_len

    def test_filter_combined_with_rotations(self, sample_gstensor):
        """Test combined filters with rotations."""
        from gsmod.config.values import FilterValues

        values = FilterValues(
            min_opacity=0.1,
            box_min=(-2.0, -2.0, -2.0),
            box_max=(2.0, 2.0, 2.0),
            box_rot=(0.0, np.pi / 4, 0.0),
            ellipsoid_center=(0.0, 0.0, 0.0),
            ellipsoid_radii=(3.0, 2.0, 2.0),
            ellipsoid_rot=(0.0, 0.0, np.pi / 6),
        )

        original_len = len(sample_gstensor)
        result = sample_gstensor.clone()
        result.filter(values, inplace=True)

        assert len(result) <= original_len

    def test_filter_box_rotation_vs_aabb(self, sample_gstensor):
        """Test that box with no rotation matches AABB."""
        from gsmod.config.values import FilterValues

        # With rotation=None (AABB path)
        values_aabb = FilterValues(
            box_min=(-1.0, -1.0, -1.0),
            box_max=(1.0, 1.0, 1.0),
            box_rot=None,
        )

        # With rotation=(0,0,0) (OBB path with identity rotation)
        values_obb = FilterValues(
            box_min=(-1.0, -1.0, -1.0),
            box_max=(1.0, 1.0, 1.0),
            box_rot=(0.0, 0.0, 0.0),
        )

        result_aabb = sample_gstensor.clone()
        result_aabb.filter(values_aabb, inplace=True)

        result_obb = sample_gstensor.clone()
        result_obb.filter(values_obb, inplace=True)

        # Both should produce same result
        assert len(result_aabb) == len(result_obb)


class TestPipelines:
    """Test GPU pipeline classes."""

    def test_color_pipeline(self, sample_gstensor):
        """Test ColorGPU pipeline."""
        pipeline = ColorGPU().brightness(1.2).contrast(1.1).saturation(1.3)

        original = sample_gstensor.clone()
        result = pipeline(sample_gstensor, inplace=False)

        # Check that original unchanged
        assert torch.allclose(sample_gstensor.sh0, original.sh0)
        # Check that result changed
        assert not torch.allclose(result.sh0, original.sh0)

    def test_transform_pipeline(self, sample_gstensor):
        """Test TransformGPU pipeline."""
        pipeline = TransformGPU().translate([1, 0, 0]).scale(2.0).rotate_euler([0, np.pi / 4, 0])

        original = sample_gstensor.clone()
        result = pipeline(sample_gstensor, inplace=False)

        # Check that original unchanged
        assert torch.allclose(sample_gstensor.means, original.means)
        # Check that result changed
        assert not torch.allclose(result.means, original.means)

    def test_filter_pipeline(self, sample_gstensor):
        """Test FilterGPU pipeline."""
        pipeline = FilterGPU().within_sphere(radius=2.0).min_opacity(0.1).max_scale(0.5)

        mask = pipeline.compute_mask(sample_gstensor)
        filtered = sample_gstensor[mask]

        assert len(filtered) <= len(sample_gstensor)

    def test_unified_pipeline(self, sample_gstensor):
        """Test PipelineGPU unified pipeline."""
        pipeline = (
            PipelineGPU()
            .within_sphere(radius=3.0)
            .min_opacity(0.1)
            .translate([1, 0, 0])
            .scale(2.0)
            .brightness(1.2)
            .saturation(1.3)
        )

        original_len = len(sample_gstensor)
        result = pipeline(sample_gstensor, inplace=False)

        # Check filtering applied
        assert len(result) <= original_len
        # Check transforms applied (if any points remain)
        if len(result) > 0:
            assert result.means.shape[1] == 3
