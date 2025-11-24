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
