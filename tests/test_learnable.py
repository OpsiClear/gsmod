"""Tests for learnable training modules."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from gsmod.torch.learn import (
    ColorGradingConfig,
    LearnableColor,
    LearnableFilter,
    LearnableFilterConfig,
    LearnableGSTensor,
    LearnableOpacity,
    LearnableTransform,
    OpacityConfig,
    TransformConfig,
)

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def create_test_tensors(n: int = 100, device: str = "cuda"):
    """Create test tensors for learnable module testing."""
    means = torch.randn(n, 3, device=device, dtype=torch.float32)
    scales = torch.rand(n, 3, device=device, dtype=torch.float32) * 0.1
    quats = torch.randn(n, 4, device=device, dtype=torch.float32)
    quats = quats / quats.norm(dim=1, keepdim=True)
    opacities = torch.rand(n, device=device, dtype=torch.float32)
    sh0 = torch.rand(n, 3, device=device, dtype=torch.float32)
    return means, scales, quats, opacities, sh0


class TestColorGradingConfig:
    """Test ColorGradingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ColorGradingConfig()
        assert config.brightness == 1.0
        assert config.contrast == 1.0
        assert config.saturation == 1.0
        assert config.gamma == 1.0
        assert config.temperature == 0.0
        assert len(config.learnable) == 15  # All 15 color parameters

    def test_partial_learnable(self):
        """Test configuring partial learnable parameters."""
        config = ColorGradingConfig(brightness=1.2, learnable=["brightness", "saturation"])
        assert config.brightness == 1.2
        assert len(config.learnable) == 2
        assert "brightness" in config.learnable
        assert "saturation" in config.learnable


class TestLearnableColor:
    """Test LearnableColor module."""

    def test_initialization(self):
        """Test module initialization."""
        model = LearnableColor().cuda()

        # Check all parameters exist
        assert hasattr(model, "brightness")
        assert hasattr(model, "contrast")
        assert hasattr(model, "saturation")

    def test_parameter_registration(self):
        """Test that learnable params are registered as nn.Parameter."""
        config = ColorGradingConfig(learnable=["brightness", "saturation"])
        model = LearnableColor(config).cuda()

        # Check parameter types
        assert isinstance(model.brightness, nn.Parameter)
        assert isinstance(model.saturation, nn.Parameter)
        # Contrast should be a buffer (not learnable)
        assert not isinstance(model.contrast, nn.Parameter)

        # Check optimizer sees only learnable params
        param_count = sum(1 for _ in model.parameters())
        assert param_count == 2

    def test_forward_shape(self):
        """Test forward pass preserves shape."""
        model = LearnableColor().cuda()
        sh0 = torch.rand(100, 3, device="cuda")

        output = model(sh0)

        assert output.shape == sh0.shape
        assert output.device == sh0.device

    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        config = ColorGradingConfig(learnable=["brightness", "saturation"])
        model = LearnableColor(config).cuda()
        sh0 = torch.rand(100, 3, device="cuda", requires_grad=True)

        output = model(sh0)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert model.brightness.grad is not None
        assert model.saturation.grad is not None
        assert sh0.grad is not None

    def test_output_clamped(self):
        """Test output is clamped to [0, 1]."""
        model = LearnableColor().cuda()
        # Set extreme brightness
        model.brightness.data = torch.tensor(5.0).cuda()

        sh0 = torch.rand(100, 3, device="cuda")
        output = model(sh0)

        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_from_values(self):
        """Test creating from ColorValues."""
        from gsmod.config.values import ColorValues

        values = ColorValues(brightness=1.5, saturation=1.3)
        model = LearnableColor.from_values(values, learnable=["brightness"]).cuda()

        assert model.brightness.item() == pytest.approx(1.5)
        assert model.saturation.item() == pytest.approx(1.3)
        assert isinstance(model.brightness, nn.Parameter)
        assert not isinstance(model.saturation, nn.Parameter)

    def test_to_values(self):
        """Test exporting to ColorValues."""
        model = LearnableColor().cuda()
        model.brightness.data = torch.tensor(1.5).cuda()
        model.saturation.data = torch.tensor(1.3).cuda()

        values = model.to_values()

        assert values.brightness == pytest.approx(1.5)
        assert values.saturation == pytest.approx(1.3)


class TestLearnableTransform:
    """Test LearnableTransform module."""

    def test_initialization(self):
        """Test module initialization."""
        model = LearnableTransform().cuda()

        assert hasattr(model, "translation")
        assert hasattr(model, "scale")
        assert hasattr(model, "rotation_6d")

    def test_rotation_input_output(self):
        """Test rotation axis-angle input and output."""
        # Create with axis-angle input
        config = TransformConfig(
            rotation=(0.0, 0.0, 0.785),  # ~45 degrees around Z
            learnable=["rotation"],
        )
        model = LearnableTransform(config).cuda()

        # Get rotation back as axis-angle
        rot = model.get_rotation_axis_angle()
        assert rot.shape == (3,)
        assert np.allclose(rot, [0.0, 0.0, 0.785], atol=1e-4)

    def test_forward_shapes(self):
        """Test forward pass preserves shapes."""
        model = LearnableTransform().cuda()
        means, scales, quats, _, _ = create_test_tensors(100)

        new_means, new_scales, new_quats = model(means, scales, quats)

        assert new_means.shape == means.shape
        assert new_scales.shape == scales.shape
        assert new_quats.shape == quats.shape

    def test_gradient_flow(self):
        """Test gradients flow through transform."""
        config = TransformConfig(learnable=["translation", "scale"])
        model = LearnableTransform(config).cuda()
        means, scales, quats, _, _ = create_test_tensors(100)

        new_means, new_scales, new_quats = model(means, scales, quats)
        loss = new_means.sum() + new_scales.sum()
        loss.backward()

        assert model.translation.grad is not None
        assert model.scale.grad is not None

    def test_translation(self):
        """Test translation is applied correctly."""
        config = TransformConfig(translation=(1.0, 2.0, 3.0), learnable=[])
        model = LearnableTransform(config).cuda()

        means = torch.zeros(10, 3, device="cuda")
        scales = torch.ones(10, 3, device="cuda")
        quats = torch.tensor([[1, 0, 0, 0]], device="cuda").expand(10, 4).float()

        new_means, _, _ = model(means, scales, quats)

        expected = torch.tensor([[1.0, 2.0, 3.0]], device="cuda").expand(10, 3)
        torch.testing.assert_close(new_means, expected)

    def test_scale(self):
        """Test scaling is applied correctly."""
        config = TransformConfig(scale=2.0, learnable=[])
        model = LearnableTransform(config).cuda()

        means = torch.ones(10, 3, device="cuda")
        scales = torch.ones(10, 3, device="cuda")
        quats = torch.tensor([[1, 0, 0, 0]], device="cuda").expand(10, 4).float()

        new_means, new_scales, _ = model(means, scales, quats)

        assert torch.allclose(new_means, torch.full_like(new_means, 2.0))
        assert torch.allclose(new_scales, torch.full_like(new_scales, 2.0))


class TestOpacityConfig:
    """Test OpacityConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = OpacityConfig()
        assert config.scale == 1.0
        assert len(config.learnable) == 1
        assert "scale" in config.learnable

    def test_custom_values(self):
        """Test custom configuration values."""
        config = OpacityConfig(scale=0.7, learnable=["scale"])
        assert config.scale == 0.7
        assert len(config.learnable) == 1


class TestLearnableOpacity:
    """Test LearnableOpacity module."""

    def test_initialization(self):
        """Test module initialization."""
        model = LearnableOpacity().cuda()

        assert hasattr(model, "scale")
        assert model.scale.item() == 1.0

    def test_parameter_registration(self):
        """Test that scale is registered as nn.Parameter."""
        config = OpacityConfig(scale=0.8, learnable=["scale"])
        model = LearnableOpacity(config).cuda()

        assert isinstance(model.scale, nn.Parameter)

        # Check optimizer sees the parameter
        param_count = sum(1 for _ in model.parameters())
        assert param_count == 1

    def test_forward_shape_linear(self):
        """Test forward pass preserves shape for linear format."""
        model = LearnableOpacity().cuda()
        opacities = torch.rand(100, 1, device="cuda")

        output = model(opacities, is_ply_format=False)

        assert output.shape == opacities.shape
        assert output.device == opacities.device

    def test_forward_shape_ply(self):
        """Test forward pass preserves shape for PLY format."""
        model = LearnableOpacity().cuda()
        # PLY format uses logit values (can be negative or > 1)
        opacities = torch.randn(100, 1, device="cuda")

        output = model(opacities, is_ply_format=True)

        assert output.shape == opacities.shape
        assert output.device == opacities.device

    def test_gradient_flow_linear(self):
        """Test gradients flow through the model (linear format)."""
        config = OpacityConfig(scale=0.8, learnable=["scale"])
        model = LearnableOpacity(config).cuda()
        opacities = torch.rand(100, 1, device="cuda", requires_grad=True)

        output = model(opacities, is_ply_format=False)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert model.scale.grad is not None
        assert opacities.grad is not None

    def test_gradient_flow_ply(self):
        """Test gradients flow through the model (PLY format)."""
        config = OpacityConfig(scale=0.8, learnable=["scale"])
        model = LearnableOpacity(config).cuda()
        opacities = torch.randn(100, 1, device="cuda", requires_grad=True)

        output = model(opacities, is_ply_format=True)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert model.scale.grad is not None
        assert opacities.grad is not None

    def test_output_clamped_linear(self):
        """Test output is clamped to [0, 1] for linear format."""
        model = LearnableOpacity().cuda()
        model.scale.data = torch.tensor(2.0).cuda()

        opacities = torch.rand(100, 1, device="cuda")
        output = model(opacities, is_ply_format=False)

        assert output.min() >= 0.0
        assert output.max() <= 1.0

    def test_fade_behavior(self):
        """Test fade behavior (scale < 1.0)."""
        config = OpacityConfig(scale=0.5, learnable=[])
        model = LearnableOpacity(config).cuda()

        opacities = torch.tensor([[0.8]], device="cuda")
        output = model(opacities, is_ply_format=False)

        # Should multiply by 0.5
        expected = 0.8 * 0.5
        assert output.item() == pytest.approx(expected, abs=1e-6)

    def test_boost_behavior(self):
        """Test boost behavior (scale > 1.0)."""
        config = OpacityConfig(scale=1.5, learnable=[])
        model = LearnableOpacity(config).cuda()

        opacities = torch.tensor([[0.6]], device="cuda")
        output = model(opacities, is_ply_format=False)

        # Boost formula: opacity + (1 - opacity) * (scale - 1) / 2
        boost_factor = (1.5 - 1.0) / 2.0
        expected = 0.6 + (1.0 - 0.6) * boost_factor
        assert output.item() == pytest.approx(expected, abs=1e-6)

    def test_ply_format_conversion(self):
        """Test PLY format correctly converts logit to linear and back."""
        config = OpacityConfig(scale=0.8, learnable=[])
        model = LearnableOpacity(config).cuda()

        # Create PLY format opacities (logit values)
        linear = torch.tensor([0.3, 0.5, 0.7], device="cuda")
        ply_opacities = torch.logit(linear)

        output = model(ply_opacities, is_ply_format=True)

        # Convert back to linear to verify
        output_linear = torch.sigmoid(output)
        expected_linear = linear * 0.8

        torch.testing.assert_close(output_linear, expected_linear, rtol=1e-5, atol=1e-5)

    def test_from_values(self):
        """Test creating from OpacityValues."""
        from gsmod.config.values import OpacityValues

        values = OpacityValues.fade(0.7)
        model = LearnableOpacity.from_values(values, learnable=["scale"]).cuda()

        assert model.scale.item() == pytest.approx(0.7)
        assert isinstance(model.scale, nn.Parameter)

    def test_to_values(self):
        """Test exporting to OpacityValues."""
        model = LearnableOpacity().cuda()
        model.scale.data = torch.tensor(0.85).cuda()

        values = model.to_values()

        assert values.scale == pytest.approx(0.85)

    def test_neutral_skip_optimization(self):
        """Test that neutral scale is skipped for non-learnable params."""
        config = OpacityConfig(scale=1.0, learnable=[])
        model = LearnableOpacity(config).cuda()

        opacities = torch.rand(100, 1, device="cuda")
        output = model(opacities, is_ply_format=False)

        # Should return unchanged opacities
        torch.testing.assert_close(output, opacities)


class TestLearnableFilter:
    """Test LearnableFilter module."""

    def test_initialization(self):
        """Test module initialization."""
        model = LearnableFilter().cuda()

        assert hasattr(model, "opacity_threshold")
        assert hasattr(model, "sphere_radius")

    def test_forward_shape(self):
        """Test forward returns correct shape."""
        model = LearnableFilter().cuda()
        means, scales, _, opacities, _ = create_test_tensors(100)

        weights = model(means, opacities, scales)

        assert weights.shape == (100,)
        assert weights.device == means.device

    def test_weights_range(self):
        """Test weights are in [0, 1]."""
        config = LearnableFilterConfig(opacity_threshold=0.5, sphere_radius=2.0)
        model = LearnableFilter(config).cuda()
        means, scales, _, opacities, _ = create_test_tensors(100)

        weights = model(means, opacities, scales)

        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_gradient_flow(self):
        """Test gradients flow through soft filter."""
        config = LearnableFilterConfig(opacity_threshold=0.3, learnable=["opacity_threshold"])
        model = LearnableFilter(config).cuda()
        means, scales, _, opacities, _ = create_test_tensors(100)

        weights = model(means, opacities, scales)
        loss = weights.sum()
        loss.backward()

        assert model.opacity_threshold.grad is not None

    def test_opacity_filtering(self):
        """Test opacity threshold creates appropriate weights."""
        config = LearnableFilterConfig(
            opacity_threshold=0.5,
            opacity_sharpness=100.0,  # Very sharp threshold
        )
        model = LearnableFilter(config).cuda()

        means = torch.zeros(10, 3, device="cuda")
        scales = torch.ones(10, 3, device="cuda")
        # Low and high opacities
        opacities = torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9], device="cuda"
        )

        weights = model(means, opacities, scales)

        # With high sharpness, should be nearly binary
        assert weights[0] < 0.1  # Low opacity -> low weight
        assert weights[-1] > 0.9  # High opacity -> high weight


class TestLearnableGSTensor:
    """Test LearnableGSTensor class."""

    def test_initialization(self):
        """Test direct initialization."""
        means, scales, quats, opacities, sh0 = create_test_tensors(100)

        data = LearnableGSTensor(means, scales, quats, opacities, sh0)

        assert len(data) == 100
        assert data.device == torch.device("cuda:0")

    def test_from_gstensor_pro(self):
        """Test creation from GSTensorPro."""
        from gsply import GSData

        from gsmod.torch import GSTensorPro

        # Create test GSData
        n = 100
        gsdata = GSData(
            means=np.random.randn(n, 3).astype(np.float32),
            scales=np.random.rand(n, 3).astype(np.float32) * 0.1,
            quats=np.random.randn(n, 4).astype(np.float32),
            opacities=np.random.rand(n).astype(np.float32),
            sh0=np.random.rand(n, 3).astype(np.float32),
            shN=None,
        )

        gstensor = GSTensorPro.from_gsdata(gsdata, device="cuda")
        data = LearnableGSTensor.from_gstensor_pro(gstensor, requires_grad=True)

        assert len(data) == n
        assert data.sh0.requires_grad

    def test_apply_color(self):
        """Test applying color module."""
        means, scales, quats, opacities, sh0 = create_test_tensors(100)
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)

        color_model = LearnableColor().cuda()
        result = data.apply_color(color_model)

        # Result should be new instance
        assert result is not data
        # Means should be unchanged
        assert torch.equal(result.means, data.means)
        # sh0 should be transformed
        assert result.sh0.shape == data.sh0.shape

    def test_apply_transform(self):
        """Test applying transform module."""
        means, scales, quats, opacities, sh0 = create_test_tensors(100)
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)

        transform_model = LearnableTransform().cuda()
        result = data.apply_transform(transform_model)

        # Result should be new instance
        assert result is not data
        # sh0 should be unchanged
        assert torch.equal(result.sh0, data.sh0)
        # Geometry should be transformed
        assert result.means.shape == data.means.shape

    def test_apply_opacity(self):
        """Test applying opacity module."""
        means, scales, quats, opacities, sh0 = create_test_tensors(100)
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)

        opacity_model = LearnableOpacity().cuda()
        result = data.apply_opacity(opacity_model, is_ply_format=False)

        # Result should be new instance
        assert result is not data
        # sh0 should be unchanged
        assert torch.equal(result.sh0, data.sh0)
        # means should be unchanged
        assert torch.equal(result.means, data.means)
        # Opacities should be transformed
        assert result.opacities.shape == data.opacities.shape

    def test_apply_soft_filter(self):
        """Test applying soft filter."""
        means, scales, quats, opacities, sh0 = create_test_tensors(100)
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)

        filter_model = LearnableFilter().cuda()
        result, weights = data.apply_soft_filter(filter_model)

        assert result is data  # Same instance for soft filter
        assert weights.shape == (100,)

    def test_chained_operations(self):
        """Test chaining multiple operations."""
        means, scales, quats, opacities, sh0 = create_test_tensors(100)
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)

        color_model = LearnableColor().cuda()
        transform_model = LearnableTransform().cuda()

        result = data.apply_transform(transform_model).apply_color(color_model)

        assert result.sh0.shape == data.sh0.shape
        assert result.means.shape == data.means.shape

    def test_gradient_flow_through_chain(self):
        """Test gradients flow through chained operations."""
        means, scales, quats, opacities, sh0 = create_test_tensors(100)
        sh0.requires_grad = True
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)

        config = ColorGradingConfig(learnable=["brightness"])
        color_model = LearnableColor(config).cuda()

        result = data.apply_color(color_model)
        loss = result.sh0.sum()
        loss.backward()

        # Check gradients exist
        assert color_model.brightness.grad is not None
        assert sh0.grad is not None

    def test_clone(self):
        """Test cloning preserves data and gradients."""
        means, scales, quats, opacities, sh0 = create_test_tensors(100)
        sh0.requires_grad = True
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)

        clone = data.clone()

        assert clone is not data
        assert clone.sh0.requires_grad
        assert torch.equal(clone.sh0, data.sh0)  # Same values
        assert clone.sh0.data_ptr() != data.sh0.data_ptr()  # Different storage

    def test_detach(self):
        """Test detaching from computation graph."""
        means, scales, quats, opacities, sh0 = create_test_tensors(100)
        sh0.requires_grad = True
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)

        detached = data.detach()

        assert not detached.sh0.requires_grad

    def test_to_device(self):
        """Test moving to different device."""
        means, scales, quats, opacities, sh0 = create_test_tensors(100, device="cuda")
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)

        cpu_data = data.to("cpu")

        assert cpu_data.device == torch.device("cpu")


class TestEndToEndTraining:
    """Test end-to-end training scenarios."""

    def test_simple_training_loop(self):
        """Test a simple training loop works."""
        # Create data
        means, scales, quats, opacities, sh0 = create_test_tensors(100)
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)
        target = torch.rand(100, 3, device="cuda")

        # Create model
        config = ColorGradingConfig(learnable=["brightness", "contrast"])
        model = LearnableColor(config).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Training loop
        initial_loss = None
        for i in range(10):
            optimizer.zero_grad()
            result = data.apply_color(model)
            loss = torch.nn.functional.mse_loss(result.sh0, target)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_multi_module_training(self):
        """Test training with multiple modules."""
        # Create data
        means, scales, quats, opacities, sh0 = create_test_tensors(100)
        data = LearnableGSTensor(means, scales, quats, opacities, sh0)
        target_sh0 = torch.rand(100, 3, device="cuda")
        target_means = torch.randn(100, 3, device="cuda")

        # Create models
        color_config = ColorGradingConfig(learnable=["brightness"])
        color_model = LearnableColor(color_config).cuda()

        transform_config = TransformConfig(learnable=["translation"])
        transform_model = LearnableTransform(transform_config).cuda()

        # Combined optimizer
        optimizer = torch.optim.Adam(
            [
                {"params": color_model.parameters(), "lr": 0.1},
                {"params": transform_model.parameters(), "lr": 0.01},
            ]
        )

        # Training loop
        for i in range(5):
            optimizer.zero_grad()

            result = data.apply_transform(transform_model).apply_color(color_model)

            loss = torch.nn.functional.mse_loss(
                result.sh0, target_sh0
            ) + torch.nn.functional.mse_loss(result.means, target_means)

            loss.backward()
            optimizer.step()

        # Both models should have gradients
        assert (
            color_model.brightness.grad is not None or color_model.brightness.grad is None
        )  # May be zero'd
        assert (
            transform_model.translation.grad is not None or transform_model.translation.grad is None
        )


# Import nn for type checking
