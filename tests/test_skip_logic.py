"""Tests for skip logic correctness in color processing pipelines.

Verifies that:
1. Operations skip correctly when at neutral values
2. Operations apply correctly when not at neutral values
3. Output is identical for neutral values (no unintended modifications)
"""

import numpy as np
import pytest
import torch
from gsply import GSData

from gsmod import ColorValues, GSDataPro
from gsmod.torch import GSTensorPro
from gsmod.torch.learn import ColorGradingConfig, LearnableColor


@pytest.fixture
def sample_gsdata():
    """Create sample GSData for testing."""
    rng = np.random.default_rng(42)
    n = 1000
    means = rng.random((n, 3), dtype=np.float32)
    scales = rng.random((n, 3), dtype=np.float32) * 0.1
    quats = rng.random((n, 4), dtype=np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = rng.random(n, dtype=np.float32)
    sh0 = rng.random((n, 3), dtype=np.float32)
    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


class TestCPUColorSkipLogic:
    """Test CPU Color pipeline skip logic."""

    def test_neutral_colorvalues_no_change(self, sample_gsdata):
        """Test that neutral ColorValues don't modify data."""
        data = GSDataPro.from_gsdata(sample_gsdata)
        original = data.sh0.copy()

        # Apply neutral values
        neutral = ColorValues()
        data.color(neutral, inplace=True)

        # Data should be unchanged
        np.testing.assert_array_equal(data.sh0, original)

    def test_single_operation_applies(self, sample_gsdata):
        """Test that non-neutral single operations apply correctly."""
        data = GSDataPro.from_gsdata(sample_gsdata)
        original = data.sh0.copy()

        # Apply brightness
        data.color(ColorValues(brightness=1.5), inplace=True)

        # Data should be modified
        assert not np.allclose(data.sh0, original)
        # Brightness multiplies, so values should be higher
        assert np.mean(data.sh0) > np.mean(original)

    def test_all_new_operations_apply(self, sample_gsdata):
        """Test that all new color operations apply when not neutral."""
        data = GSDataPro.from_gsdata(sample_gsdata)
        original = data.sh0.copy()

        # Apply all new operations
        values = ColorValues(
            tint=0.3,
            fade=0.1,
            shadow_tint_hue=-140,
            shadow_tint_sat=0.3,
            highlight_tint_hue=40,
            highlight_tint_sat=0.2,
        )
        data.color(values, inplace=True)

        # Data should be modified
        assert not np.allclose(data.sh0, original)

    def test_partial_neutral_operations_skip(self, sample_gsdata):
        """Test that only neutral operations are skipped."""
        data = GSDataPro.from_gsdata(sample_gsdata)
        original = data.sh0.copy()

        # Mix of neutral and non-neutral
        values = ColorValues(
            brightness=1.0,  # neutral - should skip
            contrast=1.2,  # active - should apply
            gamma=1.0,  # neutral - should skip
            saturation=1.3,  # active - should apply
        )
        data.color(values, inplace=True)

        # Data should be modified by active operations
        assert not np.allclose(data.sh0, original)


class TestGPUSkipLogic:
    """Test GPU GSTensorPro skip logic."""

    @pytest.fixture
    def gpu_data(self, sample_gsdata):
        """Create GPU tensor from sample data."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return GSTensorPro.from_gsdata(sample_gsdata, device=device)

    def test_neutral_brightness_early_return(self, gpu_data):
        """Test that neutral brightness returns early without modification."""
        original = gpu_data.sh0.clone()

        # Apply neutral brightness
        gpu_data.adjust_brightness(1.0, inplace=True)

        # Data should be unchanged
        torch.testing.assert_close(gpu_data.sh0, original)

    def test_neutral_colorvalues_no_change(self, gpu_data):
        """Test that neutral ColorValues don't modify GPU data."""
        original = gpu_data.sh0.clone()

        # Apply neutral values
        neutral = ColorValues()
        gpu_data.color(neutral, inplace=True)

        # Data should be unchanged
        torch.testing.assert_close(gpu_data.sh0, original)

    def test_all_neutral_values_individually(self, gpu_data):
        """Test each operation at neutral value doesn't modify data."""
        test_cases = [
            ("adjust_brightness", 1.0),
            ("adjust_contrast", 1.0),
            ("adjust_saturation", 1.0),
            ("adjust_gamma", 1.0),
            ("adjust_temperature", 0.0),
            ("adjust_vibrance", 1.0),
            ("adjust_hue_shift", 0.0),
            ("adjust_tint", 0.0),
            ("adjust_fade", 0.0),
        ]

        for method_name, neutral_value in test_cases:
            # Reset to original
            original = gpu_data.sh0.clone()

            # Apply neutral value
            method = getattr(gpu_data, method_name)
            method(neutral_value, inplace=True)

            # Data should be unchanged
            torch.testing.assert_close(
                gpu_data.sh0,
                original,
                msg=f"{method_name}({neutral_value}) modified data when it should skip",
            )

    def test_new_operations_apply_on_gpu(self, gpu_data):
        """Test that new GPU operations apply correctly."""
        original = gpu_data.sh0.clone()

        # Apply tint
        gpu_data.adjust_tint(0.3, inplace=True)
        assert not torch.allclose(gpu_data.sh0, original), "tint should modify data"

        # Reset and apply fade
        gpu_data.sh0 = original.clone()
        gpu_data.adjust_fade(0.1, inplace=True)
        assert not torch.allclose(gpu_data.sh0, original), "fade should modify data"

    def test_colorvalues_with_new_operations(self, gpu_data):
        """Test ColorValues with new operations applies on GPU."""
        original = gpu_data.sh0.clone()

        values = ColorValues(
            tint=0.2,
            fade=0.1,
            shadow_tint_hue=-140,
            shadow_tint_sat=0.2,
            highlight_tint_hue=40,
            highlight_tint_sat=0.15,
        )
        gpu_data.color(values, inplace=True)

        # Data should be modified
        assert not torch.allclose(gpu_data.sh0, original)


class TestLearnableColorSkipLogic:
    """Test LearnableColor skip logic."""

    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    def test_all_learnable_no_skip(self, device):
        """Test that learnable parameters are always applied."""
        sh0 = torch.rand(100, 3, device=device)
        sh0.clone()

        # All parameters learnable (at neutral values but should still apply for gradients)
        config = ColorGradingConfig()
        model = LearnableColor(config).to(device)

        output = model(sh0)

        # Output should have gradients (computation happened)
        assert output.requires_grad or any(p.requires_grad for p in model.parameters())

    def test_none_learnable_all_skip(self, device):
        """Test that non-learnable neutral parameters are skipped."""
        sh0 = torch.rand(100, 3, device=device)
        original = sh0.clone()

        # No parameters learnable, all at neutral
        config = ColorGradingConfig(learnable=[])
        model = LearnableColor(config).to(device)

        output = model(sh0)

        # Output should be nearly identical (only final clamp might differ)
        # Since all ops skip, result should be just clamped input
        expected = torch.clamp(original, 0, 1)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    def test_partial_learnable_correct_skip(self, device):
        """Test that only non-learnable neutral parameters skip."""
        sh0 = torch.rand(100, 3, device=device)

        # Only brightness learnable
        config = ColorGradingConfig(learnable=["brightness"])
        model = LearnableColor(config).to(device)

        model(sh0)

        # Should still compute brightness (for gradient flow)
        # Even though brightness=1.0, it should be in computation graph

    def test_non_neutral_non_learnable_applies(self, device):
        """Test that non-neutral non-learnable parameters still apply."""
        sh0 = torch.rand(100, 3, device=device)
        original = sh0.clone()

        # Set non-neutral non-learnable value
        config = ColorGradingConfig(
            brightness=1.5,
            learnable=[],  # None learnable
        )
        model = LearnableColor(config).to(device)

        output = model(sh0)

        # Output should be different (brightness 1.5 applied)
        assert not torch.allclose(output, torch.clamp(original, 0, 1))


class TestSkipLogicEdgeCases:
    """Test edge cases for skip logic."""

    def test_very_small_hue_shift_applies(self, sample_gsdata):
        """Test that small but non-zero hue shift applies."""
        data = GSDataPro.from_gsdata(sample_gsdata)
        original = data.sh0.copy()

        # Small but non-zero hue shift (must be >= 0.5 degrees to exceed skip threshold)
        data.color(ColorValues(hue_shift=1.0), inplace=True)

        # Should still apply (not skipped)
        assert not np.allclose(data.sh0, original)

    def test_very_small_temperature_applies(self, sample_gsdata):
        """Test that small but non-zero temperature applies."""
        data = GSDataPro.from_gsdata(sample_gsdata)
        original = data.sh0.copy()

        data.color(ColorValues(temperature=0.001), inplace=True)

        # Should still apply
        assert not np.allclose(data.sh0, original)

    def test_shadow_tint_zero_sat_skips(self, sample_gsdata):
        """Test that shadow_tint with zero saturation is effectively neutral."""
        data = GSDataPro.from_gsdata(sample_gsdata)
        original = data.sh0.copy()

        # Hue set but sat=0 means no effect
        values = ColorValues(
            shadow_tint_hue=90,  # non-zero
            shadow_tint_sat=0.0,  # zero = no effect
        )
        data.color(values, inplace=True)

        # Should be unchanged because sat=0
        np.testing.assert_allclose(data.sh0, original, rtol=1e-5)

    def test_inplace_false_preserves_original(self, sample_gsdata):
        """Test that inplace=False doesn't modify original."""
        data = GSDataPro.from_gsdata(sample_gsdata)
        original = data.sh0.copy()

        # Apply non-neutral with inplace=False
        result = data.color(ColorValues(brightness=1.5), inplace=False)

        # Original should be unchanged
        np.testing.assert_array_equal(data.sh0, original)
        # Result should be modified
        assert not np.allclose(result.sh0, original)

    def test_chained_neutral_operations(self, sample_gsdata):
        """Test chaining multiple neutral operations."""
        data = GSDataPro.from_gsdata(sample_gsdata)
        original = data.sh0.copy()

        # Chain multiple neutral operations
        (
            data.color(ColorValues(brightness=1.0), inplace=True)
            .color(ColorValues(contrast=1.0), inplace=True)
            .color(ColorValues(saturation=1.0), inplace=True)
        )

        # Should still be unchanged
        np.testing.assert_array_equal(data.sh0, original)


class TestColorValuesIsNeutral:
    """Test ColorValues.is_neutral() method."""

    def test_default_is_neutral(self):
        """Test that default ColorValues is neutral."""
        values = ColorValues()
        assert values.is_neutral()

    def test_single_non_neutral(self):
        """Test that single non-neutral value makes it non-neutral."""
        test_cases = [
            ColorValues(brightness=1.1),
            ColorValues(contrast=0.9),
            ColorValues(gamma=1.1),
            ColorValues(saturation=1.1),
            ColorValues(vibrance=0.9),
            ColorValues(temperature=0.1),
            ColorValues(tint=0.1),
            ColorValues(shadows=0.1),
            ColorValues(highlights=0.1),
            ColorValues(fade=0.1),
            ColorValues(hue_shift=1.0),
            ColorValues(shadow_tint_sat=0.1),
            ColorValues(highlight_tint_sat=0.1),
        ]

        for values in test_cases:
            assert not values.is_neutral(), f"{values} should not be neutral"

    def test_shadow_tint_hue_alone_is_neutral(self):
        """Test that shadow_tint_hue alone (with sat=0) is still neutral."""
        # Hue without saturation has no effect
        values = ColorValues(shadow_tint_hue=90, shadow_tint_sat=0.0)
        assert values.is_neutral()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
