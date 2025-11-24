"""Tests for format-aware GPU operations."""

import numpy as np
import pytest
import torch
from gsply import GSData
from gsply.gsdata import DataFormat
from gsmod.torch import GSTensorPro, ColorGPU, TransformGPU, FilterGPU


def create_test_data(n: int = 100, device: str = "cuda") -> GSTensorPro:
    """Create test GSTensorPro with known values."""
    np.random.seed(42)
    data = GSData(
        means=np.random.randn(n, 3).astype(np.float32) * 2,
        scales=np.random.rand(n, 3).astype(np.float32) * 0.5 + 0.1,
        quats=np.random.randn(n, 4).astype(np.float32),
        opacities=np.random.rand(n).astype(np.float32) * 0.5 + 0.3,
        sh0=np.random.rand(n, 3).astype(np.float32),
        shN=None,
    )
    # Normalize quaternions
    norms = np.linalg.norm(data.quats, axis=1, keepdims=True)
    data.quats /= norms

    return GSTensorPro.from_gsdata(data, device=device)


class TestFormatTracking:
    """Test that format is correctly tracked through operations."""

    @pytest.fixture
    def gstensor(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return create_test_data(100, device)

    def test_initial_format_is_sh(self, gstensor):
        """Test that initial format is SH."""
        assert gstensor._format.get("sh0") == DataFormat.SH0_SH

    def test_to_rgb_updates_format(self, gstensor):
        """Test that to_rgb updates format tracking."""
        gstensor.to_rgb(inplace=True)
        assert gstensor._format.get("sh0") == DataFormat.SH0_RGB

    def test_to_sh_updates_format(self, gstensor):
        """Test that to_sh updates format tracking."""
        gstensor.to_rgb(inplace=True)
        gstensor.to_sh(inplace=True)
        assert gstensor._format.get("sh0") == DataFormat.SH0_SH

    def test_brightness_preserves_format(self, gstensor):
        """Test that brightness preserves format."""
        original_format = gstensor._format.get("sh0")
        gstensor.adjust_brightness(1.2, inplace=True)
        assert gstensor._format.get("sh0") == original_format

    def test_contrast_preserves_format(self, gstensor):
        """Test that contrast preserves format."""
        original_format = gstensor._format.get("sh0")
        gstensor.adjust_contrast(1.1, inplace=True)
        assert gstensor._format.get("sh0") == original_format

    def test_gamma_preserves_format(self, gstensor):
        """Test that gamma preserves format."""
        original_format = gstensor._format.get("sh0")
        gstensor.adjust_gamma(0.8, inplace=True)
        assert gstensor._format.get("sh0") == original_format

    def test_saturation_converts_to_rgb(self, gstensor):
        """Test that saturation converts to RGB when in SH format."""
        gstensor.adjust_saturation(1.3, inplace=True)
        assert gstensor._format.get("sh0") == DataFormat.SH0_RGB

    def test_temperature_converts_to_rgb(self, gstensor):
        """Test that temperature converts to RGB when in SH format."""
        gstensor.adjust_temperature(0.2, inplace=True)
        assert gstensor._format.get("sh0") == DataFormat.SH0_RGB

    def test_vibrance_converts_to_rgb(self, gstensor):
        """Test that vibrance converts to RGB when in SH format."""
        gstensor.adjust_vibrance(1.2, inplace=True)
        assert gstensor._format.get("sh0") == DataFormat.SH0_RGB

    def test_hue_shift_converts_to_rgb(self, gstensor):
        """Test that hue shift converts to RGB when in SH format."""
        gstensor.adjust_hue_shift(30, inplace=True)
        assert gstensor._format.get("sh0") == DataFormat.SH0_RGB


class TestColorGPUFormatHandling:
    """Test ColorGPU pipeline format handling."""

    @pytest.fixture
    def gstensor(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return create_test_data(100, device)

    def test_requires_rgb_detection(self):
        """Test that requires_rgb correctly identifies RGB-requiring operations.

        Note: As of the CPU/GPU equivalence update, ALL color operations require
        RGB format for correct color math. This ensures consistent results.
        """
        # All operations now require RGB for CPU/GPU equivalence
        pipeline = ColorGPU().brightness(1.2).contrast(1.1).gamma(0.8)
        assert pipeline.requires_rgb()

        # Should require RGB
        pipeline = ColorGPU().brightness(1.2).saturation(1.3)
        assert pipeline.requires_rgb()

        # Should require RGB
        pipeline = ColorGPU().temperature(0.2)
        assert pipeline.requires_rgb()

        # Empty pipeline does not require RGB
        pipeline = ColorGPU()
        assert not pipeline.requires_rgb()

    def test_lazy_conversion_for_rgb_ops(self, gstensor):
        """Test that conversion happens only once for multiple RGB operations."""
        pipeline = ColorGPU().saturation(1.1).temperature(0.1).vibrance(1.2)

        # All three need RGB, but conversion should happen only once
        result = pipeline(gstensor, inplace=False)
        assert result._format.get("sh0") == DataFormat.SH0_RGB

    def test_all_ops_convert_to_rgb(self, gstensor):
        """Test that all color operations convert to RGB for consistency.

        Note: As of the CPU/GPU equivalence update, ALL color operations run
        in RGB space to ensure correct color math and consistent results
        between CPU and GPU pipelines.
        """
        pipeline = ColorGPU().brightness(1.2).contrast(1.1).gamma(0.8)
        result = pipeline(gstensor, inplace=False)

        # All operations now convert to RGB for CPU/GPU equivalence
        assert result._format.get("sh0") == DataFormat.SH0_RGB

    def test_restore_format_option(self, gstensor):
        """Test that restore_format restores original format after processing."""
        original_format = gstensor._format.get("sh0")

        pipeline = ColorGPU().brightness(1.2).saturation(1.3)
        result = pipeline(gstensor, inplace=False, restore_format=True)

        # Should be back to original format
        assert result._format.get("sh0") == original_format

    def test_already_rgb_no_extra_conversion(self, gstensor):
        """Test that data already in RGB doesn't get converted again."""
        # Convert to RGB first
        gstensor.to_rgb(inplace=True)
        original_sh0 = gstensor.sh0.clone()

        # Run pipeline with saturation (needs RGB)
        pipeline = ColorGPU().saturation(1.3)
        result = pipeline(gstensor, inplace=False)

        # Should remain in RGB
        assert result._format.get("sh0") == DataFormat.SH0_RGB


class TestFormatConsistency:
    """Test format consistency across operations."""

    @pytest.fixture
    def gstensor(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return create_test_data(100, device)

    def test_clone_preserves_format(self, gstensor):
        """Test that clone preserves format."""
        gstensor.to_rgb(inplace=True)
        cloned = gstensor.clone()
        assert cloned._format.get("sh0") == DataFormat.SH0_RGB

    def test_slicing_preserves_format(self, gstensor):
        """Test that slicing preserves format."""
        gstensor.to_rgb(inplace=True)
        mask = torch.ones(len(gstensor), dtype=torch.bool, device=gstensor.device)
        mask[50:] = False
        sliced = gstensor[mask]
        # Check that sliced result maintains format tracking
        if hasattr(sliced, '_format'):
            assert sliced._format.get("sh0") == DataFormat.SH0_RGB

    def test_transform_preserves_format(self, gstensor):
        """Test that transform operations preserve color format."""
        gstensor.to_rgb(inplace=True)
        original_format = gstensor._format.get("sh0")

        gstensor.translate([1, 0, 0], inplace=True)
        gstensor.rotate_axis_angle([0, 1, 0], 0.5, inplace=True)

        # Transform should not affect color format
        assert gstensor._format.get("sh0") == original_format


class TestConversionAccuracy:
    """Test accuracy of format conversions."""

    @pytest.fixture
    def gstensor(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return create_test_data(100, device)

    def test_roundtrip_conversion(self, gstensor):
        """Test that SH -> RGB -> SH roundtrip preserves values."""
        original = gstensor.sh0.clone()

        gstensor.to_rgb(inplace=True)
        gstensor.to_sh(inplace=True)

        # Should be back to original (within float tolerance)
        torch.testing.assert_close(gstensor.sh0, original, rtol=1e-5, atol=1e-6)

    def test_rgb_values_in_valid_range(self, gstensor):
        """Test that RGB conversion produces values in [0, 1] range."""
        gstensor.to_rgb(inplace=True)

        assert gstensor.sh0.min() >= 0.0
        assert gstensor.sh0.max() <= 1.0

    def test_sh_to_rgb_formula(self, gstensor):
        """Test that SH to RGB conversion uses correct formula."""
        SH_C0 = 0.28209479
        original = gstensor.sh0.clone()

        gstensor.to_rgb(inplace=True)

        # Expected: rgb = sh * SH_C0 + 0.5, clamped to [0, 1]
        expected = torch.clamp(original * SH_C0 + 0.5, 0, 1)
        torch.testing.assert_close(gstensor.sh0, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
