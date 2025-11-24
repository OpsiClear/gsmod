"""Tests for Color (chained color operations with LUT compilation)."""

import numpy as np
import pytest
from gsply import GSData

from gsmod.color.pipeline import Color


@pytest.fixture
def sample_gsdata():
    """Generate sample GSData for testing."""
    rng = np.random.default_rng(42)
    n = 10000

    # Create sample Gaussian data
    means = rng.random((n, 3), dtype=np.float32)
    scales = rng.random((n, 3), dtype=np.float32) * 0.1
    quats = rng.random((n, 4), dtype=np.float32)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    opacities = rng.random(n, dtype=np.float32)
    sh0 = rng.random((n, 3), dtype=np.float32)  # RGB colors

    return GSData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh0=sh0,
        shN=None,
    )


class TestColor:
    """Test Color functionality."""

    def test_initialization(self):
        """Test Color initialization."""
        pipeline = Color(lut_size=1024)

        assert pipeline.lut_size == 1024
        assert pipeline._compiled_lut is None
        assert pipeline._is_dirty

    def test_method_chaining(self):
        """Test that methods return self for chaining."""
        pipeline = Color()

        # Test chaining
        result = pipeline.temperature(0.6).brightness(1.2).contrast(1.1).gamma(1.05).saturation(1.3)

        assert result is pipeline

    def test_compilation(self, sample_gsdata):
        """Test that compilation creates LUT."""
        pipeline = Color()

        # Before compilation
        assert pipeline._compiled_lut is None
        assert pipeline._is_dirty

        # Add operations and compile
        pipeline.brightness(1.2).contrast(1.1).compile()

        # After compilation
        assert pipeline._compiled_lut is not None
        assert not pipeline._is_dirty
        assert pipeline._compiled_lut.shape == (1024, 3)  # Interleaved LUT

    def test_auto_compilation_on_apply(self, sample_gsdata):
        """Test that apply() auto-compiles if needed."""
        pipeline = Color()
        pipeline.brightness(1.2).contrast(1.1)

        # Not compiled yet
        assert pipeline._compiled_lut is None

        # Apply triggers compilation
        pipeline.apply(sample_gsdata, inplace=False)

        # Now compiled
        assert pipeline._compiled_lut is not None
        assert not pipeline._is_dirty

    def test_recompilation_on_change(self, sample_gsdata):
        """Test that LUT is recompiled when parameters change."""
        pipeline = Color()

        # First compilation
        pipeline.brightness(1.2).compile()
        lut1 = pipeline._compiled_lut.copy()

        # Change parameters
        pipeline.brightness(1.5)
        assert pipeline._is_dirty  # Should be marked dirty

        # Apply triggers recompilation
        pipeline.apply(sample_gsdata, inplace=False)
        lut2 = pipeline._compiled_lut

        # LUTs should be different
        assert not np.allclose(lut1, lut2)

    def test_phase1_operations(self, sample_gsdata):
        """Test Phase 1 (LUT-capable) operations."""
        pipeline = Color()
        original_colors = sample_gsdata.sh0.copy()

        # Apply Phase 1 operations
        result = (
            pipeline.temperature(0.7)
            .brightness(1.2)
            .contrast(1.1)
            .gamma(0.95)
            .apply(sample_gsdata, inplace=False)
        )

        # Result should be different from input
        assert not np.allclose(result.sh0, original_colors)
        # Result should be in valid range
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

    def test_phase2_operations(self, sample_gsdata):
        """Test Phase 2 (sequential) operations."""
        pipeline = Color()
        original_colors = sample_gsdata.sh0.copy()

        # Apply Phase 2 operations (shadows/highlights now use additive -1 to 1 range)
        result = (
            pipeline.saturation(1.3)
            .shadows(0.1)
            .highlights(-0.1)
            .apply(sample_gsdata, inplace=False)
        )

        # Result should be different from input
        assert not np.allclose(result.sh0, original_colors)
        # Result should be in valid range
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

    def test_combined_phases(self, sample_gsdata):
        """Test combining Phase 1 and Phase 2 operations."""
        pipeline = Color()
        original_colors = sample_gsdata.sh0.copy()

        # Apply both Phase 1 and Phase 2 operations
        # Temperature now uses -1 to 1 range (0=neutral), shadows uses -1 to 1 (0=neutral)
        result = (
            pipeline.temperature(0.1)  # Phase 1 (warm)
            .brightness(1.2)  # Phase 1
            .contrast(1.1)  # Phase 1
            .saturation(1.3)  # Phase 2
            .shadows(0.1)  # Phase 2 (lighten shadows)
            .apply(sample_gsdata, inplace=False)
        )

        # Result should be valid
        assert result.sh0.shape == original_colors.shape
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

    def test_inplace_vs_copy(self, sample_gsdata):
        """Test inplace vs copy behavior."""
        pipeline = Color().brightness(1.2)
        original_colors = sample_gsdata.sh0.copy()

        # Test copy
        result_copy = pipeline.apply(sample_gsdata, inplace=False)
        assert np.allclose(sample_gsdata.sh0, original_colors)  # Original unchanged
        assert result_copy is not sample_gsdata  # Different GSData

        # Test inplace
        result_inplace = pipeline.apply(sample_gsdata, inplace=True)
        assert not np.allclose(sample_gsdata.sh0, original_colors)  # Original modified
        assert result_inplace is sample_gsdata  # Same GSData

    def test_callable_interface(self, sample_gsdata):
        """Test that pipeline can be called as a function."""
        pipeline = Color().brightness(1.2).contrast(1.1)
        original_colors = sample_gsdata.sh0.copy()

        # Should work as a callable
        result = pipeline(sample_gsdata, inplace=False)

        assert result.sh0.shape == original_colors.shape
        assert not np.allclose(result.sh0, original_colors)

    def test_reset(self, sample_gsdata):
        """Test reset functionality."""
        pipeline = Color()

        # Set some operations and compile
        pipeline.brightness(1.2).contrast(1.1).compile()
        assert pipeline._compiled_lut is not None

        # Reset
        pipeline.reset()

        # Should be back to defaults
        # Temperature/shadows/highlights now use additive semantics (0.0 = neutral)
        params = pipeline.get_params()
        assert params["temperature"] == 0.0  # Additive neutral
        assert params["brightness"] == 1.0
        assert params["contrast"] == 1.0
        assert params["gamma"] == 1.0
        assert params["saturation"] == 1.0
        assert params["shadows"] == 0.0  # Additive neutral
        assert params["highlights"] == 0.0  # Additive neutral
        assert pipeline._compiled_lut is None
        assert pipeline._is_dirty

    def test_get_params(self):
        """Test getting current parameters."""
        # Temperature now uses -1 to 1 range (0 = neutral)
        pipeline = (
            Color().temperature(0.1).brightness(1.2).contrast(1.1).gamma(0.95).saturation(1.3)
        )

        params = pipeline.get_params()

        assert params["temperature"] == 0.1  # Warm temperature
        assert params["brightness"] == 1.2
        assert params["contrast"] == 1.1
        assert params["gamma"] == 0.95
        assert params["saturation"] == 1.3
        assert params["shadows"] == 0.0  # Additive neutral
        assert params["highlights"] == 0.0  # Additive neutral

    def test_parameter_validation(self):
        """Test parameter validation."""
        pipeline = Color()

        # Invalid temperature
        with pytest.raises(ValueError):
            pipeline.temperature(1.5)  # Out of range

        # Invalid brightness
        with pytest.raises(ValueError):
            pipeline.brightness(-0.5)  # Negative

        # Invalid gamma
        with pytest.raises(ValueError):
            pipeline.gamma(0.0)  # Zero

        # Invalid types
        with pytest.raises(TypeError):
            pipeline.brightness("invalid")

    def test_repr(self):
        """Test string representation."""
        pipeline = Color().brightness(1.2).contrast(1.1)
        repr_str = repr(pipeline)

        assert "Color" in repr_str
        assert "bright=1.20" in repr_str  # Uses abbreviated format
        assert "contrast=1.10" in repr_str
        assert "not compiled" in repr_str

        # After compilation
        pipeline.compile()
        repr_str = repr(pipeline)
        assert "compiled" in repr_str
        assert "not compiled" not in repr_str

    def test_extreme_values(self, sample_gsdata):
        """Test with extreme parameter values."""
        pipeline = Color()

        # Apply extreme adjustments
        result = (
            pipeline.temperature(1.0)  # Maximum warm
            .brightness(3.0)  # Very bright
            .contrast(3.0)  # Very high contrast
            .gamma(0.2)  # Very low gamma
            .saturation(3.0)  # Very saturated
            .apply(sample_gsdata, inplace=False)
        )

        # Output should still be clamped to [0, 1]
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

    def test_tint_operation(self, sample_gsdata):
        """Test tint (green/magenta) operation."""
        pipeline = Color().tint(0.5)  # Magenta tint
        result = pipeline.apply(sample_gsdata, inplace=False)

        # Output should be valid
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

        # Tint should affect colors (magenta = less green, more R/B)
        # With positive tint, green should decrease relative to R/B

    def test_tint_validation(self):
        """Test tint parameter validation."""
        pipeline = Color()

        # Valid range
        pipeline.tint(-1.0)  # Max green
        pipeline.tint(1.0)  # Max magenta

        # Invalid range
        with pytest.raises(ValueError):
            pipeline.tint(1.5)

        with pytest.raises(ValueError):
            pipeline.tint(-1.5)

    def test_fade_operation(self, sample_gsdata):
        """Test fade (black point lift) operation."""
        pipeline = Color().fade(0.2)
        result = pipeline.apply(sample_gsdata, inplace=False)

        # Output should be valid
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

        # Fade lifts black point, so minimum values should increase
        # The formula is: output = fade + input * (1 - fade)
        # So minimum output should be at least 0.2

    def test_fade_validation(self):
        """Test fade parameter validation."""
        pipeline = Color()

        # Valid range
        pipeline.fade(0.0)
        pipeline.fade(0.5)
        pipeline.fade(1.0)

        # Invalid range
        with pytest.raises(ValueError):
            pipeline.fade(-0.1)

        with pytest.raises(ValueError):
            pipeline.fade(1.5)

    def test_shadow_tint_operation(self, sample_gsdata):
        """Test shadow_tint (split toning) operation."""
        pipeline = Color().shadow_tint(-140, 0.3)  # Blue shadows
        result = pipeline.apply(sample_gsdata, inplace=False)

        # Output should be valid
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

    def test_shadow_tint_validation(self):
        """Test shadow_tint parameter validation."""
        pipeline = Color()

        # Valid hue range
        pipeline.shadow_tint(-180, 0.5)
        pipeline.shadow_tint(180, 0.5)

        # Invalid hue range
        with pytest.raises(ValueError):
            pipeline.shadow_tint(200, 0.5)

        with pytest.raises(ValueError):
            pipeline.shadow_tint(-200, 0.5)

        # Invalid saturation range
        with pytest.raises(ValueError):
            pipeline.shadow_tint(0, -0.1)

        with pytest.raises(ValueError):
            pipeline.shadow_tint(0, 1.5)

    def test_highlight_tint_operation(self, sample_gsdata):
        """Test highlight_tint (split toning) operation."""
        pipeline = Color().highlight_tint(40, 0.2)  # Warm highlights
        result = pipeline.apply(sample_gsdata, inplace=False)

        # Output should be valid
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

    def test_highlight_tint_validation(self):
        """Test highlight_tint parameter validation."""
        pipeline = Color()

        # Valid ranges
        pipeline.highlight_tint(-180, 0.0)
        pipeline.highlight_tint(180, 1.0)

        # Invalid hue range
        with pytest.raises(ValueError):
            pipeline.highlight_tint(200, 0.5)

        # Invalid saturation range
        with pytest.raises(ValueError):
            pipeline.highlight_tint(0, 1.5)

    def test_combined_new_operations(self, sample_gsdata):
        """Test all new operations combined."""
        pipeline = (
            Color()
            .temperature(0.2)
            .tint(-0.1)
            .brightness(1.1)
            .fade(0.05)
            .shadow_tint(-140, 0.2)
            .highlight_tint(40, 0.15)
        )
        result = pipeline.apply(sample_gsdata, inplace=False)

        # Output should be valid
        assert np.all(result.sh0 >= 0.0)
        assert np.all(result.sh0 <= 1.0)

        # Pipeline should have correct operation count
        assert len(pipeline) == 6

    def test_new_operations_is_identity(self):
        """Test is_identity with new operations."""
        # Empty pipeline is identity
        pipeline = Color()
        assert pipeline.is_identity()

        # Adding tint makes it non-identity
        pipeline.tint(0.1)
        assert not pipeline.is_identity()

        # Reset and add fade
        pipeline = Color().fade(0.1)
        assert not pipeline.is_identity()

        # Reset and add shadow_tint
        pipeline = Color().shadow_tint(0, 0.1)
        assert not pipeline.is_identity()

        # Reset and add highlight_tint
        pipeline = Color().highlight_tint(0, 0.1)
        assert not pipeline.is_identity()
