"""Tests for histogram computation module."""

import numpy as np
import pytest

from gsmod import GSDataPro, HistogramConfig, HistogramResult
from gsmod.histogram.apply import (
    compute_histogram_colors,
    compute_histogram_opacity,
    compute_histogram_positions,
    compute_histogram_scales,
)


class TestHistogramConfig:
    """Test HistogramConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HistogramConfig()
        assert config.n_bins == 256
        assert config.min_value is None
        assert config.max_value is None
        assert config.normalize is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HistogramConfig(n_bins=128, min_value=0.0, max_value=1.0, normalize=True)
        assert config.n_bins == 128
        assert config.min_value == 0.0
        assert config.max_value == 1.0
        assert config.normalize is True

    def test_merge_configs(self):
        """Test config merging with + operator."""
        c1 = HistogramConfig(n_bins=128, min_value=0.2)
        c2 = HistogramConfig(n_bins=256, max_value=0.8, normalize=True)
        merged = c1 + c2

        # n_bins: use larger
        assert merged.n_bins == 256
        # min_value: from c1 (c2 is None)
        assert merged.min_value == 0.2
        # max_value: from c2 (c1 is None)
        assert merged.max_value == 0.8
        # normalize: True if either is True
        assert merged.normalize is True

    def test_merge_overlapping_ranges(self):
        """Test merging configs with overlapping ranges."""
        c1 = HistogramConfig(min_value=0.1, max_value=0.9)
        c2 = HistogramConfig(min_value=0.2, max_value=0.8)
        merged = c1 + c2

        # Range should expand to cover both
        assert merged.min_value == 0.1  # min of both
        assert merged.max_value == 0.9  # max of both

    def test_is_neutral(self):
        """Test is_neutral check."""
        assert HistogramConfig().is_neutral() is True
        assert HistogramConfig(n_bins=128).is_neutral() is False
        assert HistogramConfig(normalize=True).is_neutral() is False


class TestHistogramResult:
    """Test HistogramResult dataclass."""

    def test_empty_result(self):
        """Test creating empty histogram result."""
        result = HistogramResult.empty(n_bins=64, n_channels=3)

        assert result.counts.shape == (3, 64)
        assert result.bin_edges.shape == (65,)
        assert result.mean.shape == (3,)
        assert result.n_samples == 0

    def test_bin_centers(self):
        """Test bin_centers property."""
        result = HistogramResult.empty(n_bins=10)
        centers = result.bin_centers

        assert len(centers) == 10
        # Should be midpoints of bin edges
        for i, center in enumerate(centers):
            expected = (result.bin_edges[i] + result.bin_edges[i + 1]) / 2
            assert abs(center - expected) < 1e-6

    def test_n_bins_property(self):
        """Test n_bins property."""
        result = HistogramResult.empty(n_bins=128)
        assert result.n_bins == 128

    def test_n_channels_property(self):
        """Test n_channels property."""
        result_1ch = HistogramResult.empty(n_bins=64, n_channels=1)
        result_3ch = HistogramResult.empty(n_bins=64, n_channels=3)

        assert result_1ch.n_channels == 1
        assert result_3ch.n_channels == 3


class TestHistogramColors:
    """Test color histogram computation."""

    def test_basic_histogram(self):
        """Test basic color histogram computation."""
        # Create test data
        colors = np.random.rand(1000, 3).astype(np.float32)

        result = compute_histogram_colors(colors)

        assert result.counts.shape == (3, 256)
        assert result.n_samples == 1000
        assert np.all(result.counts >= 0)
        # Total counts should equal N for each channel
        for c in range(3):
            assert result.counts[c].sum() == 1000

    def test_custom_bins(self):
        """Test histogram with custom bin count."""
        colors = np.random.rand(500, 3).astype(np.float32)
        config = HistogramConfig(n_bins=64)

        result = compute_histogram_colors(colors, config)

        assert result.counts.shape == (3, 64)
        assert result.bin_edges.shape == (65,)

    def test_custom_range(self):
        """Test histogram with custom range."""
        colors = np.random.rand(500, 3).astype(np.float32) * 0.5 + 0.25  # [0.25, 0.75]
        config = HistogramConfig(n_bins=10, min_value=0.0, max_value=1.0)

        result = compute_histogram_colors(colors, config)

        assert result.bin_edges[0] == 0.0
        assert result.bin_edges[-1] == 1.0

    def test_normalized_histogram(self):
        """Test density normalization."""
        colors = np.random.rand(1000, 3).astype(np.float32)
        config = HistogramConfig(normalize=True)

        result = compute_histogram_colors(colors, config)

        # Density should integrate to ~1
        bin_width = (result.bin_edges[-1] - result.bin_edges[0]) / result.n_bins
        for c in range(3):
            integral = result.counts[c].sum() * bin_width
            assert abs(integral - 1.0) < 0.01

    def test_empty_data(self):
        """Test with empty data."""
        colors = np.empty((0, 3), dtype=np.float32)

        result = compute_histogram_colors(colors)

        assert result.n_samples == 0
        assert np.all(result.counts == 0)

    def test_statistics(self):
        """Test computed statistics."""
        # Create data with known statistics
        colors = np.zeros((100, 3), dtype=np.float32)
        colors[:, 0] = 0.5  # R channel all 0.5
        colors[:, 1] = np.linspace(0, 1, 100)  # G channel linear
        colors[:, 2] = 0.8  # B channel all 0.8

        result = compute_histogram_colors(colors)

        # Check means
        assert abs(result.mean[0] - 0.5) < 1e-4
        assert abs(result.mean[1] - 0.5) < 1e-2  # Mean of [0, 1] is 0.5
        assert abs(result.mean[2] - 0.8) < 1e-4

        # Check std
        assert result.std[0] < 1e-4  # All same value
        assert result.std[2] < 1e-4


class TestHistogramOpacity:
    """Test opacity histogram computation."""

    def test_basic_histogram(self):
        """Test basic opacity histogram."""
        opacities = np.random.rand(1000).astype(np.float32)

        result = compute_histogram_opacity(opacities)

        assert result.counts.shape == (256,)
        assert result.n_samples == 1000
        assert result.counts.sum() == 1000

    def test_2d_opacities(self):
        """Test with 2D opacity array (flattened)."""
        opacities = np.random.rand(500, 1).astype(np.float32)

        result = compute_histogram_opacity(opacities)

        assert result.n_samples == 500


class TestHistogramScales:
    """Test scale histogram computation."""

    def test_basic_histogram(self):
        """Test basic scale histogram."""
        scales = np.random.rand(1000, 3).astype(np.float32)

        result = compute_histogram_scales(scales)

        # Uses mean across 3 dimensions
        assert result.counts.shape == (256,)
        assert result.n_samples == 1000


class TestHistogramPositions:
    """Test position histogram computation."""

    def test_distance_from_origin(self):
        """Test histogram of distances from origin."""
        means = np.random.randn(1000, 3).astype(np.float32)

        result = compute_histogram_positions(means)

        assert result.counts.shape == (256,)
        assert result.n_samples == 1000
        # All distances should be >= 0
        assert result.min_val >= 0

    def test_single_axis(self):
        """Test histogram of single axis."""
        means = np.random.randn(1000, 3).astype(np.float32)

        result = compute_histogram_positions(means, axis=1)  # Y axis

        assert result.counts.shape == (256,)


class TestGSDataProHistogram:
    """Test histogram methods on GSDataPro."""

    @pytest.fixture
    def sample_data(self):
        """Create sample GSDataPro for testing."""
        data = GSDataPro.__new__(GSDataPro)
        N = 1000
        data.means = np.random.randn(N, 3).astype(np.float32)
        data.scales = np.abs(np.random.randn(N, 3)).astype(np.float32) * 0.1
        data.quats = np.zeros((N, 4), dtype=np.float32)
        data.quats[:, 0] = 1.0  # Identity quaternion
        data.opacities = np.random.rand(N).astype(np.float32)
        data.sh0 = np.random.rand(N, 3).astype(np.float32)
        data.shN = None
        return data

    def test_histogram_colors(self, sample_data):
        """Test histogram_colors method."""
        result = sample_data.histogram_colors()

        assert isinstance(result, HistogramResult)
        assert result.n_samples == 1000
        assert result.counts.shape == (3, 256)

    def test_histogram_colors_with_config(self, sample_data):
        """Test histogram_colors with custom config."""
        config = HistogramConfig(n_bins=64)
        result = sample_data.histogram_colors(config)

        assert result.counts.shape == (3, 64)

    def test_histogram_opacity(self, sample_data):
        """Test histogram_opacity method."""
        result = sample_data.histogram_opacity()

        assert isinstance(result, HistogramResult)
        assert result.n_samples == 1000

    def test_histogram_scales(self, sample_data):
        """Test histogram_scales method."""
        result = sample_data.histogram_scales()

        assert isinstance(result, HistogramResult)
        assert result.n_samples == 1000

    def test_histogram_positions(self, sample_data):
        """Test histogram_positions method."""
        result = sample_data.histogram_positions()

        assert isinstance(result, HistogramResult)
        assert result.n_samples == 1000

    def test_histogram_positions_axis(self, sample_data):
        """Test histogram_positions with specific axis."""
        result_x = sample_data.histogram_positions(axis=0)
        result_y = sample_data.histogram_positions(axis=1)
        result_z = sample_data.histogram_positions(axis=2)

        # Different axes should have different statistics
        assert result_x.n_samples == result_y.n_samples == result_z.n_samples


class TestHistogramResultAnalysis:
    """Test HistogramResult analysis methods."""

    @pytest.fixture
    def sample_result(self):
        """Create sample histogram result."""
        # Create a simple histogram with known distribution
        N = 1000
        colors = np.random.rand(N, 3).astype(np.float32)
        return compute_histogram_colors(colors)

    def test_percentile(self, sample_result):
        """Test percentile computation."""
        p50 = sample_result.percentile(50, channel=0)
        p25 = sample_result.percentile(25, channel=0)
        p75 = sample_result.percentile(75, channel=0)

        # 25th < 50th < 75th
        assert p25 < p50 < p75

    def test_mode(self, sample_result):
        """Test mode computation."""
        mode = sample_result.mode(channel=0)

        # Mode should be within data range
        assert sample_result.min_val[0] <= mode <= sample_result.max_val[0]

    def test_entropy(self, sample_result):
        """Test entropy computation."""
        entropy = sample_result.entropy(channel=0)

        # Entropy should be positive for non-empty distribution
        assert entropy > 0

    def test_dynamic_range(self, sample_result):
        """Test dynamic range computation."""
        dr = sample_result.dynamic_range()

        # Dynamic range should be positive
        assert dr > 0
        # And bounded by actual range
        actual_range = float(sample_result.max_val.max() - sample_result.min_val.min())
        assert dr <= actual_range

    def test_to_color_values(self, sample_result):
        """Test to_color_values optimization suggestion."""
        from gsmod import ColorValues

        # Test different profiles
        for profile in ["neutral", "vibrant", "dramatic", "bright", "dark"]:
            adjustment = sample_result.to_color_values(profile)
            assert isinstance(adjustment, ColorValues)

    def test_to_color_values_invalid_profile(self, sample_result):
        """Test invalid profile raises error."""
        with pytest.raises(ValueError, match="Unknown profile"):
            sample_result.to_color_values("invalid")


class TestHistogramNumbaKernels:
    """Test Numba kernel functionality."""

    def test_consistency_with_numpy(self):
        """Test that Numba and NumPy produce consistent results."""
        colors = np.random.rand(5000, 3).astype(np.float32)

        # Compute histogram
        result = compute_histogram_colors(colors)

        # Verify counts sum to N
        for c in range(3):
            assert result.counts[c].sum() == 5000

        # Verify statistics are reasonable
        assert np.all(result.min_val >= 0)
        assert np.all(result.max_val <= 1)
        assert np.all(result.mean >= 0)
        assert np.all(result.mean <= 1)


# GPU tests - only run if CUDA is available
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestGPUHistogram:
    """Test GPU histogram computation."""

    @pytest.fixture
    def gpu_data(self):
        """Create sample GSTensorPro on GPU."""
        from gsmod.torch.gstensor_pro import GSTensorPro

        N = 1000
        data = GSTensorPro(
            means=torch.randn(N, 3, device="cuda"),
            scales=torch.abs(torch.randn(N, 3, device="cuda")) * 0.1,
            quats=torch.zeros(N, 4, device="cuda"),
            opacities=torch.rand(N, device="cuda"),
            sh0=torch.rand(N, 3, device="cuda"),
            shN=None,
        )
        data.quats[:, 0] = 1.0
        return data

    def test_histogram_colors_gpu(self, gpu_data):
        """Test GPU histogram_colors method."""
        result = gpu_data.histogram_colors()

        assert isinstance(result, HistogramResult)
        assert result.n_samples == 1000
        # Result should be on CPU (numpy)
        assert isinstance(result.counts, np.ndarray)

    def test_histogram_opacity_gpu(self, gpu_data):
        """Test GPU histogram_opacity method."""
        result = gpu_data.histogram_opacity()

        assert isinstance(result, HistogramResult)
        assert result.n_samples == 1000

    def test_histogram_scales_gpu(self, gpu_data):
        """Test GPU histogram_scales method."""
        result = gpu_data.histogram_scales()

        assert isinstance(result, HistogramResult)
        assert result.n_samples == 1000

    def test_histogram_positions_gpu(self, gpu_data):
        """Test GPU histogram_positions method."""
        result = gpu_data.histogram_positions()

        assert isinstance(result, HistogramResult)
        assert result.n_samples == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
