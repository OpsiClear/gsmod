"""Tests for histogram-based learning of color adjustments."""

import numpy as np
import pytest
import torch

from gsmod import ColorValues, HistogramConfig
from gsmod.histogram import (
    MomentMatchingLoss,
    soft_histogram,
    soft_histogram_rgb,
)
from gsmod.histogram.apply import compute_histogram_colors


class TestSoftHistogram:
    """Test differentiable soft histogram computation."""

    def test_basic_soft_histogram(self):
        """Test basic soft histogram computation."""
        x = torch.rand(1000)
        hist = soft_histogram(x, n_bins=32)

        assert hist.shape == (32,)
        # Should sum to 1 (normalized)
        assert abs(hist.sum().item() - 1.0) < 1e-5
        # All values non-negative
        assert torch.all(hist >= 0)

    def test_soft_histogram_gradient(self):
        """Test that soft histogram is differentiable."""
        x = torch.rand(100, requires_grad=True)
        hist = soft_histogram(x, n_bins=16)

        # Compute loss that varies with histogram shape
        # (sum is always 1 due to normalization, so use weighted sum)
        weights = torch.arange(16, dtype=torch.float32)
        loss = (hist * weights).sum()
        loss.backward()

        # Gradients should exist and be non-zero
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_soft_histogram_rgb(self):
        """Test RGB soft histogram."""
        colors = torch.rand(500, 3)
        hists = soft_histogram_rgb(colors, n_bins=32)

        assert hists.shape == (3, 32)
        # Each channel should sum to 1
        for c in range(3):
            assert abs(hists[c].sum().item() - 1.0) < 1e-5

    def test_soft_histogram_empty(self):
        """Test soft histogram with empty input."""
        x = torch.tensor([])
        hist = soft_histogram(x, n_bins=16)

        assert hist.shape == (16,)
        assert torch.all(hist == 0)

    def test_soft_histogram_custom_range(self):
        """Test soft histogram with custom value range."""
        # Data in [0.2, 0.8]
        x = torch.rand(1000) * 0.6 + 0.2
        hist = soft_histogram(x, n_bins=32, min_val=0.0, max_val=1.0)

        # Edge bins should have very low values
        assert hist[0] < hist[15]  # Middle bins should be higher
        assert hist[-1] < hist[15]


class TestMomentMatchingLoss:
    """Test moment matching loss function."""

    @pytest.fixture
    def target_histogram(self):
        """Create target histogram."""
        colors = np.random.rand(1000, 3).astype(np.float32)
        return compute_histogram_colors(colors)

    def test_loss_computation(self, target_histogram):
        """Test basic moment loss computation."""
        source = torch.rand(500, 3)
        loss_fn = MomentMatchingLoss()

        loss = loss_fn(source, target_histogram)

        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_loss_gradient(self, target_histogram):
        """Test moment loss is differentiable."""
        source = torch.rand(500, 3, requires_grad=True)
        loss_fn = MomentMatchingLoss()

        loss = loss_fn(source, target_histogram)
        loss.backward()

        assert source.grad is not None

    def test_perfect_match(self, target_histogram):
        """Test that matching moments gives low loss."""
        # Create source with same mean and std as target
        target_mean = target_histogram.mean
        target_std = target_histogram.std

        # Generate data with similar statistics
        source = torch.randn(1000, 3)
        for c in range(3):
            source[:, c] = source[:, c] * target_std[c] + target_mean[c]

        loss_fn = MomentMatchingLoss()
        loss = loss_fn(source, target_histogram)

        # Should be very low
        assert loss.item() < 0.01

    def test_loss_with_skewness(self, target_histogram):
        """Test loss with skewness included."""
        source = torch.rand(500, 3)
        loss_fn = MomentMatchingLoss(include_skewness=True)

        loss = loss_fn(source, target_histogram)

        assert loss.ndim == 0
        assert loss.item() >= 0


class TestHistogramResultLearnFrom:
    """Test HistogramResult.learn_from() method."""

    @pytest.fixture
    def target_histogram(self):
        """Create bright target histogram."""
        # Bright distribution (mean ~0.7)
        colors = np.random.rand(1000, 3).astype(np.float32) * 0.4 + 0.5
        return compute_histogram_colors(colors)

    @pytest.fixture
    def source_colors(self):
        """Create dark source colors (mean ~0.3)."""
        colors = np.random.rand(1000, 3).astype(np.float32) * 0.4 + 0.1
        return torch.tensor(colors)

    def test_learn_from_basic(self, target_histogram, source_colors):
        """Test basic learn_from functionality."""
        learned = target_histogram.learn_from(
            source_colors,
            params=["brightness"],
            n_epochs=50,
            lr=0.05,
        )

        assert isinstance(learned, ColorValues)
        # Brightness should be > 1 to brighten dark source
        assert learned.brightness > 1.0

    def test_learn_from_multiple_params(self, target_histogram, source_colors):
        """Test learning multiple parameters."""
        learned = target_histogram.learn_from(
            source_colors,
            params=["brightness", "contrast", "gamma"],
            n_epochs=50,
            lr=0.05,
        )

        assert isinstance(learned, ColorValues)

    def test_learn_from_improves_histogram(self, target_histogram, source_colors):
        """Test that learned values improve histogram match."""
        # Learn color values (on CPU)
        learned = target_histogram.learn_from(
            source_colors.cpu(),  # Ensure CPU
            params=["brightness", "gamma"],
            n_epochs=100,
            lr=0.05,
        )

        # Apply learned values (on CPU)
        from gsmod.torch.learn import LearnableColor

        model = LearnableColor.from_values(learned, None).cpu()
        adjusted = model(source_colors.cpu())

        # Compare losses
        loss_fn = MomentMatchingLoss()
        loss_before = loss_fn(source_colors.cpu(), target_histogram)
        loss_after = loss_fn(adjusted.detach(), target_histogram)

        # Loss should decrease
        assert loss_after.item() < loss_before.item()


# GPU tests
try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
class TestHistogramLearningGPU:
    """Test histogram learning on GPU."""

    @pytest.fixture
    def target_histogram(self):
        """Create target histogram."""
        colors = np.random.rand(1000, 3).astype(np.float32)
        return compute_histogram_colors(colors, HistogramConfig(n_bins=64))

    def test_soft_histogram_gpu(self):
        """Test soft histogram on GPU."""
        x = torch.rand(1000, device="cuda")
        hist = soft_histogram(x, n_bins=32)

        assert hist.device.type == "cuda"
        assert abs(hist.sum().item() - 1.0) < 1e-5

    def test_loss_gpu(self, target_histogram):
        """Test moment matching loss on GPU."""
        source = torch.rand(500, 3, device="cuda")
        loss_fn = MomentMatchingLoss().cuda()

        loss = loss_fn(source, target_histogram)

        assert loss.device.type == "cuda"
        assert loss.item() >= 0

    def test_learn_from_gpu(self, target_histogram):
        """Test learn_from on GPU."""
        source = torch.rand(1000, 3, device="cuda") * 0.5

        learned = target_histogram.learn_from(
            source,
            params=["brightness", "contrast"],
            n_epochs=50,
            lr=0.05,
        )

        assert isinstance(learned, ColorValues)


class TestEndToEndHistogramMatching:
    """End-to-end tests for histogram matching workflow."""

    def test_full_workflow(self):
        """Test complete histogram matching workflow."""
        # 1. Create reference with specific color characteristics
        reference_colors = np.random.rand(1000, 3).astype(np.float32)
        reference_colors = reference_colors * 0.3 + 0.5  # Bright, low contrast
        reference_hist = compute_histogram_colors(reference_colors)

        # 2. Create source with different characteristics
        source_colors = np.random.rand(1000, 3).astype(np.float32)
        source_colors = source_colors * 0.6 + 0.2  # Darker, high contrast

        # 3. Learn color adjustment to match (on CPU)
        source_tensor = torch.tensor(source_colors)
        learned = reference_hist.learn_from(
            source_tensor,
            params=["brightness", "contrast", "gamma"],
            n_epochs=100,
            lr=0.05,
        )

        # 4. Apply and verify (ensure CPU)
        from gsmod.torch.learn import LearnableColor

        model = LearnableColor.from_values(learned, None).cpu()
        adjusted = model(source_tensor.cpu()).detach().numpy()

        # Check that adjusted colors are closer to reference distribution
        adjusted_mean = adjusted.mean()
        reference_mean = reference_colors.mean()
        source_mean = source_colors.mean()

        # Adjusted mean should be closer to reference than source was
        assert abs(adjusted_mean - reference_mean) < abs(source_mean - reference_mean)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
