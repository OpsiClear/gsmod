"""Differentiable histogram loss functions for learning color adjustments.

This module provides differentiable approximations to histogram matching,
enabling gradient-based optimization of ColorValues parameters to match
target histogram distributions.

The key challenge is that torch.histc() is NOT differentiable. We solve this
by using soft histogram approximations with Gaussian kernels.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from gsmod.histogram.result import HistogramResult

logger = logging.getLogger(__name__)


def soft_histogram(
    x: torch.Tensor,
    n_bins: int = 64,
    min_val: float = 0.0,
    max_val: float = 1.0,
    sigma: float | None = None,
) -> torch.Tensor:
    """Compute differentiable soft histogram using Gaussian kernels.

    Instead of hard binning, uses soft assignment via Gaussian RBF kernels.
    This allows gradients to flow through the histogram computation.

    :param x: Input tensor (any shape, will be flattened)
    :param n_bins: Number of histogram bins
    :param min_val: Minimum value for binning
    :param max_val: Maximum value for binning
    :param sigma: Gaussian kernel bandwidth (default: auto from bin width)
    :returns: Soft histogram [n_bins], normalized to sum to 1

    Example:
        >>> colors = torch.rand(1000, 3, requires_grad=True)
        >>> hist = soft_histogram(colors[:, 0], n_bins=64)
        >>> loss = hist.sum()  # Some loss
        >>> loss.backward()  # Gradients flow!
    """
    device = x.device
    dtype = x.dtype

    # Flatten input
    x_flat = x.view(-1)
    N = x_flat.shape[0]

    if N == 0:
        return torch.zeros(n_bins, device=device, dtype=dtype)

    # Compute bin centers
    bin_width = (max_val - min_val) / n_bins
    bin_centers = torch.linspace(
        min_val + bin_width / 2,
        max_val - bin_width / 2,
        n_bins,
        device=device,
        dtype=dtype,
    )

    # Default sigma based on bin width
    if sigma is None:
        sigma = bin_width / 2

    # Reshape for broadcasting: x_flat [N, 1], bin_centers [1, n_bins]
    x_expanded = x_flat.unsqueeze(1)  # [N, 1]
    centers_expanded = bin_centers.unsqueeze(0)  # [1, n_bins]

    # Gaussian kernel weights
    weights = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * sigma**2))

    # Sum weights to get histogram
    hist = weights.sum(dim=0)

    # Normalize to probability distribution
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum

    return hist


def soft_histogram_rgb(
    colors: torch.Tensor,
    n_bins: int = 64,
    min_val: float = 0.0,
    max_val: float = 1.0,
    sigma: float | None = None,
) -> torch.Tensor:
    """Compute soft histogram for RGB colors (per channel).

    :param colors: RGB colors [N, 3]
    :param n_bins: Number of histogram bins per channel
    :param min_val: Minimum value
    :param max_val: Maximum value
    :param sigma: Gaussian kernel bandwidth
    :returns: Soft histograms [3, n_bins]
    """
    hists = []
    for c in range(3):
        hist = soft_histogram(colors[:, c], n_bins, min_val, max_val, sigma)
        hists.append(hist)

    return torch.stack(hists)


class MomentMatchingLoss(nn.Module):
    """Moment matching loss for learning ColorValues parameters.

    Matches mean and standard deviation between source colors and target histogram.
    This is the fastest and most accurate method for learning color adjustment
    parameters because it directly constrains the solution space.

    Advantages:
    - Fast: Only computes mean/std (no histogram binning)
    - Accurate: Uniquely constrains parameters (avoids non-uniqueness problem)
    - Stable: Simple gradients, no numerical issues

    Example:
        >>> from gsmod.torch.learn import LearnableColor
        >>> from gsmod import ColorValues
        >>>
        >>> # Target histogram from reference
        >>> target = reference.histogram_colors()
        >>>
        >>> # Create learnable model
        >>> model = LearnableColor.from_values(ColorValues(), ['brightness', 'gamma'])
        >>>
        >>> # Loss function
        >>> loss_fn = MomentMatchingLoss()
        >>>
        >>> # Training
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        >>> for epoch in range(100):
        ...     adjusted = model(source_colors)
        ...     loss = loss_fn(adjusted, target)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(self, include_skewness: bool = False):
        """Initialize moment matching loss.

        :param include_skewness: Include third moment (skewness)
        """
        super().__init__()
        self.include_skewness = include_skewness

    def forward(
        self,
        source_colors: torch.Tensor,
        target_hist: HistogramResult,
    ) -> torch.Tensor:
        """Compute moment matching loss.

        :param source_colors: Source colors [N, 3]
        :param target_hist: Target HistogramResult
        :returns: Scalar loss
        """
        device = source_colors.device
        dtype = source_colors.dtype

        target_mean = torch.tensor(target_hist.mean, device=device, dtype=dtype)
        target_std = torch.tensor(target_hist.std, device=device, dtype=dtype)

        if target_mean.ndim == 0:
            target_mean = target_mean.unsqueeze(0)
            target_std = target_std.unsqueeze(0)

        n_channels = min(len(target_mean), source_colors.shape[-1])

        loss = torch.tensor(0.0, device=device, dtype=dtype)

        for c in range(n_channels):
            if source_colors.ndim == 1:
                src = source_colors
            else:
                src = source_colors[:, c]

            src_mean = src.mean()
            src_std = src.std()

            # Mean and std loss
            loss = loss + (src_mean - target_mean[c]) ** 2
            loss = loss + (src_std - target_std[c]) ** 2

            # Optional skewness
            if self.include_skewness:
                src_skew = ((src - src_mean) ** 3).mean() / (src_std**3 + 1e-8)
                # Penalize skewness (assumes symmetric target)
                loss = loss + src_skew**2 * 0.1

        return loss / n_channels


class PerceptualColorLoss(nn.Module):
    """Comprehensive loss for visually pleasing color optimization.

    Addresses the flat histogram problem by combining:
    - Moment matching (mean, std)
    - Dynamic range preservation (5th/95th percentiles)
    - Contrast preservation (penalizes contrast reduction)
    - Parameter regularization (keeps parameters near neutral)

    This loss produces results that look good, not just statistically matched.

    Example:
        >>> from gsmod.torch.learn import LearnableColor
        >>> from gsmod import ColorValues
        >>>
        >>> model = LearnableColor.from_values(ColorValues(), learnable=['brightness', 'contrast'])
        >>> loss_fn = PerceptualColorLoss(
        ...     contrast_weight=0.5,
        ...     dynamic_range_weight=0.3,
        ...     regularization_weight=0.1
        ... )
        >>>
        >>> adjusted = model(source_colors)
        >>> loss = loss_fn(adjusted, target_hist, source_colors, model)
        >>> loss.backward()
    """

    def __init__(
        self,
        moment_weight: float = 1.0,
        contrast_weight: float = 0.5,
        dynamic_range_weight: float = 0.3,
        regularization_weight: float = 0.1,
        min_contrast_ratio: float = 0.7,
    ):
        """Initialize perceptual color loss.

        :param moment_weight: Weight for mean/std matching
        :param contrast_weight: Weight for contrast preservation
        :param dynamic_range_weight: Weight for dynamic range matching
        :param regularization_weight: Weight for parameter regularization
        :param min_contrast_ratio: Minimum allowed contrast ratio vs original
        """
        super().__init__()
        self.moment_weight = moment_weight
        self.contrast_weight = contrast_weight
        self.dynamic_range_weight = dynamic_range_weight
        self.regularization_weight = regularization_weight
        self.min_contrast_ratio = min_contrast_ratio

    def forward(
        self,
        adjusted_colors: torch.Tensor,
        target_hist: HistogramResult,
        original_colors: torch.Tensor,
        model: nn.Module | None = None,
    ) -> torch.Tensor:
        """Compute perceptual color loss.

        :param adjusted_colors: Adjusted colors [N, 3]
        :param target_hist: Target HistogramResult
        :param original_colors: Original colors before adjustment [N, 3]
        :param model: LearnableColor model for regularization (optional)
        :returns: Scalar loss
        """
        device = adjusted_colors.device
        dtype = adjusted_colors.dtype
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        # 1. Moment matching (mean, std)
        if self.moment_weight > 0:
            moment_loss = self._moment_loss(adjusted_colors, target_hist)
            loss = loss + self.moment_weight * moment_loss

        # 2. Contrast preservation
        if self.contrast_weight > 0:
            contrast_loss = self._contrast_loss(adjusted_colors, original_colors)
            loss = loss + self.contrast_weight * contrast_loss

        # 3. Dynamic range preservation
        if self.dynamic_range_weight > 0:
            range_loss = self._dynamic_range_loss(adjusted_colors, target_hist)
            loss = loss + self.dynamic_range_weight * range_loss

        # 4. Parameter regularization
        if self.regularization_weight > 0 and model is not None:
            reg_loss = self._regularization_loss(model)
            loss = loss + self.regularization_weight * reg_loss

        return loss

    def _moment_loss(self, colors: torch.Tensor, target_hist: HistogramResult) -> torch.Tensor:
        """Compute moment matching loss."""
        device = colors.device
        dtype = colors.dtype

        target_mean = torch.tensor(target_hist.mean, device=device, dtype=dtype)
        target_std = torch.tensor(target_hist.std, device=device, dtype=dtype)

        if target_mean.ndim == 0:
            target_mean = target_mean.unsqueeze(0)
            target_std = target_std.unsqueeze(0)

        n_channels = min(len(target_mean), colors.shape[-1])
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        for c in range(n_channels):
            src = colors[:, c] if colors.ndim > 1 else colors
            loss = loss + (src.mean() - target_mean[c]) ** 2
            loss = loss + (src.std() - target_std[c]) ** 2

        return loss / n_channels

    def _contrast_loss(self, adjusted: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Penalize contrast reduction.

        Measures contrast as std deviation and penalizes if adjusted
        has lower contrast than original * min_contrast_ratio.
        """
        device = adjusted.device
        dtype = adjusted.dtype
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        n_channels = adjusted.shape[-1] if adjusted.ndim > 1 else 1

        for c in range(n_channels):
            adj = adjusted[:, c] if adjusted.ndim > 1 else adjusted
            orig = original[:, c] if original.ndim > 1 else original

            adj_std = adj.std()
            orig_std = orig.std()

            # Penalize if contrast drops below threshold
            min_std = orig_std * self.min_contrast_ratio
            contrast_deficit = torch.relu(min_std - adj_std)
            loss = loss + contrast_deficit**2

        return loss / n_channels

    def _dynamic_range_loss(
        self, colors: torch.Tensor, target_hist: HistogramResult
    ) -> torch.Tensor:
        """Match dynamic range using percentiles."""
        device = colors.device
        dtype = colors.dtype

        # Use percentiles from target histogram if available
        if hasattr(target_hist, "percentile_5") and hasattr(target_hist, "percentile_95"):
            target_low = torch.tensor(target_hist.percentile_5, device=device, dtype=dtype)
            target_high = torch.tensor(target_hist.percentile_95, device=device, dtype=dtype)
        else:
            # Fall back to min/max from mean +/- 2*std
            target_mean = torch.tensor(target_hist.mean, device=device, dtype=dtype)
            target_std = torch.tensor(target_hist.std, device=device, dtype=dtype)
            target_low = target_mean - 2 * target_std
            target_high = target_mean + 2 * target_std

        if target_low.ndim == 0:
            target_low = target_low.unsqueeze(0)
            target_high = target_high.unsqueeze(0)

        n_channels = min(len(target_low), colors.shape[-1] if colors.ndim > 1 else 1)
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        for c in range(n_channels):
            src = colors[:, c] if colors.ndim > 1 else colors

            # Compute source percentiles (differentiable approximation)
            sorted_src, _ = torch.sort(src)
            n = len(sorted_src)
            idx_5 = int(0.05 * n)
            idx_95 = int(0.95 * n)

            src_low = sorted_src[idx_5]
            src_high = sorted_src[idx_95]

            # Match percentiles
            loss = loss + (src_low - target_low[c].clamp(0, 1)) ** 2
            loss = loss + (src_high - target_high[c].clamp(0, 1)) ** 2

        return loss / n_channels

    def _regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Regularize parameters toward neutral values."""
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        # Multiplicative parameters (neutral = 1.0)
        mult_params = ["brightness", "contrast", "saturation", "gamma", "vibrance"]
        for name in mult_params:
            if hasattr(model, name):
                param = getattr(model, name)
                if isinstance(param, nn.Parameter):
                    loss = loss + (param - 1.0) ** 2

        # Additive parameters (neutral = 0.0)
        add_params = [
            "temperature",
            "tint",
            "shadows",
            "highlights",
            "fade",
            "hue_shift",
            "shadow_tint_sat",
            "highlight_tint_sat",
        ]
        for name in add_params:
            if hasattr(model, name):
                param = getattr(model, name)
                if isinstance(param, nn.Parameter):
                    loss = loss + param**2

        return loss


class ContrastPreservationLoss(nn.Module):
    """Loss that specifically prevents contrast reduction.

    Use this in combination with other losses when flat histograms are a problem.

    Example:
        >>> loss_fn = ContrastPreservationLoss(min_ratio=0.8)
        >>> adjusted = model(original)
        >>> loss = loss_fn(adjusted, original)
    """

    def __init__(self, min_ratio: float = 0.7, use_percentile: bool = True):
        """Initialize contrast preservation loss.

        :param min_ratio: Minimum contrast ratio vs original (0.7 = 70%)
        :param use_percentile: Use percentile range instead of std
        """
        super().__init__()
        self.min_ratio = min_ratio
        self.use_percentile = use_percentile

    def forward(self, adjusted: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """Compute contrast preservation loss.

        :param adjusted: Adjusted colors [N, 3]
        :param original: Original colors [N, 3]
        :returns: Scalar loss
        """
        device = adjusted.device
        dtype = adjusted.dtype
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        n_channels = adjusted.shape[-1] if adjusted.ndim > 1 else 1

        for c in range(n_channels):
            adj = adjusted[:, c] if adjusted.ndim > 1 else adjusted
            orig = original[:, c] if original.ndim > 1 else original

            if self.use_percentile:
                # Use 5th-95th percentile range
                adj_sorted, _ = torch.sort(adj)
                orig_sorted, _ = torch.sort(orig)
                n = len(adj_sorted)
                idx_5, idx_95 = int(0.05 * n), int(0.95 * n)

                adj_range = adj_sorted[idx_95] - adj_sorted[idx_5]
                orig_range = orig_sorted[idx_95] - orig_sorted[idx_5]
            else:
                # Use standard deviation
                adj_range = adj.std()
                orig_range = orig.std()

            # Penalize if contrast drops below threshold
            min_range = orig_range * self.min_ratio
            contrast_deficit = torch.relu(min_range - adj_range)
            loss = loss + contrast_deficit**2

        return loss / n_channels


class ParameterBoundsLoss(nn.Module):
    """Soft penalty for parameters outside reasonable bounds.

    Prevents degenerate solutions by penalizing extreme parameter values.

    Example:
        >>> loss_fn = ParameterBoundsLoss()
        >>> reg_loss = loss_fn(model)
    """

    def __init__(
        self,
        brightness_range: tuple[float, float] = (0.5, 2.0),
        contrast_range: tuple[float, float] = (0.5, 2.0),
        saturation_range: tuple[float, float] = (0.3, 2.0),
        gamma_range: tuple[float, float] = (0.5, 2.0),
    ):
        """Initialize parameter bounds loss.

        :param brightness_range: (min, max) for brightness
        :param contrast_range: (min, max) for contrast
        :param saturation_range: (min, max) for saturation
        :param gamma_range: (min, max) for gamma
        """
        super().__init__()
        self.bounds = {
            "brightness": brightness_range,
            "contrast": contrast_range,
            "saturation": saturation_range,
            "gamma": gamma_range,
        }

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute parameter bounds loss.

        :param model: LearnableColor model
        :returns: Scalar loss
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        loss = torch.tensor(0.0, device=device, dtype=dtype)

        for name, (low, high) in self.bounds.items():
            if hasattr(model, name):
                param = getattr(model, name)
                if isinstance(param, nn.Parameter):
                    # Soft penalty outside bounds
                    below = torch.relu(low - param)
                    above = torch.relu(param - high)
                    loss = loss + below**2 + above**2

        return loss


def create_balanced_loss(
    target_hist: HistogramResult,
    contrast_weight: float = 0.5,
    regularization_weight: float = 0.1,
) -> nn.Module:
    """Create a balanced loss function for good-looking results.

    Factory function that creates a PerceptualColorLoss with sensible defaults.

    :param target_hist: Target histogram to match
    :param contrast_weight: How much to preserve contrast (0-1)
    :param regularization_weight: How much to regularize parameters (0-1)
    :returns: Configured loss module

    Example:
        >>> target = reference_data.histogram_colors()
        >>> loss_fn = create_balanced_loss(target, contrast_weight=0.5)
        >>>
        >>> for epoch in range(100):
        ...     adjusted = model(source_colors)
        ...     loss = loss_fn(adjusted, target, source_colors, model)
        ...     loss.backward()
    """
    return PerceptualColorLoss(
        moment_weight=1.0,
        contrast_weight=contrast_weight,
        dynamic_range_weight=0.3,
        regularization_weight=regularization_weight,
        min_contrast_ratio=0.7,
    )
