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
    :return: Soft histogram [n_bins], normalized to sum to 1

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
    weights = torch.exp(-((x_expanded - centers_expanded) ** 2) / (2 * sigma ** 2))

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
    :return: Soft histograms [3, n_bins]
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
        target_hist: "HistogramResult",
    ) -> torch.Tensor:
        """Compute moment matching loss.

        :param source_colors: Source colors [N, 3]
        :param target_hist: Target HistogramResult
        :return: Scalar loss
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
                src_skew = ((src - src_mean) ** 3).mean() / (src_std ** 3 + 1e-8)
                # Penalize skewness (assumes symmetric target)
                loss = loss + src_skew ** 2 * 0.1

        return loss / n_channels
