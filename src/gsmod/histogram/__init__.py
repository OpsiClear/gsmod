"""Histogram computation module.

Compute histograms and statistics for Gaussian Splatting data analysis.

Example:
    >>> from gsmod import GSDataPro, HistogramConfig
    >>>
    >>> data = GSDataPro.from_ply("scene.ply")
    >>> result = data.histogram_colors(HistogramConfig(n_bins=256))
    >>> print(f"Mean RGB: {result.mean}")
    >>>
    >>> # Get optimization suggestions
    >>> adjustment = result.to_color_values("vibrant")
    >>> data.color(adjustment)
"""

from gsmod.histogram.apply import (
    compute_histogram_colors,
    compute_histogram_opacity,
    compute_histogram_positions,
    compute_histogram_scales,
)
from gsmod.histogram.loss import (
    MomentMatchingLoss,
    soft_histogram,
    soft_histogram_rgb,
)
from gsmod.histogram.result import HistogramResult

__all__ = [
    "compute_histogram_colors",
    "compute_histogram_opacity",
    "compute_histogram_scales",
    "compute_histogram_positions",
    "HistogramResult",
    # Differentiable loss
    "MomentMatchingLoss",
    "soft_histogram",
    "soft_histogram_rgb",
]
