"""
Utility functions for LUT operations (NumPy/CPU implementation)

Provides helper functions for linear interpolation, nearest neighbor lookup,
and opacity adjustments.
"""

from __future__ import annotations

import logging

import numpy as np
from gsply import GSData

logger = logging.getLogger(__name__)


def linear_interp_1d(x: np.ndarray, centers: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Perform 1D linear interpolation using sorted cluster centers.

    :param x: Input values to interpolate [N]
    :param centers: Sorted cluster centers [K]
    :param values: Precomputed output values at cluster centers [K]
    :returns: Interpolated values [N]

    Example:
        >>> centers = np.array([0.0, 0.5, 1.0])
        >>> values = np.array([0.0, 0.25, 1.0])
        >>> x = np.array([0.25, 0.75])
        >>> result = linear_interp_1d(x, centers, values)
        >>> print(result)
        [0.125 0.625]
    """
    indices = np.searchsorted(centers, x)
    indices = np.clip(indices, 1, len(centers) - 1)

    left_idx = indices - 1
    right_idx = indices

    left_centers = centers[left_idx]
    right_centers = centers[right_idx]
    left_values = values[left_idx]
    right_values = values[right_idx]

    alpha = (x - left_centers) / (right_centers - left_centers + 1e-8)
    result = left_values + alpha * (right_values - left_values)

    return result


def nearest_neighbor_1d(x: np.ndarray, centers: np.ndarray, values: np.ndarray) -> np.ndarray:
    """
    Perform 1D nearest neighbor lookup.

    :param x: Input values [N]
    :param centers: Cluster centers [K] (need not be sorted)
    :param values: Precomputed output values at cluster centers [K]
    :returns: Values at nearest neighbors [N]

    Example:
        >>> centers = np.array([0.0, 0.5, 1.0])
        >>> values = np.array([0.0, 0.25, 1.0])
        >>> x = np.array([0.1, 0.7])
        >>> result = nearest_neighbor_1d(x, centers, values)
        >>> print(result)
        [0.   0.25]
    """
    # Compute distances from each x to each center
    x_expanded = x[:, np.newaxis]  # [N, 1]
    centers_expanded = centers[np.newaxis, :]  # [1, K]
    distances = np.abs(x_expanded - centers_expanded)  # [N, K]

    nearest_idx = np.argmin(distances, axis=1)
    return values[nearest_idx]


def multiply_opacity(data: GSData, factor: float, inplace: bool = True) -> GSData:
    """
    Multiply all Gaussian opacity values by a factor.

    Useful for fading scenes in/out or adjusting overall transparency.

    :param data: GSData object containing Gaussian data
    :param factor: Opacity multiplier (1.0=no change, >1.0=more opaque, <1.0=more transparent)
    :param inplace: If True, modifies input GSData directly
    :returns: GSData with adjusted opacity values

    Example:
        >>> # Fade scene to 50% opacity
        >>> faded = multiply_opacity(data, 0.5, inplace=True)
        >>>
        >>> # Make scene more opaque
        >>> opaque = multiply_opacity(data, 1.5, inplace=False)

    Note:
        Opacity values are clamped to [0, 1] after multiplication.
    """
    if factor <= 0:
        raise ValueError(
            f"factor={factor} must be positive (> 0). "
            "Use values <1.0 for transparency, >1.0 for opacity."
        )

    # Make a copy if not inplace
    if not inplace:
        data = data.copy()

    # Multiply opacity and clamp to [0, 1]
    data.opacities[:] = np.clip(data.opacities * factor, 0.0, 1.0)

    logger.info(f"[multiply_opacity] Applied factor={factor:.2f} to {len(data)} Gaussians")
    return data
