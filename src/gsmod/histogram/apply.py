"""Histogram computation functions."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from gsmod.config.values import HistogramConfig
from gsmod.histogram.kernels import (
    NUMBA_AVAILABLE,
    compute_stats_numba,
    compute_stats_rgb_numba,
    histogram_1d_numba,
    histogram_rgb_numba,
)
from gsmod.histogram.result import HistogramResult

logger = logging.getLogger(__name__)


def compute_histogram_colors(
    sh0: NDArray[np.float32],
    config: HistogramConfig | None = None,
) -> HistogramResult:
    """Compute histogram of RGB color values.

    :param sh0: Color values [N, 3] in [0, 1]
    :param config: Histogram configuration
    :return: HistogramResult with counts and statistics
    """
    if config is None:
        config = HistogramConfig()

    N = sh0.shape[0]
    n_bins = config.n_bins

    # Handle empty data
    if N == 0:
        return HistogramResult.empty(n_bins, n_channels=3)

    # Ensure input is float32 and contiguous
    if sh0.dtype != np.float32:
        sh0 = sh0.astype(np.float32)
    if not sh0.flags["C_CONTIGUOUS"]:
        sh0 = np.ascontiguousarray(sh0)

    # Determine range
    if config.min_value is not None:
        min_val = config.min_value
    else:
        min_val = float(sh0.min())

    if config.max_value is not None:
        max_val = config.max_value
    else:
        max_val = float(sh0.max())

    # Allocate outputs
    counts_r = np.zeros(n_bins, dtype=np.int64)
    counts_g = np.zeros(n_bins, dtype=np.int64)
    counts_b = np.zeros(n_bins, dtype=np.int64)

    # Compute histograms
    if NUMBA_AVAILABLE:
        histogram_rgb_numba(sh0, n_bins, min_val, max_val, counts_r, counts_g, counts_b)
    else:
        # NumPy fallback
        counts_r, _ = np.histogram(sh0[:, 0], bins=n_bins, range=(min_val, max_val))
        counts_g, _ = np.histogram(sh0[:, 1], bins=n_bins, range=(min_val, max_val))
        counts_b, _ = np.histogram(sh0[:, 2], bins=n_bins, range=(min_val, max_val))

    # Stack counts
    counts = np.stack([counts_r, counts_g, counts_b])

    # Compute statistics
    mean = np.zeros(3, dtype=np.float64)
    std = np.zeros(3, dtype=np.float64)
    min_arr = np.zeros(3, dtype=np.float64)
    max_arr = np.zeros(3, dtype=np.float64)

    if NUMBA_AVAILABLE:
        compute_stats_rgb_numba(sh0, mean, std, min_arr, max_arr)
    else:
        mean = np.mean(sh0, axis=0).astype(np.float64)
        std = np.std(sh0, axis=0).astype(np.float64)
        min_arr = np.min(sh0, axis=0).astype(np.float64)
        max_arr = np.max(sh0, axis=0).astype(np.float64)

    # Normalize if requested
    if config.normalize:
        bin_width = (max_val - min_val) / n_bins
        counts = counts.astype(np.float64) / (N * bin_width)

    return HistogramResult(
        counts=counts,
        bin_edges=np.linspace(min_val, max_val, n_bins + 1),
        mean=mean,
        std=std,
        min_val=min_arr,
        max_val=max_arr,
        n_samples=N,
    )


def compute_histogram_opacity(
    opacities: NDArray[np.float32],
    config: HistogramConfig | None = None,
) -> HistogramResult:
    """Compute histogram of opacity values.

    :param opacities: Opacity values [N] in [0, 1]
    :param config: Histogram configuration
    :return: HistogramResult with counts and statistics
    """
    if config is None:
        config = HistogramConfig()

    # Flatten if needed
    if opacities.ndim > 1:
        opacities = opacities.flatten()

    N = opacities.shape[0]
    n_bins = config.n_bins

    # Handle empty data
    if N == 0:
        return HistogramResult.empty(n_bins, n_channels=1)

    # Ensure input is float32 and contiguous
    if opacities.dtype != np.float32:
        opacities = opacities.astype(np.float32)
    if not opacities.flags["C_CONTIGUOUS"]:
        opacities = np.ascontiguousarray(opacities)

    # Determine range
    if config.min_value is not None:
        min_val = config.min_value
    else:
        min_val = float(opacities.min())

    if config.max_value is not None:
        max_val = config.max_value
    else:
        max_val = float(opacities.max())

    # Allocate output
    counts = np.zeros(n_bins, dtype=np.int64)

    # Compute histogram
    if NUMBA_AVAILABLE:
        histogram_1d_numba(opacities, n_bins, min_val, max_val, counts)
    else:
        counts, _ = np.histogram(opacities, bins=n_bins, range=(min_val, max_val))

    # Compute statistics
    if NUMBA_AVAILABLE:
        mean, std, min_stat, max_stat = compute_stats_numba(opacities)
    else:
        mean = float(np.mean(opacities))
        std = float(np.std(opacities))
        min_stat = float(np.min(opacities))
        max_stat = float(np.max(opacities))

    # Normalize if requested
    if config.normalize:
        bin_width = (max_val - min_val) / n_bins
        counts = counts.astype(np.float64) / (N * bin_width)

    return HistogramResult(
        counts=counts,
        bin_edges=np.linspace(min_val, max_val, n_bins + 1),
        mean=np.array(mean, dtype=np.float64),
        std=np.array(std, dtype=np.float64),
        min_val=np.array(min_stat, dtype=np.float64),
        max_val=np.array(max_stat, dtype=np.float64),
        n_samples=N,
    )


def compute_histogram_scales(
    scales: NDArray[np.float32],
    config: HistogramConfig | None = None,
) -> HistogramResult:
    """Compute histogram of scale values.

    Uses the mean scale across all 3 dimensions for each Gaussian.

    :param scales: Scale values [N, 3]
    :param config: Histogram configuration
    :return: HistogramResult with counts and statistics
    """
    if config is None:
        config = HistogramConfig()

    N = scales.shape[0]
    n_bins = config.n_bins

    # Handle empty data
    if N == 0:
        return HistogramResult.empty(n_bins, n_channels=1)

    # Compute mean scale per Gaussian
    if scales.ndim > 1:
        mean_scales = np.mean(scales, axis=1).astype(np.float32)
    else:
        mean_scales = scales.astype(np.float32)

    # Ensure contiguous
    if not mean_scales.flags["C_CONTIGUOUS"]:
        mean_scales = np.ascontiguousarray(mean_scales)

    # Determine range
    if config.min_value is not None:
        min_val = config.min_value
    else:
        min_val = float(mean_scales.min())

    if config.max_value is not None:
        max_val = config.max_value
    else:
        max_val = float(mean_scales.max())

    # Allocate output
    counts = np.zeros(n_bins, dtype=np.int64)

    # Compute histogram
    if NUMBA_AVAILABLE:
        histogram_1d_numba(mean_scales, n_bins, min_val, max_val, counts)
    else:
        counts, _ = np.histogram(mean_scales, bins=n_bins, range=(min_val, max_val))

    # Compute statistics
    if NUMBA_AVAILABLE:
        mean, std, min_stat, max_stat = compute_stats_numba(mean_scales)
    else:
        mean = float(np.mean(mean_scales))
        std = float(np.std(mean_scales))
        min_stat = float(np.min(mean_scales))
        max_stat = float(np.max(mean_scales))

    # Normalize if requested
    if config.normalize:
        bin_width = (max_val - min_val) / n_bins
        counts = counts.astype(np.float64) / (N * bin_width)

    return HistogramResult(
        counts=counts,
        bin_edges=np.linspace(min_val, max_val, n_bins + 1),
        mean=np.array(mean, dtype=np.float64),
        std=np.array(std, dtype=np.float64),
        min_val=np.array(min_stat, dtype=np.float64),
        max_val=np.array(max_stat, dtype=np.float64),
        n_samples=N,
    )


def compute_histogram_positions(
    means: NDArray[np.float32],
    config: HistogramConfig | None = None,
    axis: int | None = None,
) -> HistogramResult:
    """Compute histogram of position values.

    :param means: Position values [N, 3]
    :param config: Histogram configuration
    :param axis: Axis to histogram (0=X, 1=Y, 2=Z, None=distance from origin)
    :return: HistogramResult with counts and statistics
    """
    if config is None:
        config = HistogramConfig()

    N = means.shape[0]
    n_bins = config.n_bins

    # Handle empty data
    if N == 0:
        return HistogramResult.empty(n_bins, n_channels=1)

    # Extract values based on axis
    if axis is not None:
        values = means[:, axis].astype(np.float32)
    else:
        # Distance from origin
        values = np.linalg.norm(means, axis=1).astype(np.float32)

    # Ensure contiguous
    if not values.flags["C_CONTIGUOUS"]:
        values = np.ascontiguousarray(values)

    # Determine range
    if config.min_value is not None:
        min_val = config.min_value
    else:
        min_val = float(values.min())

    if config.max_value is not None:
        max_val = config.max_value
    else:
        max_val = float(values.max())

    # Allocate output
    counts = np.zeros(n_bins, dtype=np.int64)

    # Compute histogram
    if NUMBA_AVAILABLE:
        histogram_1d_numba(values, n_bins, min_val, max_val, counts)
    else:
        counts, _ = np.histogram(values, bins=n_bins, range=(min_val, max_val))

    # Compute statistics
    if NUMBA_AVAILABLE:
        mean, std, min_stat, max_stat = compute_stats_numba(values)
    else:
        mean = float(np.mean(values))
        std = float(np.std(values))
        min_stat = float(np.min(values))
        max_stat = float(np.max(values))

    # Normalize if requested
    if config.normalize:
        bin_width = (max_val - min_val) / n_bins
        counts = counts.astype(np.float64) / (N * bin_width)

    return HistogramResult(
        counts=counts,
        bin_edges=np.linspace(min_val, max_val, n_bins + 1),
        mean=np.array(mean, dtype=np.float64),
        std=np.array(std, dtype=np.float64),
        min_val=np.array(min_stat, dtype=np.float64),
        max_val=np.array(max_stat, dtype=np.float64),
        n_samples=N,
    )
