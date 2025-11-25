"""Numba-optimized histogram computation kernels."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Track Numba availability
NUMBA_AVAILABLE = False

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
    logger.debug("Numba available for histogram kernels")
except ImportError:
    logger.debug("Numba not available, using NumPy fallback for histogram")

    # Fallback decorators
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args):
        return range(*args)


# Numba-optimized histogram kernel for RGB data
# Note: Not using parallel=True because histogram accumulation has race conditions
@njit(fastmath=True, cache=True, nogil=True)
def histogram_rgb_numba(
    colors: NDArray[np.float32],
    n_bins: int,
    min_val: float,
    max_val: float,
    out_r: NDArray[np.int64],
    out_g: NDArray[np.int64],
    out_b: NDArray[np.int64],
) -> None:
    """Compute RGB histogram with Numba parallel optimization.

    Note: For parallel safety, this uses thread-local histograms and reduces.

    :param colors: Input colors [N, 3]
    :param n_bins: Number of bins
    :param min_val: Minimum value for binning
    :param max_val: Maximum value for binning
    :param out_r: Output histogram for R channel [n_bins]
    :param out_g: Output histogram for G channel [n_bins]
    :param out_b: Output histogram for B channel [n_bins]
    """
    N = colors.shape[0]
    if N == 0:
        return

    scale = n_bins / (max_val - min_val) if max_val > min_val else 1.0

    # Sequential accumulation (parallel histogramming has race conditions)
    for i in range(N):
        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        # Compute bin indices
        r_bin = int((r - min_val) * scale)
        g_bin = int((g - min_val) * scale)
        b_bin = int((b - min_val) * scale)

        # Clamp to valid range
        r_bin = max(0, min(n_bins - 1, r_bin))
        g_bin = max(0, min(n_bins - 1, g_bin))
        b_bin = max(0, min(n_bins - 1, b_bin))

        out_r[r_bin] += 1
        out_g[g_bin] += 1
        out_b[b_bin] += 1


# Note: Not using parallel=True because histogram accumulation has race conditions
@njit(fastmath=True, cache=True, nogil=True)
def histogram_1d_numba(
    values: NDArray[np.float32],
    n_bins: int,
    min_val: float,
    max_val: float,
    out: NDArray[np.int64],
) -> None:
    """Compute 1D histogram with Numba optimization.

    :param values: Input values [N] or [N, C] (uses first column if 2D)
    :param n_bins: Number of bins
    :param min_val: Minimum value for binning
    :param max_val: Maximum value for binning
    :param out: Output histogram [n_bins]
    """
    N = values.shape[0]
    if N == 0:
        return

    scale = n_bins / (max_val - min_val) if max_val > min_val else 1.0

    for i in range(N):
        if values.ndim == 1:
            v = values[i]
        else:
            v = values[i, 0]

        bin_idx = int((v - min_val) * scale)
        bin_idx = max(0, min(n_bins - 1, bin_idx))
        out[bin_idx] += 1


# Note: Sequential Welford's algorithm - not parallelizable
@njit(fastmath=True, cache=True, nogil=True)
def compute_stats_numba(
    data: NDArray[np.float32],
) -> tuple[float, float, float, float]:
    """Compute mean, std, min, max with Numba.

    Uses Welford's online algorithm for numerical stability.

    :param data: Input data [N]
    :returns: Tuple of (mean, std, min_val, max_val)
    """
    N = data.shape[0]
    if N == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Welford's algorithm for mean and variance
    mean = 0.0
    M2 = 0.0
    min_val = data[0]
    max_val = data[0]

    for i in range(N):
        x = data[i]
        delta = x - mean
        mean += delta / (i + 1)
        delta2 = x - mean
        M2 += delta * delta2

        if x < min_val:
            min_val = x
        if x > max_val:
            max_val = x

    variance = M2 / N if N > 0 else 0.0
    std = np.sqrt(variance)

    return mean, std, min_val, max_val


# Note: Sequential per-channel stats - not parallelizable
@njit(fastmath=True, cache=True, nogil=True)
def compute_stats_rgb_numba(
    colors: NDArray[np.float32],
    out_mean: NDArray[np.float64],
    out_std: NDArray[np.float64],
    out_min: NDArray[np.float64],
    out_max: NDArray[np.float64],
) -> None:
    """Compute per-channel statistics for RGB data.

    :param colors: Input colors [N, 3]
    :param out_mean: Output mean [3]
    :param out_std: Output std [3]
    :param out_min: Output min [3]
    :param out_max: Output max [3]
    """
    N = colors.shape[0]
    if N == 0:
        for c in range(3):
            out_mean[c] = 0.0
            out_std[c] = 0.0
            out_min[c] = 0.0
            out_max[c] = 0.0
        return

    # Process each channel
    for c in range(3):
        mean = 0.0
        M2 = 0.0
        min_val = colors[0, c]
        max_val = colors[0, c]

        for i in range(N):
            x = colors[i, c]
            delta = x - mean
            mean += delta / (i + 1)
            delta2 = x - mean
            M2 += delta * delta2

            if x < min_val:
                min_val = x
            if x > max_val:
                max_val = x

        variance = M2 / N if N > 0 else 0.0
        out_mean[c] = mean
        out_std[c] = np.sqrt(variance)
        out_min[c] = min_val
        out_max[c] = max_val


def warmup_histogram_kernels() -> None:
    """Warm up Numba JIT compilation for histogram kernels.

    Should be called on module import to avoid first-call overhead.
    """
    if not NUMBA_AVAILABLE:
        return

    # Dummy data
    colors = np.random.rand(100, 3).astype(np.float32)
    values = np.random.rand(100).astype(np.float32)

    # Allocate outputs
    out_r = np.zeros(32, dtype=np.int64)
    out_g = np.zeros(32, dtype=np.int64)
    out_b = np.zeros(32, dtype=np.int64)
    out_1d = np.zeros(32, dtype=np.int64)
    out_mean = np.zeros(3, dtype=np.float64)
    out_std = np.zeros(3, dtype=np.float64)
    out_min = np.zeros(3, dtype=np.float64)
    out_max = np.zeros(3, dtype=np.float64)

    # Trigger compilation
    histogram_rgb_numba(colors, 32, 0.0, 1.0, out_r, out_g, out_b)
    histogram_1d_numba(values, 32, 0.0, 1.0, out_1d)
    compute_stats_numba(values)
    compute_stats_rgb_numba(colors, out_mean, out_std, out_min, out_max)

    logger.debug("Histogram Numba kernels warmed up")


# Warmup on import
warmup_histogram_kernels()
