"""NumPy-based Spherical Harmonics utilities for CPU color operations.

This module provides NumPy equivalents of the PyTorch SH utilities for
CPU-based color editing with full SH support.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Constants
# =============================================================================

# SH DC normalization constant: 1 / (2 * sqrt(pi))
SH_C0 = 0.28209479177387814

# Luminance weights (Rec. 709)
LUMA_R = 0.299
LUMA_G = 0.587
LUMA_B = 0.114


# =============================================================================
# Conversion Functions
# =============================================================================


def sh_to_rgb(sh: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert SH DC to RGB format.

    Formula: rgb = sh * SH_C0 + 0.5

    :param sh: SH DC coefficients [N, 3]
    :returns: RGB values [N, 3] in range [0, 1]
    """
    return np.clip(sh * SH_C0 + 0.5, 0, 1)


def rgb_to_sh(rgb: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert RGB to SH DC format.

    Formula: sh = (rgb - 0.5) / SH_C0

    :param rgb: RGB values [N, 3] in range [0, 1]
    :returns: SH DC coefficients [N, 3]
    """
    return (rgb - 0.5) / SH_C0


def compute_luminance(
    rgb: NDArray[np.float32], weights: NDArray[np.float32] | None = None
) -> NDArray[np.float32]:
    """Compute luminance from RGB values.

    :param rgb: RGB tensor [..., 3]
    :param weights: Optional custom weights [3], defaults to Rec. 709
    :returns: Luminance values [..., 1]
    """
    if weights is None:
        weights = np.array([LUMA_R, LUMA_G, LUMA_B], dtype=np.float32)
    return np.sum(rgb * weights, axis=-1, keepdims=True)


# =============================================================================
# Matrix Builders
# =============================================================================


def build_saturation_matrix(factor: float) -> NDArray[np.float32]:
    """Build 3×3 saturation adjustment matrix.

    :param factor: Saturation factor (1.0 = no change, 0 = grayscale, >1 = more saturated)
    :returns: 3×3 saturation matrix
    """
    s = factor
    w = np.array([LUMA_R, LUMA_G, LUMA_B], dtype=np.float32)

    # Build matrix: M[i,j] = (1-s) * w[j] + s * δ[i,j]
    M = (1 - s) * np.ones((3, 1), dtype=np.float32) @ w.reshape(1, 3)
    M = M + s * np.eye(3, dtype=np.float32)

    return M


def build_temperature_matrix(temp: float) -> NDArray[np.float32]:
    """Build temperature adjustment matrix.

    :param temp: Temperature adjustment (-1.0 to 1.0)
    :returns: 3×3 diagonal temperature matrix
    """
    return np.diag(np.array([1.0 + temp * 0.3, 1.0, 1.0 - temp * 0.3], dtype=np.float32))


def build_tint_matrix(tint: float) -> NDArray[np.float32]:
    """Build tint adjustment matrix.

    :param tint: Tint adjustment (-1.0 to 1.0)
    :returns: 3×3 diagonal tint matrix
    """
    return np.diag(np.array([1.0, 1.0 - tint * 0.1, 1.0], dtype=np.float32))


def build_hue_matrix(degrees: float) -> NDArray[np.float32]:
    """Build luminance-preserving hue rotation matrix.

    :param degrees: Hue rotation in degrees (-180 to 180)
    :returns: 3×3 hue rotation matrix
    """
    theta = math.radians(degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    return np.array(
        [
            [
                0.213 + 0.787 * cos_t - 0.213 * sin_t,
                0.715 - 0.715 * cos_t - 0.715 * sin_t,
                0.072 - 0.072 * cos_t + 0.928 * sin_t,
            ],
            [
                0.213 - 0.213 * cos_t + 0.143 * sin_t,
                0.715 + 0.285 * cos_t + 0.140 * sin_t,
                0.072 - 0.072 * cos_t - 0.283 * sin_t,
            ],
            [
                0.213 - 0.213 * cos_t - 0.787 * sin_t,
                0.715 - 0.715 * cos_t + 0.715 * sin_t,
                0.072 + 0.928 * cos_t + 0.072 * sin_t,
            ],
        ],
        dtype=np.float32,
    )


# =============================================================================
# SH Operations
# =============================================================================


def apply_matrix_to_sh(
    sh0: NDArray[np.float32],
    shN: NDArray[np.float32] | None,
    matrix: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """Apply 3×3 color matrix to all SH bands.

    :param sh0: DC component [N, 3]
    :param shN: Higher-order coefficients [N, K, 3] or None
    :param matrix: 3×3 transformation matrix
    :returns: Tuple of (transformed_sh0, transformed_shN)
    """
    # Apply to DC (sh0): [N, 3] @ [3, 3]^T = [N, 3]
    sh0_out = sh0 @ matrix.T

    # Apply to higher-order SH if present: [N, K, 3] @ [3, 3]^T = [N, K, 3]
    shN_out = None
    if shN is not None:
        shN_out = shN @ matrix.T

    return sh0_out, shN_out


def apply_scale_to_sh(
    sh0: NDArray[np.float32], shN: NDArray[np.float32] | None, scale: float
) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """Apply multiplicative scaling to all SH bands.

    :param sh0: DC component [N, 3]
    :param shN: Higher-order coefficients [N, K, 3] or None
    :param scale: Scaling factor
    :returns: Tuple of (scaled_sh0, scaled_shN)
    """
    sh0_out = sh0 * scale
    shN_out = shN * scale if shN is not None else None
    return sh0_out, shN_out


def apply_contrast_to_sh(
    sh0: NDArray[np.float32],
    shN: NDArray[np.float32] | None,
    factor: float,
    is_rgb_format: bool,
) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
    """Apply contrast adjustment to SH data.

    :param sh0: DC component [N, 3]
    :param shN: Higher-order coefficients [N, K, 3] or None
    :param factor: Contrast factor
    :param is_rgb_format: True if sh0 is in RGB format [0,1]
    :returns: Tuple of (contrasted_sh0, contrasted_shN)
    """
    if is_rgb_format:
        # RGB format: contrast around 0.5
        mid = 0.5
        offset = mid * (1 - factor)
        sh0_out = sh0 * factor + offset
    else:
        # SH format: contrast around 0.0
        sh0_out = sh0 * factor

    # Scale shN (no offset, as it represents variation)
    shN_out = shN * factor if shN is not None else None

    return sh0_out, shN_out


def apply_fade_to_sh(
    sh0: NDArray[np.float32],
    amount: float,
    is_rgb_format: bool,
) -> NDArray[np.float32]:
    """Apply fade (black point lift) to DC component only.

    :param sh0: DC component [N, 3]
    :param amount: Fade amount (0.0 = no change, 1.0 = full lift)
    :param is_rgb_format: True if sh0 is in RGB format
    :returns: Faded sh0
    """
    if is_rgb_format:
        return sh0 * (1.0 - amount) + amount
    else:
        # In SH space, need to convert the offset
        sh_offset = amount / SH_C0
        return sh0 + sh_offset * (1.0 - amount)
