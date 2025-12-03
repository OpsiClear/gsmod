"""Spherical Harmonics utilities for color operations.

This module provides constants, conversion functions, and matrix builders
for mathematically correct color operations on spherical harmonics data.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

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


def sh_to_rgb(sh: Tensor) -> Tensor:
    """Convert SH DC to RGB format.

    Formula: rgb = sh * SH_C0 + 0.5

    :param sh: SH DC coefficients [N, 3]
    :returns: RGB values [N, 3] in range [0, 1]
    """
    return (sh * SH_C0 + 0.5).clamp(0, 1)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    """Convert RGB to SH DC format.

    Formula: sh = (rgb - 0.5) / SH_C0

    :param rgb: RGB values [N, 3] in range [0, 1]
    :returns: SH DC coefficients [N, 3]
    """
    return (rgb - 0.5) / SH_C0


def compute_luminance(rgb: Tensor, weights: Tensor | None = None) -> Tensor:
    """Compute luminance from RGB values.

    :param rgb: RGB tensor [..., 3]
    :param weights: Optional custom weights [3], defaults to Rec. 709
    :returns: Luminance values [..., 1]
    """
    if weights is None:
        weights = torch.tensor([LUMA_R, LUMA_G, LUMA_B], device=rgb.device, dtype=rgb.dtype)
    return torch.sum(rgb * weights, dim=-1, keepdim=True)


# =============================================================================
# Matrix Builders
# =============================================================================


def build_saturation_matrix(factor: float, device: torch.device | str = "cpu") -> Tensor:
    """Build 3×3 saturation adjustment matrix.

    Saturation is a linear operation: c' = L + s·(c - L)
    where L is luminance. This expands to a 3×3 matrix multiplication.

    Matrix formula:
        M = (1-s) * outer([1,1,1], [w_r, w_g, w_b]) + s * I

    Example (s=1.3):
        [[1.2103, -0.1761, -0.0342],
         [-0.0897,  1.1239, -0.0342],
         [-0.0897, -0.1761,  1.2658]]

    :param factor: Saturation factor (1.0 = no change, 0 = grayscale, >1 = more saturated)
    :param device: Target device
    :returns: 3×3 saturation matrix
    """
    s = factor
    w = torch.tensor([LUMA_R, LUMA_G, LUMA_B], device=device, dtype=torch.float32)

    # Build matrix: M[i,j] = (1-s) * w[j] + s * δ[i,j]
    M = (1 - s) * torch.ones(3, 1, device=device, dtype=torch.float32) @ w.unsqueeze(0)
    M = M + s * torch.eye(3, device=device, dtype=torch.float32)

    return M


def build_temperature_matrix(temp: float, device: torch.device | str = "cpu") -> Tensor:
    """Build temperature adjustment matrix.

    Positive temp = warmer (boost red, reduce blue)
    Negative temp = cooler (reduce red, boost blue)

    :param temp: Temperature adjustment (-1.0 to 1.0)
    :param device: Target device
    :returns: 3×3 diagonal temperature matrix
    """
    return torch.diag(
        torch.tensor(
            [1.0 + temp * 0.3, 1.0, 1.0 - temp * 0.3],
            device=device,
            dtype=torch.float32,
        )
    )


def build_tint_matrix(tint: float, device: torch.device | str = "cpu") -> Tensor:
    """Build tint adjustment matrix.

    Positive tint = magenta (reduce green)
    Negative tint = green (boost green)

    This is the complement to temperature for white balance control.

    :param tint: Tint adjustment (-1.0 to 1.0)
    :param device: Target device
    :returns: 3×3 diagonal tint matrix
    """
    return torch.diag(
        torch.tensor([1.0, 1.0 - tint * 0.1, 1.0], device=device, dtype=torch.float32)
    )


def build_hue_matrix(degrees: float, device: torch.device | str = "cpu") -> Tensor:
    """Build luminance-preserving hue rotation matrix.

    This matrix rotates colors in RGB space while preserving luminance,
    approximating a hue shift in HSL/HSV color space.

    :param degrees: Hue rotation in degrees (-180 to 180)
    :param device: Target device
    :returns: 3×3 hue rotation matrix
    """
    theta = math.radians(degrees)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    # Luminance-preserving hue rotation matrix
    # Based on standard RGB-to-HSV rotation with luminance preservation
    return torch.tensor(
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
        device=device,
        dtype=torch.float32,
    )


# =============================================================================
# SH Operations
# =============================================================================


def apply_matrix_to_sh(
    sh0: Tensor, shN: Tensor | None, matrix: Tensor
) -> tuple[Tensor, Tensor | None]:
    """Apply 3×3 color matrix to all SH bands.

    Linear color operations (saturation, temperature, tint, hue) can be
    expressed as 3×3 matrix multiplications. Because SH is a linear
    representation, we can apply the matrix to all bands.

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
    sh0: Tensor, shN: Tensor | None, scale: float
) -> tuple[Tensor, Tensor | None]:
    """Apply multiplicative scaling to all SH bands.

    Used for brightness adjustment.

    :param sh0: DC component [N, 3]
    :param shN: Higher-order coefficients [N, K, 3] or None
    :param scale: Scaling factor
    :returns: Tuple of (scaled_sh0, scaled_shN)
    """
    sh0_out = sh0 * scale
    shN_out = shN * scale if shN is not None else None
    return sh0_out, shN_out


def apply_contrast_to_sh(
    sh0: Tensor,
    shN: Tensor | None,
    factor: float,
    is_rgb_format: bool,
) -> tuple[Tensor, Tensor | None]:
    """Apply contrast adjustment to SH data.

    Contrast: c' = (c - mid) * factor + mid

    For SH format, mid = 0.0 (neutral SH is 0)
    For RGB format, mid = 0.5

    The offset only affects DC (sh0), while scale affects all bands.

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
        # In SH space, 0 is the neutral value
        sh0_out = sh0 * factor

    # Scale shN (no offset, as it represents variation)
    shN_out = shN * factor if shN is not None else None

    return sh0_out, shN_out


def apply_fade_to_sh(
    sh0: Tensor,
    amount: float,
    is_rgb_format: bool,
) -> Tensor:
    """Apply fade (black point lift) to DC component only.

    Fade formula: c' = c * (1 - fade) + fade

    This only affects the DC term (average color), not the directional
    variation stored in higher-order SH.

    :param sh0: DC component [N, 3]
    :param amount: Fade amount (0.0 = no change, 1.0 = full lift)
    :param is_rgb_format: True if sh0 is in RGB format
    :returns: Faded sh0
    """
    if is_rgb_format:
        return sh0 * (1.0 - amount) + amount
    else:
        # In SH space, need to convert the offset
        # RGB offset 'amount' becomes SH offset 'amount / SH_C0'
        sh_offset = amount / SH_C0
        return sh0 + sh_offset * (1.0 - amount)
