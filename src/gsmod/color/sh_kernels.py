"""Numba-optimized kernels for SH-aware color operations.

Provides JIT-compiled kernels that correctly handle all SH bands (sh0 + shN)
while maintaining 10-30x speedup over pure NumPy.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray

# =============================================================================
# SH Constants
# =============================================================================

SH_C0 = 0.28209479177387814
LUMA_R = 0.299
LUMA_G = 0.587
LUMA_B = 0.114


# =============================================================================
# Core SH Kernels
# =============================================================================


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def apply_matrix_to_sh_numba(
    sh0: NDArray[np.float32],
    shN: NDArray[np.float32],
    matrix: NDArray[np.float32],
    sh0_out: NDArray[np.float32],
    shN_out: NDArray[np.float32],
) -> None:
    """Apply 3×3 matrix to all SH bands (optimized).

    :param sh0: DC component [N, 3]
    :param shN: Higher-order SH [N, K, 3] or empty
    :param matrix: 3×3 transformation matrix
    :param sh0_out: Output DC [N, 3]
    :param shN_out: Output higher-order [N, K, 3] or empty
    """
    N = sh0.shape[0]

    # Apply to sh0: [N, 3] @ [3, 3]^T
    for i in prange(N):
        for c in range(3):
            val = 0.0
            for j in range(3):
                val += sh0[i, j] * matrix[c, j]
            sh0_out[i, c] = val

    # Apply to shN if present
    if shN.size > 0:
        K = shN.shape[1]
        for i in prange(N):
            for k in range(K):
                for c in range(3):
                    val = 0.0
                    for j in range(3):
                        val += shN[i, k, j] * matrix[c, j]
                    shN_out[i, k, c] = val


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def apply_scale_to_sh_numba(
    sh0: NDArray[np.float32],
    shN: NDArray[np.float32],
    scale: float,
    sh0_out: NDArray[np.float32],
    shN_out: NDArray[np.float32],
) -> None:
    """Apply uniform scaling to all SH bands (optimized).

    :param sh0: DC component [N, 3]
    :param shN: Higher-order SH [N, K, 3] or empty
    :param scale: Scaling factor
    :param sh0_out: Output DC [N, 3]
    :param shN_out: Output higher-order [N, K, 3] or empty
    """
    N = sh0.shape[0]

    # Scale sh0
    for i in prange(N):
        for c in range(3):
            sh0_out[i, c] = sh0[i, c] * scale

    # Scale shN if present
    if shN.size > 0:
        K = shN.shape[1]
        for i in prange(N):
            for k in range(K):
                for c in range(3):
                    shN_out[i, k, c] = shN[i, k, c] * scale


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def apply_contrast_to_sh_numba(
    sh0: NDArray[np.float32],
    shN: NDArray[np.float32],
    factor: float,
    is_rgb: bool,
    sh0_out: NDArray[np.float32],
    shN_out: NDArray[np.float32],
) -> None:
    """Apply contrast to SH data (optimized).

    :param sh0: DC component [N, 3]
    :param shN: Higher-order SH [N, K, 3] or empty
    :param factor: Contrast factor
    :param is_rgb: True if RGB format, False if SH format
    :param sh0_out: Output DC [N, 3]
    :param shN_out: Output higher-order [N, K, 3] or empty
    """
    N = sh0.shape[0]

    if is_rgb:
        # RGB: contrast around 0.5
        offset = 0.5 * (1.0 - factor)
        for i in prange(N):
            for c in range(3):
                sh0_out[i, c] = sh0[i, c] * factor + offset
    else:
        # SH: contrast around 0.0
        for i in prange(N):
            for c in range(3):
                sh0_out[i, c] = sh0[i, c] * factor

    # Scale shN (no offset)
    if shN.size > 0:
        K = shN.shape[1]
        for i in prange(N):
            for k in range(K):
                for c in range(3):
                    shN_out[i, k, c] = shN[i, k, c] * factor


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def apply_gamma_to_sh_numba(
    sh0: NDArray[np.float32],
    shN: NDArray[np.float32],
    gamma: float,
    is_rgb: bool,
    sh0_out: NDArray[np.float32],
    shN_out: NDArray[np.float32],
) -> None:
    """Apply gamma correction with SH approximation (optimized).

    :param sh0: DC component [N, 3]
    :param shN: Higher-order SH [N, K, 3] or empty
    :param gamma: Gamma value
    :param is_rgb: True if RGB format
    :param sh0_out: Output DC [N, 3]
    :param shN_out: Output higher-order [N, K, 3] or empty
    """
    N = sh0.shape[0]
    has_sh = shN.size > 0

    if not has_sh:
        # No shN: apply gamma directly
        if is_rgb:
            for i in prange(N):
                for c in range(3):
                    val = max(sh0[i, c], 1e-6)
                    sh0_out[i, c] = val**gamma
        else:
            # Convert to RGB, apply gamma, convert back
            for i in prange(N):
                for c in range(3):
                    rgb = sh0[i, c] * SH_C0 + 0.5
                    rgb = max(min(rgb, 1.0), 1e-6)
                    rgb_gamma = rgb**gamma
                    sh0_out[i, c] = (rgb_gamma - 0.5) / SH_C0
    else:
        # Has shN: approximate by scaling proportionally to luminance change
        K = shN.shape[1]

        for i in prange(N):
            # Compute luminance before and after
            if is_rgb:
                lum_before = LUMA_R * sh0[i, 0] + LUMA_G * sh0[i, 1] + LUMA_B * sh0[i, 2]
                for c in range(3):
                    val = max(sh0[i, c], 1e-6)
                    sh0_out[i, c] = val**gamma
                lum_after = LUMA_R * sh0_out[i, 0] + LUMA_G * sh0_out[i, 1] + LUMA_B * sh0_out[i, 2]
            else:
                # Work in RGB space
                r = sh0[i, 0] * SH_C0 + 0.5
                g = sh0[i, 1] * SH_C0 + 0.5
                b = sh0[i, 2] * SH_C0 + 0.5
                lum_before = LUMA_R * r + LUMA_G * g + LUMA_B * b

                r = max(min(r, 1.0), 1e-6) ** gamma
                g = max(min(g, 1.0), 1e-6) ** gamma
                b = max(min(b, 1.0), 1e-6) ** gamma

                sh0_out[i, 0] = (r - 0.5) / SH_C0
                sh0_out[i, 1] = (g - 0.5) / SH_C0
                sh0_out[i, 2] = (b - 0.5) / SH_C0

                lum_after = LUMA_R * r + LUMA_G * g + LUMA_B * b

            # Scale shN proportionally
            scale = lum_after / (lum_before + 1e-8)
            for k in range(K):
                for c in range(3):
                    shN_out[i, k, c] = shN[i, k, c] * scale


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def apply_vibrance_to_sh_numba(
    sh0: NDArray[np.float32],
    shN: NDArray[np.float32],
    factor: float,
    is_rgb: bool,
    matrix: NDArray[np.float32],  # Pre-computed saturation matrix with mean adaptive factor
    sh0_out: NDArray[np.float32],
    shN_out: NDArray[np.float32],
) -> None:
    """Apply vibrance with SH support (optimized).

    Uses pre-computed matrix for simplicity and speed.

    :param sh0: DC component [N, 3]
    :param shN: Higher-order SH [N, K, 3] or empty
    :param factor: Vibrance factor
    :param is_rgb: True if RGB format
    :param matrix: Pre-computed saturation matrix
    :param sh0_out: Output DC [N, 3]
    :param shN_out: Output higher-order [N, K, 3] or empty
    """
    # For simplicity with Numba, just apply the pre-computed matrix
    # The matrix should already incorporate the mean adaptive factor
    apply_matrix_to_sh_numba(sh0, shN, matrix, sh0_out, shN_out)


@njit(parallel=True, fastmath=True, cache=True, nogil=True)
def apply_fade_to_sh_numba(
    sh0: NDArray[np.float32],
    amount: float,
    is_rgb: bool,
    sh0_out: NDArray[np.float32],
) -> None:
    """Apply fade (black point lift) to DC only (optimized).

    :param sh0: DC component [N, 3]
    :param amount: Fade amount
    :param is_rgb: True if RGB format
    :param sh0_out: Output DC [N, 3]
    """
    N = sh0.shape[0]

    if is_rgb:
        inv_amount = 1.0 - amount
        for i in prange(N):
            for c in range(3):
                sh0_out[i, c] = sh0[i, c] * inv_amount + amount
    else:
        # In SH space, offset needs conversion
        sh_offset = amount / SH_C0
        inv_amount = 1.0 - amount
        for i in prange(N):
            for c in range(3):
                sh0_out[i, c] = sh0[i, c] + sh_offset * inv_amount


# =============================================================================
# Warmup
# =============================================================================


def warmup_sh_kernels() -> None:
    """Warmup Numba JIT compilation for SH kernels."""
    # Dummy data
    sh0 = np.random.rand(100, 3).astype(np.float32)
    shN = np.random.rand(100, 8, 3).astype(np.float32)
    sh0_out = np.empty_like(sh0)
    shN_out = np.empty_like(shN)
    matrix = np.eye(3, dtype=np.float32)

    # Warmup all kernels
    apply_matrix_to_sh_numba(sh0, shN, matrix, sh0_out, shN_out)
    apply_scale_to_sh_numba(sh0, shN, 1.2, sh0_out, shN_out)
    apply_contrast_to_sh_numba(sh0, shN, 1.1, True, sh0_out, shN_out)
    apply_gamma_to_sh_numba(sh0, shN, 0.9, True, sh0_out, shN_out)
    apply_vibrance_to_sh_numba(sh0, shN, 1.2, True, matrix, sh0_out, shN_out)
    apply_fade_to_sh_numba(sh0, 0.1, True, sh0_out)


# Warmup on import
warmup_sh_kernels()
