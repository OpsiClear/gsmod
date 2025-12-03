"""Apply color values to Gaussian colors.

This module provides the core color application function that matches
the GPU (GSTensorPro) behavior exactly for CPU/GPU consistency.

GPU Ground Truth (PyTorch fallback):
- Brightness: sh0 + shN (both scaled)
- Saturation: sh0 + shN (both via matrix)
- All other ops: sh0 ONLY (contrast, gamma, temperature, tint, hue, vibrance, fade)
"""

from __future__ import annotations

import numpy as np

from gsmod.config.values import ColorValues


def apply_color_values(
    sh0: np.ndarray,
    values: ColorValues,
    shN: np.ndarray | None = None,
    is_sh0_rgb: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply color values to SH coefficients (matching GPU behavior).

    Applies color transformations following GPU ground truth:
    - Brightness and Saturation: apply to BOTH sh0 and shN
    - All other operations: sh0 ONLY

    :param sh0: SH0/DC coefficients [N, 3]
    :param values: Color parameters
    :param shN: Higher-order SH coefficients [N, K, 3] or None
    :param is_sh0_rgb: True if sh0 is in RGB format [0,1], False if SH format
    :returns: Tuple of (modified_sh0, modified_shN)
    """
    from gsmod.color.sh_kernels import (
        apply_fade_to_sh_numba,
        apply_matrix_to_sh_numba,
        apply_scale_to_sh_numba,
    )
    from gsmod.color.sh_utils import (
        build_saturation_matrix,
        compute_luminance,
        sh_to_rgb,
    )

    if values.is_neutral():
        return sh0, shN

    # Ensure input is float32 and contiguous
    if sh0.dtype != np.float32:
        sh0 = sh0.astype(np.float32)
    if not sh0.flags["C_CONTIGUOUS"]:
        sh0 = np.ascontiguousarray(sh0)

    # Copy to avoid modifying input
    sh0 = sh0.copy()
    shN_empty = np.empty((0, 0, 0), dtype=np.float32)  # Empty array for Numba
    if shN is not None:
        shN = shN.copy()
        if not shN.flags["C_CONTIGUOUS"]:
            shN = np.ascontiguousarray(shN)
    else:
        shN = shN_empty

    # Allocate output buffers
    sh0_out = np.empty_like(sh0)
    shN_out = np.empty_like(shN) if shN.size > 0 else shN_empty

    # Apply operations matching GPU ground truth order and behavior

    # 1. Temperature (sh0 only - R/B channel scaling, matching GPU)
    if values.temperature != 0.0:
        r_factor = 1.0 + values.temperature * 0.3
        b_factor = 1.0 - values.temperature * 0.3
        sh0[:, 0] *= r_factor
        sh0[:, 2] *= b_factor

    # 2. Tint (sh0 only - G channel offset, matching GPU)
    if values.tint != 0.0:
        tint_offset_g = -values.tint * 0.1
        tint_offset_rb = values.tint * 0.05
        sh0[:, 0] += tint_offset_rb  # R
        sh0[:, 1] += tint_offset_g  # G
        sh0[:, 2] += tint_offset_rb  # B

    # 3. Brightness (sh0 + shN - both scaled, matching GPU)
    if values.brightness != 1.0:
        apply_scale_to_sh_numba(sh0, shN, values.brightness, sh0_out, shN_out)
        sh0, shN = sh0_out, shN_out
        sh0_out, shN_out = np.empty_like(sh0), np.empty_like(shN)

    # 4. Contrast (sh0 only - matching GPU)
    if values.contrast != 1.0:
        # GPU formula: (val - 0.5) * factor + 0.5
        sh0 = (sh0 - 0.5) * values.contrast + 0.5

    # 5. Gamma (sh0 only - matching GPU)
    if values.gamma != 1.0:
        # Apply gamma directly (GPU ground truth: pow(val, gamma))
        # gamma > 1 = darker, gamma < 1 = brighter
        sh0 = np.power(np.maximum(sh0, 1e-8), values.gamma)

    # 6. Saturation (sh0 + shN - both via matrix, matching GPU)
    if values.saturation != 1.0:
        M = build_saturation_matrix(values.saturation)
        apply_matrix_to_sh_numba(sh0, shN, M, sh0_out, shN_out)
        sh0, shN = sh0_out, shN_out
        sh0_out, shN_out = np.empty_like(sh0), np.empty_like(shN)

    # 7. Hue shift (sh0 only - matching GPU)
    if values.hue_shift != 0.0:
        angle = values.hue_shift * np.pi / 180.0
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # GPU hue rotation matrix (approximation in RGB space)
        mat = np.array(
            [
                [
                    0.299 + 0.701 * cos_a + 0.168 * sin_a,
                    0.587 - 0.587 * cos_a + 0.330 * sin_a,
                    0.114 - 0.114 * cos_a - 0.497 * sin_a,
                ],
                [
                    0.299 - 0.299 * cos_a - 0.328 * sin_a,
                    0.587 + 0.413 * cos_a + 0.035 * sin_a,
                    0.114 - 0.114 * cos_a + 0.292 * sin_a,
                ],
                [
                    0.299 - 0.300 * cos_a + 1.250 * sin_a,
                    0.587 - 0.588 * cos_a - 1.050 * sin_a,
                    0.114 + 0.886 * cos_a - 0.203 * sin_a,
                ],
            ],
            dtype=np.float32,
        )
        sh0 = sh0 @ mat.T

    # 8. Vibrance (sh0 only - matching GPU)
    if values.vibrance != 1.0:
        if is_sh0_rgb:
            rgb = sh0
        else:
            rgb = sh_to_rgb(sh0)

        max_rgb = np.max(rgb, axis=-1, keepdims=True)
        min_rgb = np.min(rgb, axis=-1, keepdims=True)
        saturation = max_rgb - min_rgb
        luminance = compute_luminance(rgb)

        # Vibrance: boost less-saturated colors more
        boost = (1.0 - saturation) * (values.vibrance - 1.0) + 1.0
        sh0 = luminance + (sh0 - luminance) * boost

    # 9. Shadows/Highlights (sh0 only - using smoothstep curves like supersplat)
    if values.shadows != 0.0 or values.highlights != 0.0:
        if is_sh0_rgb:
            lum = compute_luminance(sh0)
        else:
            lum = compute_luminance(sh_to_rgb(sh0))

        # Smoothstep curves matching supersplat shader
        # shadowCurve = 1.0 - smoothstep(0.0, 0.5, lum)
        # highlightCurve = smoothstep(0.5, 1.0, lum)
        shadow_curve = 1.0 - np.clip((lum - 0.0) / 0.5, 0.0, 1.0) ** 2 * (
            3.0 - 2.0 * np.clip((lum - 0.0) / 0.5, 0.0, 1.0)
        )
        highlight_curve = np.clip((lum - 0.5) / 0.5, 0.0, 1.0) ** 2 * (
            3.0 - 2.0 * np.clip((lum - 0.5) / 0.5, 0.0, 1.0)
        )

        # Multiplicative adjustment: color * (1 + shadows * shadowCurve + highlights * highlightCurve)
        adjustment = 1.0 + values.shadows * shadow_curve + values.highlights * highlight_curve
        sh0 = sh0 * adjustment

    # 10. Fade (offset to DC only)
    if values.fade != 0.0:
        sh0_out = np.empty_like(sh0)
        apply_fade_to_sh_numba(sh0, values.fade, is_sh0_rgb, sh0_out)
        sh0 = sh0_out

    # Clamp if in RGB format
    if is_sh0_rgb:
        sh0 = np.clip(sh0, 0, 1)

    # Return None for shN if it was originally None
    if shN.size == 0:
        shN = None

    return sh0, shN


def _hue_to_rgb_offset(hue_deg: float) -> tuple[float, float, float]:
    """Convert hue angle to RGB offset (centered at 0).

    :param hue_deg: Hue angle in degrees
    :returns: Tuple of (r, g, b) offsets
    """
    hue_rad = hue_deg * (np.pi / 180.0)
    # Use HSL-like conversion with L=0.5, S=1.0
    r = np.cos(hue_rad) * 0.5
    g = np.cos(hue_rad - 2.0 * np.pi / 3.0) * 0.5
    b = np.cos(hue_rad - 4.0 * np.pi / 3.0) * 0.5
    return float(r), float(g), float(b)


def _build_lut_interleaved(values: ColorValues, size: int = 256) -> np.ndarray:
    """Build interleaved RGB LUT from color values.

    :param values: Color parameters
    :param size: LUT size (default 256)
    :returns: Interleaved LUT [size, 3]
    """
    x = np.linspace(0, 1, size, dtype=np.float32)

    # Start with identity
    r = x.copy()
    g = x.copy()
    b = x.copy()

    # Temperature (affects R and B channels)
    if values.temperature != 0:
        r = np.clip(r + values.temperature * 0.1, 0, 1)
        b = np.clip(b - values.temperature * 0.1, 0, 1)

    # Tint (affects G and M channels - green/magenta)
    if values.tint != 0:
        g = np.clip(g + values.tint * 0.1, 0, 1)

    # Brightness
    if values.brightness != 1.0:
        r = np.clip(r * values.brightness, 0, 1)
        g = np.clip(g * values.brightness, 0, 1)
        b = np.clip(b * values.brightness, 0, 1)

    # Contrast
    if values.contrast != 1.0:
        r = np.clip((r - 0.5) * values.contrast + 0.5, 0, 1)
        g = np.clip((g - 0.5) * values.contrast + 0.5, 0, 1)
        b = np.clip((b - 0.5) * values.contrast + 0.5, 0, 1)

    # Gamma
    if values.gamma != 1.0:
        r = np.power(r, 1.0 / values.gamma)
        g = np.power(g, 1.0 / values.gamma)
        b = np.power(b, 1.0 / values.gamma)

    return np.stack([r, g, b], axis=1)


def _compute_hue_rotation_matrix(hue_shift_deg: float) -> np.ndarray:
    """Compute RGB rotation matrix for hue shift.

    Uses Rodrigues' rotation formula around the (1,1,1) axis.

    :param hue_shift_deg: Hue shift in degrees
    :returns: 3x3 rotation matrix
    """
    if abs(hue_shift_deg) < 0.5:
        return np.eye(3, dtype=np.float32)

    angle = np.radians(hue_shift_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Rodrigues' rotation formula around (1,1,1)/sqrt(3)
    sqrt3 = np.sqrt(3.0)

    return np.array(
        [
            [
                cos_a + (1 - cos_a) / 3,
                (1 - cos_a) / 3 - sin_a / sqrt3,
                (1 - cos_a) / 3 + sin_a / sqrt3,
            ],
            [
                (1 - cos_a) / 3 + sin_a / sqrt3,
                cos_a + (1 - cos_a) / 3,
                (1 - cos_a) / 3 - sin_a / sqrt3,
            ],
            [
                (1 - cos_a) / 3 - sin_a / sqrt3,
                (1 - cos_a) / 3 + sin_a / sqrt3,
                cos_a + (1 - cos_a) / 3,
            ],
        ],
        dtype=np.float32,
    )
