"""Apply color values to Gaussian colors.

This module provides the core color application function that uses
fused Numba kernels for maximum performance.
"""

from __future__ import annotations

import numpy as np

from gsmod.config.values import ColorValues
from gsmod.color.kernels import (
    fused_color_pipeline_interleaved_lut_numba,
    fused_color_pipeline_skip_lut_numba,
    apply_lut_only_interleaved_numba,
)


def apply_color_values(sh0: np.ndarray, values: ColorValues) -> np.ndarray:
    """Apply color values to SH0 coefficients.

    Uses fused Numba kernels for maximum performance.

    :param sh0: SH0 coefficients [N, 3] in [0, 1]
    :param values: Color parameters
    :return: Modified colors [N, 3]
    """
    if values.is_neutral():
        return sh0

    # Ensure input is float32 and contiguous
    if sh0.dtype != np.float32:
        sh0 = sh0.astype(np.float32)
    if not sh0.flags['C_CONTIGUOUS']:
        sh0 = np.ascontiguousarray(sh0)

    N = sh0.shape[0]
    out = np.empty((N, 3), dtype=np.float32)

    # Check if we can skip LUT (Phase 1 all defaults)
    skip_lut = (
        values.temperature == 0.0
        and values.tint == 0.0
        and values.brightness == 1.0
        and values.contrast == 1.0
        and values.gamma == 1.0
    )

    # Check if we can skip Phase 2
    skip_phase2 = (
        values.saturation == 1.0
        and values.vibrance == 1.0
        and abs(values.hue_shift) < 0.5
        and values.shadows == 0.0
        and values.highlights == 0.0
        and values.fade == 0.0
        and values.shadow_tint_sat == 0.0
        and values.highlight_tint_sat == 0.0
    )

    # Check if we can use simplified Phase 2 (only saturation, shadows, highlights)
    # The fast path kernel doesn't support hue_shift, vibrance, fade, or split toning
    can_use_simple_phase2 = (
        values.vibrance == 1.0
        and abs(values.hue_shift) < 0.5
        and values.fade == 0.0
        and values.shadow_tint_sat == 0.0
        and values.highlight_tint_sat == 0.0
    )

    if skip_lut and skip_phase2:
        # Nothing to do
        return sh0

    if skip_lut and can_use_simple_phase2:
        # Fast path: Simple Phase 2 only (saturation, shadows, highlights)
        fused_color_pipeline_skip_lut_numba(
            sh0,
            values.saturation,
            values.shadows,
            values.highlights,
            out
        )
    elif skip_phase2:
        # Fast path: LUT only
        lut = _build_lut_interleaved(values)
        apply_lut_only_interleaved_numba(sh0, lut, out)
    else:
        # Full pipeline with LUT + Phase 2
        lut = _build_lut_interleaved(values)

        # Compute hue rotation matrix
        m = _compute_hue_rotation_matrix(values.hue_shift)

        # Compute RGB offsets for split toning
        shadow_r, shadow_g, shadow_b = _hue_to_rgb_offset(values.shadow_tint_hue)
        highlight_r, highlight_g, highlight_b = _hue_to_rgb_offset(values.highlight_tint_hue)

        fused_color_pipeline_interleaved_lut_numba(
            sh0,
            lut,
            values.saturation,
            values.vibrance,
            values.hue_shift,
            values.shadows,
            values.highlights,
            out,
            m[0, 0], m[0, 1], m[0, 2],
            m[1, 0], m[1, 1], m[1, 2],
            m[2, 0], m[2, 1], m[2, 2],
            values.fade,
            shadow_r, shadow_g, shadow_b, values.shadow_tint_sat,
            highlight_r, highlight_g, highlight_b, values.highlight_tint_sat,
        )

    return out


def _hue_to_rgb_offset(hue_deg: float) -> tuple[float, float, float]:
    """Convert hue angle to RGB offset (centered at 0).

    :param hue_deg: Hue angle in degrees
    :return: Tuple of (r, g, b) offsets
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
    :return: Interleaved LUT [size, 3]
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
    :return: 3x3 rotation matrix
    """
    if abs(hue_shift_deg) < 0.5:
        return np.eye(3, dtype=np.float32)

    angle = np.radians(hue_shift_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    # Rodrigues' rotation formula around (1,1,1)/sqrt(3)
    sqrt3 = np.sqrt(3.0)

    return np.array([
        [cos_a + (1 - cos_a) / 3, (1 - cos_a) / 3 - sin_a / sqrt3, (1 - cos_a) / 3 + sin_a / sqrt3],
        [(1 - cos_a) / 3 + sin_a / sqrt3, cos_a + (1 - cos_a) / 3, (1 - cos_a) / 3 - sin_a / sqrt3],
        [(1 - cos_a) / 3 - sin_a / sqrt3, (1 - cos_a) / 3 + sin_a / sqrt3, cos_a + (1 - cos_a) / 3],
    ], dtype=np.float32)
