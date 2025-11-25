"""Apply opacity values to Gaussian opacities.

This module provides format-aware opacity adjustment that works correctly
with both linear [0, 1] and PLY (logit) opacity formats.
"""

from __future__ import annotations

import numpy as np

from gsmod.config.values import OpacityValues


def apply_opacity_values(
    opacities: np.ndarray,
    values: OpacityValues,
    is_ply_format: bool = False,
) -> np.ndarray:
    """Apply opacity scaling to opacity values.

    Handles both linear [0, 1] and PLY (logit) formats correctly.
    The scale factor is always applied in linear space.

    :param opacities: Opacity values [N] or [N, 1]
    :param values: Opacity parameters
    :param is_ply_format: True if opacities are in PLY (logit) format
    :returns: Modified opacities (same shape as input)
    """
    if values.is_neutral():
        return opacities

    scale = values.scale
    original_shape = opacities.shape
    opacities_flat = opacities.ravel()

    if is_ply_format:
        # PLY format: stored as logit(opacity)
        # Convert to linear, scale, convert back
        linear = 1.0 / (1.0 + np.exp(-opacities_flat))  # sigmoid

        if scale <= 1.0:
            # Simple scaling for fade
            scaled = linear * scale
        else:
            # Boost: move toward 1.0 with diminishing returns
            boost_factor = (scale - 1.0) / 2.0
            scaled = linear + (1.0 - linear) * boost_factor

        # Clamp to valid range for logit (avoid inf)
        scaled = np.clip(scaled, 1e-7, 1.0 - 1e-7)

        # Convert back to logit
        result = np.log(scaled / (1.0 - scaled))
    else:
        # Linear format: direct scaling
        if scale <= 1.0:
            result = np.clip(opacities_flat * scale, 0.0, 1.0)
        else:
            # Boost: move toward 1.0 with diminishing returns
            boost_factor = (scale - 1.0) / 2.0
            result = np.clip(
                opacities_flat + (1.0 - opacities_flat) * boost_factor,
                0.0,
                1.0,
            )

    return result.reshape(original_shape).astype(opacities.dtype)
