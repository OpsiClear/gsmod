"""Apply transform values to Gaussian parameters.

This module provides the core transform application function.
"""

from __future__ import annotations

import numpy as np

from gsmod.config.values import TransformValues
from gsmod.transform.api import (
    _apply_homogeneous_transform_numpy,
    _quaternion_multiply_numpy,
)
from gsmod.transform.kernels import (
    elementwise_add_scalar_numba,
    elementwise_multiply_scalar_numba,
)


def apply_transform_values(
    means: np.ndarray,
    quats: np.ndarray,
    scales: np.ndarray,
    values: TransformValues,
    is_scales_ply: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply transform values to Gaussian parameters.

    Modifies arrays inplace for performance.

    :param means: Positions [N, 3]
    :param quats: Quaternions [N, 4] wxyz
    :param scales: Scales [N, 3]
    :param values: Transform parameters
    :param is_scales_ply: If True, scales are in log space (PLY format)
    :returns: Tuple of (means, quats, scales) - all modified inplace
    """
    if values.is_neutral():
        return means, quats, scales

    # Get transformation matrix
    matrix = values.to_matrix()

    # Apply to positions
    _apply_homogeneous_transform_numpy(means, matrix, out=means)

    # Apply rotation to quaternions
    if values.rotation != (1.0, 0.0, 0.0, 0.0):
        rot_quat = np.array(values.rotation, dtype=quats.dtype)
        _quaternion_multiply_numpy(rot_quat[np.newaxis, :], quats, out=quats)

    # Apply scale (handle log-space for PLY format)
    # Scale is now a tuple (sx, sy, sz)
    scale_arr = np.array(values.scale, dtype=scales.dtype)
    is_uniform = np.allclose(scale_arr, scale_arr[0])
    is_identity = np.allclose(scale_arr, 1.0)

    if not is_identity:
        if is_scales_ply:
            # In log space: add log(scale) instead of multiplying
            log_scale = np.log(scale_arr)
            if is_uniform:
                elementwise_add_scalar_numba(scales, float(log_scale[0]), scales)
            else:
                scales += log_scale  # Broadcast add for non-uniform
        else:
            # In linear space: multiply directly
            if is_uniform:
                elementwise_multiply_scalar_numba(scales, float(scale_arr[0]), scales)
            else:
                scales *= scale_arr  # Broadcast multiply for non-uniform

    return means, quats, scales
