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
from gsmod.transform.kernels import elementwise_multiply_scalar_numba


def apply_transform_values(
    means: np.ndarray,
    quats: np.ndarray,
    scales: np.ndarray,
    values: TransformValues,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply transform values to Gaussian parameters.

    Modifies arrays inplace for performance.

    :param means: Positions [N, 3]
    :param quats: Quaternions [N, 4] wxyz
    :param scales: Scales [N, 3]
    :param values: Transform parameters
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

    # Apply scale
    if values.scale != 1.0:
        elementwise_multiply_scalar_numba(scales, values.scale, scales)

    return means, quats, scales
