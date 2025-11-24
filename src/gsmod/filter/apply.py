"""Apply filter values to compute masks for Gaussian filtering.

This module provides the core filter mask computation function.
"""

from __future__ import annotations

import numpy as np

from gsmod.config.values import FilterValues
from gsmod.filter.kernels import (
    combined_filter_fused,
    compute_output_indices_and_count,
    cuboid_filter_numba,
    filter_gaussians_fused_parallel,
    sphere_filter_numba,
)


def _axis_angle_to_rotation_matrix(axis_angle: tuple[float, float, float] | None) -> np.ndarray:
    """Convert axis-angle rotation to 3x3 rotation matrix.

    :param axis_angle: Rotation vector [3] where magnitude is angle in radians
    :return: Rotation matrix [3, 3]
    """
    if axis_angle is None:
        return np.eye(3, dtype=np.float32)

    axis_angle = np.asarray(axis_angle, dtype=np.float32)
    angle = np.linalg.norm(axis_angle)

    if angle < 1e-8:
        return np.eye(3, dtype=np.float32)

    # Normalize axis
    axis = axis_angle / angle

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=np.float32)

    # R = I + sin(angle) * K + (1 - cos(angle)) * K^2
    R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

    return R


def compute_filter_mask(data, values: FilterValues) -> np.ndarray:
    """Compute boolean filter mask from FilterValues.

    All criteria are combined with AND logic using fused Numba kernel.

    :param data: Object with means, opacities, scales attributes
    :param values: Filter parameters
    :return: Boolean mask [N]
    """
    N = len(data.means)

    # Use fused Numba kernel for all filters in single pass
    positions = np.ascontiguousarray(data.means, dtype=np.float32)
    opacities = np.ascontiguousarray(data.opacities.flatten(), dtype=np.float32)
    scales = np.ascontiguousarray(data.scales, dtype=np.float32)

    # Prepare sphere parameters
    has_sphere = values.sphere_radius < float('inf')
    sphere_center = np.array(values.sphere_center, dtype=np.float32) if has_sphere else np.zeros(3, dtype=np.float32)
    sphere_radius_sq = values.sphere_radius ** 2 if has_sphere else 0.0

    # Prepare box parameters
    has_box = values.box_min is not None and values.box_max is not None
    box_min = np.array(values.box_min, dtype=np.float32) if has_box else np.zeros(3, dtype=np.float32)
    box_max = np.array(values.box_max, dtype=np.float32) if has_box else np.zeros(3, dtype=np.float32)

    # Prepare ellipsoid parameters
    has_ellipsoid = values.ellipsoid_radii is not None
    if has_ellipsoid:
        ellipsoid_center = np.array(values.ellipsoid_center, dtype=np.float32)
        ellipsoid_radii = np.array(values.ellipsoid_radii, dtype=np.float32)
        # Transpose for world-to-local transformation
        ellipsoid_rotation = _axis_angle_to_rotation_matrix(values.ellipsoid_rotation).T
    else:
        ellipsoid_center = np.zeros(3, dtype=np.float32)
        ellipsoid_radii = np.ones(3, dtype=np.float32)
        ellipsoid_rotation = np.eye(3, dtype=np.float32)

    # Prepare frustum parameters
    has_frustum = values.frustum_position is not None
    if has_frustum:
        frustum_pos = np.array(values.frustum_position, dtype=np.float32)
        # Transpose for world-to-camera transformation
        frustum_rotation = _axis_angle_to_rotation_matrix(values.frustum_rotation).T
        tan_half_fov_y = np.tan(values.frustum_fov / 2)
        tan_half_fov_x = tan_half_fov_y * values.frustum_aspect
        frustum_near = values.frustum_near
        frustum_far = values.frustum_far
    else:
        frustum_pos = np.zeros(3, dtype=np.float32)
        frustum_rotation = np.eye(3, dtype=np.float32)
        tan_half_fov_x = 1.0
        tan_half_fov_y = 1.0
        frustum_near = 0.1
        frustum_far = 100.0

    # Output mask
    mask = np.empty(N, dtype=np.bool_)

    # Run fused kernel
    combined_filter_fused(
        positions,
        opacities,
        scales,
        sphere_center,
        sphere_radius_sq,
        box_min,
        box_max,
        values.min_opacity,
        values.max_opacity,
        values.min_scale,
        values.max_scale,
        has_sphere,
        has_box,
        ellipsoid_center,
        ellipsoid_radii,
        ellipsoid_rotation,
        has_ellipsoid,
        frustum_pos,
        frustum_rotation,
        tan_half_fov_x,
        tan_half_fov_y,
        frustum_near,
        frustum_far,
        has_frustum,
        mask,
    )

    return mask


def apply_mask_fused(data, mask: np.ndarray) -> tuple:
    """Apply boolean mask to GSData using fused parallel scatter.

    Uses Numba parallel kernel to scatter all attributes in single pass,
    avoiding 5-6 separate NumPy fancy indexing operations.

    :param data: Object with means, scales, quats, opacities, sh0, shN attributes
    :param mask: Boolean mask [N]
    :return: Tuple of (means, scales, quats, opacities, sh0, shN)
    """
    N = len(mask)

    # Compute output indices and count
    out_indices = np.empty(N, dtype=np.int64)
    n_kept = compute_output_indices_and_count(mask, out_indices)

    if n_kept == 0:
        # Return empty arrays
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            None if data.shN is None else np.empty((0, data.shN.shape[1], 3), dtype=np.float32),
        )

    # Prepare input arrays (ensure contiguous)
    positions = np.ascontiguousarray(data.means, dtype=np.float32)
    quaternions = np.ascontiguousarray(data.quats, dtype=np.float32)
    scales = np.ascontiguousarray(data.scales, dtype=np.float32)
    opacities = np.ascontiguousarray(data.opacities.flatten(), dtype=np.float32)
    colors = np.ascontiguousarray(data.sh0, dtype=np.float32)
    shN = np.ascontiguousarray(data.shN, dtype=np.float32) if data.shN is not None else None

    # Allocate output arrays
    out_positions = np.empty((n_kept, 3), dtype=np.float32)
    out_quaternions = np.empty((n_kept, 4), dtype=np.float32)
    out_scales = np.empty((n_kept, 3), dtype=np.float32)
    out_opacities = np.empty(n_kept, dtype=np.float32)
    out_colors = np.empty((n_kept, 3), dtype=np.float32)
    out_shN = np.empty((n_kept, shN.shape[1], 3), dtype=np.float32) if shN is not None else None

    # Run fused parallel scatter
    filter_gaussians_fused_parallel(
        mask,
        out_indices,
        positions,
        quaternions,
        scales,
        opacities,
        colors,
        shN,
        out_positions,
        out_quaternions,
        out_scales,
        out_opacities,
        out_colors,
        out_shN,
    )

    return out_positions, out_scales, out_quaternions, out_opacities, out_colors, out_shN
