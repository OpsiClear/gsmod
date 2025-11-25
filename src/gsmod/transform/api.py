"""
3D Geometric Transformations for Gaussian Splatting.

CPU-optimized using NumPy and Numba for maximum performance.

Functions:

- ``transform()``: Unified transformation interface (translation, rotation, scaling).
  Uses fused Numba kernel for 4-5x speedup. All operations are in-place by default.
- Quaternion utilities: multiply, conversions, etc.
"""

from __future__ import annotations

import numpy as np

# Import from shared rotation module (canonical implementations)
from gsmod.shared.rotation import (
    _axis_angle_to_quaternion_numpy,
    _euler_to_quaternion_numpy,
    _quaternion_to_euler_numpy,
    _quaternion_to_rotation_matrix_numpy,
    _rotation_matrix_to_quaternion_numpy,
)

# Import Numba kernels at module level - Numba is required
from gsmod.transform.kernels import (
    elementwise_multiply_scalar_numba,
    elementwise_multiply_vector_numba,
    quaternion_multiply_batched_numba,
    quaternion_multiply_single_numba,
)

# Type aliases for better readability (Python 3.12+ syntax)
type ArrayLike = np.ndarray | tuple | list

# ============================================================================
# 4x4 Homogeneous Transformation Matrix Building (NumPy)
# ============================================================================


def _build_translation_matrix_4x4_numpy(translation: np.ndarray) -> np.ndarray:
    """Build 4x4 translation matrix."""
    T = np.eye(4, dtype=translation.dtype)
    T[:3, 3] = translation
    return T


def _build_rotation_matrix_4x4_numpy(rotation_3x3: np.ndarray) -> np.ndarray:
    """Build 4x4 rotation matrix from 3x3 rotation matrix."""
    R = np.eye(4, dtype=rotation_3x3.dtype)
    R[:3, :3] = rotation_3x3
    return R


def _build_scale_matrix_4x4_numpy(scale_factor: np.ndarray) -> np.ndarray:
    """Build 4x4 scale matrix."""
    S = np.eye(4, dtype=scale_factor.dtype)
    S[0, 0] = scale_factor[0]
    S[1, 1] = scale_factor[1]
    S[2, 2] = scale_factor[2]
    return S


def _compose_transform_matrix_numpy(
    translation: ArrayLike | None,
    rotation: ArrayLike | None,
    rotation_format: str,
    scale_factor: float | ArrayLike | None,
    center: ArrayLike | None,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Compose transformation matrix and auxiliary data.

    :return: Tuple of (transform_matrix, rotation_quat, scale_vec)
        - transform_matrix: 4x4 homogeneous transformation matrix
        - rotation_quat: Quaternion for orientation updates (or None)
        - scale_vec: Scale vector for size updates (or None)
    """
    # Identity matrix
    M = np.eye(4, dtype=dtype)
    rotation_quat = None
    scale_vec = None

    # Convert inputs to numpy arrays with proper dtype
    if scale_factor is not None:
        if isinstance(scale_factor, int | float):
            scale_vec = np.array([scale_factor, scale_factor, scale_factor], dtype=dtype)
        elif not isinstance(scale_factor, np.ndarray):
            scale_vec = np.array(scale_factor, dtype=dtype)
        else:
            scale_vec = scale_factor.astype(dtype)

        if scale_vec.ndim == 0:
            scale_vec = np.array([float(scale_vec)] * 3, dtype=dtype)

    if translation is not None and not isinstance(translation, np.ndarray):
        translation = np.array(translation, dtype=dtype)

    if center is not None and not isinstance(center, np.ndarray):
        center = np.array(center, dtype=dtype)

    # Build transformation: T @ R @ S (applied right-to-left)
    # With center: T(center) @ T @ R @ S @ T(-center)

    if center is not None:
        # Translate to origin
        T_neg_center = _build_translation_matrix_4x4_numpy(-center)
        M = T_neg_center @ M

    # Scale
    if scale_vec is not None:
        S = _build_scale_matrix_4x4_numpy(scale_vec)
        M = S @ M

    # Rotate
    if rotation is not None:
        # Convert rotation to quaternion and 3x3 matrix
        if rotation_format == "quaternion":
            rotation_quat = (
                np.array(rotation, dtype=dtype)
                if not isinstance(rotation, np.ndarray)
                else rotation.astype(dtype)
            )
        elif rotation_format == "matrix":
            rotation_matrix_3x3 = (
                np.array(rotation, dtype=dtype)
                if not isinstance(rotation, np.ndarray)
                else rotation.astype(dtype)
            )
            rotation_quat = _rotation_matrix_to_quaternion_numpy(rotation_matrix_3x3)
        elif rotation_format == "axis_angle":
            rotation_quat = _axis_angle_to_quaternion_numpy(np.array(rotation, dtype=dtype))
        elif rotation_format == "euler":
            rotation_quat = _euler_to_quaternion_numpy(np.array(rotation, dtype=dtype))
        else:
            raise ValueError(f"Unknown rotation format: {rotation_format}")

        # Note: Quaternion normalization is handled in _quaternion_to_rotation_matrix_numpy
        # to avoid redundant normalization

        # Convert to 3x3 rotation matrix
        rotation_matrix_3x3 = _quaternion_to_rotation_matrix_numpy(rotation_quat)
        R = _build_rotation_matrix_4x4_numpy(rotation_matrix_3x3)
        M = R @ M

    # Translate
    if translation is not None:
        T = _build_translation_matrix_4x4_numpy(translation)
        M = T @ M

    if center is not None:
        # Translate back from origin
        T_center = _build_translation_matrix_4x4_numpy(center)
        M = T_center @ M

    return M, rotation_quat, scale_vec


def _apply_homogeneous_transform_numpy(
    points: np.ndarray, matrix: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """Apply 4x4 homogeneous transformation matrix to Nx3 points.

    Optimized to avoid homogeneous coordinate overhead by extracting
    the 3x3 rotation/scale and 3-vector translation directly.

    Uses NumPy's BLAS-optimized matrix multiplication for best performance.

    :param points: Input points [N, 3]
    :param matrix: 4x4 transformation matrix
    :param out: Optional pre-allocated output buffer [N, 3]
    :return: Transformed points (same as `out` if provided)
    """
    # Extract 3x3 combined rotation/scale matrix and translation vector
    R = matrix[:3, :3]  # Upper-left 3x3
    t = matrix[:3, 3]  # Translation column

    if out is not None:
        # Use NumPy's BLAS-optimized matmul with output buffer
        np.matmul(points, R.T, out=out)
        out += t
        return out
    else:
        # Use NumPy's BLAS-optimized matmul - faster than Numba for this operation
        return points @ R.T + t


# ============================================================================
# ============================================================================


def _translate_numpy(
    means: np.ndarray,
    translation: ArrayLike,
) -> np.ndarray:
    """NumPy implementation of translate (IN-PLACE)."""
    if not isinstance(translation, np.ndarray):
        translation = np.array(translation, dtype=means.dtype)

    if translation.ndim == 1:
        translation = translation[np.newaxis, :]

    # In-place addition
    means += translation
    return means


def _scale_numpy(
    means: np.ndarray,
    scales: np.ndarray,
    scale_factor: float | ArrayLike,
    center: ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy implementation of scale (IN-PLACE).

    Modifies means and scales arrays in-place for maximum performance.
    """
    # Convert scale_factor to array
    if isinstance(scale_factor, int | float):
        scale_factor_arr = np.array([scale_factor, scale_factor, scale_factor], dtype=means.dtype)
        is_uniform_scale = True
        uniform_scale_value = float(scale_factor)
    elif not isinstance(scale_factor, np.ndarray):
        scale_factor_arr = np.array(scale_factor, dtype=means.dtype)
        is_uniform_scale = False
    else:
        scale_factor_arr = scale_factor.astype(means.dtype)
        is_uniform_scale = False

    if scale_factor_arr.ndim == 1:
        scale_factor_arr = scale_factor_arr[np.newaxis, :]

    # Handle center of scaling
    if center is not None:
        if not isinstance(center, np.ndarray):
            center = np.array(center, dtype=means.dtype)
        if center.ndim == 1:
            center = center[np.newaxis, :]

        # (means - center) * scale_factor + center (in-place)
        means -= center
        if is_uniform_scale:
            elementwise_multiply_scalar_numba(means, uniform_scale_value, means)
        else:
            elementwise_multiply_vector_numba(means, scale_factor_arr, means)
        means += center
    else:
        # means * scale_factor (in-place)
        if is_uniform_scale:
            elementwise_multiply_scalar_numba(means, uniform_scale_value, means)
        else:
            elementwise_multiply_vector_numba(means, scale_factor_arr, means)

    # scales * scale_factor (in-place)
    if is_uniform_scale:
        elementwise_multiply_scalar_numba(scales, uniform_scale_value, scales)
    else:
        elementwise_multiply_vector_numba(scales, scale_factor_arr, scales)

    return means, scales


def _rotate_numpy(
    means: np.ndarray,
    quaternions: np.ndarray,
    rotation: np.ndarray,
    center: ArrayLike | None = None,
    rotation_format: str = "quaternion",
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy implementation of rotate (IN-PLACE).

    Modifies means and quaternions arrays in-place for maximum performance.
    """
    # Convert rotation to quaternion if needed
    if rotation_format == "quaternion":
        rotation_quat = rotation
    elif rotation_format == "matrix":
        rotation_quat = _rotation_matrix_to_quaternion_numpy(rotation)
    elif rotation_format == "axis_angle":
        rotation_quat = _axis_angle_to_quaternion_numpy(rotation)
    elif rotation_format == "euler":
        rotation_quat = _euler_to_quaternion_numpy(rotation)
    else:
        raise ValueError(f"Unknown rotation format: {rotation_format}")

    # Reshape if needed (normalization handled in _quaternion_to_rotation_matrix_numpy)
    if rotation_quat.ndim == 1:
        rotation_quat = rotation_quat[np.newaxis, :]

    # Convert quaternion to rotation matrix for position rotation
    rot_matrix = _quaternion_to_rotation_matrix_numpy(rotation_quat.squeeze(0))

    # Handle center of rotation (in-place)
    if center is not None:
        if not isinstance(center, np.ndarray):
            center = np.array(center, dtype=means.dtype)
        if center.ndim == 1:
            center = center[np.newaxis, :]

        means -= center
        np.matmul(means, rot_matrix.T, out=means)
        means += center
    else:
        np.matmul(means, rot_matrix.T, out=means)

    # Rotate quaternions: q_new = q_rotation * q_old (in-place)
    _quaternion_multiply_numpy(rotation_quat, quaternions, out=quaternions)

    return means, quaternions


def _quaternion_multiply_numpy(
    q1: np.ndarray, q2: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """
    NumPy implementation of quaternion multiplication.

    Uses Numba-optimized version (200x faster than pure NumPy).

    :param q1: First quaternion [N, 4] or [4] (w, x, y, z)
    :param q2: Second quaternion [N, 4] or [4] (w, x, y, z)
    :param out: Optional pre-allocated output buffer [N, 4]
    :return: Product quaternion [N, 4] (same as `out` if provided)
    """
    # Use Numba-optimized version (always available)
    # Ensure inputs are 2D
    q1_was_1d = q1.ndim == 1
    q2_was_1d = q2.ndim == 1

    if q1_was_1d:
        q1 = q1[np.newaxis, :]
    if q2_was_1d:
        q2 = q2[np.newaxis, :]

    # Allocate or use provided output
    if q1.shape[0] == 1 and q2.shape[0] > 1:
        # Single quaternion broadcast to many
        if out is None:
            out = np.empty_like(q2)
        quaternion_multiply_single_numba(q1[0], q2, out)
    elif q1.shape[0] == q2.shape[0]:
        # Batched multiplication
        if out is None:
            out = np.empty_like(q1)
        quaternion_multiply_batched_numba(q1, q2, out)
    else:
        # Fall back to NumPy for other cases
        if q1.ndim == 1:
            q1 = q1[np.newaxis, :]
        if q2.ndim == 1:
            q2 = q2[np.newaxis, :]

        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        result = np.stack([w, x, y, z], axis=1)
        if out is not None:
            out[:] = result
            return out
        return result

    return out


# Note: _quaternion_to_rotation_matrix_numpy, _rotation_matrix_to_quaternion_numpy,
# _axis_angle_to_quaternion_numpy, _euler_to_quaternion_numpy, _quaternion_to_euler_numpy
# are imported from gsmod.shared.rotation (canonical implementations)


# ============================================================================
# Public API - Quaternion Utilities
# ============================================================================


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply quaternions.

    :param q1: First quaternion(s) [4] or [N, 4]
    :param q2: Second quaternion(s) [4] or [N, 4]
    :return: Product quaternion(s)
    """
    return _quaternion_multiply_numpy(q1, q2)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    return _quaternion_to_rotation_matrix_numpy(q)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion."""
    return _rotation_matrix_to_quaternion_numpy(R)


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to quaternion."""
    return _axis_angle_to_quaternion_numpy(axis_angle)


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """Convert Euler angles to quaternion."""
    return _euler_to_quaternion_numpy(euler)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to Euler angles."""
    return _quaternion_to_euler_numpy(q)
