"""Unified rotation utilities for CPU and GPU operations.

This module provides rotation utilities that work with both NumPy arrays
and PyTorch tensors. All functions auto-detect the input type and use
the appropriate backend.

Quaternion Convention: (w, x, y, z) - scalar first
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

# Type aliases
type ArrayLike = np.ndarray | list | tuple


# ============================================================================
# NumPy/CPU Implementation
# ============================================================================


def _quaternion_multiply_numpy(
    q1: np.ndarray, q2: np.ndarray, out: np.ndarray | None = None
) -> np.ndarray:
    """NumPy quaternion multiplication.

    :param q1: First quaternion [4] or [N, 4] (w, x, y, z)
    :param q2: Second quaternion [4] or [N, 4] (w, x, y, z)
    :param out: Optional pre-allocated output buffer
    :returns: Product quaternion
    """
    # Handle 1D inputs
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
    return result.squeeze(0) if result.shape[0] == 1 else result


def _quaternion_to_rotation_matrix_numpy(q: np.ndarray) -> np.ndarray:
    """NumPy quaternion to 3x3 rotation matrix.

    :param q: Quaternion [4] (w, x, y, z)
    :returns: 3x3 rotation matrix
    """
    q = q / np.linalg.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]

    R = np.zeros((3, 3), dtype=q.dtype)

    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)

    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)

    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)

    return R


def _rotation_matrix_to_quaternion_numpy(R: np.ndarray) -> np.ndarray:
    """NumPy 3x3 rotation matrix to quaternion.

    :param R: 3x3 rotation matrix
    :returns: Quaternion [4] (w, x, y, z)
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=R.dtype)
    return q / np.linalg.norm(q)


def _axis_angle_to_quaternion_numpy(axis_angle: np.ndarray) -> np.ndarray:
    """NumPy axis-angle to quaternion.

    :param axis_angle: Axis-angle vector [3] (axis * angle)
    :returns: Quaternion [4] (w, x, y, z)
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=axis_angle.dtype)

    axis = axis_angle / angle
    half_angle = angle / 2
    sin_half = np.sin(half_angle)

    w = np.cos(half_angle)
    x = axis[0] * sin_half
    y = axis[1] * sin_half
    z = axis[2] * sin_half

    return np.array([w, x, y, z], dtype=axis_angle.dtype)


def _euler_to_quaternion_numpy(euler: np.ndarray, order: str = "XYZ") -> np.ndarray:
    """NumPy Euler angles to quaternion.

    :param euler: Euler angles [3] (roll, pitch, yaw) in radians
    :param order: Rotation order (e.g., "XYZ", "ZYX")
    :returns: Quaternion [4] (w, x, y, z)
    """
    # For now, always use XYZ order (can be extended)
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z], dtype=euler.dtype)


def _quaternion_to_euler_numpy(q: np.ndarray) -> np.ndarray:
    """NumPy quaternion to Euler angles.

    :param q: Quaternion [4] (w, x, y, z)
    :returns: Euler angles [3] (roll, pitch, yaw) in radians
    """
    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=q.dtype)


# ============================================================================
# PyTorch/GPU Implementation
# ============================================================================


def _quaternion_multiply_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """PyTorch quaternion multiplication.

    :param q1: First quaternion [4] or [N, 4] (w, x, y, z)
    :param q2: Second quaternion [4] or [N, 4] (w, x, y, z)
    :returns: Product quaternion
    """
    import torch

    # Handle 1D inputs
    squeeze_output = False
    if q1.dim() == 1:
        q1 = q1.unsqueeze(0)
        squeeze_output = True
    if q2.dim() == 1:
        q2 = q2.unsqueeze(0)

    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = torch.stack([w, x, y, z], dim=1)
    return result.squeeze(0) if squeeze_output else result


def _quaternion_to_rotation_matrix_torch(q: torch.Tensor) -> torch.Tensor:
    """PyTorch quaternion to 3x3 rotation matrix.

    :param q: Quaternion [4] (w, x, y, z)
    :returns: 3x3 rotation matrix
    """
    import torch

    q = q / torch.norm(q)
    w, x, y, z = q[0], q[1], q[2], q[3]

    R = torch.zeros((3, 3), dtype=q.dtype, device=q.device)

    R[0, 0] = 1 - 2 * (y * y + z * z)
    R[0, 1] = 2 * (x * y - w * z)
    R[0, 2] = 2 * (x * z + w * y)

    R[1, 0] = 2 * (x * y + w * z)
    R[1, 1] = 1 - 2 * (x * x + z * z)
    R[1, 2] = 2 * (y * z - w * x)

    R[2, 0] = 2 * (x * z - w * y)
    R[2, 1] = 2 * (y * z + w * x)
    R[2, 2] = 1 - 2 * (x * x + y * y)

    return R


def _rotation_matrix_to_quaternion_torch(R: torch.Tensor) -> torch.Tensor:
    """PyTorch 3x3 rotation matrix to quaternion.

    :param R: 3x3 rotation matrix
    :returns: Quaternion [4] (w, x, y, z)
    """
    import torch

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = torch.tensor([w, x, y, z], dtype=R.dtype, device=R.device)
    return q / torch.norm(q)


def _axis_angle_to_quaternion_torch(axis_angle: torch.Tensor) -> torch.Tensor:
    """PyTorch axis-angle to quaternion.

    :param axis_angle: Axis-angle vector [3] (axis * angle)
    :returns: Quaternion [4] (w, x, y, z)
    """
    import torch

    angle = torch.norm(axis_angle)
    if angle < 1e-8:
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=axis_angle.dtype, device=axis_angle.device)

    axis = axis_angle / angle
    half_angle = angle / 2
    sin_half = torch.sin(half_angle)

    w = torch.cos(half_angle)
    x = axis[0] * sin_half
    y = axis[1] * sin_half
    z = axis[2] * sin_half

    return torch.stack([w, x, y, z])


def _euler_to_quaternion_torch(euler: torch.Tensor, order: str = "XYZ") -> torch.Tensor:
    """PyTorch Euler angles to quaternion.

    :param euler: Euler angles [3] (roll, pitch, yaw) in radians
    :param order: Rotation order (e.g., "XYZ", "ZYX")
    :returns: Quaternion [4] (w, x, y, z)
    """
    import torch

    roll, pitch, yaw = euler[0], euler[1], euler[2]

    cr = torch.cos(roll / 2)
    sr = torch.sin(roll / 2)
    cp = torch.cos(pitch / 2)
    sp = torch.sin(pitch / 2)
    cy = torch.cos(yaw / 2)
    sy = torch.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z])


def _quaternion_to_euler_torch(q: torch.Tensor) -> torch.Tensor:
    """PyTorch quaternion to Euler angles.

    :param q: Quaternion [4] (w, x, y, z)
    :returns: Euler angles [3] (roll, pitch, yaw) in radians
    """
    import torch

    w, x, y, z = q[0], q[1], q[2], q[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if torch.abs(sinp) >= 1:
        pitch = torch.copysign(torch.tensor(np.pi / 2, device=q.device), sinp)
    else:
        pitch = torch.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack([roll, pitch, yaw])


# ============================================================================
# Public API - Auto-dispatching functions
# ============================================================================


def _is_torch_tensor(x) -> bool:
    """Check if input is a PyTorch tensor without importing torch."""
    return type(x).__module__.startswith("torch")


def quaternion_multiply(q1, q2, out=None):
    """Multiply quaternions.

    Auto-dispatches to NumPy or PyTorch based on input type.

    :param q1: First quaternion [4] or [N, 4] (w, x, y, z)
    :param q2: Second quaternion [4] or [N, 4] (w, x, y, z)
    :param out: Optional output buffer (NumPy only)
    :returns: Product quaternion

    Example:
        >>> import numpy as np
        >>> q1 = np.array([1, 0, 0, 0])  # Identity
        >>> q2 = np.array([0.707, 0, 0.707, 0])  # 90 deg Y rotation
        >>> result = quaternion_multiply(q1, q2)
    """
    if _is_torch_tensor(q1):
        return _quaternion_multiply_torch(q1, q2)
    return _quaternion_multiply_numpy(np.asarray(q1), np.asarray(q2), out)


def quaternion_to_rotation_matrix(q):
    """Convert quaternion to 3x3 rotation matrix.

    Auto-dispatches to NumPy or PyTorch based on input type.

    :param q: Quaternion [4] (w, x, y, z)
    :returns: 3x3 rotation matrix

    Example:
        >>> import numpy as np
        >>> q = np.array([0.707, 0, 0.707, 0])  # 90 deg Y rotation
        >>> R = quaternion_to_rotation_matrix(q)
    """
    if _is_torch_tensor(q):
        return _quaternion_to_rotation_matrix_torch(q)
    return _quaternion_to_rotation_matrix_numpy(np.asarray(q))


def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion.

    Auto-dispatches to NumPy or PyTorch based on input type.

    :param R: 3x3 rotation matrix
    :returns: Quaternion [4] (w, x, y, z)

    Example:
        >>> import numpy as np
        >>> R = np.eye(3)  # Identity rotation
        >>> q = rotation_matrix_to_quaternion(R)
    """
    if _is_torch_tensor(R):
        return _rotation_matrix_to_quaternion_torch(R)
    return _rotation_matrix_to_quaternion_numpy(np.asarray(R))


def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle to quaternion.

    Auto-dispatches to NumPy or PyTorch based on input type.

    :param axis_angle: Axis-angle vector [3] (axis * angle)
    :returns: Quaternion [4] (w, x, y, z)

    Example:
        >>> import numpy as np
        >>> axis_angle = np.array([0, np.pi/2, 0])  # 90 deg Y rotation
        >>> q = axis_angle_to_quaternion(axis_angle)
    """
    if _is_torch_tensor(axis_angle):
        return _axis_angle_to_quaternion_torch(axis_angle)
    return _axis_angle_to_quaternion_numpy(np.asarray(axis_angle))


def euler_to_quaternion(euler, order: str = "XYZ"):
    """Convert Euler angles to quaternion.

    Auto-dispatches to NumPy or PyTorch based on input type.

    :param euler: Euler angles [3] (roll, pitch, yaw) in radians
    :param order: Rotation order (e.g., "XYZ", "ZYX")
    :returns: Quaternion [4] (w, x, y, z)

    Example:
        >>> import numpy as np
        >>> euler = np.array([0, np.pi/4, 0])  # 45 deg pitch
        >>> q = euler_to_quaternion(euler)
    """
    if _is_torch_tensor(euler):
        return _euler_to_quaternion_torch(euler, order)
    return _euler_to_quaternion_numpy(np.asarray(euler), order)


def quaternion_to_euler(q):
    """Convert quaternion to Euler angles.

    Auto-dispatches to NumPy or PyTorch based on input type.

    :param q: Quaternion [4] (w, x, y, z)
    :returns: Euler angles [3] (roll, pitch, yaw) in radians

    Example:
        >>> import numpy as np
        >>> q = np.array([0.924, 0, 0.383, 0])  # ~45 deg Y rotation
        >>> euler = quaternion_to_euler(q)
    """
    if _is_torch_tensor(q):
        return _quaternion_to_euler_torch(q)
    return _quaternion_to_euler_numpy(np.asarray(q))


# ============================================================================
# Convenience Functions
# ============================================================================


def normalize_quaternion(q):
    """Normalize quaternion to unit length.

    :param q: Quaternion [4] (w, x, y, z)
    :returns: Normalized quaternion
    """
    if _is_torch_tensor(q):
        import torch

        return q / torch.norm(q)
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def quaternion_identity(dtype=np.float32, device=None):
    """Return identity quaternion.

    :param dtype: Data type (numpy dtype or torch dtype)
    :param device: Device (for PyTorch tensors)
    :returns: Identity quaternion [1, 0, 0, 0]
    """
    if device is not None:
        import torch

        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=dtype, device=device)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=dtype)
