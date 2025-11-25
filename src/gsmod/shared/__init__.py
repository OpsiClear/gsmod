"""Shared utilities for gsmod.

This module contains utilities shared between CPU (NumPy/Numba) and
GPU (PyTorch) implementations to avoid code duplication.
"""

from gsmod.shared.format import (
    copy_format_dict,
    ensure_format_copied,
    ensure_linear,
    ensure_ply,
    ensure_rgb,
    ensure_sh,
    get_opacity_for_threshold,
    get_scale_for_threshold,
    is_linear_format,
    is_ply_format,
)
from gsmod.shared.rotation import (
    # NumPy/CPU functions
    axis_angle_to_quaternion,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)

__all__ = [
    # Rotation utilities
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "euler_to_quaternion",
    "quaternion_to_euler",
    # Format utilities
    "ensure_format_copied",
    "copy_format_dict",
    "is_ply_format",
    "is_linear_format",
    "ensure_rgb",
    "ensure_sh",
    "ensure_linear",
    "ensure_ply",
    "get_opacity_for_threshold",
    "get_scale_for_threshold",
]
