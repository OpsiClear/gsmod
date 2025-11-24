"""
3D transform module.

Provides high-performance geometric transformations for 3D Gaussian Splatting
with chainable pipeline interface for composing multiple transforms.
"""

from gsmod.transform.api import (
    axis_angle_to_quaternion,
    euler_to_quaternion,
    quaternion_multiply,
    quaternion_to_euler,
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from gsmod.transform.pipeline import Transform

__all__ = [
    "Transform",
    "quaternion_multiply",
    "quaternion_to_rotation_matrix",
    "rotation_matrix_to_quaternion",
    "axis_angle_to_quaternion",
    "euler_to_quaternion",
    "quaternion_to_euler",
]
