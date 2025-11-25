"""
3D transform module - Geometric transformations with Numba-optimized kernels.

Performance (CPU, 1M Gaussians):
  - Transform pipeline: 1.43ms (698M Gaussians/sec)
  - Matrix fusion: Combines multiple transforms into single operation
  - Quaternion utilities: 200x faster with Numba JIT compilation

Example:
    >>> from gsmod.transform import Transform
    >>> pipeline = Transform().translate([1, 0, 0]).rotate_euler(0, 45, 0).scale(2.0)
    >>> result = pipeline(data, inplace=True)
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
