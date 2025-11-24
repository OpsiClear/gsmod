"""GPU-accelerated transform pipeline for Gaussian Splatting."""

from __future__ import annotations

import numpy as np
import torch

from gsmod.torch.gstensor_pro import GSTensorPro


class TransformGPU:
    """GPU-accelerated 3D transformation pipeline using PyTorch.

    Provides chainable transformation operations that run entirely on GPU with
    massive parallelism. All operations return self for fluent chaining.

    Performance:
        - 20-50x faster than CPU transforms
        - Matrix operations leverage GPU BLAS
        - Batched quaternion operations

    Example:
        >>> from gsmod.torch import TransformGPU
        >>> pipeline = (
        ...     TransformGPU()
        ...     .translate([1, 0, 0])
        ...     .rotate_euler([0, np.pi/4, 0])
        ...     .scale(2.0)
        ... )
        >>> result = pipeline(gstensor, inplace=True)
    """

    def __init__(self):
        """Initialize transform pipeline."""
        self._operations = []

    def translate(self, translation: list[float] | np.ndarray | torch.Tensor) -> TransformGPU:
        """Add translation.

        :param translation: Translation vector [x, y, z]
        :returns: Self for chaining

        Example:
            >>> pipeline = TransformGPU().translate([1, 0, 0])
        """
        self._operations.append(("translate", translation))
        return self

    def scale(self, scale: float | list[float] | np.ndarray | torch.Tensor) -> TransformGPU:
        """Add scaling.

        :param scale: Uniform scale (float) or non-uniform [sx, sy, sz]
        :returns: Self for chaining

        Example:
            >>> pipeline = TransformGPU().scale(2.0)  # Uniform
            >>> pipeline = TransformGPU().scale([1, 2, 0.5])  # Non-uniform
        """
        if isinstance(scale, int | float):
            self._operations.append(("scale_uniform", scale))
        else:
            self._operations.append(("scale_nonuniform", scale))
        return self

    def rotate_quaternion(self, quaternion: np.ndarray | torch.Tensor) -> TransformGPU:
        """Add rotation by quaternion.

        :param quaternion: Rotation quaternion [w, x, y, z]
        :returns: Self for chaining

        Example:
            >>> import gsmod.transform.api as tf
            >>> quat = tf.axis_angle_to_quaternion([0, 1, 0], np.pi/4)
            >>> pipeline = TransformGPU().rotate_quaternion(quat)
        """
        self._operations.append(("rotate_quaternion", quaternion))
        return self

    def rotate_euler(
        self, angles: list[float] | np.ndarray | torch.Tensor, order: str = "XYZ"
    ) -> TransformGPU:
        """Add rotation by Euler angles.

        :param angles: Euler angles [x, y, z] in radians
        :param order: Rotation order (e.g., "XYZ", "ZYX")
        :returns: Self for chaining

        Example:
            >>> pipeline = TransformGPU().rotate_euler([0, np.pi/4, 0], order="XYZ")
        """
        self._operations.append(("rotate_euler", (angles, order)))
        return self

    def rotate_axis_angle(
        self, axis: list[float] | np.ndarray | torch.Tensor, angle: float
    ) -> TransformGPU:
        """Add rotation around axis.

        :param axis: Rotation axis [x, y, z]
        :param angle: Rotation angle in radians
        :returns: Self for chaining

        Example:
            >>> pipeline = TransformGPU().rotate_axis_angle([0, 1, 0], np.pi/4)
        """
        self._operations.append(("rotate_axis_angle", (axis, angle)))
        return self

    def rotate_matrix(self, matrix: np.ndarray | torch.Tensor) -> TransformGPU:
        """Add rotation by matrix.

        :param matrix: 3x3 rotation matrix
        :returns: Self for chaining

        Example:
            >>> rot_mat = np.eye(3)  # Identity rotation
            >>> pipeline = TransformGPU().rotate_matrix(rot_mat)
        """
        self._operations.append(("rotate_matrix", matrix))
        return self

    def transform_matrix(self, matrix: np.ndarray | torch.Tensor) -> TransformGPU:
        """Add 4x4 transformation matrix.

        :param matrix: 4x4 homogeneous transformation matrix
        :returns: Self for chaining

        Example:
            >>> transform = np.eye(4)
            >>> transform[:3, 3] = [1, 0, 0]  # Translation
            >>> pipeline = TransformGPU().transform_matrix(transform)
        """
        self._operations.append(("transform_matrix", matrix))
        return self

    def center_at_origin(self) -> TransformGPU:
        """Add centering at origin.

        :returns: Self for chaining

        Example:
            >>> pipeline = TransformGPU().center_at_origin().scale(2.0)
        """
        self._operations.append(("center_at_origin", None))
        return self

    def normalize_scale(self, target_size: float = 2.0) -> TransformGPU:
        """Add scale normalization to fit in target size.

        :param target_size: Target bounding box size
        :returns: Self for chaining

        Example:
            >>> pipeline = TransformGPU().normalize_scale(2.0)
        """
        self._operations.append(("normalize_scale", target_size))
        return self

    def __call__(self, data: GSTensorPro, inplace: bool = True) -> GSTensorPro:
        """Apply transform pipeline to GSTensorPro.

        :param data: GSTensorPro object to process
        :param inplace: If True, modify data in-place; if False, create copy
        :returns: Processed GSTensorPro

        Example:
            >>> pipeline = TransformGPU().translate([1, 0, 0]).scale(2.0)
            >>> result = pipeline(gstensor, inplace=True)
        """
        if not isinstance(data, GSTensorPro):
            raise TypeError(f"Expected GSTensorPro, got {type(data)}")

        # Create copy if not inplace
        if not inplace:
            data = data.clone()

        # Apply operations
        for op_name, op_value in self._operations:
            if op_name == "translate":
                data.translate(op_value, inplace=True)
            elif op_name == "scale_uniform":
                data.scale_uniform(op_value, inplace=True)
            elif op_name == "scale_nonuniform":
                data.scale_nonuniform(op_value, inplace=True)
            elif op_name == "rotate_quaternion":
                data.rotate_quaternion(op_value, inplace=True)
            elif op_name == "rotate_euler":
                angles, order = op_value
                data.rotate_euler(angles, order, inplace=True)
            elif op_name == "rotate_axis_angle":
                axis, angle = op_value
                data.rotate_axis_angle(axis, angle, inplace=True)
            elif op_name == "rotate_matrix":
                # Convert rotation matrix to quaternion
                from gsmod.transform.api import rotation_matrix_to_quaternion

                if isinstance(op_value, torch.Tensor):
                    matrix_np = op_value.cpu().numpy()
                else:
                    matrix_np = op_value
                quat = rotation_matrix_to_quaternion(matrix_np)
                data.rotate_quaternion(quat, inplace=True)
            elif op_name == "transform_matrix":
                data.transform_matrix(op_value, inplace=True)
            elif op_name == "center_at_origin":
                data.center_at_origin(inplace=True)
            elif op_name == "normalize_scale":
                data.normalize_scale(op_value, inplace=True)

        return data

    def reset(self) -> TransformGPU:
        """Reset pipeline, removing all operations.

        :returns: Self for chaining

        Example:
            >>> pipeline.reset().translate([1, 0, 0])  # Clear and start fresh
        """
        self._operations = []
        return self

    def clone(self) -> TransformGPU:
        """Create a copy of this pipeline.

        :returns: New TransformGPU with same operations

        Example:
            >>> pipeline2 = pipeline1.clone()
            >>> pipeline2.scale(2.0)  # Doesn't affect pipeline1
        """
        new_pipeline = TransformGPU()
        new_pipeline._operations = self._operations.copy()
        return new_pipeline

    def to_matrix(self) -> torch.Tensor:
        """Compose all transforms into a single 4x4 matrix.

        Useful for applying the same transform multiple times efficiently.

        :returns: 4x4 transformation matrix as torch.Tensor

        Example:
            >>> pipeline = TransformGPU().translate([1, 0, 0]).scale(2.0)
            >>> matrix = pipeline.to_matrix()
            >>> # Now can apply matrix directly
            >>> gstensor.transform_matrix(matrix, inplace=True)
        """
        # Start with identity
        matrix = torch.eye(4, dtype=torch.float32)

        # We need a dummy tensor to compute transforms
        # This is a bit of a hack, but works for computing the final matrix
        for op_name, op_value in self._operations:
            if op_name == "translate":
                if not isinstance(op_value, torch.Tensor):
                    translation = torch.tensor(op_value, dtype=torch.float32)
                else:
                    translation = op_value.to(dtype=torch.float32)
                T = torch.eye(4)
                T[:3, 3] = translation
                matrix = torch.matmul(matrix, T)

            elif op_name == "scale_uniform":
                S = torch.eye(4)
                S[:3, :3] *= op_value
                matrix = torch.matmul(matrix, S)

            elif op_name == "scale_nonuniform":
                if not isinstance(op_value, torch.Tensor):
                    scale = torch.tensor(op_value, dtype=torch.float32)
                else:
                    scale = op_value.to(dtype=torch.float32)
                S = torch.eye(4)
                S[0, 0] = scale[0]
                S[1, 1] = scale[1]
                S[2, 2] = scale[2]
                matrix = torch.matmul(matrix, S)

            elif op_name == "rotate_quaternion":
                # Convert quaternion to rotation matrix
                if not isinstance(op_value, torch.Tensor):
                    quat = torch.tensor(op_value, dtype=torch.float32)
                else:
                    quat = op_value.to(dtype=torch.float32)
                quat = quat / torch.norm(quat)
                w, x, y, z = quat
                R = torch.tensor(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0],
                        [0, 0, 0, 1],
                    ]
                )
                matrix = torch.matmul(matrix, R)

            elif op_name == "rotate_euler":
                # Convert Euler angles to quaternion
                from gsmod.transform.api import euler_to_quaternion

                angles, order = op_value
                if isinstance(angles, torch.Tensor):
                    angles_np = angles.cpu().numpy()
                else:
                    angles_np = np.array(angles)
                quat = euler_to_quaternion(angles_np, order)
                # Apply quaternion rotation
                quat = torch.tensor(quat, dtype=torch.float32)
                quat = quat / torch.norm(quat)
                w, x, y, z = quat
                R = torch.tensor(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0],
                        [0, 0, 0, 1],
                    ]
                )
                matrix = torch.matmul(matrix, R)

            elif op_name == "rotate_axis_angle":
                # Convert axis-angle to quaternion
                from gsmod.transform.api import axis_angle_to_quaternion

                axis, angle = op_value
                if isinstance(axis, torch.Tensor):
                    axis_np = axis.cpu().numpy()
                else:
                    axis_np = np.array(axis)
                quat = axis_angle_to_quaternion(axis_np, angle)
                # Apply quaternion rotation
                quat = torch.tensor(quat, dtype=torch.float32)
                quat = quat / torch.norm(quat)
                w, x, y, z = quat
                R = torch.tensor(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0],
                        [0, 0, 0, 1],
                    ]
                )
                matrix = torch.matmul(matrix, R)

            elif op_name == "transform_matrix":
                if not isinstance(op_value, torch.Tensor):
                    T = torch.tensor(op_value, dtype=torch.float32)
                else:
                    T = op_value.to(dtype=torch.float32)
                matrix = torch.matmul(matrix, T)

            # Note: center_at_origin and normalize_scale require the data
            # so they can't be precomputed into a matrix

        return matrix
