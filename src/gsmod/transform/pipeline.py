"""
Transform: Composable geometric transform pipeline with matrix pre-composition.

This module provides a fluent API for chaining 3D transformations and compiling
them into a single optimized transformation matrix for maximum performance.

Key Features:
- Explicit rotation methods (rotate_quat, rotate_euler, rotate_axis_angle, rotate_matrix)
- Method chaining for intuitive pipeline construction
- Automatic compilation of multiple transforms into single matrix
- Quaternion accumulation for orientation updates
- Scale accumulation for size updates
- Single-pass application for optimal performance
- GSData integration for unified data handling
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Self

import numpy as np

# Import GSData from gsply
from gsply import GSData

# Import constants and validators
# Import transform utilities from the existing transform module
from gsmod.transform.api import (
    ArrayLike,
    _axis_angle_to_quaternion_numpy,
    _build_rotation_matrix_4x4_numpy,
    _build_scale_matrix_4x4_numpy,
    _build_translation_matrix_4x4_numpy,
    _euler_to_quaternion_numpy,
    _quaternion_multiply_numpy,
    _quaternion_to_rotation_matrix_numpy,
    _rotation_matrix_to_quaternion_numpy,
)
from gsmod.transform.kernels import (
    elementwise_multiply_vector_numba,
    fused_transform_numba,
)

logger = logging.getLogger(__name__)


class Transform:
    """
    Composable geometric transform pipeline with matrix pre-composition.

    This class allows chaining multiple 3D transformations and compiling them
    into a single optimized transformation matrix for maximum performance.

    Supported Operations:
    - translate: Translation in 3D space
    - rotate_quat: Rotation from quaternion [w, x, y, z]
    - rotate_euler: Rotation from Euler angles [roll, pitch, yaw]
    - rotate_axis_angle: Rotation from axis and angle
    - rotate_matrix: Rotation from 3x3 rotation matrix
    - scale: Uniform or per-axis scaling
    - set_center: Set center point for rotation/scaling

    Example:
        >>> pipeline = (Transform()
        ...     .scale(2.0)
        ...     .rotate_quat([0, 0, 0, 1])
        ...     .translate([1, 0, 0])
        ...     .compile()  # Explicit compilation
        ... )
        >>> transformed_data = pipeline(data, inplace=True)
    """

    __slots__ = (
        "_transforms",
        "_compiled_matrix",
        "_compiled_quat",
        "_compiled_scale",
        "_center",
        "_is_dirty",
    )

    def __init__(self):
        """Initialize the transform pipeline."""
        # List of transform operations to apply
        self._transforms: list[tuple[str, dict]] = []

        # Compiled transformation data
        self._compiled_matrix: np.ndarray | None = None
        self._compiled_quat: np.ndarray | None = None
        self._compiled_scale: np.ndarray | None = None

        # Current center for rotation/scaling (if any)
        self._center: np.ndarray | None = None

        # Dirty flag for recompilation
        self._is_dirty: bool = True

        logger.info("[Transform] Initialized")

    def translate(self, vector: ArrayLike) -> Self:
        """
        Add a translation to the pipeline.

        :param vector: Translation vector [3] or list/tuple
        :returns: Self for method chaining
        """
        if not isinstance(vector, np.ndarray | list | tuple):
            raise TypeError(f"vector must be array-like, got {type(vector)}")

        self._transforms.append(("translate", {"translation": vector}))
        self._is_dirty = True
        return self

    def rotate_quat(self, quaternion: ArrayLike, center: ArrayLike | None = None) -> Self:
        """
        Add a quaternion rotation to the pipeline.

        :param quaternion: Rotation quaternion [w, x, y, z] (4-element array)
        :param center: Optional center point for rotation
        :returns: Self for method chaining

        Example:
            >>> Transform().rotate_quat([1, 0, 0, 0])  # Identity rotation
        """
        params = {"rotation": quaternion, "format": "quaternion"}
        if center is not None:
            params["center"] = center

        self._transforms.append(("rotate", params))
        self._is_dirty = True
        return self

    def rotate_euler(self, angles: ArrayLike, center: ArrayLike | None = None) -> Self:
        """
        Add an Euler angle rotation to the pipeline.

        :param angles: Euler angles [roll, pitch, yaw] in radians (3-element array)
        :param center: Optional center point for rotation
        :returns: Self for method chaining

        Example:
            >>> Transform().rotate_euler([0, 0, np.pi/2])  # 90deg yaw rotation
        """
        params = {"rotation": angles, "format": "euler"}
        if center is not None:
            params["center"] = center

        self._transforms.append(("rotate", params))
        self._is_dirty = True
        return self

    def rotate_axis_angle(
        self, axis: ArrayLike, angle: float, center: ArrayLike | None = None
    ) -> Self:
        """
        Add an axis-angle rotation to the pipeline.

        :param axis: Rotation axis (3-element array, will be normalized)
        :param angle: Rotation angle in radians
        :param center: Optional center point for rotation
        :returns: Self for method chaining

        Example:
            >>> Transform().rotate_axis_angle(axis=[0, 0, 1], angle=np.pi/2)  # 90deg around Z
        """
        # Encode axis-angle as scaled axis vector (magnitude = angle)
        # The internal representation uses the magnitude of the vector as the angle
        axis_arr = np.asarray(axis, dtype=np.float32)
        axis_norm = axis_arr / np.linalg.norm(axis_arr)
        axis_angle = axis_norm * angle  # Scale normalized axis by angle

        params = {"rotation": axis_angle, "format": "axis_angle"}
        if center is not None:
            params["center"] = center

        self._transforms.append(("rotate", params))
        self._is_dirty = True
        return self

    def rotate_matrix(self, matrix: ArrayLike, center: ArrayLike | None = None) -> Self:
        """
        Add a rotation matrix rotation to the pipeline.

        :param matrix: 3x3 rotation matrix
        :param center: Optional center point for rotation
        :returns: Self for method chaining

        Example:
            >>> R = np.eye(3)  # Identity rotation
            >>> Transform().rotate_matrix(R)
        """
        params = {"rotation": matrix, "format": "matrix"}
        if center is not None:
            params["center"] = center

        self._transforms.append(("rotate", params))
        self._is_dirty = True
        return self

    def scale(self, factor: float | ArrayLike, center: ArrayLike | None = None) -> Self:
        """
        Add a scaling to the pipeline.

        :param factor: Uniform scale (float) or per-axis scale [3]
        :param center: Optional center point for scaling
        :returns: Self for method chaining
        """
        params = {"scale_factor": factor}
        if center is not None:
            params["center"] = center

        self._transforms.append(("scale", params))
        self._is_dirty = True
        return self

    def set_center(self, center: ArrayLike) -> Self:
        """
        Set a center point for subsequent rotation/scaling operations.

        :param center: Center point [3]
        :returns: Self for method chaining
        """
        if not isinstance(center, np.ndarray | list | tuple):
            raise TypeError(f"center must be array-like, got {type(center)}")

        self._center = np.asarray(center, dtype=np.float32)
        return self

    def compile(self) -> Self:
        """
        Explicitly compile all transforms into a single transformation matrix.

        This method composes all transformation operations into a single 4x4
        homogeneous transformation matrix and accumulates quaternions/scales
        for efficient single-pass application.

        :returns: Self for method chaining
        """
        if not self._is_dirty and self._compiled_matrix is not None:
            logger.debug("[Transform] Already compiled, skipping")
            return self

        logger.debug("[Transform] Compiling %d transforms", len(self._transforms))

        # Initialize with identity matrix
        M = np.eye(4, dtype=np.float32)

        # Track accumulated quaternion and scale
        accumulated_quat = None
        accumulated_scale = None

        # Process each transform operation
        for op_type, params in self._transforms:
            if op_type == "translate":
                # Build translation matrix
                translation = np.asarray(params["translation"], dtype=np.float32)
                T = _build_translation_matrix_4x4_numpy(translation)
                M = T @ M

            elif op_type == "rotate":
                # Handle rotation with optional center
                rotation = params["rotation"]
                format = params["format"]
                center = params.get("center", self._center)

                # Convert rotation to quaternion
                if format == "quaternion":
                    quat = np.asarray(rotation, dtype=np.float32)
                elif format == "matrix":
                    rot_matrix = np.asarray(rotation, dtype=np.float32)
                    quat = _rotation_matrix_to_quaternion_numpy(rot_matrix)
                elif format == "axis_angle":
                    axis_angle = np.asarray(rotation, dtype=np.float32)
                    quat = _axis_angle_to_quaternion_numpy(axis_angle)
                elif format == "euler":
                    euler = np.asarray(rotation, dtype=np.float32)
                    quat = _euler_to_quaternion_numpy(euler)
                else:
                    raise ValueError(f"Unknown rotation format: {format}")

                # Accumulate quaternion
                if accumulated_quat is None:
                    accumulated_quat = quat
                else:
                    # Quaternion multiplication: q_total = q_new * q_old
                    accumulated_quat = _quaternion_multiply_numpy(quat, accumulated_quat)

                # Build rotation matrix
                rot_matrix_3x3 = _quaternion_to_rotation_matrix_numpy(quat)
                R = _build_rotation_matrix_4x4_numpy(rot_matrix_3x3)

                # Handle center if specified
                if center is not None:
                    center_arr = np.asarray(center, dtype=np.float32)
                    # T(center) @ R @ T(-center)
                    T_center = _build_translation_matrix_4x4_numpy(center_arr)
                    T_neg_center = _build_translation_matrix_4x4_numpy(-center_arr)
                    R = T_center @ R @ T_neg_center

                M = R @ M

            elif op_type == "scale":
                # Handle scaling with optional center
                scale_factor = params["scale_factor"]
                center = params.get("center", self._center)

                # Convert to scale vector
                if isinstance(scale_factor, int | float):
                    scale_vec = np.array(
                        [scale_factor, scale_factor, scale_factor], dtype=np.float32
                    )
                else:
                    scale_vec = np.asarray(scale_factor, dtype=np.float32)
                    if scale_vec.ndim == 0:
                        scale_vec = np.array([float(scale_vec)] * 3, dtype=np.float32)

                # Accumulate scale
                if accumulated_scale is None:
                    accumulated_scale = scale_vec
                else:
                    accumulated_scale = accumulated_scale * scale_vec

                # Build scale matrix
                S = _build_scale_matrix_4x4_numpy(scale_vec)

                # Handle center if specified
                if center is not None:
                    center_arr = np.asarray(center, dtype=np.float32)
                    # T(center) @ S @ T(-center)
                    T_center = _build_translation_matrix_4x4_numpy(center_arr)
                    T_neg_center = _build_translation_matrix_4x4_numpy(-center_arr)
                    S = T_center @ S @ T_neg_center

                M = S @ M

        # Store compiled results
        self._compiled_matrix = M
        self._compiled_quat = accumulated_quat
        self._compiled_scale = accumulated_scale
        self._is_dirty = False

        logger.debug("[Transform] Compilation complete")
        return self

    def _apply_to_arrays(
        self,
        means: np.ndarray,
        quaternions: np.ndarray | None = None,
        scales: np.ndarray | None = None,
        inplace: bool = True,
        make_contiguous: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """
        Internal method: Apply transform pipeline to NumPy arrays.

        :param means: Positions [N, 3] to transform
        :param quaternions: Optional orientations [N, 4] to rotate
        :param scales: Optional sizes [N, 3] to scale
        :param inplace: If True, modifies input arrays directly
        :param make_contiguous: If True, ensures C-contiguous arrays for optimal performance
        :returns: Tuple of (transformed_means, transformed_quaternions, transformed_scales)
        """
        # Auto-compile if needed
        if self._is_dirty or self._compiled_matrix is None:
            self.compile()

        # Handle copy/inplace
        if not inplace:
            means = means.copy()
            if quaternions is not None:
                quaternions = quaternions.copy()
            if scales is not None:
                scales = scales.copy()

        # Optionally ensure contiguous arrays for optimal performance
        # Note: Conversion has ~3ms overhead for 100K Gaussians
        # Only enable if doing 8+ operations (cost-benefit analysis)
        if make_contiguous:
            # Use GSData's built-in make_contiguous() if available (gsply v0.2.6+)
            # Falls back to manual np.ascontiguousarray() for compatibility
            if not means.flags["C_CONTIGUOUS"]:
                means = np.ascontiguousarray(means)
            if quaternions is not None and not quaternions.flags["C_CONTIGUOUS"]:
                quaternions = np.ascontiguousarray(quaternions)
            if scales is not None and not scales.flags["C_CONTIGUOUS"]:
                scales = np.ascontiguousarray(scales)

        # Extract rotation and translation from compiled matrix
        R = self._compiled_matrix[:3, :3]  # Upper-left 3x3
        t = self._compiled_matrix[:3, 3]  # Translation column

        # Check if we can use the fused kernel (fastest path)
        if (
            quaternions is not None
            and scales is not None
            and self._compiled_quat is not None
            and self._compiled_scale is not None
        ):
            # Use fused Numba kernel (4-5x faster)
            fused_transform_numba(
                means,
                quaternions,
                scales,
                self._compiled_quat,
                self._compiled_scale,
                t,
                R,
                means,  # In-place
                quaternions,  # In-place
                scales,  # In-place
            )
        else:
            # Standard path: separate operations
            # Transform positions
            means[:] = means @ R.T + t

            # Transform quaternions if provided
            if quaternions is not None and self._compiled_quat is not None:
                _quaternion_multiply_numpy(self._compiled_quat, quaternions, out=quaternions)

            # Transform scales if provided
            if scales is not None and self._compiled_scale is not None:
                elementwise_multiply_vector_numba(scales, self._compiled_scale, scales)

        return means, quaternions, scales

    def is_identity(self) -> bool:
        """
        Check if this pipeline applies no geometric transformations (identity operation).

        :returns: True if pipeline has no operations, False otherwise

        Note:
            Identity pipelines have zero overhead with fast-path optimization
            (measured < 0.001ms vs 3.1ms without optimization)
        """
        return not self._transforms

    def apply(self, data: GSData, inplace: bool = True, make_contiguous: bool = False) -> GSData:
        """
        Apply the transform pipeline to GSData object.

        :param data: GSData object containing Gaussian data
        :param inplace: If True, modifies input GSData directly
        :param make_contiguous: If True, ensures C-contiguous arrays for optimal Numba performance
        :returns: GSData with transformed positions, orientations, and scales

        Example:
            >>> import gsply
            >>> from gsmod import Transform
            >>> data = gsply.plyread("scene.ply")
            >>> transformed_data = Transform().rotate_quat(quat).translate([1, 0, 0]).apply(data)
            >>> # Or use __call__:
            >>> transformed_data = Transform().rotate_quat(quat).translate([1, 0, 0])(data)

        Note:
            make_contiguous defaults to False to avoid ~3ms conversion overhead for 100K
            Gaussians on PLY-loaded data (which is non-contiguous by design). For workflows
            with 8+ operations, set make_contiguous=True for 2-45x per-operation speedup.

            Uses GSData.is_contiguous() and make_contiguous() when available (gsply v0.2.6+)
            for better integration with gsply's memory management.
        """
        # Fast-path: identity pipeline (no operations)
        # Measured: 0.001ms vs 3.1ms overhead for empty pipeline
        if self.is_identity():
            if inplace:
                return data
            result = data.copy()
            # Deep copy _format since gsply.copy() does shallow copy
            if hasattr(result, "_format"):
                result._format = result._format.copy()
            return result

        # Use GSData's built-in make_contiguous() if requested and available (gsply v0.2.6+)
        if make_contiguous and hasattr(data, "make_contiguous") and hasattr(data, "is_contiguous"):
            if not data.is_contiguous():
                data.make_contiguous(inplace=True)
                logger.debug("[Transform] Made GSData arrays contiguous for optimal performance")

        # Extract arrays from GSData
        means = data.means
        quaternions = data.quats
        scales = data.scales

        # Apply transform
        transformed_means, transformed_quats, transformed_scales = self._apply_to_arrays(
            means, quaternions, scales, inplace=inplace, make_contiguous=make_contiguous
        )

        # If inplace, data arrays are already modified, return same GSData
        if inplace:
            logger.info("[Transform] Applied to %d Gaussians (in-place)", len(data))
            return data

        # If not inplace, leverage GSData.copy() and modify geometric fields
        # This is more efficient and idiomatic than manual field copying
        transformed_data = data.copy()
        # Deep copy _format since gsply.copy() does shallow copy
        if hasattr(transformed_data, "_format"):
            transformed_data._format = transformed_data._format.copy()
        transformed_data.means = transformed_means
        transformed_data.scales = transformed_scales
        transformed_data.quats = transformed_quats

        logger.info("[Transform] Applied to %d Gaussians (new copy)", len(data))
        return transformed_data

    def __call__(self, data: GSData, inplace: bool = True, make_contiguous: bool = False) -> GSData:
        """
        Apply the pipeline when called as a function.

        This is the primary way to use the pipeline - clean and Pythonic.

        :param data: GSData object containing Gaussian data
        :param inplace: If True, modifies input GSData directly
        :param make_contiguous: If True, ensures C-contiguous arrays for optimal Numba performance
        :returns: GSData with transformed geometry

        Example:
            >>> data = Transform().rotate_quat(quat).translate([1, 0, 0])(data, inplace=True)
        """
        return self.apply(data, inplace=inplace, make_contiguous=make_contiguous)

    def reset(self) -> Self:
        """
        Reset the pipeline, clearing all transforms.

        :returns: Self for method chaining
        """
        self._transforms = []
        self._compiled_matrix = None
        self._compiled_quat = None
        self._compiled_scale = None
        self._center = None
        self._is_dirty = True
        logger.debug("[Transform] Reset")
        return self

    def copy(self) -> Self:
        """
        Create a deep copy of the pipeline.

        :returns: New Transform instance with copied state
        """
        return deepcopy(self)

    def __copy__(self) -> Self:
        """Shallow copy (creates deep copy for safety)."""
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Deep copy implementation."""
        # Create new instance
        new_obj = self.__class__()

        # Deep copy all attributes
        new_obj._transforms = deepcopy(self._transforms, memo)
        new_obj._compiled_matrix = (
            self._compiled_matrix.copy() if self._compiled_matrix is not None else None
        )
        new_obj._compiled_quat = (
            self._compiled_quat.copy() if self._compiled_quat is not None else None
        )
        new_obj._compiled_scale = (
            self._compiled_scale.copy() if self._compiled_scale is not None else None
        )
        new_obj._center = self._center.copy() if self._center is not None else None
        new_obj._is_dirty = self._is_dirty

        return new_obj

    def get_matrix(self) -> np.ndarray | None:
        """
        Get the compiled 4x4 transformation matrix.

        :returns: The compiled matrix if available, None otherwise
        """
        if self._is_dirty or self._compiled_matrix is None:
            self.compile()
        return self._compiled_matrix

    @property
    def is_compiled(self) -> bool:
        """Check if the pipeline is compiled."""
        return not self._is_dirty and self._compiled_matrix is not None

    @property
    def num_operations(self) -> int:
        """Number of transform operations in the pipeline."""
        return len(self._transforms)

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        status = "compiled" if self.is_compiled else "not compiled"
        return f"Transform({self.num_operations} operations) [{status}]"

    def __len__(self) -> int:
        """Number of transform operations in the pipeline."""
        return len(self._transforms)
