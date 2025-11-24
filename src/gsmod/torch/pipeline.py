"""Unified GPU-accelerated pipeline for Gaussian Splatting."""

from __future__ import annotations

import numpy as np
import torch

from gsmod.torch.gstensor_pro import GSTensorPro
from gsmod.torch.color import ColorGPU
from gsmod.torch.transform import TransformGPU
from gsmod.torch.filter import FilterGPU


class PipelineGPU:
    """Unified GPU-accelerated pipeline combining all operations.

    Provides a single interface for chaining color, transform, and filter
    operations in any order. All operations run on GPU with massive parallelism.

    Performance:
        - Single pipeline for all operations
        - Minimal kernel launches
        - Zero CPU-GPU transfer during processing

    Example:
        >>> from gsmod.torch import PipelineGPU
        >>> pipeline = (
        ...     PipelineGPU()
        ...     .within_sphere(radius=1.0)
        ...     .min_opacity(0.1)
        ...     .translate([1, 0, 0])
        ...     .rotate_euler([0, np.pi/4, 0])
        ...     .brightness(1.2)
        ...     .saturation(1.3)
        ... )
        >>> result = pipeline(gstensor, inplace=True)
    """

    def __init__(self):
        """Initialize unified pipeline."""
        self._color_pipeline = ColorGPU()
        self._transform_pipeline = TransformGPU()
        self._filter_pipeline = FilterGPU()
        self._operation_order = []

    # ==========================================================================
    # Color Operations
    # ==========================================================================

    def brightness(self, factor: float = 1.0) -> PipelineGPU:
        """Add brightness adjustment.

        :param factor: Brightness factor (1.0 = no change)
        :returns: Self for chaining
        """
        self._color_pipeline.brightness(factor)
        self._operation_order.append("color")
        return self

    def contrast(self, factor: float = 1.0) -> PipelineGPU:
        """Add contrast adjustment.

        :param factor: Contrast factor (1.0 = no change)
        :returns: Self for chaining
        """
        self._color_pipeline.contrast(factor)
        self._operation_order.append("color")
        return self

    def saturation(self, factor: float = 1.0) -> PipelineGPU:
        """Add saturation adjustment.

        :param factor: Saturation factor (1.0 = no change)
        :returns: Self for chaining
        """
        self._color_pipeline.saturation(factor)
        self._operation_order.append("color")
        return self

    def gamma(self, value: float = 1.0) -> PipelineGPU:
        """Add gamma correction.

        :param value: Gamma value (1.0 = no change)
        :returns: Self for chaining
        """
        self._color_pipeline.gamma(value)
        self._operation_order.append("color")
        return self

    def temperature(self, temp: float = 0.0) -> PipelineGPU:
        """Add temperature adjustment.

        :param temp: Temperature (-1.0 = cold, 0 = neutral, 1.0 = warm)
        :returns: Self for chaining
        """
        self._color_pipeline.temperature(temp)
        self._operation_order.append("color")
        return self

    def vibrance(self, factor: float = 1.0) -> PipelineGPU:
        """Add vibrance adjustment.

        :param factor: Vibrance factor (1.0 = no change)
        :returns: Self for chaining
        """
        self._color_pipeline.vibrance(factor)
        self._operation_order.append("color")
        return self

    def hue_shift(self, degrees: float = 0.0) -> PipelineGPU:
        """Add hue shift.

        :param degrees: Hue shift in degrees
        :returns: Self for chaining
        """
        self._color_pipeline.hue_shift(degrees)
        self._operation_order.append("color")
        return self

    def shadows(self, factor: float = 0.0) -> PipelineGPU:
        """Adjust shadows.

        :param factor: Shadow adjustment (-1.0 to 1.0)
        :returns: Self for chaining
        """
        self._color_pipeline.shadows(factor)
        self._operation_order.append("color")
        return self

    def highlights(self, factor: float = 0.0) -> PipelineGPU:
        """Adjust highlights.

        :param factor: Highlight adjustment (-1.0 to 1.0)
        :returns: Self for chaining
        """
        self._color_pipeline.highlights(factor)
        self._operation_order.append("color")
        return self

    def color_preset(self, name: str, strength: float = 1.0) -> PipelineGPU:
        """Apply color preset.

        :param name: Preset name
        :param strength: Preset strength (0.0 to 1.0)
        :returns: Self for chaining
        """
        self._color_pipeline.preset(name, strength)
        self._operation_order.append("color")
        return self

    # ==========================================================================
    # Transform Operations
    # ==========================================================================

    def translate(self, translation: list[float] | np.ndarray | torch.Tensor) -> PipelineGPU:
        """Add translation.

        :param translation: Translation vector [x, y, z]
        :returns: Self for chaining
        """
        self._transform_pipeline.translate(translation)
        self._operation_order.append("transform")
        return self

    def scale(self, scale: float | list[float] | np.ndarray | torch.Tensor) -> PipelineGPU:
        """Add scaling.

        :param scale: Uniform scale or [sx, sy, sz]
        :returns: Self for chaining
        """
        self._transform_pipeline.scale(scale)
        self._operation_order.append("transform")
        return self

    def rotate_quaternion(self, quaternion: np.ndarray | torch.Tensor) -> PipelineGPU:
        """Add rotation by quaternion.

        :param quaternion: Rotation quaternion [w, x, y, z]
        :returns: Self for chaining
        """
        self._transform_pipeline.rotate_quaternion(quaternion)
        self._operation_order.append("transform")
        return self

    def rotate_euler(self, angles: list[float] | np.ndarray | torch.Tensor, order: str = "XYZ") -> PipelineGPU:
        """Add rotation by Euler angles.

        :param angles: Euler angles [x, y, z] in radians
        :param order: Rotation order
        :returns: Self for chaining
        """
        self._transform_pipeline.rotate_euler(angles, order)
        self._operation_order.append("transform")
        return self

    def rotate_axis_angle(self, axis: list[float] | np.ndarray | torch.Tensor, angle: float) -> PipelineGPU:
        """Add rotation around axis.

        :param axis: Rotation axis [x, y, z]
        :param angle: Rotation angle in radians
        :returns: Self for chaining
        """
        self._transform_pipeline.rotate_axis_angle(axis, angle)
        self._operation_order.append("transform")
        return self

    def transform_matrix(self, matrix: np.ndarray | torch.Tensor) -> PipelineGPU:
        """Add 4x4 transformation matrix.

        :param matrix: 4x4 homogeneous transformation matrix
        :returns: Self for chaining
        """
        self._transform_pipeline.transform_matrix(matrix)
        self._operation_order.append("transform")
        return self

    def center_at_origin(self) -> PipelineGPU:
        """Add centering at origin.

        :returns: Self for chaining
        """
        self._transform_pipeline.center_at_origin()
        self._operation_order.append("transform")
        return self

    def normalize_scale(self, target_size: float = 2.0) -> PipelineGPU:
        """Add scale normalization.

        :param target_size: Target bounding box size
        :returns: Self for chaining
        """
        self._transform_pipeline.normalize_scale(target_size)
        self._operation_order.append("transform")
        return self

    # ==========================================================================
    # Filter Operations
    # ==========================================================================

    def within_sphere(self, center: list[float] | np.ndarray | torch.Tensor = None, radius: float = 1.0) -> PipelineGPU:
        """Add spherical region filter.

        :param center: Sphere center [x, y, z] (default: origin)
        :param radius: Absolute radius in world units
        :returns: Self for chaining
        """
        self._filter_pipeline.within_sphere(center, radius)
        self._operation_order.append("filter")
        return self

    def within_box(self, min_bounds: list[float] | np.ndarray | torch.Tensor,
                   max_bounds: list[float] | np.ndarray | torch.Tensor) -> PipelineGPU:
        """Add box filter.

        :param min_bounds: Minimum bounds [x, y, z]
        :param max_bounds: Maximum bounds [x, y, z]
        :returns: Self for chaining
        """
        self._filter_pipeline.within_box(min_bounds, max_bounds)
        self._operation_order.append("filter")
        return self

    def min_opacity(self, threshold: float = 0.1) -> PipelineGPU:
        """Add minimum opacity filter.

        :param threshold: Minimum opacity threshold
        :returns: Self for chaining
        """
        self._filter_pipeline.min_opacity(threshold)
        self._operation_order.append("filter")
        return self

    def max_opacity(self, threshold: float = 0.99) -> PipelineGPU:
        """Add maximum opacity filter.

        :param threshold: Maximum opacity threshold
        :returns: Self for chaining
        """
        self._filter_pipeline.max_opacity(threshold)
        self._operation_order.append("filter")
        return self

    def min_scale(self, threshold: float = 0.001) -> PipelineGPU:
        """Add minimum scale filter.

        :param threshold: Minimum scale threshold
        :returns: Self for chaining
        """
        self._filter_pipeline.min_scale(threshold)
        self._operation_order.append("filter")
        return self

    def max_scale(self, threshold: float = 0.1) -> PipelineGPU:
        """Add maximum scale filter.

        :param threshold: Maximum scale threshold
        :returns: Self for chaining
        """
        self._filter_pipeline.max_scale(threshold)
        self._operation_order.append("filter")
        return self

    # ==========================================================================
    # Execution
    # ==========================================================================

    def __call__(self, data: GSTensorPro, inplace: bool = True, filter_mode: str = "and") -> GSTensorPro:
        """Apply pipeline to GSTensorPro.

        :param data: GSTensorPro object to process
        :param inplace: If True, modify data in-place; if False, create copy
        :param filter_mode: Filter combination mode ("and" or "or")
        :returns: Processed GSTensorPro

        Example:
            >>> pipeline = (
            ...     PipelineGPU()
            ...     .within_sphere(1.0)
            ...     .translate([1, 0, 0])
            ...     .brightness(1.2)
            ... )
            >>> result = pipeline(gstensor, inplace=True)
        """
        if not isinstance(data, GSTensorPro):
            raise TypeError(f"Expected GSTensorPro, got {type(data)}")

        # Create copy if not inplace
        if not inplace:
            data = data.clone()
            # Deep copy _format since clone() may do shallow copy
            if hasattr(data, '_format'):
                data._format = data._format.copy()

        # Get unique operation types in order
        seen = set()
        operation_sequence = []
        for op_type in self._operation_order:
            if op_type not in seen:
                operation_sequence.append(op_type)
                seen.add(op_type)

        # Apply operations in order of first appearance
        for op_type in operation_sequence:
            if op_type == "filter":
                # Apply filter
                if self._filter_pipeline._operations:
                    data = self._filter_pipeline(data, mode=filter_mode, inplace=True)
            elif op_type == "transform":
                # Apply transforms
                if self._transform_pipeline._operations:
                    data = self._transform_pipeline(data, inplace=True)
            elif op_type == "color":
                # Apply color adjustments
                if self._color_pipeline._operations:
                    data = self._color_pipeline(data, inplace=True)

        return data

    def reset(self) -> PipelineGPU:
        """Reset pipeline, removing all operations.

        :returns: Self for chaining
        """
        self._color_pipeline.reset()
        self._transform_pipeline.reset()
        self._filter_pipeline.reset()
        self._operation_order = []
        return self

    def clone(self) -> PipelineGPU:
        """Create a copy of this pipeline.

        :returns: New PipelineGPU with same operations
        """
        new_pipeline = PipelineGPU()
        new_pipeline._color_pipeline = self._color_pipeline.clone()
        new_pipeline._transform_pipeline = self._transform_pipeline.clone()
        new_pipeline._filter_pipeline = self._filter_pipeline.clone()
        new_pipeline._operation_order = self._operation_order.copy()
        return new_pipeline