"""Unified CPU Pipeline class matching GPU PipelineGPU interface.

Provides a fluent API for chaining color, transform, and filter operations.

Example:
    >>> from gsmod import Pipeline, GSDataPro
    >>>
    >>> # Create and configure pipeline
    >>> pipe = (Pipeline()
    ...     .brightness(1.2)
    ...     .saturation(1.3)
    ...     .translate([1, 0, 0])
    ...     .scale(2.0)
    ...     .min_opacity(0.1))
    >>>
    >>> # Apply to data
    >>> data = GSDataPro.from_ply("scene.ply")
    >>> result = pipe(data, inplace=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gsmod.gsdata_pro import GSDataPro

from gsmod.config.values import ColorValues, FilterValues, TransformValues
from gsmod.filter.atomic import Filter


@dataclass
class Pipeline:
    """Unified CPU pipeline for color, transform, and filter operations.

    Operations are accumulated and executed in order when called.
    Supports method chaining for fluent API.
    """

    _operations: list[tuple[str, Any]] = field(default_factory=list)

    # ========================================================================
    # Color Methods
    # ========================================================================

    def brightness(self, factor: float) -> Pipeline:
        """Add brightness adjustment.

        :param factor: Brightness multiplier (1.0 = no change)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(brightness=factor)))
        return self

    def contrast(self, factor: float) -> Pipeline:
        """Add contrast adjustment.

        :param factor: Contrast multiplier (1.0 = no change)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(contrast=factor)))
        return self

    def saturation(self, factor: float) -> Pipeline:
        """Add saturation adjustment.

        :param factor: Saturation multiplier (1.0 = no change)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(saturation=factor)))
        return self

    def gamma(self, value: float) -> Pipeline:
        """Add gamma adjustment.

        :param value: Gamma value (1.0 = linear)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(gamma=value)))
        return self

    def temperature(self, temp: float) -> Pipeline:
        """Add temperature adjustment.

        :param temp: Temperature (-1.0 cool to 1.0 warm, 0.0 = neutral)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(temperature=temp)))
        return self

    def vibrance(self, factor: float) -> Pipeline:
        """Add vibrance adjustment.

        :param factor: Vibrance multiplier (1.0 = no change)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(vibrance=factor)))
        return self

    def hue_shift(self, degrees: float) -> Pipeline:
        """Add hue shift.

        :param degrees: Hue shift in degrees (-180 to 180)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(hue_shift=degrees)))
        return self

    def shadows(self, factor: float) -> Pipeline:
        """Add shadow adjustment.

        :param factor: Shadow adjustment (-1.0 to 1.0, 0.0 = no change)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(shadows=factor)))
        return self

    def highlights(self, factor: float) -> Pipeline:
        """Add highlight adjustment.

        :param factor: Highlight adjustment (-1.0 to 1.0, 0.0 = no change)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(highlights=factor)))
        return self

    def tint(self, value: float) -> Pipeline:
        """Add tint adjustment.

        :param value: Tint value (-1.0 green to 1.0 magenta)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(tint=value)))
        return self

    def fade(self, value: float) -> Pipeline:
        """Add fade effect.

        :param value: Fade amount (0.0 to 1.0)
        :returns: Self for chaining
        """
        self._operations.append(("color", ColorValues(fade=value)))
        return self

    def color(self, values: ColorValues) -> Pipeline:
        """Add color values directly.

        :param values: ColorValues to apply
        :returns: Self for chaining
        """
        self._operations.append(("color", values))
        return self

    # ========================================================================
    # Transform Methods
    # ========================================================================

    def translate(self, translation: list[float] | tuple[float, ...]) -> Pipeline:
        """Add translation.

        :param translation: Translation vector [x, y, z]
        :returns: Self for chaining
        """
        t = translation
        self._operations.append(("transform", TransformValues.from_translation(t[0], t[1], t[2])))
        return self

    def scale(self, factor: float | list[float] | tuple[float, ...]) -> Pipeline:
        """Add scale operation.

        :param factor: Uniform scale factor or [sx, sy, sz] for non-uniform
        :returns: Self for chaining
        """
        if isinstance(factor, int | float):
            self._operations.append(("transform", TransformValues.from_scale(factor)))
        else:
            # Non-uniform scale stored as special operation
            self._operations.append(("scale_nonuniform", tuple(factor)))
        return self

    def rotate_quaternion(self, quat: list[float] | tuple[float, ...]) -> Pipeline:
        """Add quaternion rotation.

        :param quat: Quaternion [w, x, y, z]
        :returns: Self for chaining
        """
        self._operations.append(("transform", TransformValues(rotation=tuple(quat))))
        return self

    def rotate_euler(self, angles: list[float] | tuple[float, ...], order: str = "XYZ") -> Pipeline:
        """Add Euler rotation.

        :param angles: Angles [rx, ry, rz] in degrees
        :param order: Rotation order (default 'XYZ')
        :returns: Self for chaining
        """
        a = angles
        self._operations.append(
            ("transform", TransformValues.from_rotation_euler(a[0], a[1], a[2]))
        )
        return self

    def rotate_axis_angle(self, axis: list[float] | tuple[float, ...], angle: float) -> Pipeline:
        """Add axis-angle rotation.

        :param axis: Rotation axis [x, y, z]
        :param angle: Rotation angle in degrees
        :returns: Self for chaining
        """
        self._operations.append(
            ("transform", TransformValues.from_rotation_axis_angle(tuple(axis), angle))
        )
        return self

    def center_at_origin(self) -> Pipeline:
        """Add center at origin operation.

        :returns: Self for chaining
        """
        self._operations.append(("center_at_origin", None))
        return self

    def normalize_scale(self, target_size: float = 2.0) -> Pipeline:
        """Add normalize scale operation.

        :param target_size: Target size for largest dimension
        :returns: Self for chaining
        """
        self._operations.append(("normalize_scale", target_size))
        return self

    def transform(self, values: TransformValues) -> Pipeline:
        """Add transform values directly.

        :param values: TransformValues to apply
        :returns: Self for chaining
        """
        self._operations.append(("transform", values))
        return self

    # ========================================================================
    # Filter Methods
    # ========================================================================

    def min_opacity(self, threshold: float) -> Pipeline:
        """Add minimum opacity filter.

        :param threshold: Minimum opacity value
        :returns: Self for chaining
        """
        self._operations.append(("filter", FilterValues(min_opacity=threshold)))
        return self

    def max_opacity(self, threshold: float) -> Pipeline:
        """Add maximum opacity filter.

        :param threshold: Maximum opacity value
        :returns: Self for chaining
        """
        self._operations.append(("filter", FilterValues(max_opacity=threshold)))
        return self

    def min_scale(self, threshold: float) -> Pipeline:
        """Add minimum scale filter.

        :param threshold: Minimum scale value
        :returns: Self for chaining
        """
        self._operations.append(("filter", FilterValues(min_scale=threshold)))
        return self

    def max_scale(self, threshold: float) -> Pipeline:
        """Add maximum scale filter.

        :param threshold: Maximum scale value
        :returns: Self for chaining
        """
        self._operations.append(("filter", FilterValues(max_scale=threshold)))
        return self

    def within_sphere(
        self, radius: float, center: tuple[float, float, float] | None = None
    ) -> Pipeline:
        """Add sphere inclusion filter.

        :param radius: Sphere radius
        :param center: Sphere center, defaults to origin
        :returns: Self for chaining
        """
        c = center if center is not None else (0.0, 0.0, 0.0)
        self._operations.append(("filter", FilterValues(sphere_radius=radius, sphere_center=c)))
        return self

    def outside_sphere(
        self, radius: float, center: tuple[float, float, float] | None = None
    ) -> Pipeline:
        """Add sphere exclusion filter.

        :param radius: Sphere radius
        :param center: Sphere center, defaults to origin
        :returns: Self for chaining
        """
        c = center if center is not None else (0.0, 0.0, 0.0)
        self._operations.append(
            ("filter", FilterValues(sphere_radius=radius, sphere_center=c, invert=True))
        )
        return self

    def within_box(
        self,
        min_corner: tuple[float, float, float],
        max_corner: tuple[float, float, float],
        rotation: tuple[float, float, float] | None = None,
    ) -> Pipeline:
        """Add box inclusion filter.

        :param min_corner: Box minimum corner
        :param max_corner: Box maximum corner
        :param rotation: Optional axis-angle rotation
        :returns: Self for chaining
        """
        self._operations.append(
            ("filter", FilterValues(box_min=min_corner, box_max=max_corner, box_rot=rotation))
        )
        return self

    def outside_box(
        self,
        min_corner: tuple[float, float, float],
        max_corner: tuple[float, float, float],
        rotation: tuple[float, float, float] | None = None,
    ) -> Pipeline:
        """Add box exclusion filter.

        :param min_corner: Box minimum corner
        :param max_corner: Box maximum corner
        :param rotation: Optional axis-angle rotation
        :returns: Self for chaining
        """
        self._operations.append(
            (
                "filter",
                FilterValues(box_min=min_corner, box_max=max_corner, box_rot=rotation, invert=True),
            )
        )
        return self

    def filter(self, values: FilterValues | Filter) -> Pipeline:
        """Add filter values or Filter directly.

        :param values: FilterValues or Filter to apply
        :returns: Self for chaining
        """
        if isinstance(values, Filter):
            self._operations.append(("filter_atomic", values))
        else:
            self._operations.append(("filter", values))
        return self

    # ========================================================================
    # Execution
    # ========================================================================

    def _merge_operations(self) -> list[tuple[str, Any]]:
        """Merge consecutive operations of the same type for optimization.

        Consecutive color, transform, and filter operations are merged
        using their `+` operator to reduce the number of kernel calls.

        Note:
            Color merging uses mathematically correct composition rules
            (multiply for brightness/contrast/etc, add for temperature/shadows).
            Due to LUT quantization, merged results may differ from sequential
            application by up to ~2% - this is expected and acceptable.

        :returns: Optimized list of operations
        """
        if not self._operations:
            return []

        merged: list[tuple[str, Any]] = []
        current_type = None
        current_value = None

        for op_type, op_value in self._operations:
            # Types that can be merged
            if op_type in ("color", "transform", "filter"):
                if current_type == op_type:
                    # Merge with current
                    current_value = current_value + op_value
                else:
                    # Flush previous and start new
                    if current_type is not None:
                        merged.append((current_type, current_value))
                    current_type = op_type
                    current_value = op_value
            else:
                # Non-mergeable operation - flush and add
                if current_type is not None:
                    merged.append((current_type, current_value))
                    current_type = None
                    current_value = None
                merged.append((op_type, op_value))

        # Flush remaining
        if current_type is not None:
            merged.append((current_type, current_value))

        return merged

    def __call__(self, data: GSDataPro, inplace: bool = True) -> GSDataPro:
        """Execute pipeline on data.

        Consecutive operations of the same type are automatically merged
        for optimal performance.

        :param data: GSDataPro to process
        :param inplace: If True, modify in place; if False, work on copy
        :returns: Processed GSDataPro
        """
        if not inplace:
            data = data.clone()

        # Merge consecutive operations for performance
        merged_ops = self._merge_operations()

        for op_type, op_value in merged_ops:
            if op_type == "color":
                data.color(op_value, inplace=True)
            elif op_type == "transform":
                data.transform(op_value, inplace=True)
            elif op_type == "filter":
                data.filter(op_value, inplace=True)
            elif op_type == "filter_atomic":
                op_value(data, inplace=True)
            elif op_type == "scale_nonuniform":
                data.scale_nonuniform(op_value, inplace=True)
            elif op_type == "center_at_origin":
                data.center_at_origin(inplace=True)
            elif op_type == "normalize_scale":
                data.normalize_scale(target_size=op_value, inplace=True)

        return data

    def reset(self) -> Pipeline:
        """Clear all operations.

        :returns: Self for chaining
        """
        self._operations.clear()
        return self

    def clone(self) -> Pipeline:
        """Create a copy of the pipeline.

        :returns: New Pipeline with same operations
        """
        new_pipe = Pipeline()
        new_pipe._operations = list(self._operations)
        return new_pipe

    def __len__(self) -> int:
        """Return number of operations."""
        return len(self._operations)

    def __repr__(self) -> str:
        """Return string representation."""
        ops = [f"{op_type}" for op_type, _ in self._operations]
        return f"Pipeline([{', '.join(ops)}])"
