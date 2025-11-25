"""
Protocol definitions for gsmod pipeline interfaces.

Defines common interfaces for pipeline stages, format-aware operations,
and unified processing across CPU (NumPy) and GPU (PyTorch).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from gsply import GSData
    from gsply.torch import GSTensor

# Type variable for generic data containers
T = TypeVar("T", bound="GSData | GSTensor")


@runtime_checkable
class PipelineStage(Protocol[T]):
    """
    Protocol for pipeline stages (Color, Transform, Filter).

    All pipeline stages must implement this interface for compatibility
    with the unified Pipeline class. The generic type T allows stages
    to work with both GSData (CPU) and GSTensor (GPU) data.
    """

    def apply(self, data: T, inplace: bool = True) -> T:
        """
        Apply pipeline operations to data.

        :param data: GSData or GSTensor object to process
        :param inplace: If True, modify data in-place; if False, create copy
        :returns: Processed data object (same type as input)
        """
        ...

    def reset(self) -> None:
        """Reset pipeline to initial state (no operations)."""
        ...

    def is_neutral(self) -> bool:
        """Check if pipeline has no operations (identity transform)."""
        ...


@runtime_checkable
class ColorProcessor(Protocol[T]):
    """Protocol for color processing operations."""

    def apply(self, data: T, inplace: bool = True) -> T:
        """Apply color operations to data."""
        ...

    def brightness(self, factor: float) -> ColorProcessor[T]:
        """Add brightness adjustment."""
        ...

    def contrast(self, factor: float) -> ColorProcessor[T]:
        """Add contrast adjustment."""
        ...

    def saturation(self, factor: float) -> ColorProcessor[T]:
        """Add saturation adjustment."""
        ...

    def reset(self) -> ColorProcessor[T]:
        """Reset pipeline."""
        ...


@runtime_checkable
class TransformProcessor(Protocol[T]):
    """Protocol for transform processing operations."""

    def apply(self, data: T, inplace: bool = True) -> T:
        """Apply transform operations to data."""
        ...

    def translate(self, translation: list[float]) -> TransformProcessor[T]:
        """Add translation."""
        ...

    def scale(self, factor: float | list[float]) -> TransformProcessor[T]:
        """Add scaling."""
        ...

    def rotate_quaternion(self, quaternion: list[float]) -> TransformProcessor[T]:
        """Add rotation by quaternion."""
        ...

    def reset(self) -> TransformProcessor[T]:
        """Reset pipeline."""
        ...


@runtime_checkable
class FilterProcessor(Protocol[T]):
    """Protocol for filter processing operations."""

    def compute_mask(self, data: T) -> bool | object:
        """Compute filter mask for data."""
        ...

    def apply(self, data: T, inplace: bool = False) -> T:
        """Apply filter to data."""
        ...

    def within_sphere(self, center: list[float] | None, radius: float) -> FilterProcessor[T]:
        """Add spherical region filter."""
        ...

    def min_opacity(self, threshold: float) -> FilterProcessor[T]:
        """Add minimum opacity filter."""
        ...

    def max_scale(self, threshold: float) -> FilterProcessor[T]:
        """Add maximum scale filter."""
        ...

    def reset(self) -> FilterProcessor[T]:
        """Reset pipeline."""
        ...
